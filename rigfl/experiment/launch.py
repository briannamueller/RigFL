"""Expand an experiment into independently runnable task configurations.

Sweep axes form a Cartesian product; zipped replicates do not. Fixed settings
belong under ``base``; ``launch`` writes one configuration per task to
``grid.jsonl``.

    rigfl sweep configs/experiments/cifar10_sweep.yaml
"""

from __future__ import annotations

import argparse
import difflib
import itertools
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from pydantic import ValidationError

from rigfl.cli import legacy_config_argv
from rigfl.data.config import (
    BioSiloDatasetSettings,
    dataset_settings,
    inactive_replicate_seed_fields,
)
from rigfl.experiment.artifacts import (
    ResultValidationError,
    atomic_write_json,
    atomic_write_text,
    existing_result_decision,
    read_json,
    validate_run_record,
    write_run_record,
)
from rigfl.experiment.config import (
    ExperimentConfig,
    ExperimentFileConfig,
    ResolvedExperimentConfig,
    result_filename,
)
from rigfl.experiment.device import resolve_device
from rigfl.experiment.env import capture_env
from rigfl.experiment.paths import (
    filter_for_model,
    flatten_mapping,
    model_has_path,
    model_paths,
    nested_set,
)
from rigfl.experiment.registry import (
    ALL_ALGORITHMS,
    algorithm_run_fingerprint,
    config_class,
    ignored_experiment_fields,
    ignores_experiment_field,
    resolve_algorithm_config,
    resolve_algorithm_experiment,
)
from rigfl.experiment.run import resolve_experiment_data, run_one
from rigfl.experiment.storage import (
    GRID_TASK_KIND,
    GRID_TASK_SCHEMA_VERSION,
    SUBMISSION_FILE,
    SUBMISSION_KIND,
    SUBMISSION_SCHEMA_VERSION,
    read_grid_tasks,
    run_store,
    study_directory,
    validate_grid_task,
)
from rigfl.experiment.tuning import canonical_axis


def _values(spec) -> list:
    """A sweep axis value: a list stays a list; '0-2'/'0.1,0.5' expands to a list."""
    if isinstance(spec, (list, tuple)):
        return list(spec)
    out: list = []
    for part in str(spec).split(","):
        part = part.strip()
        if "-" in part and part.replace("-", "").isdigit():      # integer range 'a-b'
            lo, hi = part.split("-")
            out.extend(range(int(lo), int(hi) + 1))
        elif part:
            out.append(part)
    return out


def _suggest(name: str, known) -> str:
    close = difflib.get_close_matches(name, sorted(known), n=1)
    return f'\nDid you mean "{close[0]}"?' if close else ""


def _validate_axes(exp_axes: dict, algorithm_axes: dict, algorithms: list[str],
                   base_algorithm: dict) -> None:
    """Validate axes and fixed settings against the selected algorithms."""
    known_algorithm: set[str] = set()
    for m in algorithms:
        known_algorithm |= model_paths(config_class(m))
    known_exp = model_paths(ExperimentConfig)

    for field in algorithm_axes:
        if field not in known_algorithm:
            raise SystemExit(
                f'Unknown sweep axis: algorithm.{field}\n'
                f'No selected algorithm ({", ".join(algorithms)}) defines "{field}".'
                f'{_suggest(field, known_algorithm)}\n\n'
                f'Known algorithm fields: {", ".join(sorted(known_algorithm))}')

    for field in exp_axes:
        if field not in known_exp:
            raise SystemExit(
                f'Unknown sweep axis: experiment.{field}\n'
                f'ExperimentConfig has no field "{field}".'
                f'{_suggest(field, known_exp)}\n\n'
                f'Known experiment fields: {", ".join(sorted(known_exp))}')
        if not any(
            not ignores_experiment_field(name, field)
            for name in algorithms
        ):
            raise SystemExit(
                f"Sweep axis experiment.{field} does not apply to any selected algorithm."
            )

    # Fixed algorithm settings follow the same validation as algorithm axes.
    for field in flatten_mapping(base_algorithm):
        if field not in known_algorithm:
            raise SystemExit(
                f'Unknown algorithm setting in base.algorithm: {field}\n'
                f'No selected algorithm ({", ".join(algorithms)}) defines "{field}".'
                f'{_suggest(field, known_algorithm)}\n\n'
                f'Known algorithm fields: {", ".join(sorted(known_algorithm))}')


_REPLICATE_PATHS = {
    "partition_seed": "experiment.partition_seed",
    "split_seed": "experiment.split_seed",
    "experiment_seed": "experiment.seed",
}


def _replicate_conditions(
    spec: dict | ExperimentFileConfig,
) -> tuple[list[dict[str, int]], str | None]:
    """Return explicitly paired data and training seed conditions."""
    parsed = spec if isinstance(spec, ExperimentFileConfig) else _validate_spec(spec)
    paired = parsed.replicates
    if paired is None:
        return [], None
    return [condition.model_dump() for condition in paired], "zipped"


def _apply_replicate(experiment: dict, condition: dict[str, int]) -> None:
    experiment["partition_seed"] = condition["partition_seed"]
    experiment["split_seed"] = condition["split_seed"]
    experiment["seed"] = condition["experiment_seed"]


def _validate_spec(spec: dict) -> ExperimentFileConfig:
    if not isinstance(spec, dict):
        raise SystemExit(f"a sweep config must be a mapping, got {type(spec).__name__}")
    known = tuple(ExperimentFileConfig.model_fields)
    unknown = sorted(set(spec) - set(known))
    if unknown:
        raise SystemExit(
            f"Unknown top-level key(s) in the sweep config: {', '.join(unknown)}"
            f"{_suggest(unknown[0], known)}\n\n"
            f"Known: {', '.join(known)}")
    try:
        return ExperimentFileConfig.model_validate(spec)
    except ValidationError as error:
        first = error.errors(include_url=False)[0]
        location = ".".join(str(part) for part in first["loc"])
        value = spec.get(first["loc"][0]) if first["loc"] else spec
        if first["type"] == "dict_type" and first["loc"]:
            raise SystemExit(
                f"sweep config: '{location}' must be a mapping, "
                f"got {type(value).__name__}"
            ) from error
        raise SystemExit(f"invalid sweep config at {location}: {first['msg']}") from error


def _validate_tasks(grid: list[dict]) -> None:
    """Validate every generated task before submission."""
    for i, task in enumerate(grid, 1):
        try:
            exp = ExperimentConfig(**task["experiment"])
            cfg = config_class(task["algorithm"])(**task["algorithm_config"])
            resolve_algorithm_config(task["algorithm"], exp, cfg)
        except Exception as e:
            raise SystemExit(
                f"task {i} ({task['algorithm']}) does not validate: {e}\n\n"
                f"Fix the sweep config -- this would otherwise fail on a compute "
                f"node, once per task.")


def _validate_replicate_effects(grid: list[dict]) -> None:
    by_dataset: dict[tuple[str, str], list[dict]] = {}
    for task in grid:
        experiment = task["experiment"]
        key = (
            experiment.get("dataset", "cifar10"),
            experiment.get("dataset_config", "configs/datasets.yaml"),
        )
        by_dataset.setdefault(key, []).append(experiment)

    for (dataset, config_path), experiments in by_dataset.items():
        varied = {
            field
            for field in ("partition_seed", "split_seed")
            if len({experiment.get(field) for experiment in experiments}) > 1
        }
        if not varied:
            continue
        inactive = inactive_replicate_seed_fields(
            _dataset_settings_or_exit(dataset, config_path)
        )
        ineffective = sorted(inactive & varied)
        if ineffective:
            raise SystemExit(
                f"dataset {dataset!r} does not use the varied replicate field(s): "
                + ", ".join(ineffective)
            )


def _dataset_settings_or_exit(dataset: str, config_path: str):
    try:
        return dataset_settings(dataset, config_path)
    except (FileNotFoundError, KeyError, ValueError) as error:
        raise SystemExit(
            f"dataset {dataset!r} does not validate against {config_path}: {error}"
        ) from error


def _validate_biosilo_partitions(grid: list[dict]) -> None:
    """Require every BioSilo partition before a sweep is written or submitted."""
    from rigfl.data.biosilo import load_biosilo_partition

    checked = set()
    missing = []
    for task in grid:
        experiment = ExperimentConfig(**task["experiment"])
        key = (
            experiment.dataset,
            experiment.dataset_config,
            experiment.data_dir,
            experiment.partition_seed,
        )
        if key in checked:
            continue
        checked.add(key)
        settings = _dataset_settings_or_exit(
            experiment.dataset, experiment.dataset_config
        )
        if not isinstance(settings, BioSiloDatasetSettings):
            continue
        if experiment.partition_seed is not None:
            settings = settings.model_copy(
                update={
                    "parameters": {
                        **settings.parameters,
                        "seed": experiment.partition_seed,
                    }
                }
            )
        try:
            load_biosilo_partition(
                settings,
                data_dir=experiment.data_dir,
                dataset_name=experiment.dataset,
                dataset_config=experiment.dataset_config,
                partition_seed_override=experiment.partition_seed,
            )
        except FileNotFoundError as error:
            missing.append(str(error))
    if missing:
        raise SystemExit(
            "BioSilo partition preflight failed:\n\n" + "\n\n".join(missing)
        )


def build_grid(spec: dict) -> list[dict]:
    """Expand a sweep spec into a flat list of per-task configs.

    Each task = {algorithm, experiment: {...}, algorithm_config: {...}}. Axis keys
    use 'experiment.x' or 'algorithm.x'.

    An ``algorithm.x`` axis only multiplies the grid for algorithms that actually
    have field ``x``; for algorithms without it, that axis collapses to a single
    entry. A sweep over ``algorithm.mu`` therefore gives FedProx one task per
    value and every other algorithm exactly one -- no duplicate configurations
    and no manual per-algorithm scoping."""
    return expand(spec)[0]


def expand(spec: dict) -> tuple[list[dict], dict | None]:
    """Expand an ordinary sweep into independent run configurations."""
    if isinstance(spec, dict) and "tuning" in spec:
        raise SystemExit(
            "ordinary sweeps do not perform hyperparameter tuning; remove the "
            "tuning section or run the configuration with "
            "rigfl hpo"
        )
    parsed = _validate_spec(spec)
    replicate_conditions, _ = _replicate_conditions(parsed)
    spec = parsed.model_dump(by_alias=True, exclude_unset=True)
    base = spec.get("base") or {}
    base_exp = dict(base.get("experiment", base))            # allow flat base = experiment fields
    base_exp.pop("algorithm", None)
    base_algorithm = dict(base.get("algorithm", {}))

    sweep = {k: _values(v) for k, v in (spec.get("sweep") or {}).items()}
    declared_algorithms = sweep.pop("algorithm", None) or spec.get("algorithms")
    if not declared_algorithms:
        raise SystemExit(
            "sweep config must declare 'algorithms' explicitly; "
            f"known: {', '.join(ALL_ALGORITHMS)}"
        )
    algorithms = _values(declared_algorithms)

    if replicate_conditions:
        replicate_paths = set(_REPLICATE_PATHS.values())
        overlap = sorted(
            replicate_paths & {canonical_axis(path) for path in sweep}
        )
        if overlap:
            raise SystemExit(
                "seed fields declared under replicates cannot also "
                "appear under sweep: " + ", ".join(overlap)
            )

    exp_axes: dict[str, list] = {}
    algorithm_axes: dict[str, list] = {}
    declared_axes: set[str] = set()
    for path, vals in sweep.items():
        axis, resolved_values = canonical_axis(path), vals
        if axis in declared_axes:
            raise SystemExit(
                f"Sweep axes {path!r} and an earlier declaration both resolve to "
                f"{axis!r}; declare that setting only once.")
        declared_axes.add(axis)
        if axis.startswith("algorithm."):
            algorithm_axes[axis[len("algorithm."):]] = resolved_values
        elif axis.startswith("experiment."):
            exp_axes[axis[len("experiment."):]] = resolved_values
        else:
            raise SystemExit(f"Internal error: non-canonical sweep axis {axis!r}.")

    _validate_axes(exp_axes, algorithm_axes, algorithms, base_algorithm)

    grid: list[dict] = []
    for algorithm in algorithms:
        config_model = config_class(algorithm)
        # only the algorithm-axes this algorithm has; the rest don't multiply its grid
        m_axes = {
            k: v for k, v in algorithm_axes.items()
            if model_has_path(config_model, k)
        }
        axes = {
            f"experiment::{k}": v
            for k, v in exp_axes.items()
            if not ignores_experiment_field(algorithm, k)
        }
        axes.update({f"algorithm::{k}": v for k, v in m_axes.items()})
        keys = list(axes)
        for combo in itertools.product(*(axes[k] for k in keys)):   # () once when no axes
            for replicate in replicate_conditions or [None]:
                exp = dict(base_exp)
                for field in ignored_experiment_fields(algorithm):
                    exp.pop(field, None)
                # Fixed algorithm settings obey the same per-algorithm scoping as algorithm
                # axes: validate against the selected-algorithm union above, then apply
                # only settings this algorithm's configuration class actually defines.
                mcfg = filter_for_model(base_algorithm, config_model)
                for key, val in zip(keys, combo):
                    kind, field = key.split("::", 1)
                    if kind == "experiment":
                        nested_set(exp, field, val)
                    else:
                        nested_set(mcfg, field, val)
                if replicate is not None:
                    _apply_replicate(exp, replicate)
                task = {
                    "algorithm": algorithm,
                    "experiment": exp,
                    "algorithm_config": mcfg,
                }
                grid.append(task)

    _validate_replicate_effects(grid)
    _validate_tasks(grid)

    return grid, None


def _check_grid(text: str, expected: int) -> None:
    """Every line of the written grid parses, and none was lost."""
    lines = text.splitlines()
    if len(lines) != expected:
        raise ValueError(f"grid holds {len(lines)} task(s), expected {expected}")
    for i, line in enumerate(lines, 1):
        task = json.loads(line)
        validate_grid_task(task, source=f"grid line {i}")


def _write_grid(path: Path, grid: list[dict]) -> bool:
    """Write or replace the working grid, reusing identical contents."""
    records = [
        {
            **task,
            "kind": GRID_TASK_KIND,
            "schema_version": GRID_TASK_SCHEMA_VERSION,
        }
        for task in grid
    ]
    body = "".join(json.dumps(task) + "\n" for task in records)
    if path.exists():
        try:
            existing = path.read_text()
            _check_grid(existing, len(existing.splitlines()))
        except (OSError, ValueError, json.JSONDecodeError) as error:
            raise SystemExit(
                f"cannot replace existing grid {path}: {error}"
            ) from error
        if existing == body:
            return False
    atomic_write_text(
        path,
        body,
        validate=lambda text: _check_grid(text, len(grid)),
    )
    return True


def _difference(value, base):
    """Nested values that differ from a shared submission base."""
    if not isinstance(value, dict) or not isinstance(base, dict):
        return value
    changed = {}
    for key, item in value.items():
        if key not in base:
            changed[key] = item
            continue
        if isinstance(item, dict) and isinstance(base[key], dict):
            nested = _difference(item, base[key])
            if nested:
                changed[key] = nested
        elif item != base[key]:
            changed[key] = item
    return changed


def _resolve_task(task: dict, *, data_cache: dict | None = None):
    """Resolve one declared task to the complete configuration it will execute."""
    name = task["algorithm"]
    experiment_values = task["experiment"]
    declared = ExperimentConfig(
        **{
            key: value
            for key, value in experiment_values.items()
            if key in ExperimentConfig.model_fields
        }
    )
    Cfg = config_class(name)
    unknown = sorted(
        set(flatten_mapping(task["algorithm_config"])) - model_paths(Cfg)
    )
    if unknown:
        raise ValueError(
            f"unknown {name} algorithm setting(s): {', '.join(unknown)}; "
            f"known: {', '.join(sorted(model_paths(Cfg)))}"
        )
    cache_key = json.dumps(declared.model_dump(mode="json"), sort_keys=True)
    cached = data_cache.get(cache_key) if data_cache is not None else None
    if cached is None:
        actual, data = resolve_experiment_data(declared)
        if data_cache is not None:
            data_cache[cache_key] = (actual, data)
    else:
        actual, data = cached
    actual = resolve_algorithm_experiment(name, actual)
    if "partition_id" in experiment_values:
        exp = resolve_algorithm_experiment(
            name, ResolvedExperimentConfig(**experiment_values)
        )
        if exp.model_dump(mode="json") != actual.model_dump(mode="json"):
            raise ValueError(
                "saved submission configuration no longer resolves to the same "
                "dataset partition or model assignment"
            )
    else:
        exp = actual
    cfg = resolve_algorithm_config(name, exp, Cfg(**task["algorithm_config"]))
    return name, exp, cfg, data


def _materialize_submission(
    grid: list[dict], submission_dir: Path, results_root: str | Path
) -> Path:
    """Write a resolved, immutable submission grid."""
    algorithms = sorted({task["algorithm"] for task in grid})
    experiment_defaults = ExperimentConfig().model_dump(mode="json")
    algorithm_defaults = {
        name: config_class(name)().model_dump(mode="json") for name in algorithms
    }
    output_dir = run_store(results_root)
    resolved_tasks = []
    data_cache = {}
    for index, task in enumerate(grid, 1):
        try:
            name, exp, cfg, _ = _resolve_task(task, data_cache=data_cache)
        except Exception as error:
            raise SystemExit(
                f"task {index} ({task.get('algorithm')}): cannot resolve submission: "
                f"{error}"
            ) from error
        experiment = exp.model_dump(mode="json")
        algorithm_config = cfg.model_dump(mode="json")
        fp = algorithm_run_fingerprint(name, exp, algorithm_config)
        saved_task = {
            "kind": GRID_TASK_KIND,
            "schema_version": GRID_TASK_SCHEMA_VERSION,
            "algorithm": name,
            "experiment": _difference(experiment, experiment_defaults),
            "algorithm_config": _difference(
                algorithm_config, algorithm_defaults[name]
            ),
            "run_fingerprint": fp,
            "result_file": result_filename(exp, name, fp),
        }
        resolved_tasks.append(saved_task)

    metadata = {
        "kind": SUBMISSION_KIND,
        "schema_version": SUBMISSION_SCHEMA_VERSION,
        "experiment_defaults": experiment_defaults,
        "algorithm_defaults": algorithm_defaults,
        "submission_provenance": capture_env(),
        "results_store": str(output_dir),
    }
    atomic_write_json(submission_dir / SUBMISSION_FILE, metadata)
    snapshot = submission_dir / "grid.jsonl"
    body = "".join(json.dumps(task) + "\n" for task in resolved_tasks)
    atomic_write_text(
        snapshot,
        body,
        validate=lambda text: _check_grid(text, len(resolved_tasks)),
    )
    return snapshot


def stage_task_snapshot(
    grid_path: Path, results_root: str | Path | None = None
) -> Path:
    """Resolve and copy a grid so external tasks cannot observe later edits."""
    try:
        body = grid_path.read_text()
        task_count = len(body.splitlines())
        _check_grid(body, task_count)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise SystemExit(f"cannot stage grid {grid_path}: {error}") from error

    submissions = grid_path.parent / "submissions"
    submissions.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    submission_dir = Path(
        tempfile.mkdtemp(prefix=f"{timestamp}-", dir=submissions)
    )
    if results_root is not None:
        grid = [json.loads(line) for line in body.splitlines() if line]
        return _materialize_submission(grid, submission_dir, results_root)

    snapshot = submission_dir / "grid.jsonl"
    atomic_write_text(
        snapshot,
        body,
        validate=lambda text: _check_grid(text, task_count),
    )
    return snapshot


def run_task(grid_path: str, task_id: int, out_dir: Path,
             dry_run: bool = False, force: bool = False) -> None:
    """Run the 1-indexed task from a grid file and save its result."""
    try:
        tasks = read_grid_tasks(grid_path)
    except ValueError as error:
        raise SystemExit(error) from error
    if not 1 <= task_id <= len(tasks):
        raise SystemExit(f"task {task_id} out of range 1..{len(tasks)}")
    task = tasks[task_id - 1]
    run_config(task, out_dir, dry_run=dry_run, force=force,
               task_label=f"task {task_id}")


def execute_sweep(
    grid_path: str | Path, *, results_root: str | Path = "results"
) -> list[dict | None]:
    """Execute every task in a prepared sweep sequentially."""
    try:
        tasks = read_grid_tasks(grid_path)
    except ValueError as error:
        raise SystemExit(error) from error
    out_dir = run_store(results_root)
    print(f"Executing {len(tasks)} tasks sequentially on this machine")
    records = [
        run_config(task, out_dir, task_label=f"task {index}")
        for index, task in enumerate(tasks, 1)
    ]
    print(f"Completed sweep execution: {len(tasks)} tasks")
    return records


def run_config(task: dict, out_dir: Path, *, dry_run: bool = False,
               force: bool = False, task_label: str = "run",
               run_missing: bool = True) -> dict | None:
    """Run one resolved task mapping and return its completed record."""
    name = task["algorithm"]
    exp = ExperimentConfig(
        **{
            key: value
            for key, value in task["experiment"].items()
            if key in ExperimentConfig.model_fields
        }
    )
    if not dry_run:
        try:
            name, exp, cfg, data = _resolve_task(task)
        except (FileNotFoundError, KeyError, ValueError) as exc:
            raise SystemExit(f"{task_label}: {exc}") from exc
    else:
        Cfg = config_class(name)
        unknown = sorted(
            set(flatten_mapping(task["algorithm_config"])) - model_paths(Cfg)
        )
        if unknown:
            raise SystemExit(
                f"{task_label} ({name}): unknown algorithm setting(s): "
                f"{', '.join(unknown)}\nknown: "
                f"{', '.join(sorted(model_paths(Cfg)))}"
            )
        cfg = resolve_algorithm_config(name, exp, Cfg(**task["algorithm_config"]))
    out_dir.mkdir(parents=True, exist_ok=True)
    if dry_run:
        print(
            f"{task_label}: {name}  experiment={exp.model_dump()}  "
            f"algorithm={cfg.model_dump()}"
        )
        return None
    # Non-dry tasks resolved the experiment data (including canonical client
    # models) above; only that resolved form is eligible for run identity.
    fp = algorithm_run_fingerprint(name, exp, cfg.model_dump())
    submitted_fp = task.get("run_fingerprint")
    if submitted_fp is not None and submitted_fp != fp:
        raise SystemExit(
            f"{task_label}: resolved fingerprint {fp} does not match submitted "
            f"fingerprint {submitted_fp}; refusing to run a changed task"
        )
    path = out_dir / result_filename(exp, name, fp)
    submitted_file = task.get("result_file")
    if submitted_file is not None and submitted_file != path.name:
        raise SystemExit(
            f"{task_label}: result filename {path.name} does not match submitted "
            f"filename {submitted_file}; refusing to run a changed task"
        )
    try:
        skip, message = existing_result_decision(
            path, expected_algorithm=name, expected_fingerprint=fp,
            force=force, require_flops=exp.estimate_flops)
    except ResultValidationError as e:
        raise SystemExit(f"{task_label}: {e.report()}")
    if message:
        print(f"{task_label}: {message}")
    if skip:
        record = read_json(path)
        validate_run_record(
            record, path=path, expected_algorithm=name, expected_fingerprint=fp
        )
        record["_source_file"] = path.name
        return record
    if not run_missing:
        raise SystemExit(f"{task_label}: result has not completed: {path}")
    device = resolve_device(exp.device)
    print(f"=== {task_label}: {name} (dataset={exp.dataset}, "
          f"partition={exp.partition_id}, seed={exp.seed}, {device}) ===")
    record = run_one(name, exp, cfg, device, data=data)
    write_run_record(path, record, expected_algorithm=name, expected_fingerprint=fp)
    print(f"  wrote {path}  ({len(record['result']['evaluation_history']['evaluation_rounds'])} eval rounds, {record['wall_seconds']}s)")
    record["_source_file"] = path.name
    return record


def declare_sweep(
    config: str | Path, *, results_root: str | Path = "results"
) -> Path:
    """Expand one YAML sweep and write its working task grid."""
    import yaml

    config_path = Path(config)
    spec = yaml.safe_load(config_path.read_text()) or {}
    if not isinstance(spec, dict):
        raise SystemExit(
            f"{config_path}: the sweep config must be a mapping, "
            f"got {type(spec).__name__}"
        )
    spec.setdefault("name", config_path.stem)
    grid, _ = expand(spec)
    _validate_biosilo_partitions(grid)

    sweep_dir = study_directory(results_root, spec["name"])
    sweep_dir.mkdir(parents=True, exist_ok=True)
    grid_path = sweep_dir / "grid.jsonl"
    created = _write_grid(grid_path, grid)

    n = len(grid)
    action = "Wrote" if created else "Reused"
    print(f"{action} {n} tasks at {grid_path}")
    print(f"  task indices: 1..{n}")
    print(f"  algorithms: {sorted({c['algorithm'] for c in grid})}")
    collect = f"rigfl report --results-dir {run_store(results_root)} --grid {grid_path}"
    print(f"Collect when done:\n  {collect}")
    return grid_path


def main(argv: list[str] | None = None, *, prog: str | None = None) -> None:
    p = argparse.ArgumentParser(prog=prog, description="Declare a RigFL sweep task grid.")
    p.add_argument("config", metavar="CONFIG", help="YAML sweep file")
    p.add_argument("--results-root", default="results")
    p.add_argument(
        "--execute",
        action="store_true",
        help="execute every prepared task sequentially on this machine",
    )
    args = p.parse_args(legacy_config_argv(argv))
    grid_path = declare_sweep(args.config, results_root=args.results_root)
    if args.execute:
        execute_sweep(grid_path, results_root=args.results_root)


if __name__ == "__main__":
    main()
