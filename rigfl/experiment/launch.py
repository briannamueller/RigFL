"""Expand an experiment into independently runnable task configurations.

Sweep axes form a Cartesian product; zipped replicates do not. Fixed settings
belong under ``base``; ``rigfl sweep`` writes one task per line to ``grid.jsonl``.

    rigfl sweep configs/experiments/mnist_sweep.yaml
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

from rigfl.data.config import (
    BioSiloDatasetSettings,
    dataset_settings,
    inactive_replicate_seed_fields,
    with_seed_overrides,
)
from rigfl.experiment.artifacts import (
    ResultValidationError,
    atomic_write_json,
    atomic_write_text,
    existing_result_decision,
    read_json,
    validate_run_record,
)
from rigfl.experiment.config import (
    ExperimentConfig,
    ExperimentFileConfig,
    ReplicateCondition,
    ResolvedExperimentConfig,
    experiment_fields,
    result_filename,
)
from rigfl.experiment.device import resolve_device
from rigfl.experiment.env import capture_env
from rigfl.experiment.identity import fingerprint
from rigfl.experiment.paths import (
    filter_for_model,
    flatten_mapping,
    model_has_path,
    model_paths,
    nested_set,
)
from rigfl.experiment.registry import (
    ALL_ALGORITHMS,
    config_class,
    configuration_owner,
    ignores_experiment_field,
    resolve_algorithm_config,
    resolve_algorithm_models,
    run_identity,
    split_public_configuration,
    supports_model_arch,
)
from rigfl.experiment.run import resolve_experiment_data, run_one
from rigfl.experiment.storage import (
    GRID_TASK_KIND,
    GRID_TASK_SCHEMA_VERSION,
    SNAPSHOT_FILE,
    SNAPSHOT_KIND,
    SNAPSHOT_SCHEMA_VERSION,
    read_grid_tasks,
    run_store,
    study_directory,
    validate_grid_task,
)


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


def _validate_axes(exp_axes: dict, algorithms: list[str]) -> None:
    """Reject experiment axes that cannot vary the selected algorithms."""
    if "results_root" in exp_axes:
        raise SystemExit(
            "results_root is not a sweep dimension; set it once under base or "
            "with --results-root"
        )
    for field in exp_axes:
        if all(ignores_experiment_field(name, field) for name in algorithms):
            raise SystemExit(
                f"Sweep setting {field} does not apply to any selected algorithm."
            )


def _replicate_conditions(parsed: ExperimentFileConfig) -> list[dict[str, int]]:
    """Return explicitly paired data and training seed conditions."""
    if parsed.replicates is None:
        return []
    return [condition.model_dump() for condition in parsed.replicates]


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
    """Validate every generated task before execution or snapshotting."""
    for i, task in enumerate(grid, 1):
        try:
            exp = ExperimentConfig(**task["experiment"])
            cfg = config_class(task["algorithm"])(**task["algorithm_config"])
            resolve_algorithm_config(task["algorithm"], exp, cfg)
        except Exception as e:
            raise SystemExit(
                f"task {i} ({task['algorithm']}) does not validate: {e}")


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
    """Require every BioSilo partition before writing the grid."""
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
        settings = with_seed_overrides(settings, experiment.partition_seed)
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


def expand(spec: dict) -> list[dict]:
    """Expand a sweep spec into a flat list of per-task configs.

    Each task = {algorithm, experiment: {...}, algorithm_config: {...}}. Public
    sweep settings are flat and are split into the internal validated models.

    Algorithm-specific axes only expand algorithms that define that field, and
    model-family values of ``model_arch`` only expand algorithms that allow
    different client architectures."""
    if isinstance(spec, dict) and "tuning" in spec:
        raise SystemExit(
            "ordinary sweeps do not perform hyperparameter tuning; remove the "
            "tuning section or run the configuration with "
            "rigfl hpo"
        )
    parsed = _validate_spec(spec)
    replicate_conditions = _replicate_conditions(parsed)
    spec = parsed.model_dump(by_alias=True, exclude_unset=True)
    sweep = {k: _values(v) for k, v in (spec.get("sweep") or {}).items()}
    declared_algorithms = sweep.pop("algorithm", None) or spec.get("algorithm")
    if not declared_algorithms:
        raise SystemExit(
            "sweep config must declare 'algorithm' explicitly; "
            f"known: {', '.join(ALL_ALGORITHMS)}"
        )
    algorithms = _values(declared_algorithms)
    try:
        for algorithm in algorithms:
            config_class(algorithm)
        base_exp, base_algorithm = split_public_configuration(
            spec.get("base") or {}, algorithms
        )
    except (KeyError, ValueError) as error:
        raise SystemExit(f"invalid fixed sweep setting: {error}") from error

    if replicate_conditions:
        overlap = sorted(set(ReplicateCondition.model_fields) & set(sweep))
        if overlap:
            raise SystemExit(
                "seed fields declared under replicates cannot also "
                "appear under sweep: " + ", ".join(overlap)
            )

    exp_axes: dict[str, list] = {}
    algorithm_axes: dict[str, list] = {}
    for path, vals in sweep.items():
        try:
            owner = configuration_owner(path, algorithms)
        except ValueError as error:
            raise SystemExit(f"invalid sweep setting: {error}") from error
        (algorithm_axes if owner == "algorithm" else exp_axes)[path] = vals

    _validate_axes(exp_axes, algorithms)

    default_model_arch = ExperimentConfig.model_fields["model_arch"].default
    grid: list[dict] = []
    for algorithm in algorithms:
        tasks_before = len(grid)
        config_model = config_class(algorithm)
        # only the algorithm-axes this algorithm has; the rest don't multiply its grid
        m_axes = {
            k: v for k, v in algorithm_axes.items()
            if model_has_path(config_model, k)
        }
        # an ignored axis keeps only its first value: one task, one fingerprint
        axes = {
            f"experiment::{k}": v[:1] if ignores_experiment_field(algorithm, k) else v
            for k, v in exp_axes.items()
        }
        axes.update({f"algorithm::{k}": v for k, v in m_axes.items()})
        keys = list(axes)
        for combo in itertools.product(*(axes[k] for k in keys)):   # () once when no axes
            # model families apply only to algorithms that allow different
            # client architectures
            model_arch = dict(zip(keys, combo)).get(
                "experiment::model_arch",
                base_exp.get("model_arch", default_model_arch),
            )
            if not supports_model_arch(algorithm, model_arch):
                continue
            for replicate in replicate_conditions or [None]:
                exp = dict(base_exp)
                # keep only base settings this algorithm defines
                mcfg = filter_for_model(base_algorithm, config_model)
                for key, val in zip(keys, combo):
                    kind, field = key.split("::", 1)
                    if kind == "experiment":
                        nested_set(exp, field, val)
                    else:
                        nested_set(mcfg, field, val)
                if replicate is not None:
                    exp.update(replicate)
                task = {
                    "algorithm": algorithm,
                    "experiment": exp,
                    "algorithm_config": mcfg,
                }
                grid.append(task)
        if len(grid) == tasks_before:
            raise SystemExit(
                f"{algorithm} requires identical client architectures, but "
                "every model_arch value in this sweep is a model family"
            )

    _validate_replicate_effects(grid)
    _validate_tasks(grid)

    return grid


def _check_grid(text: str) -> None:
    """Every line of a grid parses as a task."""
    for i, line in enumerate(text.splitlines(), 1):
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
            _check_grid(existing)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            raise SystemExit(
                f"cannot replace existing grid {path}: {error}"
            ) from error
        if existing == body:
            return False
    atomic_write_text(path, body)
    return True


def _task_results_root(task: dict) -> Path:
    return Path(ExperimentConfig(**experiment_fields(task["experiment"])).results_root)


def _override_results_root(
    grid: list[dict], results_root: str | Path | None
) -> None:
    if results_root is None:
        return
    for task in grid:
        task["experiment"]["results_root"] = str(results_root)


def _shared_results_root(grid: list[dict]) -> Path:
    directories = {_task_results_root(task) for task in grid}
    if len(directories) != 1:
        raise SystemExit(
            "a sweep must use one results_root; set it under base or with "
            "--results-root"
        )
    return directories.pop()


def _resolve_task(task: dict, *, data_cache: dict | None = None):
    """Resolve one declared task to the complete configuration it will execute."""
    name = task["algorithm"]
    experiment_values = task["experiment"]
    declared = ExperimentConfig(**experiment_fields(experiment_values))
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
    actual = resolve_algorithm_models(name, actual)
    if "partition_id" in experiment_values:
        exp = resolve_algorithm_models(
            name, ResolvedExperimentConfig(**experiment_values)
        )
        if exp.model_dump(mode="json") != actual.model_dump(mode="json"):
            raise ValueError(
                "saved snapshot configuration no longer resolves to the same "
                "dataset partition or model assignment"
            )
    else:
        exp = actual
    cfg = resolve_algorithm_config(name, exp, Cfg(**task["algorithm_config"]))
    return name, exp, cfg, data


def _materialize_snapshot(grid: list[dict], snapshot_dir: Path) -> Path:
    """Write a resolved, immutable task snapshot."""
    resolved = []
    data_cache = {}
    for index, task in enumerate(grid, 1):
        try:
            name, exp, cfg, _ = _resolve_task(task, data_cache=data_cache)
        except Exception as error:
            raise SystemExit(
                f"task {index} ({task.get('algorithm')}): cannot resolve snapshot: "
                f"{error}"
            ) from error
        experiment = exp.model_dump(mode="json")
        algorithm_config = cfg.model_dump(mode="json")
        fp = fingerprint(run_identity(name, exp, algorithm_config))
        resolved.append(
            (
                name,
                experiment,
                algorithm_config,
                fp,
                result_filename(exp, name, fp),
            )
        )

    resolved_tasks = [
        {
            "kind": GRID_TASK_KIND,
            "schema_version": GRID_TASK_SCHEMA_VERSION,
            "algorithm": name,
            "experiment": experiment,
            "algorithm_config": algorithm_config,
            "run_fingerprint": fp,
            "result_file": result_file,
        }
        for name, experiment, algorithm_config, fp, result_file in resolved
    ]
    metadata = {
        "kind": SNAPSHOT_KIND,
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "snapshot_provenance": capture_env(),
    }
    atomic_write_json(snapshot_dir / SNAPSHOT_FILE, metadata)
    snapshot = snapshot_dir / "grid.jsonl"
    body = "".join(json.dumps(task) + "\n" for task in resolved_tasks)
    atomic_write_text(snapshot, body)
    return snapshot


def stage_task_snapshot(grid_path: Path) -> Path:
    """Resolve and copy a grid so external tasks cannot observe later edits."""
    try:
        body = grid_path.read_text()
        _check_grid(body)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise SystemExit(f"cannot stage grid {grid_path}: {error}") from error

    snapshots = grid_path.parent / "snapshots"
    snapshots.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    snapshot_dir = Path(
        tempfile.mkdtemp(prefix=f"{timestamp}-", dir=snapshots)
    )
    grid = [json.loads(line) for line in body.splitlines() if line]
    return _materialize_snapshot(grid, snapshot_dir)


def run_task(
    grid_path: str,
    task_id: int,
    results_root: str | Path | None = None,
    force: bool = False,
) -> None:
    """Run the 1-indexed task from a grid file and save its result."""
    try:
        tasks = read_grid_tasks(grid_path)
    except ValueError as error:
        raise SystemExit(error) from error
    if not 1 <= task_id <= len(tasks):
        raise SystemExit(f"task {task_id} out of range 1..{len(tasks)}")
    task = tasks[task_id - 1]
    if results_root is not None:
        task["experiment"]["results_root"] = str(results_root)
    out_dir = run_store(_task_results_root(task))
    run_config(task, out_dir, force=force, task_label=f"task {task_id}")


def execute_sweep(grid_path: str | Path) -> list[dict]:
    """Execute every task in a prepared sweep sequentially."""
    try:
        tasks = read_grid_tasks(grid_path)
    except ValueError as error:
        raise SystemExit(error) from error
    print(f"Executing {len(tasks)} tasks sequentially on this machine")
    records = [
        run_config(
            task,
            run_store(_task_results_root(task)),
            task_label=f"task {index}",
        )
        for index, task in enumerate(tasks, 1)
    ]
    print(f"Completed sweep execution: {len(tasks)} tasks")
    return records


def run_config(task: dict, out_dir: Path, *, force: bool = False,
               task_label: str = "run") -> dict:
    """Run one resolved task mapping and return its completed record."""
    try:
        name, exp, cfg, data = _resolve_task(task)
    except (FileNotFoundError, KeyError, ValueError) as exc:
        raise SystemExit(f"{task_label}: {exc}") from exc
    out_dir.mkdir(parents=True, exist_ok=True)
    fp = fingerprint(run_identity(name, exp, cfg.model_dump()))
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
    device = resolve_device(exp.device)
    print(f"=== {task_label}: {name} (dataset={exp.dataset}, "
          f"partition={exp.partition_id}, training_seed={exp.training_seed}, {device}) ===")
    record = run_one(name, exp, cfg, device, data=data)
    atomic_write_json(path, record)
    print(f"  wrote {path}  ({len(record['result']['evaluation_history']['evaluation_rounds'])} eval rounds, {record['wall_seconds']}s)")
    record["_source_file"] = path.name
    return record


def declare_sweep(
    config: str | Path, *, results_root: str | Path | None = None
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
    grid = expand(spec)
    _override_results_root(grid, results_root)
    _validate_biosilo_partitions(grid)

    effective_results_root = _shared_results_root(grid)
    sweep_dir = study_directory(effective_results_root, spec["name"])
    sweep_dir.mkdir(parents=True, exist_ok=True)
    grid_path = sweep_dir / "grid.jsonl"
    created = _write_grid(grid_path, grid)

    n = len(grid)
    action = "Wrote" if created else "Reused"
    print(f"{action} {n} tasks at {grid_path}")
    print(f"  task indices: 1..{n}")
    print(f"  algorithms: {sorted({c['algorithm'] for c in grid})}")
    report = "rigfl report"
    if str(effective_results_root) != "results":
        report += f" --results-root {effective_results_root}"
    print(f"When the tasks finish, summarize results with:\n  {report}")
    return grid_path


def main(argv: list[str] | None = None, *, prog: str | None = None) -> None:
    p = argparse.ArgumentParser(prog=prog, description="Declare a RigFL sweep task grid.")
    p.add_argument("config", metavar="CONFIG", help="YAML sweep file")
    p.add_argument(
        "--results-root",
        help="base results directory; overrides 'results_root' in the YAML",
    )
    action = p.add_mutually_exclusive_group()
    action.add_argument(
        "--execute",
        action="store_true",
        help="execute every prepared task sequentially on this machine",
    )
    action.add_argument(
        "--snapshot",
        action="store_true",
        help="freeze the prepared tasks for execution by an external scheduler",
    )
    args = p.parse_args(argv)
    grid_path = declare_sweep(args.config, results_root=args.results_root)
    if args.execute:
        execute_sweep(grid_path)
    elif args.snapshot:
        snapshot = stage_task_snapshot(grid_path)
        task_count = len(read_grid_tasks(snapshot))
        print(f"Wrote immutable task snapshot: {snapshot}")
        print(f"  task indices: 1..{task_count}")
        print(f"Run one indexed task with:\n  rigfl task {snapshot} TASK_INDEX")


if __name__ == "__main__":
    main()
