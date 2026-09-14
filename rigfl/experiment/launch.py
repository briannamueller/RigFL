"""Expand an experiment into runnable configurations and submit an SGE array.

Sweep axes form a Cartesian product; zipped replicates do not. Fixed settings
belong under ``base``; ``launch`` writes one configuration per task to
``grid.jsonl``.

    python -m rigfl.experiment.launch --config experiments/cifar_baselines.yaml --queue gpu
    python -m rigfl.experiment.launch --name demo --algorithms local,fedproto \
        --seeds 0-2 --sweep algorithm.lamda=0.1,1,10 --queue gpu
"""

from __future__ import annotations

import argparse
import difflib
import itertools
import json
from pathlib import Path

from pydantic import ValidationError

from rigfl.data.config import (
    BioSiloDatasetSettings,
    dataset_settings,
    inactive_replicate_seed_fields,
)
from rigfl.experiment.artifacts import (
    ResultValidationError,
    atomic_write_text,
    existing_result_decision,
    read_json,
    validate_run_record,
    write_run_record,
)
from rigfl.experiment.config import (
    ExperimentConfig,
    ExperimentFileConfig,
    result_filename,
)
from rigfl.experiment.device import resolve_device
from rigfl.experiment.paths import (
    filter_for_model,
    flatten_mapping,
    model_has_path,
    model_paths,
    nested_set,
)
from rigfl.experiment.registry import (
    ALL_ALGORITHMS,
    BASELINES,
    algorithm_run_fingerprint,
    algorithm_spec,
    config_class,
    resolve_algorithm_config,
    resolve_algorithm_models,
)
from rigfl.experiment.run import resolve_experiment_data, run_one
from rigfl.experiment.storage import run_store, study_directory
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
                f'Unknown sweep axis: exp.{field}\n'
                f'ExperimentConfig has no field "{field}".'
                f'{_suggest(field, known_exp)}\n\n'
                f'Known experiment fields: {", ".join(sorted(known_exp))}')
        if not any(
            field not in algorithm_spec(name).ignored_experiment_fields
            for name in algorithms
        ):
            raise SystemExit(
                f"Sweep axis exp.{field} does not apply to any selected algorithm."
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
    "partition_seed": "exp.partition_seed",
    "split_seed": "exp.split_seed",
    "experiment_seed": "exp.seed",
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

    Each task = {algorithm, experiment: {...}, algorithm_config: {...}}. Axis keys: 'algorithm',
    an experiment field (bare or 'exp.x'), or 'algorithm.x' (algorithm-specific).

    An ``algorithm.x`` axis only multiplies the grid for algorithms that actually have
    field ``x``; for algorithms without it, that axis collapses to a single entry. So
    ``--algorithms all --sweep algorithm.graphroute.graph.k=3,5,10`` gives
    FedDES three variants and every other algorithm exactly one -- no duplicate
    configs, no manual per-algorithm scoping."""
    return expand(spec)[0]


def expand(spec: dict) -> tuple[list[dict], dict | None]:
    """Expand an ordinary sweep into independent run configurations."""
    if isinstance(spec, dict) and "tuning" in spec:
        raise SystemExit(
            "ordinary sweeps do not perform hyperparameter tuning; remove the "
            "tuning section or run the configuration with "
            "python -m rigfl.experiment.optimize"
        )
    parsed = _validate_spec(spec)
    replicate_conditions, _ = _replicate_conditions(parsed)
    spec = parsed.model_dump(by_alias=True, exclude_unset=True)
    base = spec.get("base") or {}
    base_exp = dict(base.get("experiment", base))            # allow flat base = experiment fields
    base_exp.pop("algorithm", None)
    base_algorithm = dict(base.get("algorithm", {}))

    sweep = {k: _values(v) for k, v in (spec.get("sweep") or {}).items()}
    algorithms = sweep.pop("algorithm", None) or _values(spec.get("algorithms", BASELINES))

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
        elif axis.startswith("exp."):
            exp_axes[axis[len("exp."):]] = resolved_values
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
        ignored = set(algorithm_spec(algorithm).ignored_experiment_fields)
        axes = {f"exp::{k}": v for k, v in exp_axes.items() if k not in ignored}
        axes.update({f"algorithm::{k}": v for k, v in m_axes.items()})
        keys = list(axes)
        for combo in itertools.product(*(axes[k] for k in keys)):   # () once when no axes
            for replicate in replicate_conditions or [None]:
                exp = dict(base_exp)
                # Fixed algorithm settings obey the same per-algorithm scoping as algorithm
                # axes: validate against the selected-algorithm union above, then apply
                # only settings this algorithm's configuration class actually defines.
                mcfg = filter_for_model(base_algorithm, config_model)
                for key, val in zip(keys, combo):
                    kind, field = key.split("::", 1)
                    if kind == "exp":
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
        if "algorithm" not in task or "experiment" not in task:
            raise ValueError(f"grid line {i} is not a task")


def _write_grid(path: Path, grid: list[dict]) -> bool:
    """Write a new grid, or reuse an identical existing grid."""
    body = "".join(json.dumps(task) + "\n" for task in grid)
    if path.exists():
        try:
            existing = path.read_text()
            _check_grid(existing, len(grid))
        except (OSError, ValueError, json.JSONDecodeError) as error:
            raise SystemExit(f"cannot reuse existing grid {path}: {error}") from error
        if existing != body:
            raise SystemExit(
                f"{path} already contains a different sweep. Use a new sweep "
                "name so submitted jobs continue to reference an immutable grid."
            )
        return False
    atomic_write_text(
        path,
        body,
        validate=lambda text: _check_grid(text, len(grid)),
    )
    return True


def run_task(grid_path: str, task_id: int, out_dir: Path,
             dry_run: bool = False, force: bool = False) -> None:
    """Run the 1-indexed task from a grid file and save its result."""
    lines = Path(grid_path).read_text().splitlines()
    if not 1 <= task_id <= len(lines):
        raise SystemExit(f"task {task_id} out of range 1..{len(lines)}")
    task = json.loads(lines[task_id - 1])
    run_config(task, out_dir, dry_run=dry_run, force=force,
               task_label=f"task {task_id}")


def run_config(task: dict, out_dir: Path, *, dry_run: bool = False,
               force: bool = False, task_label: str = "run",
               run_missing: bool = True) -> dict | None:
    """Run one resolved task mapping and return its completed record."""
    name = task["algorithm"]
    exp = ExperimentConfig(**task["experiment"])
    if not dry_run:
        try:
            exp, data = resolve_experiment_data(exp)
            exp = resolve_algorithm_models(name, exp)
        except (FileNotFoundError, KeyError, ValueError) as exc:
            raise SystemExit(f"{task_label}: {exc}") from exc
    Cfg = config_class(name)
    # Grid tasks use the same algorithm-setting validation as single runs.
    unknown = sorted(
        set(flatten_mapping(task["algorithm_config"])) - model_paths(Cfg)
    )
    if unknown:
        raise SystemExit(
            f"{task_label} ({name}): unknown algorithm setting(s): {', '.join(unknown)}\n"
            f"known: {', '.join(sorted(model_paths(Cfg)))}")
    cfg = Cfg(**task["algorithm_config"])
    cfg = resolve_algorithm_config(name, exp, cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    if dry_run:
        print(f"{task_label}: {name}  exp={exp.model_dump()}  algorithm={cfg.model_dump()}")
        return None
    # Non-dry tasks resolved the experiment data (including canonical client
    # models) above; only that resolved form is eligible for run identity.
    fp = algorithm_run_fingerprint(name, exp, cfg.model_dump())
    path = out_dir / result_filename(exp, name, fp)
    try:
        skip, message = existing_result_decision(
            path, expected_algorithm=name, expected_fingerprint=fp,
            force=force)
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


def _spec_from_args(args) -> dict:
    if args.config:
        import yaml
        spec = yaml.safe_load(Path(args.config).read_text()) or {}
        spec.setdefault("name", Path(args.config).stem)
        return spec
    sweep = {"seed": args.seeds}
    for s in args.sweep:                                     # --sweep algorithm.lamda=0.1,1,10
        key, vals = s.split("=", 1)
        sweep[key] = vals
    return {"name": args.name, "algorithms": args.algorithms, "sweep": sweep}


def main() -> None:
    p = argparse.ArgumentParser(description="Declare + submit a RigFL sweep.")
    p.add_argument("--config", help="YAML sweep file (overrides the CLI sweep flags)")
    p.add_argument("--name", default="sweep")
    p.add_argument("--queue", help="cluster queue for the printed qsub line (e.g. gpu)")
    p.add_argument("--algorithms", default="baselines", help="'all' | 'baselines' | comma list")
    p.add_argument("--seeds", default="0-2")
    p.add_argument("--sweep", nargs="*", default=[], help="extra axes, e.g. algorithm.lamda=0.1,1,10")
    p.add_argument("--results-root", default="results")
    p.add_argument("--grid-task", type=int, help="run the Nth config from --grid")
    p.add_argument(
        "--grid",
        help="existing grid.jsonl to inspect, submit, or run with --grid-task",
    )
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--force", action="store_true", help="re-run tasks even if the result exists")
    p.add_argument("--submit", action="store_true", help="run qsub instead of printing it")
    args = p.parse_args()

    if args.grid_task is not None:                            # ── per-task execution ──
        if not args.grid:
            raise SystemExit("--grid-task requires --grid")
        run_task(args.grid, args.grid_task, run_store(args.results_root),
                 dry_run=args.dry_run, force=args.force)
        return

    if args.grid:
        grid_path = Path(args.grid)
        try:
            lines = grid_path.read_text().splitlines()
            _check_grid("\n".join(lines) + ("\n" if lines else ""), len(lines))
            grid = [json.loads(line) for line in lines]
        except (OSError, ValueError, json.JSONDecodeError) as error:
            raise SystemExit(f"cannot read grid {grid_path}: {error}") from error
        _validate_biosilo_partitions(grid)
        n = len(grid)
        qsub = (
            f"qsub -t 1-{n} -q {args.queue or '<gpu-queue>'} -l ngpus=1 "
            f"scripts/run_grid.sh {grid_path} {args.results_root}"
        )
        print(f"Grid contains {n} tasks: {grid_path}")
        print(f"\nSubmit:\n  {qsub}")
        if args.submit:
            if not args.queue:
                raise SystemExit("--submit requires --queue (e.g. --queue gpu)")
            import subprocess

            subprocess.run(qsub.split(), check=True)
        return

    spec = _spec_from_args(args)
    if spec.get("algorithms") in ("all", None):
        spec["algorithms"] = ALL_ALGORITHMS
    elif spec.get("algorithms") == "baselines":
        spec["algorithms"] = BASELINES
    grid, _ = expand(spec)
    _validate_biosilo_partitions(grid)

    sweep_dir = study_directory(args.results_root, spec.get("name", "sweep"))
    sweep_dir.mkdir(parents=True, exist_ok=True)
    grid_path = sweep_dir / "grid.jsonl"
    created = _write_grid(grid_path, grid)

    n = len(grid)
    action = "Wrote" if created else "Reused"
    print(f"{action} {n} tasks at {grid_path}")
    print(f"  algorithms: {sorted({c['algorithm'] for c in grid})}")
    qsub = (
        f"qsub -t 1-{n} -q {args.queue or '<gpu-queue>'} -l ngpus=1 "
        f"scripts/run_grid.sh {grid_path} {args.results_root}"
    )
    print(f"\nSubmit:\n  {qsub}")
    collect = (
        "python -m rigfl.experiment.collect --results-dir "
        f"{run_store(args.results_root)}"
    )
    print(f"Collect when done:\n  {collect}")
    if args.submit:
        if not args.queue:
            raise SystemExit("--submit requires --queue (e.g. --queue gpu)")
        import subprocess
        subprocess.run(qsub.split(), check=True)


if __name__ == "__main__":
    main()
