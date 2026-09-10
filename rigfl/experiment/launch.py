"""Expand a sweep into runnable configurations and submit an SGE array.

Sweep axes form a Cartesian product. Fixed settings belong under ``base``;
``launch`` writes one resolved configuration per task to ``grid.jsonl``.

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

from rigfl.experiment.artifacts import (ResultValidationError, atomic_write_text,
                                        existing_result_decision, write_run_record)
from rigfl.experiment.config import ExperimentConfig, result_filename
from rigfl.experiment.device import resolve_device
from rigfl.experiment.registry import (ALL_ALGORITHMS, BASELINES,
                                       algorithm_run_fingerprint, algorithm_spec,
                                       config_class, resolve_algorithm_config,
                                       resolve_algorithm_models)
from rigfl.experiment.paths import (filter_for_model, flatten_mapping,
                                    model_has_path, model_paths, nested_set)
from rigfl.experiment.run import resolve_experiment_data, run_one
from rigfl.experiment.tuning import (applicable_parameters, build_candidates,
                                     build_manifest,
                                     candidate_index, canonical_axis, parse_tuning,
                                     write_manifest, _norm)


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


#: Top-level keys accepted by a sweep file.
_SPEC_KEYS = ("name", "algorithms", "base", "sweep", "tuning")


def _validate_spec(spec: dict) -> None:
    if not isinstance(spec, dict):
        raise SystemExit(f"a sweep config must be a mapping, got {type(spec).__name__}")
    unknown = sorted(set(spec) - set(_SPEC_KEYS))
    if unknown:
        raise SystemExit(
            f"Unknown top-level key(s) in the sweep config: {', '.join(unknown)}"
            f"{_suggest(unknown[0], _SPEC_KEYS)}\n\n"
            f"Known: {', '.join(_SPEC_KEYS)}")
    for key in ("base", "sweep", "tuning"):
        if key in spec and spec[key] is not None and not isinstance(spec[key], dict):
            raise SystemExit(f"sweep config: '{key}' must be a mapping, "
                             f"got {type(spec[key]).__name__}")


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


def _canonical_tuning_block(block: dict | None) -> tuple[dict | None, list[str] | None]:
    """Normalize tuning parameter paths."""
    if not block or not isinstance(block, dict):
        return block, None
    normalized = dict(block)
    raw = block.get("parameters") or []
    raw = [raw] if isinstance(raw, str) else list(raw)
    declared = [canonical_axis(path) for path in raw]
    normalized["parameters"] = declared
    if "replicate_axis" in normalized:
        normalized["replicate_axis"] = canonical_axis(normalized["replicate_axis"])
    return normalized, declared


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
    """The grid, and the tuning manifest when the sweep declares one.

    Declaring ``tuning:`` does not change which tasks are produced -- it names
    which of the axes already being swept jointly define a candidate, and which
    one is a replicate. The manifest records those relationships without changing
    the run configurations, so the same sweep with and without a tuning block runs
    exactly the same jobs.
    """
    _validate_spec(spec)
    base = spec.get("base") or {}
    base_exp = dict(base.get("experiment", base))            # allow flat base = experiment fields
    base_exp.pop("algorithm", None)
    base_algorithm = dict(base.get("algorithm", {}))

    sweep = {k: _values(v) for k, v in spec.get("sweep", {}).items()}
    algorithms = sweep.pop("algorithm", None) or _values(spec.get("algorithms", BASELINES))

    exp_axes: dict[str, list] = {}
    algorithm_axes: dict[str, list] = {}
    axis_values: dict[str, list] = {}
    declared_axes: list[dict] = []
    for path, vals in sweep.items():
        axis, resolved_values = canonical_axis(path), vals
        if axis in axis_values:
            raise SystemExit(
                f"Sweep axes {path!r} and an earlier declaration both resolve to "
                f"{axis!r}; declare that setting only once.")
        axis_values[axis] = resolved_values
        declared_axes.append({"path": axis, "declared_as": path,
                              "values": list(resolved_values)})
        if axis.startswith("algorithm."):
            algorithm_axes[axis[len("algorithm."):]] = resolved_values
        elif axis.startswith("exp."):
            exp_axes[axis[len("exp."):]] = resolved_values
        else:
            raise SystemExit(f"Internal error: non-canonical sweep axis {axis!r}.")

    _validate_axes(exp_axes, algorithm_axes, algorithms, base_algorithm)

    tuning_block, declared_tuning_parameters = _canonical_tuning_block(
        spec.get("tuning"))
    tuning = parse_tuning(tuning_block, axis_values)
    candidates = index = None
    if tuning:
        unsupported = [
            name for name in algorithms
            if tuning.replicate_axis not in applicable_parameters(
                name, [tuning.replicate_axis]
            )
        ]
        if unsupported:
            raise SystemExit(
                f"tuning.replicate_axis {tuning.replicate_axis} does not apply to "
                f"{', '.join(unsupported)}."
            )
        candidates = build_candidates(algorithms, tuning.parameters, axis_values)
        index = candidate_index(candidates)

    grid: list[dict] = []
    manifest_tasks: list[dict] = []
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
            exp = dict(base_exp)
            # Fixed algorithm settings obey the same per-algorithm scoping as algorithm
            # axes: validate against the selected-algorithm union above, then apply
            # only settings this algorithm's configuration class actually defines.
            mcfg = filter_for_model(base_algorithm, config_model)
            assigned: dict[str, object] = {}
            for key, val in zip(keys, combo):
                kind, field = key.split("::", 1)
                if kind == "exp":
                    nested_set(exp, field, val)
                else:
                    nested_set(mcfg, field, val)
                assigned[f"{'exp' if kind == 'exp' else 'algorithm'}.{field}"] = val
            task = {"algorithm": algorithm, "experiment": exp, "algorithm_config": mcfg}
            grid.append(task)
            if tuning:
                metadata = _tuning_task_metadata(algorithm, assigned, tuning, index)
                manifest_tasks.append({"task_id": len(grid), "algorithm": algorithm,
                                       **metadata})

    _validate_tasks(grid)

    manifest = None
    if tuning:
        manifest = build_manifest(
            name=spec.get("name", "sweep"), tuning=tuning, algorithms=algorithms,
            base={"experiment": base_exp, "algorithm": base_algorithm},
            axis_values=axis_values, declared_axes=declared_axes,
            candidates=candidates, tasks=manifest_tasks,
            declared_tuning_parameters=declared_tuning_parameters)
    return grid, manifest


def _check_grid(text: str, expected: int) -> None:
    """Every line of the written grid parses, and none was lost."""
    lines = text.splitlines()
    if len(lines) != expected:
        raise ValueError(f"grid holds {len(lines)} task(s), expected {expected}")
    for i, line in enumerate(lines, 1):
        task = json.loads(line)
        if "algorithm" not in task or "experiment" not in task:
            raise ValueError(f"grid line {i} is not a task")


def _tuning_task_metadata(algorithm: str, assigned: dict,
                          tuning, index: dict) -> dict:
    """The candidate, replicate and condition a generated task belongs to.

    Stored only in the sweep manifest. Completed runs are identified entirely by
    their resolved experiment and algorithm configurations.
    """
    params = {path: assigned[path] for path in tuning.parameters
              if path in assigned}
    key = (algorithm, tuple(sorted((k, _norm(v)) for k, v in params.items())))
    cid = index.get(key)
    if cid is None:                       # unreachable: candidates are built from
        raise SystemExit(                 # the same axes and the same scoping rule
            f"internal error: task for {algorithm} with {params} matches no candidate")
    condition = {k: v for k, v in assigned.items()
                 if k not in tuning.parameters
                 and k != tuning.replicate_axis}
    return {"candidate_id": cid, "replicate": assigned.get(tuning.replicate_axis),
            "condition": condition}


def run_task(grid_path: str, task_id: int, out_dir: Path,
             dry_run: bool = False, force: bool = False) -> None:
    """Run the 1-indexed task from a grid file and save its result."""
    lines = Path(grid_path).read_text().splitlines()
    if not 1 <= task_id <= len(lines):
        raise SystemExit(f"task {task_id} out of range 1..{len(lines)}")
    task = json.loads(lines[task_id - 1])
    name = task["algorithm"]
    exp = ExperimentConfig(**task["experiment"])
    if not dry_run:
        try:
            exp, data = resolve_experiment_data(exp)
            exp = resolve_algorithm_models(name, exp)
        except (FileNotFoundError, KeyError, ValueError) as exc:
            raise SystemExit(f"task {task_id}: {exc}") from exc
    Cfg = config_class(name)
    # Grid tasks use the same algorithm-setting validation as single runs.
    unknown = sorted(
        set(flatten_mapping(task["algorithm_config"])) - model_paths(Cfg)
    )
    if unknown:
        raise SystemExit(
            f"task {task_id} ({name}): unknown algorithm setting(s): {', '.join(unknown)}\n"
            f"known: {', '.join(sorted(model_paths(Cfg)))}")
    cfg = Cfg(**task["algorithm_config"])
    cfg = resolve_algorithm_config(name, exp, cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    if dry_run:
        print(f"task {task_id}: {name}  exp={exp.model_dump()}  algorithm={cfg.model_dump()}")
        return
    # Non-dry tasks resolved the experiment data (including canonical client
    # models) above; only that resolved form is eligible for run identity.
    fp = algorithm_run_fingerprint(name, exp, cfg.model_dump())
    path = out_dir / result_filename(exp, name, fp)
    try:
        skip, message = existing_result_decision(
            path, expected_algorithm=name, expected_fingerprint=fp,
            force=force)
    except ResultValidationError as e:
        raise SystemExit(f"task {task_id}: {e.report()}")
    if message:
        print(f"task {task_id}: {message}")
    if skip:
        return
    device = resolve_device(exp.device)
    print(f"=== task {task_id}: {name} (dataset={exp.dataset}, "
          f"partition={exp.partition_id}, seed={exp.seed}, {device}) ===")
    record = run_one(name, exp, cfg, device, data=data)
    write_run_record(path, record, expected_algorithm=name, expected_fingerprint=fp)
    print(f"  wrote {path}  ({len(record['result']['evaluation_history']['evaluation_rounds'])} eval rounds, {record['wall_seconds']}s)")


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
    p.add_argument("--grid", help="grid.jsonl path (with --grid-task)")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--force", action="store_true", help="re-run tasks even if the result exists")
    p.add_argument("--submit", action="store_true", help="run qsub instead of printing it")
    args = p.parse_args()

    if args.grid_task is not None:                            # ── per-task execution ──
        run_task(args.grid, args.grid_task, Path(args.grid).resolve().parent,
                 dry_run=args.dry_run, force=args.force)
        return

    spec = _spec_from_args(args)
    if spec.get("algorithms") in ("all", None):
        spec["algorithms"] = ALL_ALGORITHMS
    elif spec.get("algorithms") == "baselines":
        spec["algorithms"] = BASELINES
    grid, manifest = expand(spec)

    sweep_dir = Path(args.results_root) / spec.get("name", "sweep")
    sweep_dir.mkdir(parents=True, exist_ok=True)
    grid_path = sweep_dir / "grid.jsonl"
    atomic_write_text(grid_path, "".join(json.dumps(c) + "\n" for c in grid),
                      validate=lambda text: _check_grid(text, len(grid)))

    n = len(grid)
    print(f"Wrote {n} tasks to {grid_path}")
    print(f"  algorithms: {sorted({c['algorithm'] for c in grid})}")
    if manifest:
        mpath = write_manifest(manifest, sweep_dir)
        displayed_parameters = manifest.get(
            "declared_tuning_parameters", manifest["tuning_parameters"])
        print(f"  tuning: {len(manifest['candidates'])} candidate(s) over "
              f"{', '.join(displayed_parameters)}, "
              f"replicates on {manifest['replicate_axis']}"
              f"={manifest['replicate_values']} -> {mpath.name}")
    qsub = f"qsub -t 1-{n} -q {args.queue or '<gpu-queue>'} -l ngpus=1 scripts/run_grid.sh {grid_path}"
    print(f"\nSubmit:\n  {qsub}")
    collect = f"python -m rigfl.experiment.collect --results-dir {sweep_dir}"
    if manifest:
        collect += " --selection-metric <metric> --rank"
    print(f"Collect when done:\n  {collect}")
    if args.submit:
        if not args.queue:
            raise SystemExit("--submit requires --queue (e.g. --queue gpu)")
        import subprocess
        subprocess.run(qsub.split(), check=True)


if __name__ == "__main__":
    main()
