"""Run validation-based Optuna studies over complete configurations."""

from __future__ import annotations

import argparse
import importlib
import math
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from rigfl.cli import legacy_config_argv
from rigfl.eval.metrics import canonical, direction_of
from rigfl.eval.report import selected_metric_value, selection_for
from rigfl.experiment.config import ExperimentConfig
from rigfl.experiment.launch import (
    _apply_replicate,
    _replicate_conditions,
    _validate_replicate_effects,
    _validate_spec,
    _validate_tasks,
    _values,
    run_config,
)
from rigfl.experiment.paths import (
    filter_for_model,
    flatten_mapping,
    model_paths,
    nested_set,
)
from rigfl.experiment.registry import config_class, ignores_experiment_field
from rigfl.experiment.storage import run_store, study_directory
from rigfl.experiment.tuning import (
    MANIFEST_KIND,
    MANIFEST_SCHEMA_VERSION,
    Candidate,
    candidate_hash,
    canonical_axis,
    rank,
    write_manifest,
    write_ranking,
)

_FORBIDDEN_EXPERIMENT_SEARCH_FIELDS = {
    "data_dir",
    "dataset",
    "dataset_config",
    "device",
    "out_dir",
    "partition_seed",
    "quiet",
    "seed",
    "split_seed",
    "wandb",
    "wandb_project",
}


class OptimizationSpec(BaseModel):
    """A resolved Optuna study ready for execution."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    algorithm: str
    base_experiment: dict
    base_algorithm: dict
    replicate_axis: str
    replicates: list[int]
    replicate_conditions: list[dict[str, int]]
    replicate_mode: Literal["zipped"] | None
    search_space: dict[str, dict]
    trials: int
    sampler_class: str
    sampler_options: dict
    metric: str
    selection_view: Literal["global", "per-client"]
    selection_aggregation: Literal["mean", "weighted_mean"]
    tie_break: Literal["earliest", "latest"]


def parse_optimization(spec: dict) -> OptimizationSpec:
    """Validate an adaptive-search configuration."""
    experiment_file = _validate_spec(spec)
    tuning = experiment_file.tuning
    if tuning is None:
        raise SystemExit("an Optuna configuration requires a tuning section")

    algorithms = _values(experiment_file.algorithms or [])
    if len(algorithms) != 1:
        raise SystemExit("an Optuna study must configure exactly one algorithm")
    algorithm = algorithms[0]
    try:
        algorithm_model = config_class(algorithm)
    except KeyError as error:
        raise SystemExit(str(error)) from error

    base = experiment_file.base.root
    base_experiment = dict(base.get("experiment", base))
    base_experiment.pop("algorithm", None)
    base_algorithm = dict(base.get("algorithm", {}))
    unknown_base = sorted(
        set(flatten_mapping(base_algorithm)) - model_paths(algorithm_model)
    )
    if unknown_base:
        raise SystemExit(
            "unknown algorithm setting(s) in base.algorithm: " + ", ".join(unknown_base)
        )

    conditions, replicate_mode = _replicate_conditions(experiment_file)
    if not conditions:
        raise SystemExit(
            "an Optuna study requires top-level replicates with paired "
            "partition_seed, split_seed, and experiment_seed values"
        )
    replicate_axis = "experiment.seed"
    if experiment_file.sweep and experiment_file.sweep.root:
        raise SystemExit(
            "an Optuna configuration does not use sweep; searched parameters "
            "belong under tuning.search_space"
        )
    replicates = [condition["experiment_seed"] for condition in conditions]
    if not replicates or len({_stable(value) for value in replicates}) != len(
        replicates
    ):
        raise SystemExit(
            "the replicate seed list must be nonempty and contain no duplicates"
        )
    if any(
        not isinstance(value, int) or isinstance(value, bool) or value < 0
        for value in replicates
    ):
        raise SystemExit("replicate seeds must be nonnegative integers")

    search_space = {}
    experiment_paths = model_paths(ExperimentConfig)
    algorithm_paths = model_paths(algorithm_model)
    for declared_path, distribution in tuning.search_space.items():
        path = canonical_axis(declared_path)
        if path in search_space:
            raise SystemExit(f"duplicate search parameter: {path}")
        section, name = path.split(".", 1)
        if section == "experiment" and name not in experiment_paths:
            raise SystemExit(f"unknown experiment search parameter: {path}")
        if section == "experiment" and name in _FORBIDDEN_EXPERIMENT_SEARCH_FIELDS:
            raise SystemExit(
                f"{path} cannot be optimized; it identifies the dataset, a "
                "replicate condition, or the execution environment"
            )
        if (
            section == "experiment"
            and ignores_experiment_field(algorithm, name)
        ):
            raise SystemExit(f"{path} does not apply to {algorithm}")
        if section == "algorithm" and name not in algorithm_paths:
            raise SystemExit(f"unknown {algorithm} search parameter: {path}")
        if path == replicate_axis:
            raise SystemExit(f"{path} cannot be both searched and used as a replicate")
        search_space[path] = distribution.model_dump(exclude_unset=True)

    sampler_class = tuning.sampler.class_path
    sampler_options = tuning.sampler.options
    grid_sampler = sampler_class.rsplit(".", 1)[-1] == "GridSampler"
    if grid_sampler:
        if "trials" in tuning.model_fields_set:
            raise SystemExit(
                "tuning.trials is not used with GridSampler; the declared grid "
                "determines the number of trials"
            )
        noncategorical = [
            path for path, distribution in search_space.items()
            if distribution["type"] != "categorical"
        ]
        if noncategorical:
            raise SystemExit(
                "GridSampler requires categorical values for every search parameter: "
                + ", ".join(noncategorical)
            )
        trials = math.prod(
            len(distribution["values"]) for distribution in search_space.values()
        )
    else:
        trials = tuning.trials

    parsed = OptimizationSpec(
        name=experiment_file.name or "optimization",
        algorithm=algorithm,
        base_experiment=base_experiment,
        base_algorithm=base_algorithm,
        replicate_axis=replicate_axis,
        replicates=replicates,
        replicate_conditions=conditions,
        replicate_mode=replicate_mode,
        search_space=search_space,
        trials=trials,
        sampler_class=sampler_class,
        sampler_options=dict(sampler_options),
        metric=canonical(tuning.metric),
        selection_view=tuning.selection_view,
        selection_aggregation=tuning.selection_aggregation,
        tie_break=tuning.tie_break,
    )
    sample = {
        path: _first_value(distribution) for path, distribution in search_space.items()
    }
    sample_tasks = _tasks(parsed, sample)
    _validate_replicate_effects(sample_tasks)
    _validate_tasks(sample_tasks)
    return parsed


def _stable(value) -> str:
    return repr(value)


def _first_value(distribution: dict):
    if distribution["type"] == "categorical":
        return distribution["values"][0]
    return distribution["low"]


def _tasks(spec: OptimizationSpec, parameters: dict) -> list[dict]:
    tasks = []
    algorithm_model = config_class(spec.algorithm)
    if spec.replicate_conditions:
        conditions = spec.replicate_conditions
    else:
        conditions = [{"experiment_seed": seed} for seed in spec.replicates]
    for condition in conditions:
        experiment = deepcopy(spec.base_experiment)
        algorithm_config = filter_for_model(spec.base_algorithm, algorithm_model)
        if set(condition) == {"experiment_seed"}:
            experiment["seed"] = condition["experiment_seed"]
        else:
            _apply_replicate(experiment, condition)
        for path, value in parameters.items():
            section, name = path.split(".", 1)
            nested_set(
                experiment if section == "experiment" else algorithm_config,
                name,
                value,
            )
        tasks.append(
            {
                "algorithm": spec.algorithm,
                "experiment": experiment,
                "algorithm_config": algorithm_config,
            }
        )
    return tasks


def _suggest(trial, path: str, distribution: dict):
    kind = distribution["type"]
    if kind == "categorical":
        return trial.suggest_categorical(path, distribution["values"])
    if kind == "int":
        return trial.suggest_int(
            path,
            distribution["low"],
            distribution["high"],
            step=distribution.get("step", 1),
            log=distribution.get("log", False),
        )
    options = {"log": distribution.get("log", False)}
    if distribution.get("step") is not None:
        options["step"] = distribution["step"]
    return trial.suggest_float(
        path, distribution["low"], distribution["high"], **options
    )


def _validation_value(record: dict, spec: OptimizationSpec) -> float:
    selected = selection_for(
        record,
        spec.metric,
        view=spec.selection_view,
        aggregation=spec.selection_aggregation,
        tie_break=spec.tie_break,
        include_test=False,
    )
    name = canonical(spec.metric)
    score = selected_metric_value(
        selected, name, "validation", spec.selection_aggregation
    )
    if score is None:
        raise ValueError(f"run produced no validation {spec.metric}")
    return score


def _objective(spec: OptimizationSpec, results_dir: Path, force: bool):
    def objective(trial):
        parameters = {
            path: _suggest(trial, path, distribution)
            for path, distribution in spec.search_space.items()
        }
        trial.set_user_attr("replicates", list(spec.replicates))
        if spec.replicate_conditions:
            trial.set_user_attr(
                "replicate_conditions", list(spec.replicate_conditions)
            )
        scores, sources = [], []
        try:
            tasks = _tasks(spec, parameters)
            _validate_tasks(tasks)
            for seed, task in zip(spec.replicates, tasks):
                record = run_config(
                    task,
                    results_dir,
                    force=force,
                    task_label=f"trial {trial.number}, seed {seed}",
                )
                scores.append(_validation_value(record, spec))
                sources.append(record.get("_source_file"))
                trial.set_user_attr("source_results", list(sources))
                trial.set_user_attr("validation_scores", list(scores))
        except (Exception, SystemExit) as error:
            trial.set_user_attr("failure_reason", f"{type(error).__name__}: {error}")
            raise RuntimeError(str(error)) from error
        trial.set_user_attr("source_results", sources)
        trial.set_user_attr("validation_scores", scores)
        return sum(scores) / len(scores)

    return objective


def _manifest(spec: OptimizationSpec, study, target_trials: int | None = None) -> dict:
    completed = [trial for trial in study.trials if trial.state.name == "COMPLETE"]
    unique: dict[tuple, dict] = {}
    for trial in completed:
        key = tuple(
            sorted((path, _stable(value)) for path, value in trial.params.items())
        )
        entry = unique.setdefault(
            key, {"parameters": dict(trial.params), "trials": [], "source_results": []}
        )
        entry["trials"].append(trial.number)
        entry["source_results"].extend(
            getattr(trial, "user_attrs", {}).get("source_results", [])
        )
    candidates = [
        Candidate(
            id=index,
            algorithm=spec.algorithm,
            parameters=entry["parameters"],
            config_hash=candidate_hash(spec.algorithm, entry["parameters"]),
        )
        for index, entry in enumerate(unique.values())
    ]
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "kind": MANIFEST_KIND,
        "name": spec.name,
        "engine": "optuna",
        "algorithm": spec.algorithm,
        "base": {
            "experiment": spec.base_experiment,
            "algorithm": spec.base_algorithm,
        },
        "search_space": spec.search_space,
        "tuning_parameters": list(spec.search_space),
        "replicate_axis": spec.replicate_axis,
        "replicate_values": list(spec.replicates),
        "replicate_fields": (
            [
                "experiment.partition_seed",
                "experiment.split_seed",
                "experiment.seed",
            ]
            if spec.replicate_conditions
            else [spec.replicate_axis]
        ),
        "replicate_mode": spec.replicate_mode,
        "replicate_conditions": spec.replicate_conditions or None,
        "selection_protocol": {
            "metric": spec.metric,
            "selection_view": spec.selection_view,
            "selection_aggregation": spec.selection_aggregation,
            "round_tie_break": spec.tie_break,
            "fixed": True,
        },
        "condition_axes": [],
        "candidates": [candidate.to_dict() for candidate in candidates],
        "run_store": "../runs",
    }
    trial_numbers = {
        candidate_hash(spec.algorithm, entry["parameters"]): entry["trials"]
        for entry in unique.values()
    }
    for candidate in manifest["candidates"]:
        candidate["optuna_trial_numbers"] = trial_numbers[candidate["config_hash"]]
    manifest["evaluations"] = [
        {
            "candidate_id": candidate.id,
            "replicates": list(spec.replicates),
            "replicate_conditions": list(spec.replicate_conditions),
            "source_results": sorted(set(entry["source_results"])),
        }
        for candidate, entry in zip(candidates, unique.values())
    ]
    counts = Counter(trial.state.name.lower() for trial in study.trials)
    trial_records = [
        {
            "number": trial.number,
            "state": trial.state.name.lower(),
            "parameters": dict(trial.params),
            "validation_objective": getattr(trial, "value", None),
            "validation_scores": getattr(trial, "user_attrs", {}).get(
                "validation_scores", []
            ),
            "source_results": getattr(trial, "user_attrs", {}).get(
                "source_results", []
            ),
            "failure_reason": getattr(trial, "user_attrs", {}).get("failure_reason"),
        }
        for trial in study.trials
    ]
    manifest["optimization"] = {
        "study_name": study.study_name,
        "sampler": {
            "class": spec.sampler_class,
            "options": spec.sampler_options,
        },
        "metric": spec.metric,
        "direction": direction_of(spec.metric),
        "selection_view": spec.selection_view,
        "selection_aggregation": spec.selection_aggregation,
        "round_tie_break": spec.tie_break,
        "target_trials": target_trials or spec.trials,
        "trial_counts": dict(counts),
        "trials": trial_records,
    }
    return manifest


def _optuna():
    try:
        import optuna
    except ImportError as error:
        raise SystemExit(
            "Optuna optimization requires the optional dependency: "
            "pip install 'rigfl[hpo]'"
        ) from error
    return optuna


def _storage_url(results_dir: Path, supplied: str | None) -> str:
    if supplied:
        return supplied
    return "sqlite:///" + str((results_dir / "optuna.db").resolve())


def _study_protocol(spec: OptimizationSpec) -> dict:
    return {
        "kind": "rigfl.hpo_protocol",
        "schema_version": 1,
        "algorithm": spec.algorithm,
        "base_experiment": spec.base_experiment,
        "base_algorithm": spec.base_algorithm,
        "replicate_axis": spec.replicate_axis,
        "replicates": spec.replicates,
        "replicate_conditions": spec.replicate_conditions,
        "replicate_mode": spec.replicate_mode,
        "search_space": spec.search_space,
        "sampler": {
            "class": spec.sampler_class,
            "options": spec.sampler_options,
        },
        "metric": spec.metric,
        "selection_view": spec.selection_view,
        "selection_aggregation": spec.selection_aggregation,
        "round_tie_break": spec.tie_break,
    }


def _bind_study_protocol(study, spec: OptimizationSpec) -> None:
    expected = _study_protocol(spec)
    stored = study.user_attrs.get("rigfl_protocol")
    if stored is None:
        if study.trials:
            raise SystemExit(
                "the existing Optuna study has no RigFL protocol metadata; "
                "use a new study name or storage location"
            )
        study.set_user_attr("rigfl_protocol", expected)
        return
    if stored != expected:
        raise SystemExit(
            "the Optuna study was created from a different RigFL configuration; "
            "use a new study name or storage location"
        )


def _sampler(spec: OptimizationSpec):
    optuna = _optuna()
    module_name, separator, class_name = spec.sampler_class.rpartition(".")
    if not separator:
        module_name = "optuna.samplers"
        class_name = spec.sampler_class
    try:
        module = importlib.import_module(module_name)
        sampler_class = getattr(module, class_name)
    except (ImportError, AttributeError) as error:
        raise SystemExit(
            f"cannot import Optuna sampler {spec.sampler_class!r}: {error}"
        ) from error
    if not isinstance(sampler_class, type) or not issubclass(
        sampler_class, optuna.samplers.BaseSampler
    ):
        raise SystemExit(
            "tuning.sampler.class must inherit optuna.samplers.BaseSampler: "
            f"{spec.sampler_class!r}"
        )
    options = dict(spec.sampler_options)
    if issubclass(sampler_class, optuna.samplers.GridSampler):
        if "search_space" in options:
            raise SystemExit(
                "do not repeat the grid under tuning.sampler.options; RigFL derives "
                "it from tuning.search_space"
            )
        options["search_space"] = {
            path: list(distribution["values"])
            for path, distribution in spec.search_space.items()
        }
    try:
        return sampler_class(**options)
    except (TypeError, ValueError) as error:
        raise SystemExit(
            f"cannot construct Optuna sampler {spec.sampler_class!r}: {error}"
        ) from error


def run_study(
    spec: OptimizationSpec,
    study_dir: Path,
    *,
    storage: str | None,
    study_name: str | None,
    target_trials: int | None,
    force: bool = False,
    export_only: bool = False,
):
    """Run or resume a study and export its tuning manifest."""
    optuna = _optuna()
    study_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = run_store(study_dir.parent)
    runs_dir.mkdir(parents=True, exist_ok=True)
    sampler = _sampler(spec)
    study = optuna.create_study(
        study_name=study_name or spec.name,
        storage=_storage_url(study_dir, storage),
        load_if_exists=True,
        direction=direction_of(spec.metric),
        sampler=sampler,
    )
    _bind_study_protocol(study, spec)
    target = target_trials or spec.trials
    if not export_only:
        remaining = max(0, target - len(study.trials))
        if remaining:
            if study.trials:
                study.sampler.reseed_rng()
            study.optimize(
                _objective(spec, runs_dir, force),
                n_trials=remaining,
                catch=(Exception,),
                gc_after_trial=True,
            )
    complete = [trial for trial in study.trials if trial.state.name == "COMPLETE"]
    best_trial = None
    if complete:
        chooser = min if direction_of(spec.metric) == "minimize" else max
        best_trial = chooser(complete, key=lambda trial: trial.value)
    manifest = _manifest(spec, study, target)
    manifest["optimization"]["best_trial"] = (
        best_trial.number if best_trial is not None else None
    )
    if best_trial is not None:
        manifest["optimization"]["best_validation_value"] = best_trial.value
        manifest["optimization"]["best_parameters"] = best_trial.params
    study_path = write_manifest(manifest, study_dir)
    manifest["_path"] = str(study_path.resolve())
    return study, manifest


def write_study_selection(
    spec: OptimizationSpec, study_dir: Path, manifest: dict
) -> list[Path]:
    from rigfl.experiment.collect import load_results

    by_algorithm = load_results(run_store(study_dir.parent), None)
    source_names = {
        Path(source).name
        for trial in manifest.get("optimization", {}).get("trials", [])
        for source in trial.get("source_results", [])
        if source
    }
    records = [
        record
        for values in by_algorithm.values()
        for record in values
        if Path(record.get("_source_file", "")).name in source_names
    ]
    if not manifest.get("candidates"):
        raise SystemExit(
            "cannot select an Optuna configuration because no trial completed "
            "successfully; inspect the trial records in study.json for the reasons"
        )
    artifact = rank(
        records,
        manifest,
        metric=spec.metric,
        views=[spec.selection_view],
        aggregation=spec.selection_aggregation,
        tie_break=spec.tie_break,
    )
    return write_ranking(artifact, records, manifest, study_dir)


def main(argv: list[str] | None = None, *, prog: str | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Run or resume a validation-based Optuna study."
    )
    parser.add_argument("config", metavar="CONFIG", help="YAML HPO study file")
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--study-name")
    parser.add_argument("--trials", type=int, help="target total number of trials")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--export-only", action="store_true")
    args = parser.parse_args(legacy_config_argv(argv))

    import yaml

    raw = yaml.safe_load(Path(args.config).read_text()) or {}
    raw.setdefault("name", Path(args.config).stem)
    spec = parse_optimization(raw)
    if args.trials is not None and args.trials < 1:
        raise SystemExit("--trials must be a positive integer")
    if args.trials is not None and spec.sampler_class.rsplit(".", 1)[-1] == "GridSampler":
        raise SystemExit(
            "--trials is not used with GridSampler; the declared grid determines "
            "the number of trials"
        )
    results_dir = study_directory(args.results_root, spec.name)
    study, manifest = run_study(
        spec,
        results_dir,
        storage=None,
        study_name=args.study_name,
        target_trials=args.trials,
        force=args.force,
        export_only=args.export_only,
    )
    complete = sum(trial.state.name == "COMPLETE" for trial in study.trials)
    print(f"study {study.study_name}: {complete} completed trial(s)")
    print(f"wrote {results_dir / 'study.json'}")
    written = write_study_selection(spec, results_dir, manifest)
    for path in written:
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
