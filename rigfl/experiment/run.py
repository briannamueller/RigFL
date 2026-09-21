"""Run one experiment and save its resolved configuration and results.

    python -m rigfl.experiment.run --algorithm fedproto --config configs/experiments/cifar10_run.yaml
    python -m rigfl.experiment.run --algorithm fedproto --set algorithm.lamda=10
    python -m rigfl.experiment.run --algorithm fedtgp \
        --set experiment.rounds=50 algorithm.server_epochs=100

Use :mod:`rigfl.experiment.launch` for multi-configuration sweeps.
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from pydantic import ValidationError

from rigfl.data.config import (
    BioSiloDatasetSettings,
    DatasetSettings,
    FlowerDatasetSettings,
    dataset_settings,
)
from rigfl.data.partitions import (
    build_partition_clients,
    generate_partition,
)
from rigfl.eval.resources import ResourceMonitor
from rigfl.experiment.artifacts import (
    ResultValidationError,
    existing_result_decision,
    make_run_record,
    write_run_record,
)
from rigfl.experiment.config import (
    ExperimentConfig,
    ResolvedExperimentConfig,
    RunFileConfig,
    result_filename,
)
from rigfl.experiment.device import resolve_device
from rigfl.experiment.env import capture_env
from rigfl.experiment.paths import (
    filter_for_model,
    flatten_mapping,
    model_paths,
    nested_set,
)
from rigfl.experiment.registry import (
    BASELINES,
    adapter_factory,
    algorithm_run_fingerprint,
    algorithm_spec,
    build_algorithm,
    config_class,
    resolve_algorithm_config,
    resolve_algorithm_experiment,
)
from rigfl.experiment.storage import run_store
from rigfl.experiment.tracking import make_tracker
from rigfl.models.registry import instantiate_backbones, resolve_models


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def partition_summary(clients, num_classes: int, artifact=None, handle=None) -> dict:
    """Per-client split sizes, label histograms, and partition identity."""
    out = []
    for c in clients:
        hist = [0] * num_classes
        for _, y in c.train_loader:
            for label in y.tolist():
                hist[label] += 1
        out.append({
            "train": len(c.train_loader.dataset),
            "val": len(c.val_loader.dataset) if c.val_loader else 0,
            "test": len(c.test_loader.dataset) if c.test_loader else 0,
            "train_label_hist": hist,
        })
    summary = {"per_client": out}
    if artifact is not None:
        summary["generated"] = {
            "partition_id": artifact.partition_id,
            "settings": artifact.settings.model_dump(mode="json"),
        }
    if handle is not None:
        for client, source_id in zip(out, handle.client_ids):
            client["source_client_id"] = source_id
        summary["biosilo"] = {
            "dataset": handle.dataset,
            "partition_id": handle.partition_id,
            "schema_version": handle.schema_version,
            "settings": handle.settings,
            "provenance": handle.provenance,
        }
    return summary


@dataclass(frozen=True)
class ResolvedData:
    """The backend object and metadata selected by a dataset-registry entry."""

    settings: DatasetSettings
    artifact: Any = None
    handle: Any = None


def resolve_experiment_data(
    exp: ExperimentConfig,
) -> tuple[ResolvedExperimentConfig, ResolvedData]:
    """Resolve one dataset alias to the partition facts used by an experiment."""
    experiment_input = {
        name: getattr(exp, name) for name in ExperimentConfig.model_fields
    }
    experiment_input.pop("partition_seed")
    experiment_input.pop("split_seed")
    settings = dataset_settings(exp.dataset, exp.dataset_config)

    if isinstance(settings, FlowerDatasetSettings):
        overrides = {
            name: value
            for name, value in {
                "partition_seed": exp.partition_seed,
                "split_seed": exp.split_seed,
            }.items()
            if value is not None
        }
        if overrides:
            settings = settings.model_copy(
                update={"partition": settings.partition.model_copy(update=overrides)}
            )
        artifact, _ = generate_partition(
            exp.dataset,
            config_path=exp.dataset_config,
            data_dir=exp.data_dir,
            settings=settings,
        )
        target_spec = artifact.manifest["target_spec"]
        if artifact.manifest["task"] != "classification":
            raise ValueError(
                "RigFL's experiment algorithms currently support classification only; "
                f"dataset {exp.dataset!r} generated a "
                f"{artifact.manifest['task']} partition"
            )
        manifest_input_spec = dict(artifact.manifest["input_spec"])
        input_kind = manifest_input_spec.pop("kind")
        input_spec = {"input_kind": input_kind, **manifest_input_spec}
        resolved = ResolvedExperimentConfig(
            **experiment_input,
            data_backend="flower",
            partition_id=artifact.partition_id,
            partition_seed=settings.partition.partition_seed,
            split_seed=settings.partition.split_seed,
            partition_scheme=settings.partition.scheme,
            num_clients=artifact.manifest["num_clients"],
            num_classes=target_spec["num_classes"],
            validation_fraction=(
                settings.client_split.validation_fraction
                if settings.client_split is not None
                else settings.partition.val_frac
            ),
            input_kind=input_kind,
            input_spec=input_spec,
            resolved_models=resolve_models(
                model=exp.model,
                model_family=exp.model_family,
                input_kind=input_kind,
            ),
        )
        return resolved, ResolvedData(settings=settings, artifact=artifact)

    if isinstance(settings, BioSiloDatasetSettings):
        from rigfl.data.biosilo import (
            biosilo_input_spec,
            load_biosilo_partition,
        )

        overrides = {}
        if exp.partition_seed is not None:
            overrides["parameters"] = {
                **settings.parameters,
                "seed": exp.partition_seed,
            }
        if exp.split_seed is not None:
            overrides["split_seed"] = exp.split_seed
        if overrides:
            settings = settings.model_copy(update=overrides)

        handle = load_biosilo_partition(
            settings,
            data_dir=exp.data_dir,
            dataset_name=exp.dataset,
            dataset_config=exp.dataset_config,
            partition_seed_override=exp.partition_seed,
        )
        input_kind, input_spec = biosilo_input_spec(handle)
        partition_seed = handle.settings.get("seed")
        if (
            not isinstance(partition_seed, int)
            or isinstance(partition_seed, bool)
            or partition_seed < 0
        ):
            partition_seed = None
        resolved = ResolvedExperimentConfig(
            **experiment_input,
            data_backend="biosilo",
            partition_id=handle.partition_id,
            partition_seed=partition_seed,
            split_seed=settings.split_seed,
            partition_scheme=None,
            num_clients=handle.num_clients,
            num_classes=handle.num_classes,
            validation_fraction=settings.validation_fraction,
            input_kind=input_kind,
            input_spec=input_spec,
            resolved_models=resolve_models(
                model=exp.model,
                model_family=exp.model_family,
                input_kind=input_kind,
            ),
        )
        return resolved, ResolvedData(settings=settings, handle=handle)

    raise ValueError(f"unsupported dataset backend: {settings.backend!r}")


def run_one(name, exp: ExperimentConfig, cfg, device, *, data: ResolvedData | None = None) -> dict:
    if data is None:
        exp, data = resolve_experiment_data(exp)
    elif not isinstance(exp, ResolvedExperimentConfig):
        raise TypeError("pre-resolved data requires a ResolvedExperimentConfig")
    exp = resolve_algorithm_experiment(name, exp)
    cfg = resolve_algorithm_config(name, exp, cfg)
    spec = algorithm_spec(name)
    set_seed(exp.seed)                                    # training + model determinism
    adapter = adapter_factory(name) if spec.requires_client_model else None
    model_input_spec = dict(exp.input_spec)
    if "shape" in model_input_spec:
        model_input_spec["shape"] = tuple(model_input_spec["shape"])
    backbone_names = exp.resolved_models
    backbones = instantiate_backbones(backbone_names, input_spec=model_input_spec)
    if exp.data_backend == "flower":
        clients = build_partition_clients(
            data.artifact,
            shared_dim=exp.shared_dim,
            batch=exp.batch,
            seed=exp.seed,
            split_seed=exp.split_seed,
            validation_fraction=exp.validation_fraction,
            adapter=adapter,
            backbones=backbones,
            build_models=spec.requires_client_model,
        )
    elif exp.data_backend == "biosilo":
        from rigfl.data.biosilo import build_biosilo_clients

        clients = build_biosilo_clients(
            data.handle,
            shared_dim=exp.shared_dim,
            batch=exp.batch,
            seed=exp.seed,
            split_seed=exp.split_seed,
            adapter=adapter,
            backbones=backbones,
            validation_fraction=exp.validation_fraction,
            build_models=spec.requires_client_model,
        )
    else:
        raise RuntimeError(f"unresolved data backend: {exp.data_backend!r}")
    algorithm = build_algorithm(
        name, exp, cfg, model_input_spec=model_input_spec,
        model_template=clients[0].model)
    tracker = make_tracker(name, exp, cfg)               # W&B if exp.wandb else no-op
    monitor = ResourceMonitor(device, estimate_flops=exp.estimate_flops)
    runner = spec.runner
    with monitor:
        result = runner(algorithm, clients, num_rounds=exp.rounds, device=device,
                        num_classes=exp.num_classes, eval_gap=exp.eval_gap,
                        verbose=not exp.quiet, tracker=tracker,
                        early_stopping=exp.early_stopping,
                        resource_monitor=monitor)
    resources = monitor.to_dict()

    # The fingerprint is computed from the resolved experiment, including the
    # identity and metadata of the partition that was actually loaded.
    record = make_run_record(
        algorithm=name,
        experiment=exp.model_dump(), algorithm_config=cfg.model_dump(),
        run_fingerprint=algorithm_run_fingerprint(name, exp, cfg.model_dump()),
        result=result,
        env=capture_env(), device=str(device),
        wall_seconds=round(resources["observed"]["wall_seconds"]["total"], 1),
        resources=resources,
        partition=partition_summary(
            clients,
            exp.num_classes,
            artifact=data.artifact,
            handle=data.handle,
        ),
    )
    if tracker is not None:
        update_resources = getattr(tracker, "update_resources", None)
        if callable(update_resources):
            update_resources(resources)
        tracker.finish(result)
    return record


#: Sections accepted by a single-run YAML file.
_CONFIG_SECTIONS = tuple(RunFileConfig.model_fields)
#: Prefixes accepted by ``--set``.
_SET_SECTIONS = {"experiment", "algorithm"}


def load_run_config(path: str) -> tuple[dict, dict]:
    """The ``experiment:`` and ``algorithm:`` sections of a single-run YAML."""
    import yaml

    loaded = yaml.safe_load(Path(path).read_text())
    if loaded is None:
        return {}, {}
    if not isinstance(loaded, dict):
        raise SystemExit(f"{path}: the config must be a mapping with "
                         f"{' and '.join(_CONFIG_SECTIONS)} sections, "
                         f"got {type(loaded).__name__}")
    known = _CONFIG_SECTIONS
    unknown = sorted(set(loaded) - set(known))
    if unknown:
        raise SystemExit(
            f"{path}: unknown top-level section(s): {', '.join(unknown)}"
            f"{_suggest(unknown[0], known)}\n"
            f"Known: {', '.join(known)}. (Sweep files with base/sweep "
            f"sections go to rigfl.experiment.launch, not here.)")
    try:
        parsed = RunFileConfig.model_validate(loaded)
    except ValidationError as error:
        first = error.errors(include_url=False)[0]
        location = ".".join(str(part) for part in first["loc"])
        value = loaded.get(first["loc"][0]) if first["loc"] else loaded
        if first["type"] in {"dict_type", "model_type"} and first["loc"]:
            raise SystemExit(
                f"{path}: '{location}' must be a mapping, "
                f"got {type(value).__name__}"
            ) from error
        raise SystemExit(f"{path}: invalid setting at {location}: {first['msg']}") from error
    experiment = parsed.experiment.model_dump(exclude_unset=True)
    return experiment, dict(parsed.algorithm.root)


def _suggest(name: str, known) -> str:
    import difflib
    close = difflib.get_close_matches(name, sorted(known), n=1)
    return f' Did you mean "{close[0]}"?' if close else ""


def build_configs(args) -> tuple[ExperimentConfig, dict]:
    """Resolve YAML configuration followed by any ``--set`` overrides."""
    exp_over: dict = {}
    algorithm_over: dict = {}
    if args.config:
        exp_over, algorithm_over = load_run_config(args.config)
    for kv in args.set:
        if "=" not in kv or "." not in kv.split("=", 1)[0]:
            raise SystemExit(f"--set {kv}: expected <section>.<field>=<value>, "
                             f"where section is one of {', '.join(sorted(_SET_SECTIONS))}")
        key, val = kv.split("=", 1)
        section, field = key.split(".", 1)
        if section not in _SET_SECTIONS:
            raise SystemExit(
                f'--set {kv}: unknown section "{section}".'
                f'{_suggest(section, _SET_SECTIONS)}\n'
                f"Use {', '.join(f'{s}.<field>' for s in sorted(_SET_SECTIONS))}.")
        nested_set(
            exp_over if section == "experiment" else algorithm_over,
            field,
            val,
        )
    return ExperimentConfig(**exp_over), algorithm_over


def _run_resolved_experiment(name: str, exp: ResolvedExperimentConfig, cfg, *,
                             data: ResolvedData, force: bool = False) -> Path:
    """Run and save one fully resolved experiment configuration."""
    device = resolve_device(exp.device)
    out_dir = run_store(exp.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fp = algorithm_run_fingerprint(name, exp, cfg.model_dump())
    path = out_dir / result_filename(exp, name, fp)
    skip, message = existing_result_decision(
        path, expected_algorithm=name, expected_fingerprint=fp, force=force
    )
    if message:
        print(message)
    if skip:
        return path

    print(f"\n=== {name}  ({exp.dataset}, {exp.num_clients} clients, "
          f"partition={exp.partition_id}, seed={exp.seed}, device={device}) ===")
    record = run_one(name, exp, cfg, device, data=data)
    write_run_record(path, record, expected_algorithm=name, expected_fingerprint=fp)
    print(f"  wrote {path}  "
          f"({len(record['result']['evaluation_history']['evaluation_rounds'])} "
          f"eval rounds, {record['wall_seconds']}s)")
    return path


def run_experiment(algorithm: str, config: str | Path, *,
                   force: bool = False) -> Path:
    """Run one YAML-defined experiment from Python and return its result path."""
    experiment, algorithm_config = load_run_config(str(config))
    exp = ExperimentConfig(**experiment)
    exp, data = resolve_experiment_data(exp)
    exp = resolve_algorithm_experiment(algorithm, exp)

    Cfg = config_class(algorithm)
    unknown = sorted(
        set(flatten_mapping(algorithm_config)) - model_paths(Cfg)
    )
    if unknown:
        raise ValueError(
            f"unknown {algorithm} algorithm setting(s): {', '.join(unknown)}; "
            f"known: {', '.join(sorted(model_paths(Cfg)))}"
        )
    cfg = resolve_algorithm_config(algorithm, exp, Cfg(**algorithm_config))
    return _run_resolved_experiment(algorithm, exp, cfg, data=data, force=force)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run one RigFL experiment.")
    p.add_argument("--algorithm", default="baselines",
                   help="algorithm name, 'baselines', or 'all'")
    p.add_argument("--config", help="YAML with 'experiment:' and 'algorithm:' sections")
    p.add_argument("--set", nargs="*", default=[],
                   help="overrides, e.g. experiment.rounds=50 algorithm.lamda=10")
    p.add_argument("--force", action="store_true",
                   help="re-run even if the result JSON exists")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    exp, algorithm_over = build_configs(args)
    try:
        exp, data = resolve_experiment_data(exp)
    except (FileNotFoundError, KeyError, ValueError) as exc:
        raise SystemExit(exc) from exc
    from rigfl.experiment.registry import ALL_ALGORITHMS
    algorithms = (ALL_ALGORITHMS if args.algorithm == "all" else
                  BASELINES if args.algorithm == "baselines" else [args.algorithm])

    known = set().union(*(model_paths(config_class(n)) for n in algorithms))
    unknown = sorted(set(flatten_mapping(algorithm_over)) - known)
    if unknown:
        raise SystemExit(
            f"unknown algorithm setting(s): {', '.join(unknown)}\n"
            f"known for {', '.join(algorithms)}: {', '.join(sorted(known))}")

    for name in algorithms:
        Cfg = config_class(name)
        algorithm_exp = resolve_algorithm_experiment(name, exp)
        # Shared overrides are applied only to algorithms that define the field.
        cfg = Cfg(**filter_for_model(algorithm_over, Cfg))
        cfg = resolve_algorithm_config(name, algorithm_exp, cfg)
        try:
            _run_resolved_experiment(
                name, algorithm_exp, cfg, data=data, force=args.force)
        except ResultValidationError as e:
            raise SystemExit(e.report())


if __name__ == "__main__":
    main()
