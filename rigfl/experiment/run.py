"""Run one experiment and save its resolved configuration and results.

    rigfl run configs/experiments/mnist_run.yaml
    rigfl run configs/experiments/mnist_run.yaml --set rounds=50 lr=0.05

Use ``rigfl sweep`` for multi-configuration sweeps.
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rigfl.data.builder import build_clients
from rigfl.data.config import (
    BioSiloDatasetSettings,
    DatasetSettings,
    FlowerDatasetSettings,
    dataset_settings,
    with_seed_overrides,
)
from rigfl.data.partitions import (
    build_partition_clients,
    load_partition,
)
from rigfl.eval.resources import ResourceMonitor
from rigfl.experiment.artifacts import (
    ResultValidationError,
    make_run_record,
)
from rigfl.experiment.config import (
    ExperimentConfig,
    ResolvedExperimentConfig,
)
from rigfl.experiment.env import capture_env
from rigfl.experiment.identity import fingerprint
from rigfl.experiment.paths import nested_set
from rigfl.experiment.registry import (
    adapter_factory,
    algorithm_spec,
    build_algorithm,
    config_class,
    recorded_experiment,
    run_identity,
    split_public_configuration,
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
            "source_manifest_version": handle.schema_version,
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
    settings = with_seed_overrides(
        dataset_settings(exp.dataset, exp.dataset_config),
        exp.partition_seed,
        exp.split_seed,
    )

    if isinstance(settings, FlowerDatasetSettings):
        artifact = load_partition(
            exp.dataset,
            config_path=exp.dataset_config,
            data_dir=exp.data_dir,
            settings=settings,
        )
        target_spec = artifact.manifest["target_spec"]
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
            resolved_models=resolve_models(exp.model_arch, input_kind=input_kind),
        )
        return resolved, ResolvedData(settings=settings, artifact=artifact)

    if isinstance(settings, BioSiloDatasetSettings):
        from rigfl.data.biosilo import (
            biosilo_input_spec,
            load_biosilo_partition,
        )

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
            resolved_models=resolve_models(exp.model_arch, input_kind=input_kind),
        )
        return resolved, ResolvedData(settings=settings, handle=handle)

    raise ValueError(f"unsupported dataset backend: {settings.backend!r}")


def run_one(
    name, exp: ResolvedExperimentConfig, cfg, device, *, data: ResolvedData
) -> dict:
    """Run one task whose experiment and config are resolved for the algorithm."""
    identity_input = run_identity(name, exp, cfg.model_dump())
    spec = algorithm_spec(name)
    # Capture provenance before training starts.
    start_env = capture_env(spec.provenance_packages)
    set_seed(exp.training_seed)                                    # training + model determinism
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
            seed=exp.training_seed,
            split_seed=exp.split_seed,
            validation_fraction=exp.validation_fraction,
            adapter=adapter,
            backbones=backbones,
            build_models=spec.requires_client_model,
        )
    elif exp.data_backend == "biosilo":
        handle = data.handle
        clients = build_clients(
            handle.client,
            handle.num_clients,
            handle.num_classes,
            backbones,
            exp.shared_dim,
            val_frac=exp.validation_fraction,
            batch=exp.batch,
            adapter=adapter,
            seed=exp.training_seed,
            split_seed=exp.split_seed,
            build_models=spec.requires_client_model,
        )
    else:
        raise RuntimeError(f"unresolved data backend: {exp.data_backend!r}")
    algorithm = build_algorithm(
        name, exp, cfg, model_input_spec=model_input_spec,
        model_template=clients[0].model,
        initial_client_models=(
            tuple(client.model for client in clients)
            if spec.requires_client_model else None
        ),
        client_sample_counts=tuple(
            len(client.train_loader.dataset) for client in clients
        ),
    )
    tracker = make_tracker(name, exp, cfg)
    monitor = ResourceMonitor(device, estimate_flops=exp.estimate_flops)
    runner = spec.runner
    with monitor:
        result = runner(algorithm, clients, num_rounds=exp.rounds, device=device,
                        num_classes=exp.num_classes, eval_gap=exp.eval_gap,
                        verbose=not exp.quiet, tracker=tracker,
                        early_stopping=exp.early_stopping,
                        resource_monitor=monitor)
    resources = monitor.to_dict()

    record = make_run_record(
        algorithm=name,
        experiment=recorded_experiment(name, exp), algorithm_config=cfg.model_dump(),
        run_fingerprint=fingerprint(identity_input),
        identity_input=identity_input,
        result=result,
        env=start_env, device=str(device),
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


def _read_run_config(path: str | Path) -> dict:
    """Read one flat, self-contained public run configuration."""
    import yaml

    loaded = yaml.safe_load(Path(path).read_text())
    if not isinstance(loaded, dict):
        kind = type(loaded).__name__ if loaded is not None else "null"
        raise SystemExit(f"{path}: the config must be a mapping, got {kind}")
    return loaded


def _split_run_config(
    loaded: dict,
    *,
    source: str,
    algorithm_override: str | None = None,
) -> tuple[str, dict, dict]:
    """Validate and split the flat public mapping for internal execution."""
    values = dict(loaded)
    configured_algorithm = values.pop("algorithm", None)
    algorithm = algorithm_override or configured_algorithm
    if not isinstance(algorithm, str) or not algorithm:
        raise SystemExit(
            f"{source}: specify one algorithm with 'algorithm' in the YAML "
            "or --algorithm"
        )
    try:
        config_class(algorithm)
        experiment, algorithm_config = split_public_configuration(values, [algorithm])
        experiment = ExperimentConfig(**experiment).model_dump(exclude_unset=True)
        config_class(algorithm)(**algorithm_config)
    except (KeyError, ValueError) as error:
        raise SystemExit(f"{source}: {error}") from error
    return algorithm, experiment, algorithm_config


def build_configs(args) -> tuple[str, ExperimentConfig, dict]:
    """Resolve YAML configuration followed by any ``--set`` overrides."""
    import yaml

    values = _read_run_config(args.config)
    for kv in args.set:
        if "=" not in kv:
            raise SystemExit(f"--set {kv}: expected <setting>=<value>")
        key, val = kv.split("=", 1)
        nested_set(values, key, yaml.safe_load(val))
    if getattr(args, "results_root", None) is not None:
        values["results_root"] = str(args.results_root)
    algorithm, experiment, algorithm_config = _split_run_config(
        values,
        source=str(args.config),
        algorithm_override=getattr(args, "algorithm", None),
    )
    return algorithm, ExperimentConfig(**experiment), algorithm_config


def run_experiment(
    config: str | Path,
    *,
    algorithm: str | None = None,
    results_root: str | Path | None = None,
    overrides: list[str] | None = None,
    force: bool = False,
) -> Path:
    """Run one YAML-defined experiment from Python and return its result path."""
    from types import SimpleNamespace

    from rigfl.experiment.launch import run_config

    algorithm, exp, algorithm_config = build_configs(
        SimpleNamespace(
            config=str(config),
            algorithm=algorithm,
            results_root=results_root,
            set=overrides or [],
        )
    )
    task = {
        "algorithm": algorithm,
        "experiment": exp.model_dump(mode="json"),
        "algorithm_config": algorithm_config,
    }
    out_dir = run_store(exp.results_root)
    record = run_config(task, out_dir, force=force)
    return out_dir / record["_source_file"]


def parse_args(
    argv: list[str] | None = None, *, prog: str | None = None
) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog=prog, description="Run one RigFL experiment.")
    p.add_argument(
        "config",
        metavar="CONFIG",
        help="self-contained experiment YAML",
    )
    p.add_argument(
        "--algorithm",
        help="algorithm name; overrides 'algorithm' in the YAML",
    )
    p.add_argument(
        "--results-root",
        help="base results directory; overrides 'results_root' in the YAML",
    )
    p.add_argument("--set", nargs="*", default=[],
                   help="overrides, e.g. rounds=50 lr=0.05")
    p.add_argument("--force", action="store_true",
                   help="re-run even if the result JSON exists")
    return p.parse_args(argv)


def main(argv: list[str] | None = None, *, prog: str | None = None) -> None:
    args = parse_args(argv, prog=prog)
    try:
        run_experiment(
            args.config,
            algorithm=args.algorithm,
            results_root=args.results_root,
            overrides=args.set,
            force=args.force,
        )
    except (FileNotFoundError, KeyError, ValueError, ResultValidationError) as exc:
        if isinstance(exc, ResultValidationError):
            raise SystemExit(exc.report()) from exc
        raise SystemExit(exc) from exc


if __name__ == "__main__":
    main()
