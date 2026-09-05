"""Run one experiment and save its resolved configuration and results.

    python -m rigfl.experiment.run --algorithm fedproto --dataset cifar10 --seed 0
    python -m rigfl.experiment.run --algorithm fedproto --set algorithm.lamda=10
    python -m rigfl.experiment.run --algorithm fedtgp --set algorithm.server_epochs=100 --rounds 50

Use :mod:`rigfl.experiment.launch` for multi-configuration sweeps.
"""

from __future__ import annotations

import argparse
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rigfl.data.partitions import build_partition_clients, load_partition
from rigfl.data.config import BioSiloDatasetSettings, FlowerDatasetSettings, dataset_settings
from rigfl.experiment.artifacts import (ResultValidationError, existing_result_decision,
                                        make_run_record, write_run_record)
from rigfl.experiment.config import (ExperimentConfig, ResolvedExperimentConfig,
                                     result_filename, run_fingerprint)
from rigfl.experiment.device import resolve_device
from rigfl.experiment.env import capture_env
from rigfl.experiment.registry import (BASELINES, adapter_factory, algorithm_spec,
                                       build_algorithm, config_class,
                                       resolve_algorithm_config)
from rigfl.experiment.tracking import make_tracker
from rigfl.models.registry import (instantiate_backbones,
                                   resolve_model_architectures)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def partition_summary(clients, num_classes: int, handle=None, artifact=None) -> dict:
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
        summary["biosilo"] = {"partition_id": handle.partition_id,
                              "provenance": handle.provenance}
    return summary


@dataclass(frozen=True)
class ResolvedData:
    """The backend object and metadata selected by a dataset-registry entry."""

    settings: Any
    artifact: Any = None
    handle: Any = None


def resolve_experiment_data(
    exp: ExperimentConfig,
) -> tuple[ResolvedExperimentConfig, ResolvedData]:
    """Resolve one dataset alias to the partition facts used by an experiment."""
    experiment_input = {
        name: getattr(exp, name) for name in ExperimentConfig.model_fields
    }
    settings = dataset_settings(exp.dataset, exp.dataset_config)

    if isinstance(settings, FlowerDatasetSettings):
        artifact = load_partition(
            exp.dataset, config_path=exp.dataset_config, data_dir=exp.data_dir
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
            partition_scheme=settings.partition.scheme,
            num_clients=artifact.manifest["num_clients"],
            num_classes=target_spec["num_classes"],
            validation_fraction=settings.partition.val_frac,
            input_kind=input_kind,
            input_spec=input_spec,
        )
        return resolve_experiment_architectures(
            resolved, input_kind=resolved.input_kind
        ), ResolvedData(settings=settings, artifact=artifact)

    if isinstance(settings, BioSiloDatasetSettings):
        import biosilo

        handle = biosilo.load(
            settings.source_dataset,
            root=settings.data_root,
            partition=settings.partition,
        )
        from rigfl.data.biosilo import temporal_dims
        n_ts, n_static = temporal_dims(handle)
        input_spec = {
            "input_kind": "temporal",
            "n_ts": n_ts,
            "n_static": n_static,
            "seq_len": handle.inputs[0]["shape"][0],
        }
        resolved = ResolvedExperimentConfig(
            **experiment_input,
            data_backend="biosilo",
            partition_id=handle.partition_id,
            partition_scheme=None,
            num_clients=handle.num_clients,
            num_classes=handle.num_classes,
            validation_fraction=settings.validation_fraction,
            input_kind="temporal",
            input_spec=input_spec,
        )
        return resolve_experiment_architectures(
            resolved, input_kind=resolved.input_kind
        ), ResolvedData(settings=settings, handle=handle)

    raise ValueError(f"unsupported dataset backend: {settings.backend!r}")


def resolve_experiment_architectures(
    exp: ExperimentConfig, *, input_kind: str
) -> ExperimentConfig:
    """Canonicalize the model architectures used by an experiment.

    A named family, an equivalent explicit ordered list, and the default family
    all resolve to one recorded representation. This happens before run identity
    is computed, so configuration records and fingerprints describe the models
    that were actually constructed.
    """
    names = resolve_model_architectures(
        architecture_family=exp.model_architecture_family,
        architectures=exp.model_architectures,
        input_kind=input_kind,
    )
    return exp.model_copy(
        update={"model_architecture_family": None, "model_architectures": names}
    )


def run_one(name, exp: ExperimentConfig, cfg, device, *, data: ResolvedData | None = None) -> dict:
    if data is None:
        exp, data = resolve_experiment_data(exp)
    elif not isinstance(exp, ResolvedExperimentConfig):
        raise TypeError("pre-resolved data requires a ResolvedExperimentConfig")
    cfg = resolve_algorithm_config(name, exp, cfg)
    set_seed(exp.seed)                                    # training + model determinism
    adapter = adapter_factory(name)                       # the algorithm's paper alignment
    aux_backbone = None                                  # FML/FedKD shared aux model
    model_input_spec = dict(exp.input_spec)
    handle = data.handle
    generated_artifact = data.artifact
    if exp.data_backend == "biosilo":
        # BioSilo provides an existing temporal partition; RigFL derives the
        # validation split and constructs the client models for this run.
        from rigfl.data.biosilo import build_biosilo_clients, temporal_dims
        from rigfl.models.eicu import GRUTabularBackbone
        n_ts, n_static = temporal_dims(handle)
        backbone_names = resolve_model_architectures(
            architecture_family=exp.model_architecture_family,
            architectures=exp.model_architectures,
            input_kind="temporal",
        )
        backbones = instantiate_backbones(backbone_names, input_spec=model_input_spec)
        clients, handle = build_biosilo_clients(
            data.settings.source_dataset, exp.shared_dim, backbones,
            root=data.settings.data_root, partition=data.settings.partition,
            val_frac=exp.validation_fraction, batch=exp.batch, adapter=adapter)
        # FML/FedKD's shared meme/mentee must consume the same multi-input as
        # the clients. FedDES builds its pool from the experiment's same resolved
        # architectures and input description below.
        aux_backbone = lambda: GRUTabularBackbone(n_ts, n_static, 64, exp.shared_dim)
    elif exp.data_backend == "flower":
        input_kind = exp.input_kind
        if "shape" in model_input_spec:
            model_input_spec["shape"] = tuple(model_input_spec["shape"])
        backbone_names = resolve_model_architectures(
            architecture_family=exp.model_architecture_family,
            architectures=exp.model_architectures,
            input_kind=input_kind,
        )
        backbones = instantiate_backbones(backbone_names, input_spec=model_input_spec)
        aux_backbone = backbones[0]
        clients = build_partition_clients(
            generated_artifact,
            shared_dim=exp.shared_dim,
            batch=exp.batch,
            adapter=adapter,
            backbones=backbones,
        )
    else:
        raise RuntimeError(f"unresolved data backend: {exp.data_backend!r}")
    algorithm = build_algorithm(
        name, exp, cfg, aux_backbone=aux_backbone,
        model_input_spec=model_input_spec,
        model_template=clients[0].model)
    tracker = make_tracker(name, exp, cfg)               # W&B if exp.wandb else no-op
    t0 = time.time()
    runner = algorithm_spec(name).runner
    result = runner(algorithm, clients, num_rounds=exp.rounds, device=device,
                    num_classes=exp.num_classes, eval_gap=exp.eval_gap,
                    verbose=not exp.quiet, tracker=tracker,
                    early_stopping=exp.early_stopping)

    # The fingerprint is computed from the resolved experiment, including the
    # identity and metadata of the partition that was actually loaded.
    record = make_run_record(
        algorithm=name,
        experiment=exp.model_dump(), algorithm_config=cfg.model_dump(),
        run_fingerprint=run_fingerprint(exp, cfg.model_dump()),
        result=result,
        env=capture_env(), device=str(device),
        wall_seconds=round(time.time() - t0, 1),
        partition=partition_summary(
            clients, exp.num_classes, handle, generated_artifact
        ),
    )
    if tracker is not None:
        tracker.finish(result)
    return record


#: Sections accepted by a single-run YAML file.
_CONFIG_SECTIONS = ("experiment", "algorithm")
#: Prefixes accepted by ``--set``.
_SET_SECTIONS = {"exp": "experiment", "algorithm": "algorithm"}


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
    unknown = sorted(set(loaded) - set(_CONFIG_SECTIONS))
    if unknown:
        raise SystemExit(
            f"{path}: unknown top-level section(s): {', '.join(unknown)}"
            f"{_suggest(unknown[0], _CONFIG_SECTIONS)}\n"
            f"Known: {', '.join(_CONFIG_SECTIONS)}. (Sweep files with base/sweep "
            f"sections go to rigfl.experiment.launch, not here.)")
    for section in _CONFIG_SECTIONS:
        if section in loaded and not isinstance(loaded[section], dict):
            raise SystemExit(f"{path}: '{section}' must be a mapping, "
                             f"got {type(loaded[section]).__name__}")
    return dict(loaded.get("experiment") or {}), dict(loaded.get("algorithm") or {})


def _suggest(name: str, known) -> str:
    import difflib
    close = difflib.get_close_matches(name, sorted(known), n=1)
    return f' Did you mean "{close[0]}"?' if close else ""


def build_configs(args) -> tuple[ExperimentConfig, dict]:
    """Resolve the experiment config plus algorithm-config overrides from
    (optional) YAML, then convenience flags, then --set. Pydantic validates."""
    exp_over: dict = {}
    algorithm_over: dict = {}
    if args.config:
        exp_over, algorithm_over = load_run_config(args.config)
    for flag in ["dataset", "dataset_config", "data_dir", "rounds", "seed",
                 "shared_dim", "eval_gap", "device", "out_dir"]:
        v = getattr(args, flag, None)
        if v is not None:
            exp_over[flag] = v
    if args.quiet:
        exp_over["quiet"] = True
    elif "quiet" not in exp_over:
        exp_over["quiet"] = False     # run.py's own default is verbose
    if args.wandb:
        exp_over["wandb"] = True
    if args.wandb_project:
        exp_over["wandb_project"] = args.wandb_project
    for flag in ["lr", "local_epochs"]:
        v = getattr(args, flag)
        if v is not None:
            algorithm_over[flag] = v
    for kv in args.set:                              # --set exp.x=1 algorithm.lamda=10
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
        (exp_over if section == "exp" else algorithm_over)[field] = val
    return ExperimentConfig(**exp_over), algorithm_over


def _run_resolved_experiment(name: str, exp: ResolvedExperimentConfig, cfg, *,
                             data: ResolvedData, force: bool = False) -> Path:
    """Run and save one fully resolved experiment configuration."""
    device = resolve_device(exp.device)
    out_dir = Path(exp.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fp = run_fingerprint(exp, cfg.model_dump())
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

    Cfg = config_class(algorithm)
    unknown = sorted(set(algorithm_config) - set(Cfg.model_fields))
    if unknown:
        raise ValueError(
            f"unknown {algorithm} algorithm setting(s): {', '.join(unknown)}; "
            f"known: {', '.join(sorted(Cfg.model_fields))}"
        )
    cfg = resolve_algorithm_config(algorithm, exp, Cfg(**algorithm_config))
    return _run_resolved_experiment(algorithm, exp, cfg, data=data, force=force)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run one RigFL experiment.")
    p.add_argument("--algorithm", default="baselines",
                   help="algorithm name, 'baselines', or 'all'")
    p.add_argument("--config", help="YAML with 'experiment:' and 'algorithm:' sections")
    p.add_argument("--dataset")
    p.add_argument("--dataset-config")
    p.add_argument("--data-dir")
    p.add_argument("--set", nargs="*", default=[],
                   help="overrides, e.g. exp.rounds=50 algorithm.lamda=10")
    # convenience flags (experiment):
    p.add_argument("--rounds", type=int)
    p.add_argument("--seed", type=int)
    p.add_argument("--shared-dim", type=int)
    p.add_argument("--eval-gap", type=int)
    p.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"])
    p.add_argument("--out-dir")
    # convenience flags (shared algorithm options):
    p.add_argument("--lr", type=float)
    p.add_argument("--local-epochs", type=int)
    p.add_argument("--quiet", action="store_true", help="suppress per-round logging")
    p.add_argument("--force", action="store_true", help="re-run even if the result JSON exists")
    p.add_argument("--wandb", action="store_true", help="log to Weights & Biases (needs rigfl[wandb])")
    p.add_argument("--wandb-project", help="W&B project name")
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

    known = set().union(*(config_class(n).model_fields for n in algorithms))
    unknown = sorted(set(algorithm_over) - known)
    if unknown:
        raise SystemExit(
            f"unknown algorithm setting(s): {', '.join(unknown)}\n"
            f"known for {', '.join(algorithms)}: {', '.join(sorted(known))}")

    for name in algorithms:
        Cfg = config_class(name)
        # Shared overrides are applied only to algorithms that define the field.
        cfg = Cfg(**{k: v for k, v in algorithm_over.items() if k in Cfg.model_fields})
        cfg = resolve_algorithm_config(name, exp, cfg)
        try:
            _run_resolved_experiment(name, exp, cfg, data=data, force=args.force)
        except ResultValidationError as e:
            raise SystemExit(e.report())


if __name__ == "__main__":
    main()
