"""Build any algorithm by name from its typed config.

Each algorithm registers its implementation, config class, and runner.
``iterative`` is the default runner, so ordinary algorithms do not repeat it.
``build_algorithm`` resolves shared experiment resources and delegates
construction to the registered implementation's standard ``from_config`` hook.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from torch import nn

from rigfl.algorithms.apple import APPLE, APPLEConfig
from rigfl.algorithms.fedamp import FedAMP, FedAMPConfig
from rigfl.algorithms.fedavg import FedAvg, FedAvgConfig
from rigfl.algorithms.feddes import FedDES, FedDESConfig
from rigfl.algorithms.fedgh import FedGH, FedGHConfig
from rigfl.algorithms.fedkd import FedKD, FedKDConfig
from rigfl.algorithms.fedpac import FedPAC, FedPACConfig
from rigfl.algorithms.fedproto import FedProto, FedProtoConfig
from rigfl.algorithms.fedprox import FedProx, FedProxConfig
from rigfl.algorithms.fedtgp import FedTGP, FedTGPConfig
from rigfl.algorithms.fml import FML, FMLConfig
from rigfl.algorithms.global_ensemble import GlobalEnsemble, GlobalEnsembleConfig
from rigfl.algorithms.lgfedavg import LGFedAvg, LGFedAvgConfig
from rigfl.algorithms.local import Local, LocalConfig
from rigfl.core import ClientModel, LearnedProjection, iterative, p2p_one_shot
from rigfl.core.adapters import AdaptivePool
from rigfl.core.config import AlgorithmConfig
from rigfl.experiment.config import (
    ExperimentConfig,
    ResolvedExperimentConfig,
    run_fingerprint,
    run_identity,
)
from rigfl.experiment.identity import fingerprint, pre_schema_identity_input
from rigfl.models.registry import instantiate_backbones, resolve_models, validate_model


@dataclass(frozen=True)
class AlgorithmSpec:
    """Everything the experiment layer needs to construct and execute an algorithm."""

    algorithm: type
    config: type[AlgorithmConfig]
    runner: Callable = iterative
    requires_client_model: bool = True
    supports_model_heterogeneity: bool = True
    ignored_experiment_fields: tuple[str, ...] = ()


REGISTRY = {
    "local":    AlgorithmSpec(Local, LocalConfig),
    "fedavg":   AlgorithmSpec(
        FedAvg,
        FedAvgConfig,
        supports_model_heterogeneity=False,
        ignored_experiment_fields=("model_family",),
    ),
    "fedprox":  AlgorithmSpec(
        FedProx,
        FedProxConfig,
        supports_model_heterogeneity=False,
        ignored_experiment_fields=("model_family",),
    ),
    "fedamp":   AlgorithmSpec(
        FedAMP,
        FedAMPConfig,
        supports_model_heterogeneity=False,
        ignored_experiment_fields=("model_family",),
    ),
    "apple":    AlgorithmSpec(
        APPLE,
        APPLEConfig,
        supports_model_heterogeneity=False,
        ignored_experiment_fields=("model_family",),
    ),
    "fedpac":   AlgorithmSpec(
        FedPAC,
        FedPACConfig,
        supports_model_heterogeneity=False,
        ignored_experiment_fields=("model_family",),
    ),
    "global":   AlgorithmSpec(GlobalEnsemble, GlobalEnsembleConfig),
    "fedproto": AlgorithmSpec(FedProto, FedProtoConfig),
    "fedgh":    AlgorithmSpec(FedGH, FedGHConfig),
    "lgfedavg": AlgorithmSpec(LGFedAvg, LGFedAvgConfig),
    "fml":      AlgorithmSpec(FML, FMLConfig),
    "fedkd":    AlgorithmSpec(FedKD, FedKDConfig),
    "fedtgp":   AlgorithmSpec(FedTGP, FedTGPConfig),
    "feddes":   AlgorithmSpec(
        FedDES,
        FedDESConfig,
        runner=p2p_one_shot,
        requires_client_model=False,
        ignored_experiment_fields=("shared_dim",),
    ),
}

_RUNNER_IGNORED_EXPERIMENT_FIELDS = {
    p2p_one_shot: ("rounds", "eval_gap", "early_stopping"),
}

# Algorithms used by the baseline sweep, plus Local as the reference condition.
# Global Ensemble and FedDES remain callable explicitly and through ``all``.
BASELINES = ["local", "fedproto", "fedgh", "lgfedavg", "fml", "fedkd", "fedtgp"]
ALL_ALGORITHMS = BASELINES + [
    "fedavg", "fedprox", "fedamp", "apple", "fedpac", "global", "feddes",
]


def config_class(name: str) -> type[AlgorithmConfig]:
    """The config class for an algorithm."""
    if name not in REGISTRY:
        raise KeyError(f"unknown algorithm '{name}'; known: {', '.join(ALL_ALGORITHMS)}")
    return REGISTRY[name].config


def algorithm_spec(name: str) -> AlgorithmSpec:
    """The registered implementation, config class, and execution runner."""
    if name not in REGISTRY:
        raise KeyError(f"unknown algorithm '{name}'; known: {', '.join(ALL_ALGORITHMS)}")
    return REGISTRY[name]


def ignored_experiment_fields(name: str) -> tuple[str, ...]:
    """Experiment fields unused by an algorithm's runner or implementation."""
    spec = algorithm_spec(name)
    runner_fields = _RUNNER_IGNORED_EXPERIMENT_FIELDS.get(spec.runner, ())
    return tuple(dict.fromkeys((*runner_fields, *spec.ignored_experiment_fields)))


def ignores_experiment_field(name: str, field: str) -> bool:
    """Whether an experiment field or nested field is inapplicable."""
    return any(
        field == ignored or field.startswith(f"{ignored}.")
        for ignored in ignored_experiment_fields(name)
    )


def algorithm_run_fingerprint(
    name: str, exp: ResolvedExperimentConfig, algorithm_dump: dict
) -> str:
    """Run identity using only experiment fields relevant to the algorithm."""
    return run_fingerprint(
        exp,
        algorithm_dump,
        algorithm=name,
        ignored_experiment_fields=ignored_experiment_fields(name),
    )


def algorithm_run_identity(
    name: str, exp: ResolvedExperimentConfig, algorithm_dump: dict
) -> dict:
    """Exact canonical mapping hashed for one registered algorithm run."""
    return run_identity(
        exp,
        algorithm_dump,
        algorithm=name,
        ignored_experiment_fields=ignored_experiment_fields(name),
    )


def legacy_algorithm_run_fingerprint(
    name: str, exp: ResolvedExperimentConfig, algorithm_dump: dict
) -> str:
    """Pre-identity-schema fingerprint used by transitional saved results."""
    return fingerprint(
        pre_schema_identity_input(
            name,
            exp.model_dump(),
            algorithm_dump,
            ignored_experiment_fields=ignored_experiment_fields(name),
        )
    )


def resolve_algorithm_config(name: str, exp: ExperimentConfig,
                             cfg: AlgorithmConfig) -> AlgorithmConfig:
    """Validate algorithm/experiment compatibility and resolve shorthand."""
    input_kind = (
        exp.input_kind if isinstance(exp, ResolvedExperimentConfig) else None
    )
    names = resolve_models(
        model=exp.model,
        model_family=exp.model_family,
        input_kind=input_kind,
        use_family=algorithm_spec(name).supports_model_heterogeneity,
    )
    if name in {"fml", "fedkd"}:
        selected = getattr(cfg, "aux_model") or (
            names[0] if exp.model_family is not None else exp.model
        )
        validate_model(selected, input_kind)
        cfg = cfg.model_copy(update={"aux_model": selected})
    return cfg


def resolve_algorithm_models(
    name: str, exp: ResolvedExperimentConfig
) -> ResolvedExperimentConfig:
    """Record the model list actually used by one algorithm."""
    names = resolve_models(
        model=exp.model,
        model_family=exp.model_family,
        input_kind=exp.input_kind,
        use_family=algorithm_spec(name).supports_model_heterogeneity,
    )
    return exp.model_copy(update={"resolved_models": names})


def resolve_algorithm_experiment(
    name: str, exp: ResolvedExperimentConfig
) -> ResolvedExperimentConfig:
    """Resolve models and reset fields that do not apply to this algorithm."""
    exp = resolve_algorithm_models(name, exp)
    values = exp.model_dump()
    for field in ignored_experiment_fields(name):
        values.pop(field, None)
    return ResolvedExperimentConfig(**values)


# Standard client-model algorithms that align representation widths by pooling.
_POOLING_ALGORITHMS = {"fedtgp"}


def adapter_factory(name: str):
    """Return the alignment used by an algorithm's standard client model."""
    if not algorithm_spec(name).requires_client_model:
        raise ValueError(f"{name} does not use a standard client model")
    if name in _POOLING_ALGORITHMS:
        return lambda native, shared: AdaptivePool(shared)
    return lambda native, shared: LearnedProjection(native, shared)


def _aux_model(backbone, shared_dim: int, num_classes: int):
    """Construct FML's meme model or FedKD's mentee model."""
    def make() -> ClientModel:
        b = backbone()
        return ClientModel(b, LearnedProjection(b.out_dim, shared_dim), nn.Linear(shared_dim, num_classes))
    return make


def build_algorithm(name: str, exp: ResolvedExperimentConfig, cfg: AlgorithmConfig,
                    aux_backbone=None, base_pool=None, model_input_spec=None,
                    model_template=None, initial_client_models=None,
                    client_sample_counts=None):
    """Construct a registered algorithm through its standard factory hook.

    ``aux_backbone`` is the backbone factory for the shared meme or mentee model
    used by FML and FedKD. ``initial_client_models`` and
    ``client_sample_counts`` describe the clients after model and data
    construction; algorithms that require round-zero client state consume and
    copy them in their own ``from_config`` implementation."""
    exp = resolve_algorithm_experiment(name, exp)
    cfg = resolve_algorithm_config(name, exp, cfg)
    sd, nc = exp.shared_dim, exp.num_classes
    if name in {"fml", "fedkd"} and aux_backbone is None:
        aux_backbone = instantiate_backbones(
            [cfg.aux_model],
            input_spec=model_input_spec or exp.input_spec,
        )[0]
    aux_factory = (
        _aux_model(aux_backbone, sd, nc) if aux_backbone is not None else None
    )
    return algorithm_spec(name).algorithm.from_config(
        cfg,
        experiment=exp,
        aux_model_factory=aux_factory,
        base_pool=base_pool,
        model_input_spec=model_input_spec,
        model_template=model_template,
        initial_client_models=initial_client_models,
        client_sample_counts=client_sample_counts,
    )
