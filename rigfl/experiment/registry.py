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
from rigfl.algorithms.fedapa import FedAPA, FedAPAConfig
from rigfl.algorithms.fedapen import FedAPEN, FedAPENConfig
from rigfl.algorithms.fedavg import FedAvg, FedAvgConfig
from rigfl.algorithms.fedcac import FedCAC, FedCACConfig
from rigfl.algorithms.fedgh import FedGH, FedGHConfig
from rigfl.algorithms.fedpac import FedPAC, FedPACConfig
from rigfl.algorithms.fedproto import FedProto, FedProtoConfig
from rigfl.algorithms.fedprox import FedProx, FedProxConfig
from rigfl.algorithms.fedtgp import FedTGP, FedTGPConfig
from rigfl.algorithms.fml import FML, FMLConfig
from rigfl.algorithms.global_ensemble import GlobalEnsemble, GlobalEnsembleConfig
from rigfl.algorithms.local import Local, LocalConfig
from rigfl.algorithms.pfedmoe import PFedMoE, PFedMoEConfig, ProxyExtractor
from rigfl.core import ClientModel, LearnedProjection, iterative
from rigfl.core.adapters import AdaptivePool
from rigfl.core.config import AlgorithmConfig
from rigfl.experiment.config import (
    ExperimentConfig,
    ResolvedExperimentConfig,
    run_fingerprint,
    run_identity,
)
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
    "fedcac":   AlgorithmSpec(
        FedCAC,
        FedCACConfig,
        supports_model_heterogeneity=False,
        ignored_experiment_fields=("model_family",),
    ),
    "fedapa":   AlgorithmSpec(
        FedAPA,
        FedAPAConfig,
        supports_model_heterogeneity=False,
        ignored_experiment_fields=("model_family",),
    ),
    "fedapen":  AlgorithmSpec(FedAPEN, FedAPENConfig),
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
    "fml":      AlgorithmSpec(FML, FMLConfig),
    "fedtgp":   AlgorithmSpec(FedTGP, FedTGPConfig),
    "pfedmoe":  AlgorithmSpec(PFedMoE, PFedMoEConfig),
}

_RUNNER_IGNORED_EXPERIMENT_FIELDS = {}

# Algorithms used by the baseline sweep, plus Local as the reference condition.
# Global Ensemble remains callable explicitly and through ``all``.
BASELINES = ["local", "fedproto", "fedgh", "fml", "fedtgp"]
ALL_ALGORITHMS = BASELINES + [
    "fedavg", "fedprox", "fedcac", "fedapa", "fedapen", "fedamp", "apple",
    "fedpac", "pfedmoe", "global",
]


def register_algorithm(name: str, spec: AlgorithmSpec) -> None:
    """Register one explicitly imported external algorithm.

    RigFL does not discover plugins. An external research repository calls this
    function before parsing or running its configurations, and owns any custom
    runner included in ``spec``.
    """
    if not name or name != name.lower() or not name.replace("_", "").isalnum():
        raise ValueError(
            "algorithm name must be a lowercase identifier containing only "
            "letters, numbers, and underscores"
        )
    if name in REGISTRY:
        raise ValueError(f"algorithm '{name}' is already registered")
    if not isinstance(spec, AlgorithmSpec):
        raise TypeError("spec must be an AlgorithmSpec")
    if not issubclass(spec.config, AlgorithmConfig):
        raise TypeError("algorithm config must inherit from AlgorithmConfig")
    REGISTRY[name] = spec
    ALL_ALGORITHMS.append(name)


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
    if name == "fml":
        selected = cfg.aux_model or (
            names[0] if exp.model_family is not None else exp.model
        )
        validate_model(selected, input_kind)
        cfg = cfg.model_copy(update={"aux_model": selected})
    elif name == "pfedmoe":
        selected = cfg.proxy_model or (
            names[0] if exp.model_family is not None else exp.model
        )
        validate_model(selected, input_kind)
        cfg = cfg.model_copy(update={"proxy_model": selected})
    elif name == "fedapen":
        selected = cfg.shared_model or (
            names[0] if exp.model_family is not None else exp.model
        )
        validate_model(selected, input_kind)
        cfg = cfg.model_copy(update={"shared_model": selected})
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


def _complete_shared_model(backbone, shared_dim: int, num_classes: int):
    """Attach an adapter and classifier to a selected shared backbone."""
    def make() -> ClientModel:
        b = backbone()
        return ClientModel(b, LearnedProjection(b.out_dim, shared_dim), nn.Linear(shared_dim, num_classes))
    return make


def _proxy_extractor(backbone, shared_dim: int):
    """Construct pFedMoE's selected backbone as a headless proxy extractor."""
    def make() -> ProxyExtractor:
        b = backbone()
        return ProxyExtractor(b, LearnedProjection(b.out_dim, shared_dim))
    return make


def build_algorithm(name: str, exp: ResolvedExperimentConfig,
                    cfg: AlgorithmConfig, aux_backbone=None,
                    proxy_backbone=None, shared_backbone=None, base_pool=None,
                    model_input_spec=None,
                    model_template=None, initial_client_models=None,
                    client_sample_counts=None):
    """Construct a registered algorithm through its standard factory hook.

    ``aux_backbone`` builds the complete meme model used by FML;
    ``proxy_backbone`` builds pFedMoE's headless proxy extractor;
    ``shared_backbone`` builds FedAPEN's complete shared classifier.
    ``initial_client_models`` and ``client_sample_counts`` describe clients
    after model and data construction; algorithms that require round-zero
    client state copy them in their own ``from_config`` implementation.
    """
    exp = resolve_algorithm_experiment(name, exp)
    cfg = resolve_algorithm_config(name, exp, cfg)
    sd, nc = exp.shared_dim, exp.num_classes
    if name == "fml" and aux_backbone is None:
        aux_backbone = instantiate_backbones(
            [cfg.aux_model],
            input_spec=model_input_spec or exp.input_spec,
        )[0]
    if name == "pfedmoe" and proxy_backbone is None:
        proxy_backbone = instantiate_backbones(
            [cfg.proxy_model],
            input_spec=model_input_spec or exp.input_spec,
        )[0]
    if name == "fedapen" and shared_backbone is None:
        shared_backbone = instantiate_backbones(
            [cfg.shared_model],
            input_spec=model_input_spec or exp.input_spec,
        )[0]
    aux_factory = (
        _complete_shared_model(aux_backbone, sd, nc)
        if aux_backbone is not None else None
    )
    return algorithm_spec(name).algorithm.from_config(
        cfg,
        experiment=exp,
        aux_model_factory=aux_factory,
        proxy_extractor_factory=(
            _proxy_extractor(proxy_backbone, sd)
            if proxy_backbone is not None else None
        ),
        shared_model_factory=(
            _complete_shared_model(shared_backbone, sd, nc)
            if shared_backbone is not None else None
        ),
        base_pool=base_pool,
        model_input_spec=model_input_spec,
        model_template=model_template,
        initial_client_models=initial_client_models,
        client_sample_counts=client_sample_counts,
    )
