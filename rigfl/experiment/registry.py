"""Build any algorithm by name from its typed config.

Each algorithm registers its implementation, config class, and runner.
``iterative`` is the default runner, so ordinary algorithms do not repeat it.
``build_algorithm`` resolves shared experiment resources and delegates
construction to the registered implementation's standard ``from_config`` hook.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch.nn as nn

from rigfl.core import ClientModel, LearnedProjection, iterative, p2p_one_shot
from rigfl.core.adapters import AdaptivePool
from rigfl.core.config import AlgorithmConfig
from rigfl.experiment.config import (ExperimentConfig, ResolvedExperimentConfig,
                                     run_fingerprint)
from rigfl.models.registry import (instantiate_backbones, resolve_models,
                                   validate_model)
from rigfl.algorithms.fedavg import FedAvg, FedAvgConfig
from rigfl.algorithms.fedprox import FedProx, FedProxConfig
from rigfl.algorithms.local import Local, LocalConfig
from rigfl.algorithms.global_ensemble import GlobalEnsemble, GlobalEnsembleConfig
from rigfl.algorithms.fedproto import FedProto, FedProtoConfig
from rigfl.algorithms.fedgh import FedGH, FedGHConfig
from rigfl.algorithms.lgfedavg import LGFedAvg, LGFedAvgConfig
from rigfl.algorithms.fml import FML, FMLConfig
from rigfl.algorithms.fedkd import FedKD, FedKDConfig
from rigfl.algorithms.fedtgp import FedTGP, FedTGPConfig
from rigfl.algorithms.feddes import FedDES, FedDESConfig


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

# Algorithms used by the baseline sweep, plus Local as the reference condition.
# Global Ensemble and FedDES remain callable explicitly and through ``all``.
BASELINES = ["local", "fedproto", "fedgh", "lgfedavg", "fml", "fedkd", "fedtgp"]
ALL_ALGORITHMS = BASELINES + ["fedavg", "fedprox", "global", "feddes"]


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


def algorithm_run_fingerprint(
    name: str, exp: ResolvedExperimentConfig, algorithm_dump: dict
) -> str:
    """Run identity using only experiment fields relevant to the algorithm."""
    return run_fingerprint(
        exp,
        algorithm_dump,
        ignored_experiment_fields=algorithm_spec(name).ignored_experiment_fields,
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
                    model_template=None):
    """Construct a registered algorithm through its standard factory hook.

    ``aux_backbone`` is the backbone factory for the shared meme or mentee model
    used by FML and FedKD."""
    exp = resolve_algorithm_models(name, exp)
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
    )
