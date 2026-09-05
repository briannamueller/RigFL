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
from rigfl.experiment.config import ExperimentConfig, ResolvedExperimentConfig
from rigfl.models.cifar import SmallCNN
from rigfl.models.registry import resolve_model_architectures
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


REGISTRY = {
    "local":    AlgorithmSpec(Local, LocalConfig),
    "fedavg":   AlgorithmSpec(FedAvg, FedAvgConfig),
    "fedprox":  AlgorithmSpec(FedProx, FedProxConfig),
    "global":   AlgorithmSpec(GlobalEnsemble, GlobalEnsembleConfig),
    "fedproto": AlgorithmSpec(FedProto, FedProtoConfig),
    "fedgh":    AlgorithmSpec(FedGH, FedGHConfig),
    "lgfedavg": AlgorithmSpec(LGFedAvg, LGFedAvgConfig),
    "fml":      AlgorithmSpec(FML, FMLConfig),
    "fedkd":    AlgorithmSpec(FedKD, FedKDConfig),
    "fedtgp":   AlgorithmSpec(FedTGP, FedTGPConfig),
    "feddes":   AlgorithmSpec(FedDES, FedDESConfig, runner=p2p_one_shot),
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


def resolve_algorithm_config(name: str, exp: ExperimentConfig,
                             cfg: AlgorithmConfig) -> AlgorithmConfig:
    """Validate algorithm/experiment compatibility and resolve shorthand."""
    if isinstance(exp, ResolvedExperimentConfig):
        input_kind = exp.input_kind
    else:
        from rigfl.data.config import BioSiloDatasetSettings, dataset_settings
        settings = dataset_settings(exp.dataset, exp.dataset_config)
        input_kind = "temporal" if isinstance(settings, BioSiloDatasetSettings) else "image"
    names = resolve_model_architectures(
        architecture_family=exp.model_architecture_family,
        architectures=exp.model_architectures,
        input_kind=input_kind,
    )
    if name in {"fedavg", "fedprox"}:
        _validate_homogeneous_model_architecture(name, exp, names)
    return cfg


def _validate_homogeneous_model_architecture(
    name: str, exp: ExperimentConfig, names: list[str]
) -> None:
    """Traditional full-model aggregation needs one resolved architecture."""
    label = "FedAvg" if name == "fedavg" else "FedProx"
    source = (
        f"model_architecture_family={exp.model_architecture_family!r} resolves "
        f"to {len(names)} architectures"
        if exp.model_architecture_family is not None
        else f"model_architectures contains {len(names)} architectures"
    )

    if len(names) != 1:
        raise ValueError(
            f"{label} requires exactly one model architecture, but {source}. "
            "Set experiment.model_architectures to a one-item list; RigFL will "
            "construct a separate fresh model for every client."
        )


# Algorithms whose own paper aligns representation widths by pooling. Every other
# algorithm's paper uses a learned projection, which is the default.
_POOLING_ALGORITHMS = {"fedtgp"}


def adapter_factory(name: str):
    """A ``(native_dim, shared_dim) -> Adapter`` factory for an algorithm, using the
    alignment its own paper specifies: FedTGP pools, everything else learns a
    linear projection. See :mod:`rigfl.core.adapters`."""
    if name in _POOLING_ALGORITHMS:
        return lambda native, shared: AdaptivePool(shared)
    return lambda native, shared: LearnedProjection(native, shared)


def _default_aux_backbone(shared_dim: int):
    """The shared aux backbone when the caller doesn't supply one: a small CIFAR
    CNN. For multi-input datasets (eICU) the caller passes a temporal backbone."""
    return lambda: SmallCNN((16, 32), shared_dim)


def _aux_model(backbone, shared_dim: int, num_classes: int):
    """A shared homogeneous aux ClientModel (FML's meme / FedKD's mentee) from a
    single backbone architecture. The backbone must suit the dataset (a temporal
    one for eICU), so the caller supplies it rather than it being a fixed CNN."""
    def make() -> ClientModel:
        b = backbone()
        return ClientModel(b, LearnedProjection(b.out_dim, shared_dim), nn.Linear(shared_dim, num_classes))
    return make


def build_algorithm(name: str, exp: ResolvedExperimentConfig, cfg: AlgorithmConfig,
                    aux_backbone=None, base_pool=None, model_input_spec=None,
                    model_template=None):
    """Construct a registered algorithm through its standard factory hook.

    ``aux_backbone`` is the (dataset-appropriate) backbone factory for the shared
    meme/mentee that FML and FedKD need; run_one supplies a temporal one for a
    multi-input (eICU) partition. Defaults to a small CIFAR CNN."""
    cfg = resolve_algorithm_config(name, exp, cfg)
    sd, nc = exp.shared_dim, exp.num_classes
    aux = aux_backbone or _default_aux_backbone(sd)
    return algorithm_spec(name).algorithm.from_config(
        cfg,
        experiment=exp,
        aux_model_factory=_aux_model(aux, sd, nc),
        base_pool=base_pool,
        model_input_spec=model_input_spec,
        model_template=model_template,
    )
