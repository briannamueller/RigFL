"""Build any algorithm by name from its typed config.

Each algorithm registers its implementation, config class, and runner.
``iterative`` is the default runner, so ordinary algorithms do not repeat it.
Installed packages add algorithms through the ``rigfl.algorithms`` entry-point
group.
``build_algorithm`` gathers shared experiment resources and delegates
construction to the registered implementation's standard ``from_config`` hook.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from importlib import metadata
from typing import Literal

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
from rigfl.algorithms.pfedmoe import PFedMoE, PFedMoEConfig
from rigfl.core import AdaptivePool, LearnedProjection, iterative
from rigfl.core.config import AlgorithmConfig
from rigfl.experiment.config import (
    ExperimentConfig,
    ResolvedExperimentConfig,
)
from rigfl.experiment.identity import (
    ENVIRONMENT_ONLY_EXPERIMENT_FIELDS,
    normalize_early_stopping,
)
from rigfl.experiment.paths import flatten_mapping, model_paths, nested_set
from rigfl.models.registry import (
    instantiate_backbones,
    is_model_family,
    resolve_models,
    validate_model,
)


@dataclass(frozen=True)
class AlgorithmSpec:
    """Everything the experiment layer needs to construct and execute an algorithm.

    ``representation_adapter`` is for algorithms that exchange or mix
    representations across clients: "projection" or "pool" maps every
    backbone to ``shared_dim``. With "none", heads sit on the backbone's native
    output and ``shared_dim`` is not part of the run.

    ``provenance_packages`` names extra distributions whose version and source
    commit are recorded with each run.
    """

    algorithm: type
    config: type[AlgorithmConfig]
    runner: Callable = iterative
    requires_client_model: bool = True
    supports_model_heterogeneity: bool = True
    ignored_experiment_fields: tuple[str, ...] = ()
    representation_adapter: Literal["none", "projection", "pool"] = "none"
    provenance_packages: tuple[str, ...] = ()


REGISTRY = {
    "local":    AlgorithmSpec(Local, LocalConfig),
    "fedproto": AlgorithmSpec(
        FedProto, FedProtoConfig, representation_adapter="projection"
    ),
    "fedgh":    AlgorithmSpec(FedGH, FedGHConfig, representation_adapter="projection"),
    "fml":      AlgorithmSpec(FML, FMLConfig),
    "fedtgp":   AlgorithmSpec(FedTGP, FedTGPConfig, representation_adapter="pool"),
    "fedavg":   AlgorithmSpec(
        FedAvg,
        FedAvgConfig,
        supports_model_heterogeneity=False,
    ),
    "fedprox":  AlgorithmSpec(
        FedProx,
        FedProxConfig,
        supports_model_heterogeneity=False,
    ),
    "fedcac":   AlgorithmSpec(
        FedCAC,
        FedCACConfig,
        supports_model_heterogeneity=False,
    ),
    "fedapa":   AlgorithmSpec(
        FedAPA,
        FedAPAConfig,
        supports_model_heterogeneity=False,
    ),
    "fedapen":  AlgorithmSpec(FedAPEN, FedAPENConfig),
    "fedamp":   AlgorithmSpec(
        FedAMP,
        FedAMPConfig,
        supports_model_heterogeneity=False,
    ),
    "apple":    AlgorithmSpec(
        APPLE,
        APPLEConfig,
        supports_model_heterogeneity=False,
    ),
    "fedpac":   AlgorithmSpec(
        FedPAC,
        FedPACConfig,
        supports_model_heterogeneity=False,
    ),
    "pfedmoe":  AlgorithmSpec(
        PFedMoE, PFedMoEConfig, representation_adapter="projection"
    ),
    "global_ensemble": AlgorithmSpec(GlobalEnsemble, GlobalEnsembleConfig),
}

# report order; register_algorithm appends to it
ALL_ALGORITHMS = list(REGISTRY)
BUILT_IN_ALGORITHMS = tuple(REGISTRY)


def register_algorithm(name: str, spec: AlgorithmSpec) -> None:
    """Register one external algorithm.

    Installed packages are registered automatically from their
    ``rigfl.algorithms`` entry points; call this directly only for code that is
    not installed as a package. The external package owns any custom runner
    included in ``spec``.
    """
    if name in REGISTRY:
        raise ValueError(f"algorithm '{name}' is already registered")
    REGISTRY[name] = spec
    ALL_ALGORITHMS.append(name)


def register_entry_points(entry_points: Iterable[metadata.EntryPoint]) -> None:
    """Register each ``rigfl.algorithms`` entry point's ``AlgorithmSpec``."""
    for ep in entry_points:
        spec = ep.load()
        # record the providing distribution with every run
        spec = replace(
            spec, provenance_packages=(ep.dist.name, *spec.provenance_packages)
        )
        register_algorithm(ep.name, spec)



def config_class(name: str) -> type[AlgorithmConfig]:
    """The config class for an algorithm."""
    return algorithm_spec(name).config


def configuration_owner(path: str, algorithms: list[str] | tuple[str, ...]) -> str:
    """Return the internal owner of one flat public configuration setting."""
    experiment_paths = model_paths(ExperimentConfig)
    algorithm_paths = {
        field
        for name in algorithms
        for field in model_paths(config_class(name))
    }
    in_experiment = path in experiment_paths
    in_algorithm = path in algorithm_paths
    if in_experiment and in_algorithm:
        raise ValueError(
            f"configuration setting {path!r} is ambiguous between the shared "
            "experiment protocol and an algorithm"
        )
    if in_experiment:
        return "experiment"
    if in_algorithm:
        return "algorithm"
    known = sorted(experiment_paths | algorithm_paths)
    import difflib

    close = difflib.get_close_matches(path, known, n=1)
    suggestion = f"; did you mean {close[0]!r}?" if close else ""
    raise ValueError(
        f"unknown configuration setting {path!r}{suggestion}; "
        f"known: {', '.join(known)}"
    )


def split_public_configuration(
    values: dict, algorithms: list[str] | tuple[str, ...]
) -> tuple[dict, dict]:
    """Split one flat public mapping into the two internal validated models."""
    experiment: dict = {}
    algorithm: dict = {}
    for path, value in flatten_mapping(values).items():
        owner = configuration_owner(path, algorithms)
        nested_set(experiment if owner == "experiment" else algorithm, path, value)
    return experiment, algorithm


def algorithm_spec(name: str) -> AlgorithmSpec:
    """The registered implementation, config class, and execution runner."""
    if name not in REGISTRY:
        raise KeyError(f"unknown algorithm '{name}'; known: {', '.join(ALL_ALGORITHMS)}")
    return REGISTRY[name]


def ignored_experiment_fields(name: str) -> tuple[str, ...]:
    """Experiment fields unused by an algorithm's runner or implementation."""
    spec = algorithm_spec(name)
    if spec.requires_client_model and spec.representation_adapter == "none":
        return (*spec.ignored_experiment_fields, "shared_dim")
    return spec.ignored_experiment_fields


def ignores_experiment_field(name: str, field: str) -> bool:
    """Whether an experiment field or nested field is inapplicable."""
    return any(
        field == ignored or field.startswith(f"{ignored}.")
        for ignored in ignored_experiment_fields(name)
    )


def run_identity(
    name: str, exp: ResolvedExperimentConfig, algorithm_dump: dict
) -> dict:
    """The canonical mapping hashed for one run of a registered algorithm."""
    experiment = exp.model_dump()
    for key in (*ENVIRONMENT_ONLY_EXPERIMENT_FIELDS, *ignored_experiment_fields(name)):
        experiment.pop(key, None)
    for key in ("partition_seed", "split_seed"):
        if experiment.get(key) is None:
            experiment.pop(key, None)
    # resolved_models carries the architectures actually assigned
    experiment.pop("model_arch", None)
    experiment["early_stopping"] = normalize_early_stopping(
        experiment.get("early_stopping")
    )
    return {"experiment": experiment, "algorithm": algorithm_dump}


def supports_model_arch(name: str, model_arch: str) -> bool:
    """Whether an algorithm can run with a model_arch value."""
    return (
        algorithm_spec(name).supports_model_heterogeneity
        or not is_model_family(model_arch)
    )


def _input_kind(exp: ExperimentConfig) -> str | None:
    return exp.input_kind if isinstance(exp, ResolvedExperimentConfig) else None


def _algorithm_models(name: str, exp: ExperimentConfig) -> list[str]:
    if not supports_model_arch(name, exp.model_arch):
        raise ValueError(
            f"{name} requires identical client architectures; model_arch must "
            f"be an architecture name, not the model family {exp.model_arch!r}"
        )
    return resolve_models(exp.model_arch, input_kind=_input_kind(exp))


# auxiliary models built from their own architecture setting
_AUX_MODEL_ALGORITHMS = ("fml", "pfedmoe", "fedapen")


def resolve_algorithm_config(name: str, exp: ExperimentConfig,
                             cfg: AlgorithmConfig) -> AlgorithmConfig:
    """Validate algorithm/experiment compatibility and resolve shorthand."""
    names = _algorithm_models(name, exp)
    if name in _AUX_MODEL_ALGORITHMS:
        selected = validate_model(cfg.aux_model_arch or names[0], _input_kind(exp))
        cfg = cfg.model_copy(update={"aux_model_arch": selected})
    return cfg


def resolve_algorithm_models(
    name: str, exp: ResolvedExperimentConfig
) -> ResolvedExperimentConfig:
    """Record the model list actually used by one algorithm."""
    return exp.model_copy(update={"resolved_models": _algorithm_models(name, exp)})


def recorded_experiment(name: str, exp: ResolvedExperimentConfig) -> dict:
    """The experiment saved with a run; fields the algorithm ignores are null.

    ``run_identity`` leaves those fields out, so one record can stand for
    every declared value of them.
    """
    experiment = exp.model_dump()
    for field in ignored_experiment_fields(name):
        if field in experiment:
            experiment[field] = None
    return experiment


def adapter_factory(name: str):
    """Return the adapter factory for an algorithm's standard client model.

    ``None`` means the head sits directly on the backbone's native output.
    """
    spec = algorithm_spec(name)
    if not spec.requires_client_model:
        raise ValueError(f"{name} does not use a standard client model")
    if spec.representation_adapter == "pool":
        return lambda native, shared: AdaptivePool(shared)
    if spec.representation_adapter == "projection":
        return LearnedProjection
    return None


def build_algorithm(name: str, exp: ResolvedExperimentConfig,
                    cfg: AlgorithmConfig, model_input_spec=None,
                    model_template=None, initial_client_models=None,
                    client_sample_counts=None):
    """Construct a registered algorithm through its standard factory hook.

    ``exp`` and ``cfg`` must already be resolved for the algorithm.
    ``aux_backbone_factory`` builds the backbone named by ``aux_model_arch``
    for algorithms with an auxiliary shared model.
    ``initial_client_models`` and ``client_sample_counts`` describe clients
    after model and data construction; algorithms that require round-zero
    client state copy them in their own ``from_config`` implementation.
    """
    aux_backbone_factory = None
    if name in _AUX_MODEL_ALGORITHMS:
        aux_backbone_factory = instantiate_backbones(
            [cfg.aux_model_arch], input_spec=model_input_spec or exp.input_spec,
        )[0]
    return algorithm_spec(name).algorithm.from_config(
        cfg,
        experiment=exp,
        aux_backbone_factory=aux_backbone_factory,
        model_input_spec=model_input_spec,
        model_template=model_template,
        initial_client_models=initial_client_models,
        client_sample_counts=client_sample_counts,
    )


# last, so an entry point can import anything defined above
register_entry_points(metadata.entry_points(group="rigfl.algorithms"))
