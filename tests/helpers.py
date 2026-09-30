"""Small factories for complete, resolved experiment records in unit tests."""

from __future__ import annotations

from pathlib import Path

from rigfl.experiment.config import ExperimentConfig, ResolvedExperimentConfig
from rigfl.experiment.registry import (
    resolve_algorithm_config,
    resolve_algorithm_models,
)
from rigfl.experiment.run import run_one

STARTER_CONFIGS = Path(__file__).parents[1] / "rigfl" / "templates" / "configs"
STARTER_DATASETS = str(STARTER_CONFIGS / "datasets.yaml")


def resolved_experiment(**overrides) -> ResolvedExperimentConfig:
    """Build a resolved image experiment without loading a partition artifact."""
    experiment = {
        key: value
        for key, value in overrides.items()
        if key in ExperimentConfig.model_fields
    }
    resolved = {
        "data_backend": "flower",
        "partition_id": "test-partition",
        "partition_scheme": "dirichlet",
        "num_clients": 2,
        "num_classes": 3,
        "validation_fraction": 0.2,
        "input_kind": "image",
        "input_spec": {"input_kind": "image", "shape": [3, 32, 32]},
        "resolved_models": [overrides.get("model_arch", "fedavg_cnn")],
    }
    resolved.update({
        key: value
        for key, value in overrides.items()
        if key in ResolvedExperimentConfig.model_fields
        and key not in ExperimentConfig.model_fields
    })
    return ResolvedExperimentConfig(**experiment, **resolved)


def run_resolved(name, exp, config, device, data) -> dict:
    """Resolve a data-resolved experiment for one algorithm, then run it."""
    exp = resolve_algorithm_models(name, exp)
    config = resolve_algorithm_config(name, exp, config)
    return run_one(name, exp, config, device, data=data)
