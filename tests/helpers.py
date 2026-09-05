"""Small factories for complete, resolved experiment records in unit tests."""

from __future__ import annotations

from rigfl.experiment.config import ExperimentConfig, ResolvedExperimentConfig


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
    }
    resolved.update({
        key: value
        for key, value in overrides.items()
        if key in ResolvedExperimentConfig.model_fields
        and key not in ExperimentConfig.model_fields
    })
    return ResolvedExperimentConfig(**experiment, **resolved)
