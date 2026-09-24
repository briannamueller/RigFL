"""The public construction and runner contract for registered algorithms."""

from __future__ import annotations

import inspect

import pytest

from rigfl.core.config import AlgorithmConfig
from rigfl.core.interfaces import Algorithm
from rigfl.experiment.registry import (
    ALL_ALGORITHMS,
    REGISTRY,
    AlgorithmSpec,
    build_algorithm,
    register_algorithm,
)


def test_default_construction_hook_stores_the_validated_configuration():
    class ExampleConfig(AlgorithmConfig):
        lr: float = 0.1

    class ExampleAlgorithm(Algorithm):
        pass

    config = ExampleConfig(lr=0.2)
    algorithm = ExampleAlgorithm.from_config(config, unused_resource=object())

    assert algorithm.config is config


def test_one_registry_entry_is_enough_to_construct_an_ordinary_algorithm(
    monkeypatch,
):
    from tests.helpers import resolved_experiment

    class ExampleConfig(AlgorithmConfig):
        lr: float = 0.1

    class ExampleAlgorithm(Algorithm):
        pass

    monkeypatch.setitem(
        REGISTRY, "example", AlgorithmSpec(ExampleAlgorithm, ExampleConfig))
    config = ExampleConfig(lr=0.2)

    algorithm = build_algorithm("example", resolved_experiment(), config)

    assert isinstance(algorithm, ExampleAlgorithm)
    assert algorithm.config is config


def test_external_algorithm_registration_is_explicit_and_refuses_replacement():
    class ExternalConfig(AlgorithmConfig):
        pass

    class ExternalAlgorithm(Algorithm):
        pass

    spec = AlgorithmSpec(
        ExternalAlgorithm,
        ExternalConfig,
        requires_client_model=False,
        ignored_experiment_fields=("rounds",),
    )
    try:
        register_algorithm("external_example", spec)

        assert REGISTRY["external_example"] is spec
        assert ALL_ALGORITHMS[-1] == "external_example"
        with pytest.raises(ValueError, match="already registered"):
            register_algorithm("external_example", spec)
    finally:
        REGISTRY.pop("external_example", None)
        if "external_example" in ALL_ALGORITHMS:
            ALL_ALGORITHMS.remove("external_example")


def test_registered_algorithms_accept_the_standard_runner_calls():
    """The runner calls positionally; local parameter names are not a contract."""
    argument_counts = {
        "init_globals": 0,
        "local_train": 2,
        "aggregate": 2,
        "predict": 3,
    }
    for algorithm_name, spec in REGISTRY.items():
        for operation, argument_count in argument_counts.items():
            method = getattr(spec.algorithm, operation, None)
            assert callable(method), f"{algorithm_name} is missing {operation}"
            arguments = [object()] * (argument_count + 1)  # unbound self + runner args
            try:
                inspect.signature(method).bind(*arguments)
            except TypeError as exc:
                raise AssertionError(
                    f"{algorithm_name}.{operation} cannot accept the standard "
                    "runner arguments"
                ) from exc
