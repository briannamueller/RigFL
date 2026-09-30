"""Named reporting filters and stable CSV output."""

from __future__ import annotations

import pytest

from rigfl.experiment.reporting_config import (
    ReportingConfigError,
    apply_named_filter,
    load_reporting_config,
)
from tests.helpers import STARTER_CONFIGS


def _record(algorithm: str, dataset: str, partition: int, split: int, seed: int):
    return {
        "algorithm": algorithm,
        "config": {
            "experiment": {
                "dataset": dataset,
                "partition_seed": partition,
                "split_seed": split,
                "training_seed": seed,
            },
            "algorithm": {"lr": 0.1},
        },
    }


def test_named_filter_uses_or_within_fields_and_and_across_fields():
    records = [
        _record("local", "cifar10", 0, 0, 0),
        _record("fedavg", "cifar10", 0, 0, 1),
        _record("fedprox", "cifar10", 0, 0, 2),
        _record("fedavg", "mnist", 0, 0, 3),
    ]
    config = {
        "filters": {
            "main": {
                "algorithm": ["local", "fedavg"],
                "dataset": ["cifar10"],
            }
        }
    }

    matches, resolved = apply_named_filter(records, config, "main")

    assert [record["algorithm"] for record in matches] == ["local", "fedavg"]
    assert resolved["dataset"] == ["cifar10"]


def test_null_experiment_setting_matches_any_filter_value():
    fixed = _record("local", "mnist", 0, 0, 0)
    fixed["config"]["experiment"]["rounds"] = 3
    ignored = _record("extalgo", "mnist", 0, 0, 0)
    ignored["config"]["experiment"]["rounds"] = None
    config = {"filters": {"five": {"rounds": 5}}}

    matches, _ = apply_named_filter([fixed, ignored], config, "five")

    assert matches == [ignored]


def test_filter_rejects_unknown_or_unobserved_paths():
    records = [_record("local", "cifar10", 0, 0, 0)]
    with pytest.raises(ReportingConfigError, match="references unknown path"):
        apply_named_filter(records, {"filters": {"x": {"method": ["local"]}}}, "x")


def test_included_reporting_example_matches_the_starter_sweep():
    config = load_reporting_config(
        STARTER_CONFIGS / "reporting.yaml"
    )

    assert config["filters"]["model_heterogeneous"] == {
        "algorithm": ["local", "fedgh", "fedproto"],
        "model_arch": ["mnist_heterogeneous_3"],
    }
