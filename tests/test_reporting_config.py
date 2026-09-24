"""Named reporting filters and stable CSV output."""

from __future__ import annotations

import csv
import io
from pathlib import Path

import pytest

from rigfl.experiment.reporting_config import (
    ReportingConfigError,
    apply_named_filter,
    csv_text,
    load_reporting_config,
    seed_summary,
    selection_defaults,
)


def _record(algorithm: str, dataset: str, partition: int, split: int, seed: int):
    return {
        "algorithm": algorithm,
        "config": {
            "experiment": {
                "dataset": dataset,
                "partition_seed": partition,
                "split_seed": split,
                "seed": seed,
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
                "config.experiment.dataset": ["cifar10"],
            }
        }
    }

    matches, resolved = apply_named_filter(records, config, "main")

    assert [record["algorithm"] for record in matches] == ["local", "fedavg"]
    assert resolved["config.experiment.dataset"] == ["cifar10"]


def test_filter_rejects_unknown_or_unobserved_paths():
    records = [_record("local", "cifar10", 0, 0, 0)]
    with pytest.raises(ReportingConfigError, match="unknown reporting path"):
        apply_named_filter(records, {"filters": {"x": {"method": ["local"]}}}, "x")
    with pytest.raises(ReportingConfigError, match="references unknown path"):
        apply_named_filter(
            records,
            {"filters": {"x": {"config.algorithm.momentum": [0.9]}}},
            "x",
        )


def test_seed_summary_names_which_components_varied():
    records = [
        _record("fedavg", "cifar10", 4, 8, 0),
        _record("fedavg", "cifar10", 4, 8, 1),
    ]

    summary = seed_summary(records)

    assert summary["varied"] == ["training"]
    assert summary["fixed"] == ["partition", "split"]
    assert summary["values"]["training"] == [0, 1]


def test_reporting_yaml_has_one_supported_confidence_level(tmp_path):
    path = tmp_path / "reporting.yaml"
    path.write_text("version: 1\ndefaults:\n  confidence_level: 0.9\n")
    config = load_reporting_config(path)

    with pytest.raises(ReportingConfigError, match="only 0.95"):
        selection_defaults(config)


def test_included_reporting_example_matches_the_starter_sweep():
    config = load_reporting_config(
        Path(__file__).parents[1] / "configs/reporting.yaml"
    )

    assert config["filters"]["main_results"]["algorithm"] == ["local", "fedavg"]
    assert set(config) == {"version", "defaults", "filters"}


def test_csv_output_preserves_columns_and_represents_missing_values_as_empty():
    rendered = csv_text([
        {"algorithm": "local", "seed_values": [0, 1], "sd": None},
        {"algorithm": "fedavg", "seed_values": [0, 1], "sd": 0.2},
    ])
    rows = list(csv.DictReader(io.StringIO(rendered)))

    assert rows[0] == {"algorithm": "local", "seed_values": "0;1", "sd": ""}
    assert rows[1]["sd"] == "0.2"
