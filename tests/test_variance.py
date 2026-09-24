"""Crossed seed variance-pilot analysis."""

from __future__ import annotations

import json

import pytest

from rigfl.experiment.artifacts import make_run_record, write_run_record
from rigfl.experiment.config import result_filename
from rigfl.experiment.registry import algorithm_run_fingerprint, config_class
from rigfl.experiment.variance import (
    VariancePilotError,
    analyze_variance_pilot,
    main,
)
from tests.helpers import resolved_experiment


def _record(partition_seed: int, split_seed: int, experiment_seed: int) -> dict:
    score = 0.5 + 0.1 * partition_seed + 0.02 * split_seed + 0.01 * experiment_seed
    partition_id = f"partition-{partition_seed}-{split_seed}"
    experiment = resolved_experiment(
        dataset="cifar10",
        partition_id=partition_id,
        partition_seed=partition_seed,
        split_seed=split_seed,
        seed=experiment_seed,
        resolved_models=["fedavg_cnn", "fedavg_cnn"],
    )
    algorithm_config = config_class("local")().model_dump()
    clients = {
        str(client_id): {
            "validation": {"accuracy": [score]},
            "test": {"accuracy": [score]},
        }
        for client_id in range(experiment.num_clients)
    }
    counts = {
        split: {
            str(client_id): [10] for client_id in range(experiment.num_clients)
        }
        for split in ("validation", "test")
    }
    record = make_run_record(
        algorithm="local",
        experiment=experiment.model_dump(mode="json"),
        algorithm_config=algorithm_config,
        run_fingerprint=algorithm_run_fingerprint(
            "local", experiment, algorithm_config
        ),
        result={
            "selection_views_supported": ["global", "per-client"],
            "evaluation_history": {
                "evaluation_rounds": [0],
                "clients": clients,
                "client_sample_counts": counts,
                "aggregate_metrics": {
                    "validation": {"accuracy": [score]},
                    "test": {"accuracy": [score]},
                },
            },
            "early_stopping": {
                "enabled": False,
                "termination_reason": "completed_all_rounds",
                "stopped_at_round": 0,
                "metric": None,
                "direction": None,
                "split": None,
                "aggregation": None,
                "patience": None,
                "min_delta": None,
                "best_round": None,
                "best_value": None,
            },
        },
        partition={
            "generated": {
                "partition_id": partition_id,
                "settings": {
                    "source_dataset": "cifar10",
                    "partition": {
                        "scheme": "dirichlet",
                        "num_clients": experiment.num_clients,
                        "alpha": 0.5,
                        "partition_seed": partition_seed,
                        "split_seed": split_seed,
                    },
                },
            }
        },
    )
    record["_source_file"] = (
        f"result-{partition_seed}-{split_seed}-{experiment_seed}.json"
    )
    return record


def _crossed_records() -> list[dict]:
    return [
        _record(partition_seed, split_seed, experiment_seed)
        for partition_seed in range(2)
        for split_seed in range(2)
        for experiment_seed in range(2)
    ]


def test_variance_pilot_reports_marginal_seed_spreads():
    artifact = analyze_variance_pilot(_crossed_records())

    assert artifact["kind"] == "rigfl.variance_pilot"
    assert artifact["selection"]["split"] == "validation"
    assert artifact["selection"]["direction"] == "maximize"
    group = artifact["groups"][0]
    assert group["runs"] == 8
    assert group["ranked_seed_sources"] == [
        "partition_seed",
        "split_seed",
        "experiment_seed",
    ]
    assert group["seed_sources"]["partition_seed"][
        "marginal_spread"
    ] == pytest.approx(0.1)
    assert group["seed_sources"]["split_seed"][
        "marginal_spread"
    ] == pytest.approx(0.02)
    assert group["seed_sources"]["experiment_seed"][
        "marginal_spread"
    ] == pytest.approx(0.01)
    assert group["cells"][0]["selected_round"] == 0


def test_variance_pilot_labels_different_datasets_and_configurations():
    records = []
    for dataset in ("cifar10", "mnist"):
        for lr in (0.01, 0.02):
            for record in _crossed_records():
                record["config"]["experiment"]["dataset"] = dataset
                record["partition"]["generated"]["settings"][
                    "source_dataset"
                ] = dataset
                record["config"]["algorithm"] = {
                    "local_epochs": 1,
                    "lr": lr,
                }
                records.append(record)

    artifact = analyze_variance_pilot(records)
    labels = [group["label"] for group in artifact["groups"]]

    assert len(set(labels)) == 4
    assert all("dataset=" in label and "lr=" in label for label in labels)


def test_variance_pilot_reports_selection_fallback_and_mixed_rounds():
    records = _crossed_records()
    for record in records:
        record["result"]["selection_views_supported"] = ["per-client"]

    artifact = analyze_variance_pilot(records, view="global")
    group = artifact["groups"][0]

    assert group["selection_view"] == "per-client"
    assert group["selection_view_fallback"] is True
    assert group["mixed_rounds"] is True


def test_variance_pilot_marks_tied_source_rankings():
    records = _crossed_records()
    for record in records:
        for client in record["result"]["evaluation_history"]["clients"].values():
            client["validation"]["accuracy"] = [0.5]

    artifact = analyze_variance_pilot(records)
    group = artifact["groups"][0]

    assert group["source_ranking_ties"] == [[
        "experiment_seed", "partition_seed", "split_seed"
    ]]
    assert {source["rank"] for source in group["seed_sources"].values()} == {1}


def test_variance_pilot_ties_spreads_that_print_identically():
    records = _crossed_records()
    for record in records:
        experiment = record["config"]["experiment"]
        value = (
            0.5
            + 0.020014 * experiment["partition_seed"]
            + 0.019986 * experiment["split_seed"]
            + 0.01 * experiment["seed"]
        )
        for client in record["result"]["evaluation_history"]["clients"].values():
            client["validation"]["accuracy"] = [value]

    group = analyze_variance_pilot(records)["groups"][0]

    assert group["seed_sources"]["partition_seed"]["rank"] == 1
    assert group["seed_sources"]["split_seed"]["rank"] == 1
    assert group["source_ranking_ties"][0] == [
        "partition_seed", "split_seed"
    ]


def test_variance_pilot_records_lower_is_better_direction():
    records = _crossed_records()
    for record in records:
        for client in record["result"]["evaluation_history"]["clients"].values():
            client["validation"]["loss"] = [1.0]

    artifact = analyze_variance_pilot(records, metric="loss")

    assert artifact["selection"]["direction"] == "minimize"


def test_variance_pilot_rejects_an_incomplete_grid():
    with pytest.raises(VariancePilotError, match="missing 1 condition"):
        analyze_variance_pilot(_crossed_records()[:-1])


def test_variance_command_reads_results_and_writes_an_artifact(
    tmp_path, monkeypatch
):
    results = tmp_path / "results"
    results.mkdir()
    for record in _crossed_records():
        record.pop("_source_file")
        experiment = resolved_experiment(**record["config"]["experiment"])
        fingerprint = record["run_fingerprint"]
        path = results / result_filename(experiment, "local", fingerprint)
        write_run_record(
            path,
            record,
            expected_algorithm="local",
            expected_fingerprint=fingerprint,
        )
    output = tmp_path / "variance.json"
    grid = tmp_path / "grid.jsonl"
    grid.write_text(
        "".join(
            json.dumps(
                {
                    "kind": "rigfl.sweep_task",
                    "schema_version": 1,
                    "algorithm": record["algorithm"],
                    "experiment": record["config"]["experiment"],
                    "algorithm_config": record["config"]["algorithm"],
                }
            )
            + "\n"
            for record in _crossed_records()
        )
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "variance",
            "--results-dir",
            str(results),
            "--grid",
            str(grid),
            "--out-json",
            str(output),
        ],
    )

    main()

    artifact = json.loads(output.read_text())
    assert artifact["groups"][0]["runs"] == 8
