"""Crossed seed sensitivity analysis."""

from __future__ import annotations

import json

import pytest

from rigfl.experiment.artifacts import atomic_write_json, make_run_record
from rigfl.experiment.config import result_filename
from rigfl.experiment.identity import fingerprint
from rigfl.experiment.registry import config_class, run_identity
from rigfl.experiment.variance import (
    VariancePilotError,
    analyze_variance_pilot,
    main,
)
from tests.helpers import resolved_experiment


def _record(partition_seed: int, split_seed: int, training_seed: int) -> dict:
    score = 0.5 + 0.1 * partition_seed + 0.02 * split_seed + 0.01 * training_seed
    partition_id = f"partition-{partition_seed}-{split_seed}"
    experiment = resolved_experiment(
        dataset="cifar10",
        partition_id=partition_id,
        partition_seed=partition_seed,
        split_seed=split_seed,
        training_seed=training_seed,
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
    identity = run_identity("local", experiment, algorithm_config)
    record = make_run_record(
        algorithm="local",
        experiment=experiment.model_dump(mode="json"),
        algorithm_config=algorithm_config,
        run_fingerprint=fingerprint(identity),
        identity_input=identity,
        result={
            "evaluation_history": {
                "evaluation_rounds": [0],
                "clients": clients,
                "client_sample_counts": counts,
            },
            "early_stopping": {
                "enabled": False,
                "termination_reason": "completed_all_rounds",
                "stopped_at_round": 0,
                "metric": None,
                "direction": None,
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
        f"result-{partition_seed}-{split_seed}-{training_seed}.json"
    )
    return record


def _crossed_records() -> list[dict]:
    return [
        _record(partition_seed, split_seed, training_seed)
        for partition_seed in range(2)
        for split_seed in range(2)
        for training_seed in range(2)
    ]


def test_variance_pilot_reports_marginal_seed_spreads():
    artifact = analyze_variance_pilot(_crossed_records())

    assert artifact["kind"] == "rigfl.seed_sensitivity"
    assert artifact["selection"]["split"] == "validation"
    assert artifact["selection"]["direction"] == "maximize"
    group = artifact["groups"][0]
    assert group["runs"] == 8
    assert group["ranked_seed_sources"] == [
        "partition_seed",
        "split_seed",
        "training_seed",
    ]
    assert group["seed_sources"]["partition_seed"][
        "marginal_spread"
    ] == pytest.approx(0.1)
    assert group["seed_sources"]["split_seed"][
        "marginal_spread"
    ] == pytest.approx(0.02)
    assert group["seed_sources"]["training_seed"][
        "marginal_spread"
    ] == pytest.approx(0.01)
    assert group["cells"][0]["selected_round"] == 0


def test_variance_pilot_rejects_an_incomplete_grid():
    with pytest.raises(VariancePilotError, match="missing 1 condition"):
        analyze_variance_pilot(_crossed_records()[:-1])


def test_variance_command_reads_results_and_writes_an_artifact(
    tmp_path, monkeypatch
):
    results_root = tmp_path / "results"
    results = results_root / "runs"
    results.mkdir(parents=True)
    for record in _crossed_records():
        record.pop("_source_file")
        experiment = resolved_experiment(**record["config"]["experiment"])
        fingerprint = record["run_fingerprint"]
        path = results / result_filename(experiment, "local", fingerprint)
        atomic_write_json(path, record)
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
            "--results-root",
            str(results_root),
            "--grid",
            str(grid),
            "--out-json",
            str(output),
        ],
    )

    main()

    artifact = json.loads(output.read_text())
    assert artifact["groups"][0]["runs"] == 8
