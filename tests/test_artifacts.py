"""Focused tests for run-result persistence and validation."""

from __future__ import annotations

import json

import pytest

from rigfl.algorithms.local import LocalConfig
from rigfl.experiment.artifacts import (
    ResultValidationError,
    atomic_write_json,
    existing_result_decision,
    make_run_record,
    validate_run_record,
)
from rigfl.experiment.identity import fingerprint
from rigfl.experiment.registry import run_identity
from tests.helpers import resolved_experiment


def _result(rounds: int = 2, clients: int = 2) -> dict:
    values = [0.5] * rounds
    per_client = {
        str(cid): {
            split: {
                "accuracy": list(values),
                "balanced_accuracy": list(values),
                "macro_f1": list(values),
                "loss": [1.0] * rounds,
                "auroc": list(values),
                "auprc": list(values),
            }
            for split in ("validation", "test")
        }
        for cid in range(clients)
    }
    counts = {
        split: {str(cid): [10] * rounds for cid in range(clients)}
        for split in ("validation", "test")
    }
    return {
        "evaluation_history": {
            "evaluation_rounds": list(range(rounds)),
            "clients": per_client,
            "client_sample_counts": counts,
        },
        "early_stopping": {
            "enabled": False,
            "termination_reason": "completed_all_rounds",
            "stopped_at_round": rounds - 1,
            "metric": None,
            "direction": None,
            "aggregation": None,
            "patience": None,
            "min_delta": None,
            "best_round": None,
            "best_value": None,
        },
    }


def _record(*, partition_id: str = "partition-a"):
    exp = resolved_experiment(
        rounds=2, eval_gap=1, num_clients=2, partition_id=partition_id
    )
    cfg = LocalConfig()
    identity = run_identity("local", exp, cfg.model_dump())
    fp = fingerprint(identity)
    return exp, cfg, fp, make_run_record(
        algorithm="local",
        experiment=exp.model_dump(),
        algorithm_config=cfg.model_dump(),
        result=_result(),
        run_fingerprint=fp,
        identity_input=identity,
    )


def test_prototype_run_record_schema_is_rejected():
    _, _, _, record = _record()
    record["schema_version"] = 5

    with pytest.raises(ResultValidationError, match="expected 1"):
        validate_run_record(record)


def test_nonstandard_numbers_are_refused(tmp_path):
    with pytest.raises(ResultValidationError, match="not valid JSON"):
        atomic_write_json(tmp_path / "bad.json", {"value": float("nan")})


def test_a_complete_matching_result_is_skipped(tmp_path):
    exp, cfg, fp, record = _record(partition_id="partition-a")
    path = tmp_path / "run.json"
    atomic_write_json(path, record)
    skip, message = existing_result_decision(
        path, expected_algorithm="local", expected_fingerprint=fp
    )
    assert skip is True
    assert "validated complete" in message

    # a different requested configuration can't reuse it
    other_exp = resolved_experiment(
        rounds=2, eval_gap=1, num_clients=2, partition_id="partition-b"
    )
    other_fp = fingerprint(
        run_identity("local", other_exp, LocalConfig().model_dump())
    )
    with pytest.raises(ResultValidationError, match="requested experiment"):
        existing_result_decision(
            path, expected_algorithm="local", expected_fingerprint=other_fp
        )


def test_malformed_existing_result_is_not_skipped(tmp_path):
    _, _, fp, record = _record()
    path = tmp_path / "run.json"
    path.write_text("{")
    with pytest.raises(ResultValidationError, match="invalid JSON"):
        existing_result_decision(
            path, expected_algorithm="local", expected_fingerprint=fp
        )
    assert path.read_text() == "{"


def test_saved_configuration_must_match_its_fingerprint():
    _, _, _, record = _record()
    record["identity"]["fingerprint_input"]["experiment"][
        "partition_id"
    ] = "partition-other"
    with pytest.raises(ResultValidationError, match="fingerprint"):
        validate_run_record(record)


def test_written_record_round_trips_through_validation(tmp_path):
    _, _, fp, record = _record()
    path = tmp_path / "run.json"
    atomic_write_json(path, record)
    parsed = json.loads(path.read_text())
    assert fingerprint(parsed["identity"]["fingerprint_input"]) == fp
    validate_run_record(parsed, expected_algorithm="local", expected_fingerprint=fp)
