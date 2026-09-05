"""Focused tests for run-result persistence and validation."""

from __future__ import annotations

import json

import pytest

from rigfl.experiment.artifacts import (
    ResultValidationError,
    atomic_write_json,
    existing_result_decision,
    make_run_record,
    read_json,
    validate_run_record,
    write_run_record,
)
from rigfl.experiment.config import run_fingerprint
from rigfl.algorithms.local import LocalConfig
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
        "schema_version": 3,
        "selection_views_supported": ["global", "per-client"],
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
            "split": None,
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
    fp = run_fingerprint(exp, cfg.model_dump())
    return exp, cfg, fp, make_run_record(
        algorithm="local",
        experiment=exp.model_dump(),
        algorithm_config=cfg.model_dump(),
        result=_result(),
        run_fingerprint=fp,
    )


def test_atomic_json_write_leaves_valid_json(tmp_path):
    path = tmp_path / "result.json"
    atomic_write_json(path, {"value": 1})
    assert read_json(path) == {"value": 1}
    assert not list(tmp_path.glob("*.partial"))


def test_nonstandard_numbers_are_refused(tmp_path):
    with pytest.raises(ResultValidationError, match="not valid JSON"):
        atomic_write_json(tmp_path / "bad.json", {"value": float("nan")})


def test_a_complete_matching_result_is_skipped(tmp_path):
    exp, cfg, fp, record = _record()
    path = tmp_path / "run.json"
    write_run_record(path, record, expected_algorithm="local", expected_fingerprint=fp)
    skip, message = existing_result_decision(
        path, expected_algorithm="local", expected_fingerprint=fp
    )
    assert skip is True
    assert "validated complete" in message


def test_malformed_existing_result_is_not_skipped(tmp_path):
    _, _, fp, record = _record()
    path = tmp_path / "run.json"
    path.write_text("{")
    with pytest.raises(ResultValidationError, match="invalid JSON"):
        existing_result_decision(
            path, expected_algorithm="local", expected_fingerprint=fp
        )
    assert path.read_text() == "{"


def test_force_allows_replacing_but_not_silently_accepting_a_bad_result(tmp_path):
    _, _, fp, _ = _record()
    path = tmp_path / "run.json"
    path.write_text("{")
    skip, message = existing_result_decision(
        path, expected_algorithm="local", expected_fingerprint=fp, force=True
    )
    assert skip is False
    assert "unusable" in message


def test_saved_configuration_must_match_its_fingerprint():
    _, _, _, record = _record()
    record["config"]["experiment"]["partition_id"] = "partition-other"
    with pytest.raises(ResultValidationError, match="fingerprint"):
        validate_run_record(record)


def test_requested_configuration_must_match_the_saved_configuration():
    _, _, fp, record = _record(partition_id="partition-a")
    other_exp = resolved_experiment(
        rounds=2, eval_gap=1, num_clients=2, partition_id="partition-b"
    )
    other_fp = run_fingerprint(other_exp, LocalConfig().model_dump())
    assert fp != other_fp
    with pytest.raises(ResultValidationError, match="requested experiment"):
        validate_run_record(record, expected_fingerprint=other_fp)


def test_history_vectors_must_align_with_evaluation_rounds():
    _, _, _, record = _record()
    record["result"]["evaluation_history"]["clients"]["0"]["validation"][
        "accuracy"
    ].pop()
    with pytest.raises(ResultValidationError, match="round-aligned"):
        validate_run_record(record)


def test_one_shot_result_validates_local_selection_provenance():
    exp = resolved_experiment(rounds=2, eval_gap=1, num_clients=2)
    cfg = LocalConfig()
    fp = run_fingerprint(exp, cfg.model_dump())
    result = _result(rounds=1)
    result["selection_views_supported"] = ["per-client"]
    result["selection_provenance"] = {
        "view": "per-client", "stage": "local_computation", "metric": "accuracy",
        "clients": {
            "0": {"selected_step": 3, "validation_value": 0.7},
            "1": {"selected_step": 5, "validation_value": 0.8},
        },
    }
    result["early_stopping"] = {
        "enabled": False, "termination_reason": "not_applicable",
        "stopped_at_round": None, "metric": None, "split": None,
        "direction": None, "aggregation": None, "patience": None,
        "min_delta": None, "best_round": None, "best_value": None,
    }
    record = make_run_record(
        algorithm="local", experiment=exp.model_dump(),
        algorithm_config=cfg.model_dump(), result=result, run_fingerprint=fp)
    assert validate_run_record(record) is record

    del result["selection_provenance"]["clients"]["1"]
    with pytest.raises(ResultValidationError, match="cover every client"):
        validate_run_record(record)


def test_an_internally_aligned_but_truncated_history_is_incomplete():
    _, _, _, record = _record()
    history = record["result"]["evaluation_history"]
    history["evaluation_rounds"].pop()
    for splits in history["clients"].values():
        for metrics in splits.values():
            for values in metrics.values():
                values.pop()
    for per_client in history["client_sample_counts"].values():
        for values in per_client.values():
            values.pop()
    with pytest.raises(ResultValidationError, match="incomplete"):
        validate_run_record(record)


def test_history_must_cover_every_configured_client():
    _, _, _, record = _record()
    del record["result"]["evaluation_history"]["clients"]["1"]
    with pytest.raises(ResultValidationError, match="client ids"):
        validate_run_record(record)


def test_metric_values_must_be_finite_numbers_or_null():
    _, _, _, record = _record()
    record["result"]["evaluation_history"]["clients"]["0"]["test"]["accuracy"][0] = True
    with pytest.raises(ResultValidationError, match="invalid value"):
        validate_run_record(record)


def test_selection_and_tuning_summaries_are_not_part_of_run_validation():
    """Derived analysis neither determines run identity nor completion."""
    _, _, _, record = _record()
    record["result"]["selection"] = {"selected_round": 999}
    record["tuning"] = {"candidate_id": 999}
    validate_run_record(record)


def test_written_record_round_trips_through_strict_validation(tmp_path):
    _, _, fp, record = _record()
    path = tmp_path / "run.json"
    write_run_record(path, record, expected_algorithm="local", expected_fingerprint=fp)
    parsed = json.loads(path.read_text())
    validate_run_record(parsed, expected_algorithm="local", expected_fingerprint=fp)
