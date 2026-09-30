"""Local-relative client transfer summaries."""

from __future__ import annotations

import pytest

from rigfl.eval.transfer import (
    TransferComparisonError,
    negative_transfer_summary,
)

_SEL = {"view": "shared", "aggregation": "uniform"}


def _record(algorithm, seed, test_values, *, validation_values=None):
    validation_values = validation_values or [[0.5] for _ in test_values]
    partition_id = f"partition-{seed}"
    clients = {
        str(i): {
            "validation": {"accuracy": list(validation_values[i])},
            "test": {"accuracy": list(values)},
        }
        for i, values in enumerate(test_values)
    }
    rounds = list(range(len(test_values[0])))
    counts = {
        split: {str(i): [10] * len(rounds) for i in range(len(test_values))}
        for split in ("validation", "test")
    }
    return {
        "algorithm": algorithm,
        "config": {
            "experiment": {
                "dataset": "test",
                "partition_id": partition_id,
                "partition_seed": seed,
                "split_seed": seed,
                "training_seed": seed,
            },
            "algorithm": {},
        },
        "partition": {
            "generated": {
                "partition_id": partition_id,
                "settings": {
                    "source_dataset": "test",
                    "partition": {
                        "scheme": "dirichlet",
                        "num_clients": len(test_values),
                        "alpha": 0.5,
                        "partition_seed": seed,
                        "split_seed": seed,
                    },
                },
            }
        },
        "result": {
            "evaluation_history": {
                "evaluation_rounds": rounds,
                "clients": clients,
                "client_sample_counts": counts,
            },
        },
    }


def _estimate(summary, name):
    return summary[name]["estimate"]


def test_gain_summary_separates_frequency_magnitude_and_burden():
    algorithm = [
        _record("fedavg", 0, [[0.7], [0.4]]),
        _record("fedavg", 1, [[0.62], [0.6]]),
    ]
    local = [
        _record("local", 0, [[0.6], [0.6]]),
        _record("local", 1, [[0.6], [0.6]]),
    ]
    summary = negative_transfer_summary(algorithm, local, "accuracy", **_SEL)

    assert _estimate(summary, "benefit_rate") == 0.5
    assert _estimate(summary, "negative_transfer_rate") == 0.25
    assert _estimate(summary, "negative_transfer_magnitude") == pytest.approx(0.2)
    assert _estimate(summary, "negative_transfer_burden") == pytest.approx(0.05)
    assert _estimate(summary, "worst_tail_gain") == pytest.approx(-0.1)
    assert _estimate(summary, "negative_transfer_burden") == pytest.approx(
        _estimate(summary, "negative_transfer_rate")
        * _estimate(summary, "negative_transfer_magnitude")
    )
    assert summary["negative_transfer_magnitude"]["harmed_client_count"] == 1
    assert summary["negative_transfer_magnitude"]["affected_replicate_count"] == 1
    assert summary["negative_transfer_magnitude"]["ci_low"] is None


def test_no_harm_has_zero_rate_and_burden_but_undefined_magnitude():
    summary = negative_transfer_summary(
        [_record("fedavg", 0, [[0.8], [0.7]])],
        [_record("local", 0, [[0.6], [0.7]])],
        "accuracy",
        **_SEL,
    )
    assert _estimate(summary, "negative_transfer_rate") == 0.0
    assert _estimate(summary, "negative_transfer_burden") == 0.0
    assert _estimate(summary, "negative_transfer_magnitude") is None
    assert summary["negative_transfer_magnitude"]["ci_low"] is None


def test_replicate_and_client_sets_must_match_exactly():
    algorithm = [_record("fedavg", 0, [[0.7], [0.7]])]
    with pytest.raises(TransferComparisonError, match="identical data"):
        negative_transfer_summary(
            algorithm, [_record("local", 1, [[0.5], [0.5]])], "accuracy", **_SEL
        )

    local = _record("local", 0, [[0.5]])
    local["partition"]["generated"]["settings"]["partition"]["num_clients"] = 2
    with pytest.raises(TransferComparisonError, match="client IDs differ"):
        negative_transfer_summary(algorithm, [local], "accuracy", **_SEL)


def test_resolved_client_models_must_match():
    algorithm = _record("fedavg", 0, [[0.7]])
    local = _record("local", 0, [[0.5]])
    algorithm["config"]["experiment"]["resolved_models"] = ["cnn"]
    local["config"]["experiment"]["resolved_models"] = ["resnet18"]
    with pytest.raises(TransferComparisonError, match="resolved client-model"):
        negative_transfer_summary([algorithm], [local], "accuracy", **_SEL)


def test_replicate_t_intervals_require_two_independent_replicates():
    algorithm = [
        _record("fedavg", 0, [[0.7], [0.6]]),
        _record("fedavg", 1, [[0.3], [0.4]]),
    ]
    local = [
        _record("local", 0, [[0.5], [0.5]]),
        _record("local", 1, [[0.5], [0.5]]),
    ]
    summary = negative_transfer_summary(algorithm, local, "accuracy", **_SEL)
    assert summary["negative_transfer_burden"]["ci_low"] is not None

    one_seed = negative_transfer_summary(
        algorithm[:1], local[:1], "accuracy", **_SEL
    )
    assert one_seed["uncertainty"]["reason"] == "fewer than two replicate estimates"
    assert one_seed["benefit_rate"]["ci_low"] is None
