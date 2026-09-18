"""Local-relative client transfer summaries."""

from __future__ import annotations

import pytest

from rigfl.eval.transfer import (
    TransferComparisonError,
    format_negative_transfer_profile,
    format_negative_transfer_table,
    negative_transfer_summary,
)

_SEL = {"view": "global", "aggregation": "mean", "tie_break": "earliest"}


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
                "seed": seed,
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
            "schema_version": 3,
            "selection_views_supported": ["global", "per-client"],
            "evaluation_history": {
                "evaluation_rounds": rounds,
                "clients": clients,
                "client_sample_counts": counts,
            },
        },
    }


def _estimate(summary, name):
    return summary[name]["estimate"]


def test_profile_separates_frequency_magnitude_and_burden():
    algorithm = [
        _record("fedavg", 0, [[0.7], [0.4]]),
        _record("fedavg", 1, [[0.62], [0.6]]),
    ]
    local = [
        _record("local", 0, [[0.6], [0.6]]),
        _record("local", 1, [[0.6], [0.6]]),
    ]
    summary = negative_transfer_summary(
        algorithm, local, "accuracy", threshold=0.05,
        profile_thresholds=(0.0, 0.1), tail_fraction=0.25, **_SEL
    )

    assert _estimate(summary, "benefit_rate") == 0.25
    assert _estimate(summary, "negative_transfer_rate") == 0.25
    assert _estimate(summary, "negative_transfer_magnitude") == pytest.approx(0.2)
    assert _estimate(summary, "negative_transfer_burden") == pytest.approx(0.05)
    assert _estimate(summary, "worst_tail_gain") == pytest.approx(-0.2)
    assert _estimate(summary, "negative_transfer_burden") == pytest.approx(
        _estimate(summary, "negative_transfer_rate")
        * _estimate(summary, "negative_transfer_magnitude")
    )
    rates = [
        item["negative_transfer_rate"]["estimate"]
        for item in summary["threshold_profile"]
    ]
    assert rates == sorted(rates, reverse=True)


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


def test_lower_is_better_metrics_are_oriented_as_gains():
    from rigfl.eval import metrics

    metrics.register("error", "minimize", fn=lambda p, y, c: 0.0)
    try:
        algorithm = _record("fedavg", 0, [[0.2], [0.8]])
        local = _record("local", 0, [[0.5], [0.5]])
        for record in (algorithm, local):
            for client in record["result"]["evaluation_history"]["clients"].values():
                client["validation"]["error"] = client["validation"]["accuracy"]
                client["test"]["error"] = client["test"]["accuracy"]
        summary = negative_transfer_summary([algorithm], [local], "error", **_SEL)
        assert [pair["gain"] for pair in summary["paired_gains"]] == pytest.approx(
            [0.3, -0.3]
        )
    finally:
        metrics.unregister("error")


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


def test_resolved_client_models_must_match_within_each_replicate():
    algorithm = [
        _record("feddes", 0, [[0.6]]),
        _record("feddes", 1, [[0.6]]),
    ]
    local = [
        _record("local", 0, [[0.5]]),
        _record("local", 1, [[0.5]]),
    ]
    algorithm[0]["config"]["experiment"]["resolved_models"] = ["cnn"]
    algorithm[1]["config"]["experiment"]["resolved_models"] = ["resnet18"]
    local[0]["config"]["experiment"]["resolved_models"] = ["resnet18"]
    local[1]["config"]["experiment"]["resolved_models"] = ["cnn"]

    with pytest.raises(TransferComparisonError, match="experiment_seed=0") as error:
        negative_transfer_summary(algorithm, local, "accuracy", **_SEL)
    assert "algorithm=['cnn']" in str(error.value)
    assert "local=['resnet18']" in str(error.value)


def test_one_shot_fallback_uses_per_client_selection_for_local():
    algorithm = _record("feddes", 0, [[0.6], [0.6]])
    algorithm["result"]["selection_views_supported"] = ["per-client"]
    local = _record(
        "local",
        0,
        [[0.5, 0.9], [0.9, 0.5]],
        validation_values=[[0.9, 0.1], [0.1, 0.9]],
    )
    summary = negative_transfer_summary([algorithm], [local], "accuracy", **_SEL)
    assert summary["selection_view"] == "per-client"
    assert _estimate(summary, "benefit_rate") == 1.0
    assert [pair["local_value"] for pair in summary["paired_gains"]] == [0.5, 0.5]


def test_hierarchical_intervals_are_reproducible_and_require_both_dimensions():
    algorithm = [
        _record("fedavg", 0, [[0.7], [0.6]]),
        _record("fedavg", 1, [[0.3], [0.4]]),
    ]
    local = [
        _record("local", 0, [[0.5], [0.5]]),
        _record("local", 1, [[0.5], [0.5]]),
    ]
    first = negative_transfer_summary(algorithm, local, "accuracy", **_SEL)
    second = negative_transfer_summary(algorithm, local, "accuracy", **_SEL)
    assert first["negative_transfer_burden"] == second[
        "negative_transfer_burden"
    ]
    assert first["negative_transfer_burden"]["ci_low"] is not None

    one_seed = negative_transfer_summary(
        algorithm[:1], local[:1], "accuracy", **_SEL
    )
    assert one_seed["uncertainty"]["available"] is False
    assert one_seed["benefit_rate"]["ci_low"] is None

    one_client = negative_transfer_summary(
        [_record("fedavg", 0, [[0.7]]), _record("fedavg", 1, [[0.3]])],
        [_record("local", 0, [[0.5]]), _record("local", 1, [[0.5]])],
        "accuracy",
        **_SEL,
    )
    assert one_client["uncertainty"]["available"] is False




def test_tables_present_the_summary_and_optional_profile():
    summary = negative_transfer_summary(
        [_record("fedavg", 0, [[0.4]])],
        [_record("local", 0, [[0.5]])],
        "accuracy",
        profile_thresholds=(0.05,),
        **_SEL,
    )
    rows = {"fedavg": {"negative_transfer": summary}}
    table = format_negative_transfer_table(rows)
    assert "| benefit rate | NTR | NTM | NTB |" in table
    assert "100.0%" in table
    assert "worst-10% gain" in table
    profile = format_negative_transfer_profile(rows)
    assert "NTR (δ=0)" in profile and "NTR (δ=0.05)" in profile


def test_table_explains_an_unavailable_comparison():
    rows = {
        "fedavg": {
            "negative_transfer": {
                "available": False,
                "reason": "client IDs differ",
            }
        }
    }
    table = format_negative_transfer_table(rows)
    assert "comparison unavailable" in table
    assert "client IDs differ" in table
