"""Round selection by validation metric, view, and client weighting."""

from __future__ import annotations

import pytest

from rigfl.eval.selection import (
    SelectionError,
    client_distribution,
    select_client_specific,
    select_shared,
)

ROUNDS = [0, 5, 10, 15]


def history():
    """Built so the three answers differ:

    * unweighted mean accuracy peaks at round 5
    * sample-weighted mean accuracy peaks at round 10 (client 1 holds 90%)
    * balanced accuracy peaks at round 15
    * per client, accuracy peaks at 5 (client 0) and 10 (client 1)

    Test values increase monotonically, so anything peeking at test would always
    pick the last round.
    """
    return {
        "evaluation_rounds": list(ROUNDS),
        "clients": {
            "0": {"validation": {"accuracy": [.50, .95, .60, .55],
                                 "balanced_accuracy": [.20, .30, .40, .95]},
                  "test": {"accuracy": [.10, .20, .30, .40],
                           "balanced_accuracy": [.11, .21, .31, .41]}},
            "1": {"validation": {"accuracy": [.40, .50, .70, .45],
                                 "balanced_accuracy": [.30, .40, .50, .99]},
                  "test": {"accuracy": [.15, .25, .35, .45],
                           "balanced_accuracy": [.16, .26, .36, .46]}},
        },
        "client_sample_counts": {"validation": {"0": [10] * 4, "1": [90] * 4},
                                 "test": {"0": [10] * 4, "1": [90] * 4}},
    }


# Metric choice
def test_accuracy_and_balanced_accuracy_choose_different_rounds():
    acc = select_shared(history(), "accuracy")["selected_round"]
    bacc = select_shared(history(), "balanced_accuracy")["selected_round"]
    assert acc != bacc, "the two metrics must be able to disagree"
    assert acc == 5 and bacc == 15


# Shared and client-specific rounds
def test_global_uses_one_round_for_every_client():
    sel = select_shared(history(), "accuracy")
    assert sel["selected_round"] == 5 and sel["mixed_rounds"] is False
    assert sel["test"]["accuracy"] == [.20, .25]        # both clients at index 1


def test_per_client_can_choose_different_rounds():
    sel = select_client_specific(history(), "accuracy")
    assert sel["selected_rounds"] == {"0": 5, "1": 10}
    assert sel["mixed_rounds"] is True


# Validation/test separation
def test_test_metrics_come_from_the_validation_selected_round():
    sel = select_client_specific(history(), "accuracy")
    assert sel["test"]["accuracy"] == [.20, .35]        # client 0 @5, client 1 @10


def test_changing_test_values_cannot_change_the_selected_round():
    h = history()
    before = select_shared(h, "accuracy")["selected_round"]
    for c in h["clients"].values():                     # make the last round best on test
        c["test"]["accuracy"] = [.01, .02, .03, .99]
    assert select_shared(h, "accuracy")["selected_round"] == before


# Combined selection output
def test_validation_only_selection_omits_test_values_and_counts():
    for select in (select_shared, select_client_specific):
        selected = select(history(), "accuracy", include_test=False)
        assert "test" not in selected
        assert set(selected["sample_counts"]) == {"validation"}


# Aggregation
def test_sample_count_uses_sample_counts():
    """Client 1 holds 90% of the samples, so its peak should carry the aggregate."""
    assert select_shared(history(), "accuracy", aggregation="sample_count")["selected_round"] == 10
    assert select_shared(history(), "accuracy", aggregation="uniform")["selected_round"] == 5


def test_sample_count_without_counts_is_refused():
    h = history()
    h["client_sample_counts"] = {}
    with pytest.raises(SelectionError, match="sample counts"):
        select_shared(h, "accuracy", aggregation="sample_count")


# Client distributions
def test_client_distribution_statistics():
    vals = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    d = client_distribution(vals, "accuracy")
    assert abs(d["p10_accuracy"] - 0.19) < 1e-9          # linear interpolation
    assert abs(d["bottom_10pct_mean_accuracy"] - 0.1) < 1e-9
