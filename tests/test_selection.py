"""Round selection: explicit metric, explicit direction, explicit view.

The fixture is built so validation **accuracy** and validation **balanced
accuracy** peak at different rounds. Any regression to a hardcoded metric makes
these fail rather than quietly reporting a different round.
"""

from __future__ import annotations

import math

import pytest

from rigfl.eval import metrics
from rigfl.eval.selection import (SelectionError, client_distribution, percentile,
                                  resolve_metric, select, select_global,
                                  select_per_client)

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


# 1 + 2 -- the configured metric decides, and different metrics decide differently
def test_accuracy_and_balanced_accuracy_choose_different_rounds():
    acc = select_global(history(), "accuracy")["selected_round"]
    bacc = select_global(history(), "balanced_accuracy")["selected_round"]
    assert acc != bacc, "the two metrics must be able to disagree"
    assert acc == 5 and bacc == 15


def test_selection_defaults_to_accuracy():
    assert resolve_metric(None) == "accuracy"
    assert select_global(history(), None)["selected_round"] == 5


# 3 -- direction comes from the registry
def test_a_minimize_metric_takes_the_smallest_value():
    metrics.register("dev_loss", "minimize")
    try:
        h = history()
        for cid, series in (("0", [3.0, 1.0, 2.0, 4.0]), ("1", [3.0, 1.5, 2.0, 4.0])):
            h["clients"][cid]["validation"]["dev_loss"] = series
            h["clients"][cid]["test"]["dev_loss"] = series
        assert select_global(h, "dev_loss")["selected_round"] == 5      # the minimum
        assert select_global(h, "dev_loss")["selection_direction"] == "minimize"
    finally:
        metrics.METRICS.pop("dev_loss", None)
        if "dev_loss" in metrics.COMPUTED_METRICS:
            metrics.COMPUTED_METRICS.remove("dev_loss")


# 4 + 5 -- one common round, versus each client's own
def test_global_uses_one_round_for_every_client():
    sel = select_global(history(), "accuracy")
    assert sel["selected_round"] == 5 and sel["mixed_rounds"] is False
    assert sel["test"]["accuracy"] == [.20, .25]        # both clients at index 1


def test_per_client_can_choose_different_rounds():
    sel = select_per_client(history(), "accuracy")
    assert sel["selected_rounds"] == {"0": 5, "1": 10}
    assert sel["mixed_rounds"] is True
    assert sel["selected_round_stats"] == {"min": 5, "max": 10, "mean": 7.5, "median": 7.5}


# 6 + 7 -- test is read from, never used to choose
def test_test_metrics_come_from_the_validation_selected_round():
    sel = select_per_client(history(), "accuracy")
    assert sel["test"]["accuracy"] == [.20, .35]        # client 0 @5, client 1 @10


def test_changing_test_values_cannot_change_the_selected_round():
    h = history()
    before = select_global(h, "accuracy")["selected_round"]
    for c in h["clients"].values():                     # make the last round best on test
        c["test"]["accuracy"] = [.01, .02, .03, .99]
    assert select_global(h, "accuracy")["selected_round"] == before


# 8 -- both views
def test_both_returns_global_and_per_client():
    out = select(history(), "accuracy", view="both")
    assert set(out) == {"global", "per-client"}
    assert out["global"]["selected_round"] == 5
    assert out["per-client"]["selected_rounds"] == {"0": 5, "1": 10}


# 9 -- tie-breaking
def test_ties_take_the_earliest_round_by_default():
    h = history()
    for c in h["clients"].values():
        c["validation"]["accuracy"] = [.9, .9, .5, .9]
    assert select_global(h, "accuracy", tie_break="earliest")["selected_round"] == 0
    assert select_global(h, "accuracy", tie_break="latest")["selected_round"] == 15


# aggregation
def test_weighted_mean_uses_sample_counts():
    """Client 1 holds 90% of the samples, so its peak should carry the aggregate."""
    assert select_global(history(), "accuracy", aggregation="weighted_mean")["selected_round"] == 10
    assert select_global(history(), "accuracy", aggregation="mean")["selected_round"] == 5


def test_weighted_mean_without_counts_is_refused():
    h = history()
    h["client_sample_counts"] = {}
    with pytest.raises(SelectionError, match="sample counts"):
        select_global(h, "accuracy", aggregation="weighted_mean")


# 11 -- distribution statistics on a deterministic fixture
def test_client_distribution_statistics():
    vals = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    d = client_distribution(vals, "accuracy", weights=[1] * 9 + [91])
    assert d["client_count"] == 10
    assert abs(d["mean_accuracy"] - 0.55) < 1e-9
    assert abs(d["std_accuracy"] - 0.30276503540974913) < 1e-9
    # weighted: nine clients at weight 1, one (the worst) at 91
    assert abs(d["weighted_mean_accuracy"] - ((5.4 * 1 + 0.1 * 91) / 100)) < 1e-9
    assert abs(d["p10_accuracy"] - percentile(vals, 10)) < 1e-9
    assert abs(d["p10_accuracy"] - 0.19) < 1e-9          # linear interpolation
    assert d["bottom_10pct_client_count"] == max(1, math.ceil(0.10 * 10))
    assert abs(d["bottom_10pct_mean_accuracy"] - 0.1) < 1e-9


def test_lower_is_better_metrics_do_not_report_tail_statistics():
    metrics.register("dev_loss", "minimize")
    try:
        d = client_distribution([0.1, 0.2, 0.3, 0.9], "dev_loss")
        assert "mean_dev_loss" in d and "std_dev_loss" in d
        assert not any(
            key.startswith(("p10_", "p90_", "bottom_10pct", "top_10pct"))
            for key in d
        )
    finally:
        metrics.METRICS.pop("dev_loss", None)
        if "dev_loss" in metrics.COMPUTED_METRICS:
            metrics.COMPUTED_METRICS.remove("dev_loss")


def test_small_client_counts_are_flagged_as_unstable():
    d = client_distribution([0.5, 0.6], "accuracy")
    assert d["bottom_10pct_client_count"] == 1
    assert "unstable" in d["tail_note"]


# the selection carries the counts it used, so downstream can match its aggregation
def test_selection_reports_the_sample_counts_it_used():
    sel = select_global(history(), "accuracy", aggregation="weighted_mean")
    assert sel["sample_counts"]["validation"] == [10, 90]
    assert sel["sample_counts"]["test"] == [10, 90]


# custom metrics are real, not just named
def test_a_registered_metric_with_a_function_is_computed():
    import torch

    from rigfl.eval.metrics import COMPUTED_METRICS, compute_all, register, unregister
    register("always_half", "maximize", fn=lambda p, y, c: 0.5)
    try:
        assert "always_half" in COMPUTED_METRICS
        assert compute_all(torch.tensor([0]), torch.tensor([0]), 2)["always_half"] == 0.5
    finally:
        unregister("always_half")
    assert "always_half" not in COMPUTED_METRICS


def test_a_registered_metric_without_a_function_is_selectable_but_not_computed():
    import torch

    from rigfl.eval.metrics import compute_all, register, unregister
    register("external_score", "maximize")
    try:
        assert "external_score" not in compute_all(torch.tensor([0]), torch.tensor([0]), 2)
        h = history()
        for c in h["clients"].values():             # supplied from outside
            c["validation"]["external_score"] = [.1, .2, .9, .3]
            c["test"]["external_score"] = [.1, .2, .3, .4]
        assert select_global(h, "external_score")["selected_round"] == 10
    finally:
        unregister("external_score")
