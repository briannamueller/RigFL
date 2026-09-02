"""Summaries and win-rate over selected rounds.

Every function here takes the selection metric explicitly. That is the point of
the redesign: there is no metric these can fall back on, so a test cannot pass
by accident under a hidden default.
"""

from __future__ import annotations

import pytest

from rigfl.eval.report import format_table, selection_for, summarize, win_rate
from rigfl.eval.selection import SelectionError

_SEL = dict(view="global", aggregation="mean", tie_break="earliest")


def _record(algorithm, seed, val_series, test_series, rounds=(0, 1, 2), n_clients=2):
    """A schema-2 record whose clients all share one metric trajectory."""
    clients = {
        str(c): {
            "validation": {"accuracy": list(val_series),
                           "balanced_accuracy": list(val_series)},
            "test": {"accuracy": list(test_series),
                     "balanced_accuracy": list(test_series)},
        } for c in range(n_clients)
    }
    counts = {s: {str(c): [10] * len(rounds) for c in range(n_clients)}
              for s in ("validation", "test")}
    return {"algorithm": algorithm,
            "config": {"experiment": {"seed": seed}, "algorithm": {}},
            "result": {"schema_version": 3,
                       "selection_views_supported": ["global", "per-client"],
                       "evaluation_history": {"evaluation_rounds": list(rounds),
                                              "clients": clients,
                                              "client_sample_counts": counts}}}


def test_summarize_reports_the_selected_rounds_test_value():
    recs = [_record("feddes", 0, [.1, .9, .2], [.5, .7, .6]),
            _record("feddes", 1, [.1, .8, .2], [.5, .9, .6])]
    s = summarize(recs, "accuracy", **_SEL)
    assert s["seeds"] == 2
    assert s["selected_rounds"] == [1, 1]          # validation peaks at index 1
    assert abs(s["test_mean"] - 0.8) < 1e-9        # mean of .7 and .9
    assert abs(s["val_mean"] - 0.85) < 1e-9


def test_summarize_single_seed_has_zero_spread():
    s = summarize([_record("local", 0, [.1, .5], [.2, .4], rounds=(0, 1))],
                  "accuracy", **_SEL)
    assert s["seeds"] == 1 and s["test_std"] == 0.0 and s["test_ci"] == 0.0


def _one_shot_record():
    record = _record("feddes", 0, [.8], [.7], rounds=(0,))
    record["result"]["selection_views_supported"] = ["per-client"]
    record["result"]["selection_provenance"] = {
        "view": "per-client", "stage": "local_computation", "metric": "accuracy",
        "clients": {
            "0": {"selected_step": 8, "validation_value": .8},
            "1": {"selected_step": 11, "validation_value": .8},
        },
    }
    return record


def test_global_request_falls_back_to_and_labels_one_shot_per_client_selection():
    record = _one_shot_record()
    selected = selection_for(record, "accuracy", view="global")
    assert selected["selection_view"] == "per-client"
    assert selected["selection_view_fallback"] is True
    assert selected["selected_steps"] == {"0": 8, "1": 11}
    assert "selected_rounds" not in selected

    table = format_table(
        {"feddes": summarize([record], "accuracy", view="global")}, "accuracy")
    assert "| feddes † ‡ | per-client |" in table
    assert "global selection was requested" in table
    assert "selected steps are not federated rounds" in table


def test_one_shot_result_cannot_claim_a_different_selection_metric():
    with pytest.raises(SelectionError, match="cannot honestly"):
        selection_for(_one_shot_record(), "loss", view="global")


def test_win_rate_counts_client_seed_pairs():
    algorithm = [_record("feddes", 0, [.9], [.8], rounds=(0,)),
              _record("feddes", 1, [.9], [.3], rounds=(0,))]
    local = [_record("local", 0, [.9], [.5], rounds=(0,)),
             _record("local", 1, [.9], [.5], rounds=(0,))]
    # seed 0: both clients win; seed 1: both lose -> 2 of 4
    assert win_rate(algorithm, local, "accuracy", **_SEL) == 0.5


def test_win_rate_ties_lose():
    algorithm = [_record("feddes", 0, [.9], [.5], rounds=(0,))]
    local = [_record("local", 0, [.9], [.5], rounds=(0,))]
    assert win_rate(algorithm, local, "accuracy", **_SEL) == 0.0


def test_win_rate_without_a_local_counterpart_is_none():
    algorithm = [_record("feddes", 0, [.9], [.8], rounds=(0,))]
    assert win_rate(algorithm, [], "accuracy", **_SEL) is None


def test_win_rate_pairs_by_seed_not_position():
    algorithm = [_record("feddes", 1, [.9], [.8], rounds=(0,))]
    local = [_record("local", 0, [.9], [.99], rounds=(0,)),   # different seed
             _record("local", 1, [.9], [.10], rounds=(0,))]   # the real counterpart
    assert win_rate(algorithm, local, "accuracy", **_SEL) == 1.0


def test_win_rate_direction_follows_the_metric():
    """A lower-is-better metric must not be compared as though larger wins."""
    from rigfl.eval import metrics
    metrics.register("val_error", "minimize")
    try:
        m = _record("feddes", 0, [.9], [.1], rounds=(0,))
        l = _record("local", 0, [.9], [.9], rounds=(0,))
        for rec in (m, l):
            for c in rec["result"]["evaluation_history"]["clients"].values():
                c["test"]["val_error"] = c["test"]["accuracy"]
                c["validation"]["val_error"] = c["validation"]["accuracy"]
        assert win_rate([m], [l], "val_error", **_SEL) == 1.0    # smaller wins
    finally:
        metrics.METRICS.pop("val_error", None)
        if "val_error" in metrics.COMPUTED_METRICS:
            metrics.COMPUTED_METRICS.remove("val_error")


def test_format_table_names_the_metric_and_flags_mixed_rounds():
    rows = {"feddes": summarize([_record("feddes", 0, [.1, .9], [.5, .7], rounds=(0, 1))],
                                "accuracy", **_SEL)}
    table = format_table(rows, "accuracy")
    assert "test accuracy" in table and "feddes" in table
    assert "*" not in table.split("\n")[2]              # global view: not mixed

    mixed = {"feddes": summarize([_record("feddes", 0, [.1, .9], [.5, .7], rounds=(0, 1))],
                                 "accuracy", view="per-client", aggregation="mean",
                                 tie_break="earliest")}
    assert "mixes rounds" in format_table(mixed, "accuracy")


def test_summarize_requires_a_metric():
    with pytest.raises(TypeError):
        summarize([_record("feddes", 0, [.1], [.1], rounds=(0,))])      # no metric


def test_summary_aggregates_the_way_selection_did():
    """A round chosen on a weighted validation mean must be reported as a
    weighted mean -- otherwise --rank orders by a number the selection never
    optimised."""
    rec = _record("feddes", 0, [.5], [.5], rounds=(0,), n_clients=2)
    hist = rec["result"]["evaluation_history"]
    hist["clients"]["0"]["test"]["accuracy"] = [1.0]
    hist["clients"]["1"]["test"]["accuracy"] = [0.0]
    hist["client_sample_counts"]["test"] = {"0": [90], "1": [10]}
    hist["client_sample_counts"]["validation"] = {"0": [90], "1": [10]}

    unweighted = summarize([rec], "accuracy", view="global", aggregation="mean",
                           tie_break="earliest")
    weighted = summarize([rec], "accuracy", view="global", aggregation="weighted_mean",
                         tie_break="earliest")
    assert abs(unweighted["test_mean"] - 0.5) < 1e-9
    assert abs(weighted["test_mean"] - 0.9) < 1e-9       # client 0 holds 90%


def test_distribution_statistics_use_every_seed_not_just_the_last():
    a = _record("feddes", 0, [.5], [.5], rounds=(0,), n_clients=2)
    b = _record("feddes", 1, [.5], [.5], rounds=(0,), n_clients=2)
    a["result"]["evaluation_history"]["clients"]["0"]["test"]["accuracy"] = [1.0]
    a["result"]["evaluation_history"]["clients"]["1"]["test"]["accuracy"] = [1.0]
    b["result"]["evaluation_history"]["clients"]["0"]["test"]["accuracy"] = [0.0]
    b["result"]["evaluation_history"]["clients"]["1"]["test"]["accuracy"] = [0.0]
    s = summarize([a, b], "accuracy", **_SEL)
    # averaged over seeds: 1.0 and 0.0 -> 0.5, not whichever came last
    assert abs(s["p10_accuracy"] - 0.5) < 1e-9
    assert abs(s["bottom_10pct_mean_accuracy"] - 0.5) < 1e-9


def test_lower_is_better_table_omits_tail_columns():
    from rigfl.eval import metrics
    metrics.register("dev_loss", "minimize", fn=lambda p, y, c: 0.0)
    try:
        rec = _record("feddes", 0, [.5], [.5], rounds=(0,))
        for c in rec["result"]["evaluation_history"]["clients"].values():
            c["validation"]["dev_loss"] = [0.3]
            c["test"]["dev_loss"] = [0.4]
        rows = {"feddes": summarize([rec], "dev_loss", **_SEL)}
        table = format_table(rows, "dev_loss")
        assert "| algorithm | selection | val dev_loss | test dev_loss | win% | seeds |" in table
        assert "p10" not in table and "p90" not in table
        assert "bottom-10%" not in table and "top-10%" not in table
    finally:
        metrics.unregister("dev_loss")


def _rec_with_test(algorithm, seed, test_by_client):
    rec = _record(algorithm, seed, [0.5], [0.5], rounds=(0,), n_clients=len(test_by_client))
    hist = rec["result"]["evaluation_history"]
    for i, v in enumerate(test_by_client):
        hist["clients"][str(i)]["test"]["accuracy"] = [v]
    return rec


def test_win_rate_pairs_by_client_id_not_position():
    """Dropping a client with no value used to shift the rest, so algorithm client 1
    was compared against Local client 0."""
    algorithm = [_rec_with_test("feddes", 0, [None, 0.90])]
    local = [_rec_with_test("local", 0, [0.95, 0.10])]
    assert win_rate(algorithm, local, "accuracy", **_SEL) == 1.0     # client 1 only


def test_weighted_mean_without_counts_raises_instead_of_silently_unweighting():
    from rigfl.eval.selection import SelectionError
    rec = _record("feddes", 0, [0.5], [0.5], rounds=(0,))
    rec["result"]["evaluation_history"]["client_sample_counts"] = {"validation": {},
                                                                   "test": {}}
    with pytest.raises(SelectionError, match="sample counts"):
        summarize([rec], "accuracy", view="global", aggregation="weighted_mean",
                  tie_break="earliest")
