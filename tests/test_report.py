"""Multi-seed summaries using explicit round-selection policies."""

from __future__ import annotations

import pytest
import torch

from rigfl.eval.report import (
    format_resource_table,
    format_table,
    selection_for,
    summarize,
)
from rigfl.eval.resources import ResourceMonitor
from rigfl.eval.selection import SelectionError

_SEL = {"view": "global", "aggregation": "mean", "tie_break": "earliest"}


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
    recs = [_record("fedprox", 0, [.1, .9, .2], [.5, .7, .6]),
            _record("fedprox", 1, [.1, .8, .2], [.5, .9, .6])]
    s = summarize(recs, "accuracy", **_SEL)
    assert s["seeds"] == 2
    assert s["runs"] == 2
    assert s["selected_rounds"] == [1, 1]          # validation peaks at index 1
    assert abs(s["test_mean"] - 0.8) < 1e-9        # mean of .7 and .9
    assert abs(s["val_mean"] - 0.85) < 1e-9


def test_summarize_single_seed_omits_a_confidence_interval():
    record = _record("local", 0, [.1, .5], [.2, .4], rounds=(0, 1))
    s = summarize([record], "accuracy", **_SEL)
    assert s["seeds"] == 1 and s["test_std"] == 0.0 and s["test_ci"] is None


def _one_shot_record():
    record = _record("fedprox", 0, [.8], [.7], rounds=(0,))
    record["result"]["selection_views_supported"] = ["per-client"]
    record["result"]["selection_provenance"] = {
        "view": "per-client", "stage": "local_computation", "metric": "accuracy",
        "clients": {
            "0": {"selected_step": 8, "validation_value": .8},
            "1": {"selected_step": 11, "validation_value": .8},
        },
    }
    return record


def test_global_request_falls_back_to_one_shot_per_client_selection():
    record = _one_shot_record()
    selected = selection_for(record, "accuracy", view="global")
    assert selected["selection_view"] == "per-client"
    assert selected["selection_view_fallback"] is True
    assert selected["selected_steps"] == {"0": 8, "1": 11}
    assert "selected_rounds" not in selected

def test_one_shot_result_cannot_claim_a_different_selection_metric():
    with pytest.raises(SelectionError, match="cannot honestly"):
        selection_for(_one_shot_record(), "loss", view="global")


def test_summarize_rejects_duplicate_seeds():
    records = [_record("fedprox", 0, [.9], [.8], rounds=(0,)),
               _record("fedprox", 0, [.9], [.7], rounds=(0,))]

    with pytest.raises(ValueError, match="more than one record for replicate condition"):
        summarize(records, "accuracy", **_SEL)


def test_summarize_keys_runs_by_the_full_seed_condition():
    records = []
    for partition_seed in range(2):
        record = _record(
            "fedprox",
            0,
            [0.5],
            [0.5 + 0.01 * partition_seed],
            rounds=(0,),
        )
        experiment = record["config"]["experiment"]
        experiment["partition_seed"] = partition_seed
        experiment["split_seed"] = partition_seed
        records.append(record)

    summary = summarize(records, "accuracy", **_SEL)

    assert summary["seeds"] == 1
    assert summary["runs"] == 2
    assert summary["test_mean"] == pytest.approx(0.505)
    assert summary["independent_replicates"] is False
    assert summary["test_ci"] is None
    table = format_table({"fedprox": summary}, "accuracy")
    assert "| fedprox § |" in table
    assert "rigfl.experiment.variance" in table


def test_crossed_resource_summary_keeps_means_and_omits_intervals():
    records = []
    for partition_seed in range(2):
        record = _record("local", 0, [0.5], [0.5], rounds=(0,))
        record["config"]["experiment"].update(
            partition_seed=partition_seed,
            split_seed=partition_seed,
        )
        record["resources"] = ResourceMonitor(
            torch.device("cpu"), estimate_flops=True
        ).to_dict()
        records.append(record)

    summary = summarize(records, "accuracy", include_resources=True, **_SEL)
    resources = summary["resources"]

    assert resources["communication_bytes_mean"] == 0
    assert resources["communication_bytes_ci"] is None
    assert resources["attributed_training_flops_mean"] == 0
    assert resources["attributed_training_flops_ci"] is None
    assert resources["attributed_training_wall_seconds_mean"] == 0
    assert resources["attributed_training_wall_seconds_ci"] is None
    table = format_resource_table({"local": summary})
    assert "| local § | 0.000 | 0.000 | 0.000 | none |" in table
    assert "resource means are shown without" in table


def test_zipped_resource_summary_keeps_confidence_intervals():
    records = []
    for seed in range(3):
        record = _record("local", seed, [0.5], [0.5], rounds=(0,))
        record["config"]["experiment"].update(
            partition_seed=seed,
            split_seed=seed,
        )
        record["resources"] = ResourceMonitor(
            torch.device("cpu"), estimate_flops=True
        ).to_dict()
        records.append(record)

    summary = summarize(records, "accuracy", include_resources=True, **_SEL)

    assert summary["resources"]["communication_bytes_ci"] == 0
    assert "| local | 0.000 ± 0.000 |" in format_resource_table({"local": summary})


def test_format_table_names_the_metric_and_flags_mixed_rounds():
    rows = {"fedprox": summarize([_record("fedprox", 0, [.1, .9], [.5, .7], rounds=(0, 1))],
                                "accuracy", **_SEL)}
    table = format_table(rows, "accuracy")
    assert "test accuracy" in table and "fedprox" in table
    assert "*" not in table.split("\n")[2]              # global view: not mixed

    mixed = {"fedprox": summarize([_record("fedprox", 0, [.1, .9], [.5, .7], rounds=(0, 1))],
                                 "accuracy", view="per-client", aggregation="mean",
                                 tie_break="earliest")}
    assert "mixes rounds" in format_table(mixed, "accuracy")


def test_summary_aggregates_the_way_selection_did():
    """A round chosen on a weighted validation mean must be reported as a
    weighted mean -- otherwise row ordering uses a number the selection never
    optimised."""
    rec = _record("fedprox", 0, [.5], [.5], rounds=(0,), n_clients=2)
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
    a = _record("fedprox", 0, [.5], [.5], rounds=(0,), n_clients=2)
    b = _record("fedprox", 1, [.5], [.5], rounds=(0,), n_clients=2)
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
        rec = _record("fedprox", 0, [.5], [.5], rounds=(0,))
        for c in rec["result"]["evaluation_history"]["clients"].values():
            c["validation"]["dev_loss"] = [0.3]
            c["test"]["dev_loss"] = [0.4]
        rows = {"fedprox": summarize([rec], "dev_loss", **_SEL)}
        table = format_table(rows, "dev_loss")
        assert (
            "| algorithm | selection | val dev_loss | test dev_loss | seeds | runs |"
            in table
        )
        assert "p10" not in table and "p90" not in table
        assert "bottom-10%" not in table and "top-10%" not in table
    finally:
        metrics.unregister("dev_loss")

def test_weighted_mean_without_counts_raises_instead_of_silently_unweighting():
    from rigfl.eval.selection import SelectionError
    rec = _record("fedprox", 0, [0.5], [0.5], rounds=(0,))
    rec["result"]["evaluation_history"]["client_sample_counts"] = {"validation": {},
                                                                   "test": {}}
    with pytest.raises(SelectionError, match="sample counts"):
        summarize([rec], "accuracy", view="global", aggregation="weighted_mean",
                  tie_break="earliest")
