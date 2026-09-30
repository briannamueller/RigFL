"""Multi-seed summaries using explicit round-selection policies."""

from __future__ import annotations

import pytest
import torch

from rigfl.eval.report import (
    format_resource_table,
    summarize,
)
from rigfl.eval.resources import ResourceMonitor

_SEL = {"view": "shared", "aggregation": "uniform"}


def _record(algorithm, seed, val_series, test_series, rounds=(0, 1, 2), n_clients=2):
    """A minimal run record whose clients all share one metric trajectory."""
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
            "config": {"experiment": {"training_seed": seed}, "algorithm": {}},
            "result": {"evaluation_history": {"evaluation_rounds": list(rounds),
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
    assert s["seeds"] == 1
    assert s["test_n"] == 1 and s["test_df"] == 0
    assert s["test_std"] is None and s["test_ci"] is None


def test_summary_separates_replicate_uncertainty_from_client_heterogeneity():
    first = _record("local", 0, [0.5], [0.5], rounds=(0,), n_clients=2)
    second = _record("local", 1, [0.5], [0.5], rounds=(0,), n_clients=2)
    first_history = first["result"]["evaluation_history"]
    second_history = second["result"]["evaluation_history"]
    first_history["clients"]["0"]["test"]["accuracy"] = [0.2]
    first_history["clients"]["1"]["test"]["accuracy"] = [0.8]
    second_history["clients"]["0"]["test"]["accuracy"] = [0.4]
    second_history["clients"]["1"]["test"]["accuracy"] = [1.0]

    summary = summarize([first, second], "accuracy", **_SEL)

    assert summary["test_mean"] == pytest.approx(0.6)
    assert summary["test_std"] == pytest.approx(2 ** 0.5 / 10)
    assert summary["test_n"] == 2 and summary["test_df"] == 1
    assert summary["test_ci"] is not None
    assert summary["pooled_client_sd"] == pytest.approx(
        (0.18) ** 0.5
    )


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
    assert "| local | — | 0.000 ± 0.000 |" in format_resource_table({"local": summary})


def test_summary_aggregates_the_way_selection_did():
    """Test values are aggregated with the same weighting used for selection."""
    rec = _record("fedprox", 0, [.5], [.5], rounds=(0,), n_clients=2)
    hist = rec["result"]["evaluation_history"]
    hist["clients"]["0"]["test"]["accuracy"] = [1.0]
    hist["clients"]["1"]["test"]["accuracy"] = [0.0]
    hist["client_sample_counts"]["test"] = {"0": [90], "1": [10]}
    hist["client_sample_counts"]["validation"] = {"0": [90], "1": [10]}

    unweighted = summarize([rec], "accuracy", view="shared", aggregation="uniform")
    weighted = summarize(
        [rec], "accuracy", view="shared", aggregation="sample_count"
    )
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
