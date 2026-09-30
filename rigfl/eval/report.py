"""Summarize selected rounds across seeds and format result tables."""

from __future__ import annotations

import json
import math
import statistics

from rigfl.eval.metrics import canonical, require_task_metric, spec
from rigfl.eval.selection import (
    SelectionError,
    aggregate,
    client_distribution,
    resolve_metric,
    select_client_specific,
    select_shared,
)

_T95 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
    7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179,
    13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110,
    18: 2.101, 19: 2.093, 20: 2.086, 21: 2.080, 22: 2.074,
    23: 2.069, 24: 2.064, 25: 2.060, 26: 2.056, 27: 2.052,
    28: 2.048, 29: 2.045, 30: 2.042,
}


REPLICATE_FIELDS = ("partition_seed", "split_seed", "training_seed")


def replicate_key(record: dict) -> tuple:
    """``(partition_seed, split_seed, training_seed)`` of one run; missing fields are None."""
    experiment = record.get("config", {}).get("experiment", {})
    return tuple(experiment.get(field) for field in REPLICATE_FIELDS)


def _records_by_seed(records: list[dict], label: str) -> dict[tuple, dict]:
    indexed = {}
    for record in records:
        condition = replicate_key(record)
        if condition[2] is None:
            raise ValueError("result record is missing config.experiment.training_seed")
        if condition in indexed:
            raise ValueError(
                f"{label} contains more than one record for replicate condition "
                f"{condition}"
            )
        indexed[condition] = record
    return indexed


def independent_replicates(records: list[dict]) -> bool:
    """Whether every completed run has a distinct training seed."""
    seeds = {
        record.get("config", {}).get("experiment", {}).get("training_seed")
        for record in records
    }
    return len(records) == len(seeds)


def _algorithm_configuration(record: dict) -> str:
    return json.dumps(record.get("config", {}).get("algorithm", {}), sort_keys=True)


def replicate_statistics(
    values: list[float], *, confidence_interval: bool = True
) -> dict:
    """Mean and replicate-level uncertainty for independent run estimates."""
    n = len(values)
    if n == 0:
        return {
            "mean": None,
            "sd": None,
            "ci_half_width": None,
            "ci_low": None,
            "ci_high": None,
            "n": 0,
            "df": 0,
        }
    mean = float(statistics.mean(values))  # float even for integer inputs like byte counts
    if n == 1:
        return {
            "mean": mean,
            "sd": None,
            "ci_half_width": None,
            "ci_low": None,
            "ci_high": None,
            "n": 1,
            "df": 0,
        }
    sd = statistics.stdev(values)
    half_width = (
        _T95.get(n - 1, 1.96) * sd / math.sqrt(n)
        if confidence_interval
        else None
    )
    return {
        "mean": mean,
        "sd": sd,
        "ci_half_width": half_width,
        "ci_low": mean - half_width if half_width is not None else None,
        "ci_high": mean + half_width if half_width is not None else None,
        "n": n,
        "df": n - 1,
    }


def pooled_within_replicate_sd(values: list[list[float]]) -> float | None:
    """Pool client deviations around each replicate's own client mean."""
    residual_sum_squares = 0.0
    degrees_of_freedom = 0
    for replicate in values:
        if len(replicate) < 2:
            continue
        mean = statistics.mean(replicate)
        residual_sum_squares += sum((value - mean) ** 2 for value in replicate)
        degrees_of_freedom += len(replicate) - 1
    if degrees_of_freedom == 0:
        return None
    return math.sqrt(residual_sum_squares / degrees_of_freedom)


def selection_for(record: dict, metric: str | None, *, view: str = "shared",
                  aggregation: str = "uniform",
                  include_test: bool = True) -> dict:
    """Select one run's reporting rounds from its validation history."""
    name = resolve_metric(metric)
    if spec(name).binary_only:
        require_task_metric(name, record["config"]["experiment"]["num_classes"])
    history = record["result"]["evaluation_history"]
    if view == "shared":
        return select_shared(history, name, aggregation=aggregation,
                             include_test=include_test)
    if view == "client-specific":
        return select_client_specific(history, name, include_test=include_test)
    raise SelectionError(f'Unknown round selection "{view}"; '
                         'use "shared" or "client-specific".')


def _values_and_weights(sel: dict, metric: str, split: str):
    """Per-client values for one metric, with their sample counts when known."""
    by_id = _by_client(sel, metric, split)
    vals = [v for v, _ in by_id.values()]
    weights = [w for _, w in by_id.values()]
    return vals, weights


def _by_client(sel: dict, metric: str, split: str) -> dict[str, tuple]:
    """``{client_id: (value, weight)}`` for clients that have a value."""
    name = canonical(metric)
    raw = sel.get(split, {}).get(name, [])
    ids = sel.get("client_ids") or [str(i) for i in range(len(raw))]
    counts = (sel.get("sample_counts") or {}).get(split) or [None] * len(raw)
    return {c: (v, w) for c, v, w in zip(ids, raw, counts) if v is not None}


def selected_metric_value(
    selected: dict,
    metric: str,
    split: str,
    aggregation: str,
) -> float | None:
    """One selected run-level value aggregated from client results."""
    name = canonical(metric)
    values, weights = _values_and_weights(selected, name, split)
    return aggregate(values, weights, aggregation)


def summarize(records: list[dict], metric: str, *, view: str = "shared",
              aggregation: str = "uniform",
              include_resources: bool = False) -> dict:
    """Summarize one configuration's replicates in a single row."""
    if len({_algorithm_configuration(record) for record in records}) > 1:
        raise ValueError("summary requires one algorithm configuration")
    replicates = _records_by_seed(records, "summary")
    name = canonical(metric)
    sels = [
        selection_for(r, name, view=view, aggregation=aggregation)
        for r in records
    ]

    test_scores, val_scores, rounds = [], [], []
    test_client_values: list[list[float]] = []
    dists: list[dict] = []
    for sel in sels:
        tv, _ = _values_and_weights(sel, name, "test")
        test_value = selected_metric_value(sel, name, "test", aggregation)
        validation_value = selected_metric_value(
            sel, name, "validation", aggregation
        )
        if test_value is not None:
            test_scores.append(test_value)
        if tv:
            dists.append(client_distribution(tv, name))
            test_client_values.append(tv)
        if validation_value is not None:
            val_scores.append(validation_value)
        if sel.get("selected_round") is not None:
            rounds.append(sel["selected_round"])
        elif "selected_rounds" in sel:
            rounds.extend(sel["selected_rounds"].values())

    replicate_independence = independent_replicates(records)
    intervals_available = replicate_independence and len(test_scores) > 1
    test_stats = replicate_statistics(
        test_scores, confidence_interval=intervals_available
    )
    validation_stats = replicate_statistics(
        val_scores, confidence_interval=intervals_available
    )
    training_seeds = {
        record.get("config", {}).get("experiment", {}).get("training_seed")
        for record in records
    }
    row = {
        "metric": name,
        "selection_view": sels[0].get("selection_view") if sels else view,
        "requested_selection_view": view,
        "selection_direction": "maximize",
        "selection_aggregation": (
            sels[0].get("selection_aggregation", aggregation)
            if sels else aggregation
        ),
        "mixed_rounds": any(s.get("mixed_rounds") for s in sels),
        "test_mean": test_stats["mean"],
        "test_std": test_stats["sd"],
        "test_ci": test_stats["ci_half_width"],
        "test_ci_low": test_stats["ci_low"],
        "test_ci_high": test_stats["ci_high"],
        "test_n": test_stats["n"],
        "test_df": test_stats["df"],
        "val_mean": validation_stats["mean"],
        "val_std": validation_stats["sd"],
        "val_ci": validation_stats["ci_half_width"],
        "val_ci_low": validation_stats["ci_low"],
        "val_ci_high": validation_stats["ci_high"],
        "val_n": validation_stats["n"],
        "val_df": validation_stats["df"],
        "pooled_client_sd": pooled_within_replicate_sd(
            test_client_values
        ),
        "selected_rounds": rounds,
        "seeds": len(training_seeds),
        "runs": len(records),
        "replicate_conditions": [
            {"partition_seed": partition, "split_seed": split,
             "training_seed": seed}
            for partition, split, seed in sorted(
                replicates,
                key=lambda key: tuple((value is not None, value) for value in key),
            )
        ],
        "independent_replicates": replicate_independence,
        "confidence_intervals_available": intervals_available,
        "confidence_interval_reason": (
            None
            if intervals_available
            else (
                "fewer than two replicate estimates"
                if len(test_scores) < 2
                else "training seeds are reused across run conditions"
            )
        ),
    }
    row.update(_average_distributions(dists))
    if include_resources:
        row["resources"] = summarize_resources(
            records, include_intervals=intervals_available
        )
    return row


def summarize_resources(records: list[dict], *, include_intervals: bool = True) -> dict:
    """Resource totals across complete run replicates."""
    saved = [record.get("resources") for record in records]
    if not saved or any(not isinstance(item, dict) for item in saved):
        return {"available": False}

    try:
        communication = [
            item["observed"]["communication_bytes"]["total"] for item in saved
        ]
        flops = [item["observed"]["flops"]["algorithm_operations"]
                 for item in saved]
        flop_signatures = {
            json.dumps(item["measurement"]["flop_estimation"], sort_keys=True)
            for item in saved
        }
        hardware = [
            item["measurement"]["timing"]["hardware"] for item in saved
        ]
        signatures = {json.dumps(item, sort_keys=True) for item in hardware}
        wall = [item["observed"]["wall_seconds"]["algorithm_operations"]
                for item in saved]
        if any(not isinstance(item, dict) for item in hardware):
            raise TypeError("invalid hardware signature")

        communication_mean, communication_ci = _mean_ci(communication)
        comparable_flops = len(flop_signatures) == 1 and not any(
            value is None for value in flops)
        flop_mean, flop_ci = ((None, None) if not comparable_flops
                              else _mean_ci(flops))
        comparable_wall = (len(signatures) == 1
                           and all(item.get("cpu_identity_source")
                                   != "generic_fallback" for item in hardware)
                           and not any(value is None for value in wall))
        wall_mean, wall_ci = (
            (None, None) if not comparable_wall else _mean_ci(wall)
        )
        if not include_intervals:
            communication_ci = None
            flop_ci = None
            wall_ci = None
        return {
            "available": True,
            "communication_bytes_mean": communication_mean,
            "communication_bytes_ci": communication_ci,
            "algorithm_flops_mean": flop_mean,
            "algorithm_flops_ci": flop_ci,
            "flops_comparable": comparable_flops,
            "flop_signatures": [json.loads(value)
                                for value in sorted(flop_signatures)],
            "algorithm_wall_seconds_mean": wall_mean,
            "algorithm_wall_seconds_ci": wall_ci,
            "wall_time_comparable": comparable_wall,
            "hardware_signatures": [
                json.loads(value) for value in sorted(signatures)
            ],
        }
    except (AttributeError, KeyError, TypeError, ValueError):
        return {"available": False}


def _mean_ci(values: list[float]) -> tuple[float, float | None]:
    stats = replicate_statistics(values)
    return stats["mean"], stats["ci_half_width"]


def _average_distributions(dists: list[dict]) -> dict:
    """Mean of each client-distribution statistic across seeds."""
    out = {}
    for key in set().union(*dists) if dists else ():
        values = [d[key] for d in dists if key in d]
        out[key] = sum(values) / len(values)
    return out


def by_condition(rows: dict) -> dict[str, dict]:
    """Group rows under their experiment-condition heading, keeping row order."""
    groups: dict[str, dict] = {}
    for label, row in rows.items():
        groups.setdefault(row.get("condition", ""), {})[label] = row
    return groups


def condition_heading(condition: str) -> list[str]:
    return [f"### {condition}", ""] if condition else []


def row_cells(label: str, row: dict) -> list[str]:
    """``algorithm`` and ``configuration`` cells of one result row."""
    return [row.get("algorithm", label), row.get("configuration", "—")]


def row_name(label: str, row: dict) -> str:
    """Algorithm and configuration in one phrase, for listings."""
    algorithm, configuration = row_cells(label, row)
    return algorithm if configuration == "—" else f"{algorithm} {configuration}"


def format_table(rows: dict, metric: str) -> str:
    """rows: {label: summary} -> markdown, one table per condition."""
    name = canonical(metric)
    any_mixed = any(s.get("mixed_rounds") for s in rows.values())
    any_dependent = any(
        not s.get("independent_replicates", True) for s in rows.values()
    )

    out = []
    for condition, group in by_condition(rows).items():
        if out:
            out.append("")
        out += condition_heading(condition)
        out += [
            (
                f"| algorithm | configuration | round selection | val {name} | "
                f"test {name} | p10 | bottom-10% | replicates |"
            ),
            "|---|---|---|---|---|---|---|---:|",
        ]
        for label, s in group.items():
            mark = " *" if s.get("mixed_rounds") else ""
            dependent = " §" if not s.get("independent_replicates", True) else ""
            algorithm, configuration = row_cells(label, s)
            cells = [f"{algorithm}{mark}", configuration, s["selection_view"],
                     _format_interval(s["val_mean"], s["val_ci"]),
                     _format_interval(s["test_mean"], s["test_ci"])]
            tail = s.get(f"p10_{name}")
            bulk = s.get(f"bottom_10pct_mean_{name}")
            cells.extend([f"{tail:.3f}" if tail is not None else "—",
                          f"{bulk:.3f}" if bulk is not None else "—"])
            cells.append(f"{s['runs']}{dependent}")
            out.append("| " + " | ".join(cells) + " |")
    if any_mixed:
        out += [
            "",
            (
                "\\* client-specific round selection: each client is reported from its own "
                "validation-selected round, so the aggregate mixes rounds and is "
                "not a single system checkpoint."
            ),
        ]
    if any_dependent:
        out += [
            "",
            (
                "§ training seeds are reused across run conditions, so ordinary "
                "replicate confidence intervals are omitted. Use "
                "`rigfl seed-sensitivity` for a crossed seed sweep."
            ),
        ]
    return "\n".join(out)


def _format_interval(mean: float | None, interval: float | None) -> str:
    if mean is None:
        return "—"
    if interval is None:
        return f"{mean:.3f}"
    return f"{mean:.3f} ± {interval:.3f}"


def format_resource_table(rows: dict) -> str:
    out = []
    for condition, group in by_condition(rows).items():
        if out:
            out.append("")
        out += condition_heading(condition)
        out += [
            ("| algorithm | configuration | communication (GiB) "
             "| algorithm FLOPs (TFLOPs) | algorithm time (min) |"),
            "|---|---|---:|---:|---:|",
        ]
        for label, summary in group.items():
            marker = " §" if not summary.get("independent_replicates", True) else ""
            algorithm, configuration = row_cells(label, summary)
            resource = summary.get("resources", {})
            if not resource.get("available"):
                out.append(f"| {algorithm}{marker} | {configuration} | — | — | — |")
                continue
            communication = _scaled_interval(
                resource["communication_bytes_mean"],
                resource["communication_bytes_ci"], 1024 ** 3)
            flops = _scaled_interval(
                resource["algorithm_flops_mean"],
                resource["algorithm_flops_ci"], 10 ** 12)
            wall = _scaled_interval(
                resource["algorithm_wall_seconds_mean"],
                resource["algorithm_wall_seconds_ci"], 60)
            out.append("| " + " | ".join([
                algorithm + marker, configuration, communication, flops, wall
            ]) + " |")
    out.extend([
        "",
        ("Communication is logical training traffic. Wall time is omitted when "
         "hardware signatures differ."),
    ])
    if any(not row.get("independent_replicates", True) for row in rows.values()):
        out.extend([
            "",
            (
                "§ training seeds are reused across run conditions; resource "
                "means are shown without ordinary replicate confidence intervals."
            ),
        ])
    return "\n".join(out)


def _scaled_interval(mean, interval, scale: float) -> str:
    if mean is None:
        return "—"
    if interval is None:
        return f"{mean / scale:.3f}"
    return f"{mean / scale:.3f} ± {interval / scale:.3f}"
