"""Summarize selected rounds across seeds and format result tables."""

from __future__ import annotations

import json
import math
import statistics

from rigfl.eval.metrics import canonical, direction_of
from rigfl.eval.selection import (
    SelectionError,
    client_distribution,
    resolve_metric,
    select,
)
from rigfl.experiment.config import algorithm_identity

_T95 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
    7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179,
    13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110,
    18: 2.101, 19: 2.093, 20: 2.086, 21: 2.080, 22: 2.074,
    23: 2.069, 24: 2.064, 25: 2.060, 26: 2.056, 27: 2.052,
    28: 2.048, 29: 2.045, 30: 2.042,
}


def _replicate_condition(record: dict) -> tuple:
    experiment = record.get("config", {}).get("experiment", {})
    if experiment.get("seed") is None:
        raise ValueError("result record is missing config.experiment.seed")
    return (
        experiment.get("partition_seed"),
        experiment.get("split_seed"),
        experiment["seed"],
    )


def _records_by_seed(records: list[dict], label: str) -> dict[tuple, dict]:
    indexed = {}
    for record in records:
        condition = _replicate_condition(record)
        if condition in indexed:
            raise ValueError(
                f"{label} contains more than one record for replicate condition "
                f"{condition}"
            )
        indexed[condition] = record
    return indexed


def independent_replicates(records: list[dict]) -> bool:
    """Whether every completed run has a distinct experiment seed."""
    seeds = {
        record.get("config", {}).get("experiment", {}).get("seed")
        for record in records
    }
    return len(records) == len(seeds)


def _algorithm_configuration(record: dict) -> str:
    config = record.get("config", {}).get("algorithm", {})
    return json.dumps(
        algorithm_identity(config, algorithm=record.get("algorithm")),
        sort_keys=True,
    )


def mean_ci(xs: list[float]) -> tuple[float, float | None]:
    """Mean and half-width of a 95% t interval."""
    if not xs:
        return 0.0, None
    mean = sum(xs) / len(xs)
    n = len(xs)
    if n < 2:
        return mean, None
    sd = statistics.stdev(xs)
    return mean, _T95.get(n - 1, 1.96) * sd / math.sqrt(n)


def selection_for(record: dict, metric: str | None, *, view: str = "global",
                  aggregation: str = "mean", tie_break: str = "earliest",
                  include_test: bool = True) -> dict:
    """Select an available reporting view, explicitly recording any fallback."""
    result = record["result"]
    supported = result.get("selection_views_supported",
                           ["global", "per-client"])
    actual = view
    if view not in supported:
        if view == "global" and supported == ["per-client"]:
            actual = "per-client"
        else:
            raise SelectionError(
                f'selection view "{view}" is not supported by this result; '
                f"supported: {', '.join(supported)}")

    provenance = result.get("selection_provenance")
    name = resolve_metric(metric)
    if provenance is not None and canonical(provenance.get("metric")) != name:
        raise SelectionError(
            f'this result retained each client model using validation metric '
            f'"{provenance.get("metric")}", so it cannot honestly be reported as '
            f'if "{name}" selected those models.')

    selected = select(result["evaluation_history"], name, view=actual,
                      aggregation=aggregation, tie_break=tie_break,
                      split="validation", include_test=include_test)
    selected["requested_selection_view"] = view
    selected["selection_view_fallback"] = actual != view
    if provenance is not None:
        selected.pop("selected_rounds", None)
        selected.pop("selected_round_stats", None)
        selected["selected_steps"] = {
            cid: values["selected_step"]
            for cid, values in provenance["clients"].items()
        }
        selected["mixed_rounds"] = False
        selected["mixed_local_selections"] = True
        selected["selection_source"] = provenance["stage"]
    return selected


def _test_values(sel: dict, metric: str) -> list[float]:
    return _values_and_weights(sel, metric, "test")[0]


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


def run_score(record: dict, metric: str, *, view: str = "global",
              aggregation: str = "mean", tie_break: str = "earliest") -> dict:
    """Reduce one run to validation and test scores at its selected round."""
    name = canonical(metric)
    sel = selection_for(record, name, view=view, aggregation=aggregation,
                        tie_break=tie_break)
    out: dict = {"selection_view": sel.get("selection_view", view)}
    for split in ("validation", "test"):
        vals, weights = _values_and_weights(sel, name, split)
        out[split] = _reduce(vals, weights, aggregation) if vals else None
    if sel.get("selected_round") is not None:
        out["selected_round"] = sel["selected_round"]
    elif "selected_rounds" in sel:
        out["selected_rounds"] = sel["selected_rounds"]
    return out


def summarize(records: list[dict], metric: str, *, view: str = "global",
              aggregation: str = "mean", tie_break: str = "earliest",
              include_resources: bool = False) -> dict:
    """Aggregate several seeds of one configuration into a row.

    Each seed is reduced with the *same* aggregation that selected its round: a
    round chosen on a sample-weighted validation mean is reported as a
    sample-weighted mean, so the number that ranks candidates is the number the
    selection optimised.

    Client-distribution statistics are computed per seed and then averaged.
    """
    if len({_algorithm_configuration(record) for record in records}) > 1:
        raise ValueError("summary requires one algorithm configuration")
    replicates = _records_by_seed(records, "summary")
    name = canonical(metric)
    sels = [selection_for(r, name, view=view, aggregation=aggregation,
                          tie_break=tie_break) for r in records]

    test_scores, val_scores, rounds, steps = [], [], [], []
    dists: list[dict] = []
    for sel in sels:
        tv, tw = _values_and_weights(sel, name, "test")
        vv, vw = _values_and_weights(sel, name, "validation")
        if tv:
            test_scores.append(_reduce(tv, tw, aggregation))
            dists.append(client_distribution(tv, name, weights=tw))
        if vv:
            val_scores.append(_reduce(vv, vw, aggregation))
        if sel.get("selected_round") is not None:
            rounds.append(sel["selected_round"])
        elif "selected_rounds" in sel:
            rounds.extend(sel["selected_rounds"].values())
        if "selected_steps" in sel:
            steps.extend(sel["selected_steps"].values())

    replicate_independence = independent_replicates(records)
    intervals_available = replicate_independence and len(records) > 1
    t_m, t_ci = mean_ci(test_scores)
    v_m, v_ci = mean_ci(val_scores)
    if not intervals_available:
        t_ci = None
        v_ci = None
    experiment_seeds = {
        record.get("config", {}).get("experiment", {}).get("seed")
        for record in records
    }
    row = {
        "metric": name,
        "selection_view": sels[0].get("selection_view") if sels else view,
        "requested_selection_view": view,
        "selection_view_fallback": any(
            s.get("selection_view_fallback") for s in sels),
        "selection_direction": direction_of(name),
        "selection_aggregation": aggregation,
        "tie_break": tie_break,
        "mixed_rounds": any(s.get("mixed_rounds") for s in sels),
        "mixed_local_selections": any(
            s.get("mixed_local_selections") for s in sels),
        "test_mean": t_m, "test_ci": t_ci,
        "test_std": statistics.stdev(test_scores) if len(test_scores) > 1 else 0.0,
        "val_mean": v_m, "val_ci": v_ci,
        "selected_rounds": rounds,
        "selected_steps": steps,
        "seeds": len(experiment_seeds),
        "runs": len(records),
        "replicate_conditions": [
            {"partition_seed": partition, "split_seed": split,
             "experiment_seed": seed}
            for partition, split, seed in sorted(replicates, key=str)
        ],
        "independent_replicates": replicate_independence,
        "confidence_intervals_available": intervals_available,
        "confidence_interval_reason": (
            None
            if intervals_available
            else (
                "fewer than two runs"
                if len(records) < 2
                else "experiment seeds are reused across run conditions"
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
    if (not saved or any(not isinstance(item, dict)
                         or item.get("schema_version") != 1
                         for item in saved)):
        return {"available": False}

    try:
        communication = [
            item["observed"]["communication_bytes"]["total"] for item in saved
        ]
        flops = [item["attributed_training"].get("flops") for item in saved]
        flop_signatures = {
            json.dumps(item["measurement"]["flop_estimation"], sort_keys=True)
            for item in saved
        }
        hardware = [
            item["measurement"]["timing"]["hardware"] for item in saved
        ]
        signatures = {json.dumps(item, sort_keys=True) for item in hardware}
        wall = [item["attributed_training"].get("wall_seconds") for item in saved]
        cache_values = [
            value for item in saved
            for value in item.get("cache_reuse", {}).values()
        ]
        if any(not isinstance(item, dict) for item in hardware):
            raise TypeError("invalid hardware signature")
        cache = _combined_cache_status(cache_values)

        communication_mean, communication_ci = mean_ci(communication)
        comparable_flops = len(flop_signatures) == 1 and not any(
            value is None for value in flops)
        flop_mean, flop_ci = ((None, None) if not comparable_flops
                              else mean_ci(flops))
        comparable_wall = (len(signatures) == 1
                           and all(item.get("cpu_identity_source")
                                   != "generic_fallback" for item in hardware)
                           and not any(value is None for value in wall))
        wall_mean, wall_ci = (
            (None, None) if not comparable_wall else mean_ci(wall)
        )
        if not include_intervals:
            communication_ci = None
            flop_ci = None
            wall_ci = None
        return {
            "available": True,
            "communication_bytes_mean": communication_mean,
            "communication_bytes_ci": communication_ci,
            "attributed_training_flops_mean": flop_mean,
            "attributed_training_flops_ci": flop_ci,
            "flops_comparable": comparable_flops,
            "flop_signatures": [json.loads(value)
                                for value in sorted(flop_signatures)],
            "attributed_training_wall_seconds_mean": wall_mean,
            "attributed_training_wall_seconds_ci": wall_ci,
            "wall_time_comparable": comparable_wall,
            "hardware_signatures": [
                json.loads(value) for value in sorted(signatures)
            ],
            "cache_reuse": cache,
        }
    except (AttributeError, KeyError, TypeError, ValueError):
        return {"available": False}


def _combined_cache_status(values: list[str]) -> str:
    if not values or all(value == "none" for value in values):
        return "none"
    if all(value == "disabled" for value in values):
        return "disabled"
    if all(value == "complete" for value in values):
        return "complete"
    return "partial"


def _reduce(values: list[float], weights: list | None, aggregation: str) -> float:
    """One seed's clients -> one number, by the aggregation that selected.

    A missing weight is an error, not a reason to switch to an unweighted mean:
    the row would be labelled weighted while being something else.
    """
    if aggregation == "weighted_mean":
        usable = [(v, w) for v, w in zip(values, weights or []) if w]
        if not usable:
            raise SelectionError(
                "weighted_mean was requested but no per-client sample counts are "
                "available for this result. Use aggregation='mean', or exclude "
                "results that do not record counts.")
        total = sum(w for _, w in usable)
        return sum(v * w for v, w in usable) / total
    return sum(values) / len(values)


def _average_distributions(dists: list[dict]) -> dict:
    """Mean of each statistic across seeds; counts are taken from the first."""
    if not dists:
        return {}
    out: dict = {}
    keys = set().union(*(d.keys() for d in dists))
    for k in keys:
        vals = [d[k] for d in dists if isinstance(d.get(k), (int, float))]
        if not vals:
            notes = [d[k] for d in dists if isinstance(d.get(k), str)]
            if notes:
                out[k] = notes[0]
            continue
        out[k] = vals[0] if k.endswith("_count") else sum(vals) / len(vals)
    return out


def format_table(rows: dict, metric: str) -> str:
    """rows: {label: summary} -> markdown. The header names what selected."""
    name = canonical(metric)
    higher_better = direction_of(name) == "maximize"
    any_mixed = any(s.get("mixed_rounds") for s in rows.values())
    any_fallback = any(s.get("selection_view_fallback") for s in rows.values())
    any_dependent = any(
        not s.get("independent_replicates", True) for s in rows.values()
    )
    has_grid = any(s.get("expected_runs") is not None for s in rows.values())
    runs_heading = "runs (done/expected)" if has_grid else "runs"

    if higher_better:
        out = [
            (
                f"| algorithm | selection | val {name} | test {name} | p10 | "
                f"bottom-10% | seeds | {runs_heading} |"
            ),
            "|---|---|---|---|---|---|---:|---:|",
        ]
    else:
        out = [
            f"| algorithm | selection | val {name} | test {name} | seeds | {runs_heading} |",
            "|---|---|---|---|---:|---:|",
        ]
    for label, s in rows.items():
        mark = " *" if s.get("mixed_rounds") else ""
        fallback = " †" if s.get("selection_view_fallback") else ""
        dependent = " §" if not s.get("independent_replicates", True) else ""
        cells = [f"{label}{mark}{fallback}{dependent}", s["selection_view"],
                 _format_interval(s["val_mean"], s["val_ci"]),
                 _format_interval(s["test_mean"], s["test_ci"])]
        if higher_better:
            tail = s.get(f"p10_{name}")
            bulk = s.get(f"bottom_10pct_mean_{name}")
            cells.extend([f"{tail:.3f}" if tail is not None else "—",
                          f"{bulk:.3f}" if bulk is not None else "—"])
        runs = str(s["runs"])
        if s.get("expected_runs") is not None:
            runs += f"/{s['expected_runs']}"
        cells.extend([str(s["seeds"]), runs])
        out.append("| " + " | ".join(cells) + " |")
    if any_mixed:
        out += [
            "",
            (
                "\\* per-client view: each client is reported from its own "
                "validation-selected round, so the aggregate mixes rounds and is "
                "not a single system checkpoint."
            ),
        ]
    if any_fallback:
        out += [
            "",
            (
                "† global selection was requested, but this algorithm only "
                "supports per-client selection; the row is explicitly reported "
                "using its per-client-selected models."
            ),
        ]
    if any_dependent:
        out += [
            "",
            (
                "§ experiment seeds are reused across run conditions, so ordinary "
                "replicate confidence intervals are omitted. Use "
                "`python -m rigfl.experiment.variance` for a crossed seed sweep."
            ),
        ]
    return "\n".join(out)


def format_replicate_details(rows: dict) -> str:
    """List completed and missing seed combinations for each result row."""
    lines = ["Completed seeds (partition, split, experiment):"]

    def render(conditions: list[dict]) -> str:
        return ", ".join(
            f"({item['partition_seed']}, {item['split_seed']}, {item['experiment_seed']})"
            for item in conditions
        ) or "none"

    for label, summary in rows.items():
        lines.append(f"- {label}: {render(summary['replicate_conditions'])}")
        if summary.get("missing_replicates"):
            lines.append(f"  - missing: {render(summary['missing_replicates'])}")
    return "\n".join(lines)


def _format_interval(mean: float, interval: float | None) -> str:
    if interval is None:
        return f"{mean:.3f}"
    return f"{mean:.3f} ± {interval:.3f}"


def format_resource_table(rows: dict) -> str:
    out = [
        ("| algorithm | communication (GiB) | attributed training FLOPs "
         "(TFLOPs) | attributed training time (min) | cache reuse |"),
        "|---|---:|---:|---:|---|",
    ]
    for label, summary in rows.items():
        marker = " §" if not summary.get("independent_replicates", True) else ""
        if summary.get("confidence_interval_reason") == "fewer than two runs":
            marker += " ¶"
        resource = summary.get("resources", {})
        if not resource.get("available"):
            out.append(f"| {label}{marker} | — | — | — | — |")
            continue
        communication = _scaled_interval(
            resource["communication_bytes_mean"],
            resource["communication_bytes_ci"], 1024 ** 3)
        flops = _scaled_interval(
            resource["attributed_training_flops_mean"],
            resource["attributed_training_flops_ci"], 10 ** 12)
        wall = _scaled_interval(
            resource["attributed_training_wall_seconds_mean"],
            resource["attributed_training_wall_seconds_ci"], 60)
        out.append("| " + " | ".join([
            label + marker, communication, flops, wall, resource["cache_reuse"]
        ]) + " |")
    out.extend([
        "",
        ("Communication is logical training traffic. Attributed totals replace "
         "cache-read work with the compatible measurements saved when each reused "
         "artifact was created. Wall time is omitted when hardware signatures "
         "differ."),
    ])
    if any(not row.get("independent_replicates", True) for row in rows.values()):
        out.extend([
            "",
            (
                "§ experiment seeds are reused across run conditions; resource "
                "means are shown without ordinary replicate confidence intervals."
            ),
        ])
    if any(row.get("confidence_interval_reason") == "fewer than two runs"
           for row in rows.values()):
        out.extend([
            "",
            (
                "¶ fewer than two runs are available; resource confidence "
                "intervals are omitted."
            ),
        ])
    return "\n".join(out)


def _scaled_interval(mean, interval, scale: float) -> str:
    if mean is None:
        return "—"
    if interval is None:
        return f"{mean / scale:.3f}"
    return f"{mean / scale:.3f} ± {interval / scale:.3f}"
