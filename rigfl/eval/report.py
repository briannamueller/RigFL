"""Summarize selected rounds across seeds and format result tables."""

from __future__ import annotations

import math
import statistics
from typing import Optional

from rigfl.eval.metrics import canonical, direction_of
from rigfl.eval.selection import (SelectionError, client_distribution,
                                  resolve_metric, select)

_T95 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
    7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179,
    13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110,
    18: 2.101, 19: 2.093, 20: 2.086, 21: 2.080, 22: 2.074,
    23: 2.069, 24: 2.064, 25: 2.060, 26: 2.056, 27: 2.052,
    28: 2.048, 29: 2.045, 30: 2.042,
}


def mean_ci(xs: list[float]) -> tuple[float, float]:
    """Mean and half-width of a 95% t interval."""
    if not xs:
        return 0.0, 0.0
    mean = sum(xs) / len(xs)
    n = len(xs)
    if n < 2:
        return mean, 0.0
    sd = statistics.stdev(xs)
    return mean, _T95.get(n - 1, 1.96) * sd / math.sqrt(n)


def selection_for(record: dict, metric: Optional[str], *, view: str = "global",
                  aggregation: str = "mean", tie_break: str = "earliest") -> dict:
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
                      split="validation")
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
              aggregation: str = "mean", tie_break: str = "earliest") -> dict:
    """Aggregate several seeds of one configuration into a row.

    Each seed is reduced with the *same* aggregation that selected its round: a
    round chosen on a sample-weighted validation mean is reported as a
    sample-weighted mean, so the number that ranks candidates is the number the
    selection optimised.

    Client-distribution statistics are computed per seed and then averaged.
    """
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

    t_m, t_ci = mean_ci(test_scores)
    v_m, v_ci = mean_ci(val_scores)
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
        "seeds": len(records),
    }
    row.update(_average_distributions(dists))
    return row


def _reduce(values: list[float], weights: Optional[list], aggregation: str) -> float:
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


def win_rate(algorithm_records: list[dict], local_records: list[dict], metric: str,
             **kw) -> Optional[float]:
    """Fraction of matched client-and-seed pairs where the algorithm beats Local."""
    name = canonical(metric)
    better = (lambda a, b: a > b) if direction_of(name) == "maximize" else (lambda a, b: a < b)

    def seed(r):
        return r.get("config", {}).get("experiment", {}).get("seed")

    local_by_seed = {seed(r): r for r in local_records}
    wins = total = 0
    for m in algorithm_records:
        l = local_by_seed.get(seed(m))
        if l is None:
            continue
        mv = _by_client(selection_for(m, name, **kw), name, "test")
        lv = _by_client(selection_for(l, name, **kw), name, "test")
        for cid in mv.keys() & lv.keys():          # same client, same seed
            wins += int(better(mv[cid][0], lv[cid][0]))
            total += 1
    return wins / total if total else None


def format_table(rows: dict, metric: str) -> str:
    """rows: {label: summary} -> markdown. The header names what selected."""
    name = canonical(metric)
    higher_better = direction_of(name) == "maximize"
    any_mixed = any(s.get("mixed_rounds") for s in rows.values())
    any_local = any(s.get("mixed_local_selections") for s in rows.values())
    any_fallback = any(s.get("selection_view_fallback") for s in rows.values())

    if higher_better:
        out = [f"| algorithm | selection | val {name} | test {name} | p10 | bottom-10% | win% | seeds |",
               "|---|---|---|---|---|---|---|---|"]
    else:
        out = [f"| algorithm | selection | val {name} | test {name} | win% | seeds |",
               "|---|---|---|---|---|---|"]
    for label, s in rows.items():
        win = f"{s['win'] * 100:.0f}%" if s.get("win") is not None else "—"
        mark = " *" if s.get("mixed_rounds") else ""
        fallback = " †" if s.get("selection_view_fallback") else ""
        local = " ‡" if s.get("mixed_local_selections") else ""
        cells = [f"{label}{mark}{fallback}{local}", s["selection_view"],
                 f"{s['val_mean']:.3f} ± {s['val_ci']:.3f}",
                 f"{s['test_mean']:.3f} ± {s['test_ci']:.3f}"]
        if higher_better:
            tail = s.get(f"p10_{name}")
            bulk = s.get(f"bottom_10pct_mean_{name}")
            cells.extend([f"{tail:.3f}" if tail is not None else "—",
                          f"{bulk:.3f}" if bulk is not None else "—"])
        cells.extend([win, str(s["seeds"])])
        out.append("| " + " | ".join(cells) + " |")
    if any_mixed:
        out += ["", "\\* per-client view: each client is reported from its own "
                    "validation-selected round, so the aggregate mixes rounds and is "
                    "not a single system checkpoint."]
    if any_fallback:
        out += ["", "† global selection was requested, but this algorithm only "
                    "supports per-client selection; the row is explicitly reported "
                    "using its per-client-selected models."]
    if any_local:
        out += ["", "‡ each client retained the model selected during its own local "
                    "computation; these selected steps are not federated rounds."]
    return "\n".join(out)
