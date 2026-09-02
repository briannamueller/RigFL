"""Choose reporting rounds from a completed evaluation history.

Two views answer different questions:

``global``      one round for the whole federation. Aggregate the validation metric across clients,
                take the best aggregate, read every client's metrics from that
                one round.
``per-client``  each client's own best round, chosen on its own validation
                metric. Useful for "how well could each client have done", but
                the aggregate mixes rounds and is not a system checkpoint. It is
                labelled as such everywhere it appears.

Selection records the metric, split, direction, aggregation, and tie-break used.
"""

from __future__ import annotations

import math
import statistics
from typing import Any, Literal, Optional

from rigfl.eval.metrics import canonical, direction_of, unavailable_reason

Aggregation = Literal["mean", "weighted_mean"]
TieBreak = Literal["earliest", "latest"]


class SelectionError(ValueError):
    """Raised when a requested selection cannot be computed honestly."""


def _better(direction: str):
    return (lambda a, b: a > b) if direction == "maximize" else (lambda a, b: a < b)


def resolve_metric(metric: Optional[str], *, source: str = "argument") -> str:
    """Canonical metric name. Accuracy is the visible, overridable default."""
    if metric is None or str(metric).strip() == "":
        return "accuracy"
    return canonical(metric)


# ── Aggregating across clients ───────────────────────────────────────────────

def aggregate(values: list[Optional[float]], weights: Optional[list[Optional[float]]],
              how: Aggregation) -> Optional[float]:
    """Combine one round's per-client values. Clients with no value are skipped."""
    pairs = [(v, (weights[i] if weights else None))
             for i, v in enumerate(values) if v is not None]
    if not pairs:
        return None
    if how == "mean":
        return sum(v for v, _ in pairs) / len(pairs)
    if how == "weighted_mean":
        usable = [(v, w) for v, w in pairs if w]
        if not usable:
            raise SelectionError(
                "weighted_mean needs per-client sample counts, and none are recorded "
                "for this run. Use aggregation='mean', or re-run with a version that "
                "records counts.")
        total = sum(w for _, w in usable)
        return sum(v * w for v, w in usable) / total
    raise SelectionError(f'Unknown aggregation "{how}"; use "mean" or "weighted_mean".')


# ── The two views ────────────────────────────────────────────────────────────

def select_global(history: dict, metric: str, *, aggregation: Aggregation = "mean",
                  tie_break: TieBreak = "earliest", split: str = "validation") -> dict:
    """One round for every client, chosen on the aggregated validation metric."""
    name = resolve_metric(metric)
    direction = direction_of(name)
    rounds = history["evaluation_rounds"]
    clients = history["clients"]
    weights = history.get("client_sample_counts", {}).get(split)

    better = _better(direction)
    allowed = range(len(rounds))
    best_i: Optional[int] = None
    best_val: Optional[float] = None
    for i in allowed:
        vals = [clients[c][split].get(name, [None] * len(rounds))[i] for c in clients]
        w = [weights[c][i] for c in clients] if weights else None
        agg = aggregate(vals, w, aggregation)
        if agg is None:
            continue
        if best_val is None or better(agg, best_val):
            best_i, best_val = i, agg
        # equal values keep the earlier index under "earliest"; take the later one
        # only when explicitly asked
        elif agg == best_val and tie_break == "latest":
            best_i = i
    if best_i is None:
        raise SelectionError(
            f'No round has a value for "{name}" on the {split} split. '
            + unavailable_reason(name))

    return {
        "selection_view": "global",
        "selection_metric": name,
        "selection_split": split,
        "selection_direction": direction,
        "selection_aggregation": aggregation,
        "tie_break": tie_break,
        "selected_round": rounds[best_i],
        "selected_index": best_i,
        # Positional lists are only interpretable alongside the ids they belong
        # to: dropping a client with no value would otherwise shift every later
        # one, and a comparison across runs would pair the wrong clients.
        "client_ids": list(clients),
        "validation": _slice(history, best_i, "validation"),
        "test": _slice(history, best_i, "test"),
        "sample_counts": _counts_at(history, best_i),
        "mixed_rounds": False,
    }


def select_per_client(history: dict, metric: str, *, tie_break: TieBreak = "earliest",
                      split: str = "validation") -> dict:
    """Each client's own best round. The aggregate mixes rounds -- see ``mixed_rounds``."""
    name = resolve_metric(metric)
    direction = direction_of(name)
    rounds = history["evaluation_rounds"]
    clients = history["clients"]
    better = _better(direction)
    allowed = range(len(rounds))

    chosen: dict[str, int] = {}
    for cid, splits in clients.items():
        series = splits[split].get(name, [])
        best_i = best_v = None
        for i in allowed:
            v = series[i] if i < len(series) else None
            if v is None:
                continue
            if best_v is None or better(v, best_v):
                best_i, best_v = i, v
            elif v == best_v and tie_break == "latest":
                best_i = i
        if best_i is not None:
            chosen[cid] = best_i
    if not chosen:
        raise SelectionError(
            f'No client has a value for "{name}" on the {split} split. '
            + unavailable_reason(name))

    per_split: dict[str, dict[str, list]] = {}
    for s in ("validation", "test"):
        per_split[s] = {}
        for m in _metric_names(history, s):
            per_split[s][m] = [clients[c][s].get(m, [None] * len(rounds))[i]
                               if c in chosen else None
                               for c, i in ((c, chosen.get(c, 0)) for c in clients)]

    picked = sorted(rounds[i] for i in chosen.values())
    return {
        "selection_view": "per-client",
        "selection_metric": name,
        "selection_split": split,
        "selection_direction": direction,
        "tie_break": tie_break,
        "selected_rounds": {c: rounds[i] for c, i in chosen.items()},
        "selected_round_stats": {
            "min": picked[0], "max": picked[-1],
            "mean": sum(picked) / len(picked),
            "median": statistics.median(picked),
        },
        "client_ids": list(clients),
        "validation": per_split["validation"],
        "test": per_split["test"],
        "sample_counts": {s: [
            (history.get("client_sample_counts", {}).get(s, {}).get(c) or [None])[chosen[c]]
            if c in chosen else None for c in clients] for s in ("validation", "test")},
        # Every client is reported from a different round, so this aggregate is
        # not any single system checkpoint. Anything rendering it must say so.
        "mixed_rounds": True,
    }


def select(history: dict, metric: str, *, view: str = "global",
           aggregation: Aggregation = "mean", tie_break: TieBreak = "earliest",
           split: str = "validation") -> dict:
    """Both views are always computable; ``view`` chooses what is returned."""
    both = {
        "global": lambda: select_global(history, metric, aggregation=aggregation,
                                        tie_break=tie_break, split=split),
        "per-client": lambda: select_per_client(history, metric, tie_break=tie_break,
                                                split=split),
    }
    if view == "both":
        return {"global": both["global"](), "per-client": both["per-client"]()}
    if view not in both:
        raise SelectionError(f'Unknown selection view "{view}"; '
                             'use "global", "per-client" or "both".')
    return both[view]()


def _counts_at(history: dict, index: int) -> dict[str, list]:
    """Per-client sample counts at one round, in client order."""
    per = history.get("client_sample_counts", {})
    return {s: [(per.get(s, {}).get(c) or [None] * (index + 1))[index] for c in history["clients"]]
            for s in ("validation", "test")}


def _metric_names(history: dict, split: str) -> list[str]:
    for splits in history["clients"].values():
        return list(splits.get(split, {}))
    return []


def _slice(history: dict, index: int, split: str) -> dict[str, list]:
    """Every client's metrics at one round, as {metric: [per-client values]}."""
    clients = history["clients"]
    return {m: [clients[c][split].get(m, [])[index] if index < len(clients[c][split].get(m, []))
                else None for c in clients]
            for m in _metric_names(history, split)}


# ── Client-distribution statistics ───────────────────────────────────────────

def percentile(values: list[float], q: float) -> float:
    """Linear-interpolation percentile on the sorted values (numpy's default).

    ``q`` in [0, 100]. With n values the rank is ``q/100 * (n-1)``, interpolating
    between neighbours. Documented because "the 10th percentile" is otherwise
    ambiguous across implementations.
    """
    if not values:
        raise ValueError("percentile of an empty list")
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    pos = (q / 100) * (len(xs) - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    return xs[lo] if lo == hi else xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def client_distribution(values: list[Optional[float]], metric: str,
                        weights: Optional[list[Optional[float]]] = None) -> dict:
    """Spread of one metric across clients, for a single reported round.

    Tail statistics are reported only for higher-is-better metrics, where the
    lowest-scoring clients represent the worst-served clients.
    """
    name = canonical(metric)
    xs = [v for v in values if v is not None]
    if not xs:
        return {"client_count": 0}

    out: dict[str, Any] = {
        "client_count": len(xs),
        f"mean_{name}": sum(xs) / len(xs),
        f"std_{name}": statistics.stdev(xs) if len(xs) > 1 else 0.0,
    }
    if weights:
        usable = [(v, w) for v, w in zip(values, weights) if v is not None and w]
        if usable:
            total = sum(w for _, w in usable)
            out[f"weighted_mean_{name}"] = sum(v * w for v, w in usable) / total

    if direction_of(name) != "maximize":
        return out

    out[f"p10_{name}"] = percentile(xs, 10)

    k = max(1, math.ceil(0.10 * len(xs)))
    tail = sorted(xs)[:k]
    out[f"bottom_10pct_mean_{name}"] = sum(tail) / len(tail)
    out["bottom_10pct_client_count"] = k
    if k < 3:
        out["tail_note"] = (f"the bottom tail holds {k} client(s); "
                            "tail statistics are unstable at this size")
    return out
