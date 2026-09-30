"""Choose reporting rounds from a completed evaluation history.

``shared``: one round for all clients, the round with the best validation
score averaged across clients. Every client's metrics are read from that
round.

``client-specific``: each client's own best round on its validation score.
The average across clients then mixes rounds, so it doesn't correspond to
a single model state.

Selection metrics are all higher-is-better (see ``resolve_metric``).
"""

from __future__ import annotations

import math
from typing import Literal, Optional

from rigfl.eval.metrics import (
    COMPUTED_METRICS,
    canonical,
    direction_of,
    unavailable_reason,
)

Aggregation = Literal["uniform", "sample_count"]


class SelectionError(ValueError):
    """A requested round selection cannot be computed."""


def resolve_metric(metric: Optional[str]) -> str:
    """Canonical selection and reporting metric; defaults to accuracy."""
    if metric is None or str(metric).strip() == "":
        return "accuracy"
    name = canonical(metric)
    if direction_of(name) != "maximize":
        choices = ", ".join(
            m for m in COMPUTED_METRICS if direction_of(m) == "maximize"
        )
        raise SelectionError(
            f'"{name}" is available only for early stopping; choose one of: {choices}.'
        )
    return name


def aggregate(values: list[Optional[float]], weights: Optional[list[Optional[float]]],
              how: Aggregation) -> Optional[float]:
    """Combine one round's per-client values. Clients with no value are skipped."""
    pairs = [(v, (weights[i] if weights else None))
             for i, v in enumerate(values) if v is not None]
    if not pairs:
        return None
    if how == "uniform":
        return sum(v for v, _ in pairs) / len(pairs)
    if how == "sample_count":
        usable = [(v, w) for v, w in pairs if w]
        if not usable:
            raise SelectionError(
                "sample-count client weighting needs per-client sample counts, "
                "and none are recorded for this run. Use uniform client weighting, "
                "or exclude results that do not record counts."
            )
        total = sum(w for _, w in usable)
        return sum(v * w for v, w in usable) / total
    raise SelectionError(f"Unknown client-weighting calculation {how!r}.")


def select_shared(history: dict, metric: str, *, aggregation: Aggregation = "uniform",
                  include_test: bool = True) -> dict:
    """One round for every client, chosen on the aggregated validation metric."""
    name = resolve_metric(metric)
    rounds = history["evaluation_rounds"]
    clients = history["clients"]
    weights = history.get("client_sample_counts", {}).get("validation")
    best_i: Optional[int] = None
    best_val: Optional[float] = None
    for i in range(len(rounds)):
        vals = [
            clients[c]["validation"].get(name, [None] * len(rounds))[i]
            for c in clients
        ]
        w = [weights[c][i] for c in clients] if weights else None
        agg = aggregate(vals, w, aggregation)
        if agg is None:
            continue
        if best_val is None or agg > best_val:
            best_i, best_val = i, agg
    if best_i is None:
        raise SelectionError(
            f'No round has a value for "{name}" on the validation split. '
            + unavailable_reason(name))

    selected = {
        "selection_view": "shared",
        "selection_metric": name,
        "selection_split": "validation",
        "selection_direction": "maximize",
        "selection_aggregation": aggregation,
        "selected_round": rounds[best_i],
        "selected_index": best_i,
        # Positional lists are only interpretable alongside the ids they belong
        # to: dropping a client with no value would otherwise shift every later
        # one, and a comparison across runs would pair the wrong clients.
        "client_ids": list(clients),
        "validation": _slice(history, best_i, "validation"),
        "sample_counts": _counts_at(
            history, best_i, ("validation", "test") if include_test
            else ("validation",)
        ),
        "mixed_rounds": False,
    }
    if include_test:
        selected["test"] = _slice(history, best_i, "test")
    return selected


def select_client_specific(history: dict, metric: str, *,
                           include_test: bool = True) -> dict:
    """Each client's own best round. The aggregate mixes rounds -- see ``mixed_rounds``."""
    name = resolve_metric(metric)
    rounds = history["evaluation_rounds"]
    clients = history["clients"]

    chosen: dict[str, int] = {}
    for cid, splits in clients.items():
        series = splits["validation"].get(name, [])
        best_i = best_v = None
        for i in range(len(rounds)):
            v = series[i] if i < len(series) else None
            if v is None:
                continue
            if best_v is None or v > best_v:
                best_i, best_v = i, v
        if best_i is not None:
            chosen[cid] = best_i
    if not chosen:
        raise SelectionError(
            f'No client has a value for "{name}" on the validation split. '
            + unavailable_reason(name))

    split_names = ("validation", "test") if include_test else ("validation",)
    per_split: dict[str, dict[str, list]] = {}
    for s in split_names:
        per_split[s] = {}
        for m in _metric_names(history, s):
            per_split[s][m] = [clients[c][s].get(m, [None] * len(rounds))[i]
                               if c in chosen else None
                               for c, i in ((c, chosen.get(c, 0)) for c in clients)]

    selected = {
        "selection_view": "client-specific",
        "selection_metric": name,
        "selection_split": "validation",
        "selection_direction": "maximize",
        "selected_rounds": {c: rounds[i] for c, i in chosen.items()},
        "client_ids": list(clients),
        "validation": per_split["validation"],
        "sample_counts": {s: [
            (history.get("client_sample_counts", {}).get(s, {}).get(c) or [None])[chosen[c]]
            if c in chosen else None for c in clients] for s in split_names},
        # Every client is reported from a different round, so this aggregate is
        # not any single system checkpoint. Anything rendering it must say so.
        "mixed_rounds": True,
    }
    if include_test:
        selected["test"] = per_split["test"]
    return selected


def _counts_at(history: dict, index: int,
               splits: tuple[str, ...] = ("validation", "test")) -> dict[str, list]:
    """Per-client sample counts at one round, in client order."""
    per = history.get("client_sample_counts", {})
    return {s: [(per.get(s, {}).get(c) or [None] * (index + 1))[index] for c in history["clients"]]
            for s in splits}


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


def percentile(values: list[float], q: float) -> float:
    """Linear-interpolation percentile on the sorted values (numpy's default).

    ``q`` in [0, 100]. With n values the rank is ``q/100 * (n-1)``, interpolating
    between neighbours.
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


def client_distribution(values: list[Optional[float]], metric: str) -> dict:
    """Lower tail of one metric across clients, for a single reported round.

    The lowest-scoring clients are the worst-served ones.
    """
    name = canonical(metric)
    xs = sorted(v for v in values if v is not None)
    if not xs:
        return {}
    k = max(1, math.ceil(0.10 * len(xs)))
    return {
        f"p10_{name}": percentile(xs, 10),
        f"bottom_10pct_mean_{name}": sum(xs[:k]) / k,
    }
