"""Algorithm runners and their shared evaluation-history recorder.

Iterative reporting-round selection is performed later from recorded history.
One-shot algorithms instead retain locally validation-selected models and record
that provenance alongside their single final evaluation.
"""

from __future__ import annotations

import os

from dataclasses import dataclass, field

import torch
from torch.utils.data import DataLoader

from rigfl.core.interfaces import (IterativeAlgorithm, LocalSelection,
                                   OneShotContext, P2POneShotAlgorithm)
from rigfl.core.model import ClientModel
from rigfl.eval.metrics import (COMPUTED_METRICS, direction_of, require_computable,
                                unavailable_reason)
from rigfl.eval.protocol import evaluate_split
from rigfl.eval.selection import aggregate


@dataclass
class Client:
    """A client's model, data, identity, and persistent algorithm state.

    Behavior lives entirely in the algorithm. ``state`` is an algorithm-defined
    dictionary that persists across this client's operations during one run.
    """

    model: ClientModel
    train_loader: DataLoader
    val_loader: DataLoader | None = None
    test_loader: DataLoader | None = None
    client_id: int | None = None
    state: dict = field(default_factory=dict)


def iterative(algorithm: IterativeAlgorithm, clients: list[Client], num_rounds: int,
              device: torch.device, num_classes: int, eval_gap: int = 1,
              verbose: bool = True, tracker=None, early_stopping=None) -> dict:
    """Train, recording every metric for every client at every evaluation round.

    Returns the canonical history. No round in it is marked selected; use
    :mod:`rigfl.eval.selection` to choose one (accuracy by default).
    """
    _require_operations(algorithm, "iterative",
                        ("init_globals", "local_train", "aggregate", "predict"))
    _start_run(algorithm, clients, device, num_rounds)
    es = _EarlyStopping(early_stopping)
    shared = algorithm.init_globals()

    rounds_evaluated: list[int] = []
    per_client: dict[str, dict[str, dict[str, list]]] = {}
    counts: dict[str, dict[str, list]] = {"validation": {}, "test": {}}
    stop_reason = "completed_all_rounds"
    last_round = -1

    for rnd in range(num_rounds):
        algorithm.round_idx = rnd
        uploads = [algorithm.local_train(client, shared) for client in clients]
        shared = algorithm.aggregate(uploads, shared)
        last_round = rnd

        if rnd % eval_gap == 0 or rnd == num_rounds - 1:
            evaluated = {
                "validation": evaluate_split(algorithm, clients, shared, device, "val",
                                             num_classes),
                "test": evaluate_split(algorithm, clients, shared, device, "test",
                                       num_classes),
            }
            rounds_evaluated.append(rnd)
            _append(per_client, counts, evaluated, len(rounds_evaluated))

            if tracker is not None:
                tracker.log_round(rnd, evaluated["validation"], evaluated["test"])
            if verbose:
                _print_round(rnd, evaluated)

            if es.update(rnd, evaluated):
                stop_reason = "early_stopping"
                if verbose:
                    print(f"  early stop at round {rnd}: {es.metric} on {es.split} "
                          f"has not improved by {es.min_delta} for {es.patience} evaluations")
                break

    history = {
        "evaluation_rounds": rounds_evaluated,
        "clients": per_client,
        "client_sample_counts": counts,
    }
    _check_alignment(history)

    return {
        "schema_version": 3,
        "selection_views_supported": ["global", "per-client"],
        "evaluation_history": history,
        "early_stopping": es.record(stop_reason, last_round),
    }


def p2p_one_shot(algorithm: P2POneShotAlgorithm, clients: list[Client],
                 num_rounds: int, device: torch.device, num_classes: int,
                 eval_gap: int = 1, verbose: bool = True, tracker=None,
                 early_stopping=None) -> dict:
    """Execute local preparation, one P2P exchange, and local computation once.

    ``num_rounds`` and ``eval_gap`` are accepted so registry runners share one
    invocation shape. They do not control this runner: its algorithm-specific
    local computation owns any internal optimization limit and validation-based
    selection.
    """
    del num_rounds, eval_gap
    _require_operations(
        algorithm, "p2p_one_shot",
        ("prepare", "one_shot_communication", "local_computation", "predict"),
    )
    _start_clients(clients)
    es = _EarlyStopping(early_stopping)
    if es.enabled:
        raise ValueError(
            "experiment early_stopping applies to iterative federated rounds and "
            "cannot be enabled for the p2p_one_shot runner. Configure the "
            "algorithm's local-computation stopping policy instead."
        )

    outgoing = []
    for cid, client in enumerate(clients):
        ctx = OneShotContext(device=device, client_id=cid,
                             client_state=client.state,
                             validation_loader=client.val_loader)
        outgoing.append(algorithm.prepare(client.model, client.train_loader, ctx))

    incoming = algorithm.one_shot_communication(outgoing)
    if not isinstance(incoming, list) or len(incoming) != len(clients):
        raise ValueError(
            "p2p_one_shot one_shot_communication must return one incoming "
            "payload per client."
        )

    local_selections: dict[str, LocalSelection] = {}
    for cid, (client, payload) in enumerate(zip(clients, incoming)):
        ctx = OneShotContext(device=device, client_id=cid,
                             client_state=client.state,
                             validation_loader=client.val_loader)
        selected = algorithm.local_computation(
            client.model, payload, client.train_loader, ctx)
        if not isinstance(selected, LocalSelection):
            raise TypeError(
                "p2p_one_shot local_computation must return LocalSelection."
            )
        local_selections[str(cid)] = selected

    metrics = {require_computable(selection.metric)
               for selection in local_selections.values()}
    if len(metrics) != 1:
        raise ValueError(
            "p2p_one_shot clients must select their retained models with the "
            "same validation metric."
        )
    selection_metric = next(iter(metrics))
    evaluated = {
        "validation": evaluate_split(
            algorithm, clients, None, device, "val", num_classes,
            shared_by_client=incoming),
        "test": evaluate_split(
            algorithm, clients, None, device, "test", num_classes,
            shared_by_client=incoming),
    }
    per_client: dict[str, dict[str, dict[str, list]]] = {}
    counts: dict[str, dict[str, list]] = {"validation": {}, "test": {}}
    _append(per_client, counts, evaluated, 1)
    history = {
        "evaluation_rounds": [0],
        "clients": per_client,
        "client_sample_counts": counts,
    }
    _check_alignment(history)

    if tracker is not None:
        tracker.log_round(0, evaluated["validation"], evaluated["test"])
    if verbose:
        _print_one_shot(evaluated)

    return {
        "schema_version": 3,
        "selection_views_supported": ["per-client"],
        "selection_provenance": {
            "view": "per-client",
            "stage": "local_computation",
            "metric": selection_metric,
            "clients": {
                cid: {
                    "selected_step": selected.selected_step,
                    "validation_value": selected.validation_value,
                }
                for cid, selected in local_selections.items()
            },
        },
        "evaluation_history": history,
        "early_stopping": {
            "enabled": False,
            "termination_reason": "not_applicable",
            "stopped_at_round": None,
            "metric": None, "split": None, "direction": None,
            "aggregation": None, "patience": None, "min_delta": None,
            "best_round": None, "best_value": None,
        },
    }


def _start_run(algorithm, clients: list[Client], device: torch.device,
               total_rounds: int) -> None:
    """Bind experiment-wide values once and reset per-client run state."""
    algorithm.device = device
    algorithm.total_rounds = total_rounds
    algorithm.round_idx = -1
    _start_clients(clients)


def _start_clients(clients: list[Client]) -> None:
    for client_id, client in enumerate(clients):
        client.client_id = client_id
        client.state.clear()


def _require_operations(algorithm, runner: str, operations: tuple[str, ...]) -> None:
    missing = [name for name in operations
               if not callable(getattr(algorithm, name, None))]
    if missing:
        raise TypeError(
            f"{runner} runner requires operations: {', '.join(operations)}; "
            f"{type(algorithm).__name__} is missing: {', '.join(missing)}"
        )


def _append(per_client, counts, evaluated, n_rounds) -> None:
    """Extend each client's metric vectors by one evaluation point.

    Vectors stay aligned with ``evaluation_rounds`` by construction: a client
    with no data this round is padded with None rather than skipped.
    """
    for split in ("validation", "test"):
        block = evaluated[split]
        for cid, metrics in block["clients"].items():
            slot = per_client.setdefault(cid, {"validation": {}, "test": {}})[split]
            for name in COMPUTED_METRICS:
                series = slot.setdefault(name, [])
                while len(series) < n_rounds - 1:      # a client seen late starts padded
                    series.append(None)
                series.append(None if metrics is None else metrics.get(name))
            cseries = counts[split].setdefault(cid, [])
            while len(cseries) < n_rounds - 1:
                cseries.append(None)
            cseries.append(block["sample_counts"].get(cid))


def _check_alignment(history: dict) -> None:
    """Every vector must be as long as evaluation_rounds -- positional alignment
    is the whole contract of this format, so it is checked, not assumed."""
    n = len(history["evaluation_rounds"])
    for cid, splits in history["clients"].items():
        for split, metrics in splits.items():
            for name, series in metrics.items():
                if len(series) != n:
                    raise RuntimeError(
                        f"evaluation history is misaligned: client {cid} "
                        f"{split}.{name} has {len(series)} values for {n} rounds")
    for split, per in history["client_sample_counts"].items():
        for cid, series in per.items():
            if len(series) != n:
                raise RuntimeError(
                    f"evaluation history is misaligned: client {cid} {split} "
                    f"sample counts have {len(series)} values for {n} rounds")


def _print_round(rnd: int, evaluated: dict) -> None:
    """Show every available validation metric for this round."""
    from rigfl.eval.protocol import mean_over_clients
    parts = []
    for m in COMPUTED_METRICS:
        v = mean_over_clients(evaluated["validation"], m)
        if v is not None:
            parts.append(f"{m} {v:.4f}")
    line = f"round {rnd:3d} | val " + " ".join(parts) if parts else f"round {rnd:3d}"
    if os.environ.get("RIGFL_LOG_TEST_ROUNDS") == "1":
        tparts = [f"{m} {v:.4f}" for m in COMPUTED_METRICS
                  if (v := mean_over_clients(evaluated["test"], m)) is not None]
        if tparts:
            line += " | test " + " ".join(tparts)
    print(line)


def _print_one_shot(evaluated: dict) -> None:
    """Show the validation metrics after one-shot local computation."""
    from rigfl.eval.protocol import mean_over_clients
    parts = []
    for metric in COMPUTED_METRICS:
        value = mean_over_clients(evaluated["validation"], metric)
        if value is not None:
            parts.append(f"{metric} {value:.4f}")
    line = "local computation | val " + " ".join(parts)
    if os.environ.get("RIGFL_LOG_TEST_ROUNDS") == "1":
        test_parts = [
            f"{metric} {value:.4f}" for metric in COMPUTED_METRICS
            if (value := mean_over_clients(evaluated["test"], metric)) is not None
        ]
        if test_parts:
            line += " | test " + " ".join(test_parts)
    print(line)


class _EarlyStopping:
    """Explicit, configurable, and independent of how results are later selected."""

    def __init__(self, cfg):
        cfg = cfg or {}
        get = cfg.get if isinstance(cfg, dict) else lambda k, d=None: getattr(cfg, k, d)
        self.enabled = bool(get("enabled", False))
        self.split = get("split", "validation") or "validation"
        if self.split != "validation":
            raise ValueError(
                "early stopping must use the validation split: stopping on test "
                "would end the run at a point chosen by the data it is then "
                "evaluated on.")

        raw_metric = get("metric", "loss") or "loss"
        if not self.enabled:
            # Disabled policies record no control settings.
            self.metric = self.direction = None
            self.aggregation = None
            self.patience = self.min_delta = None
            self.best_value = self.best_round = None
            self.stale = 0
            return

        self.metric = require_computable(raw_metric)
        self.direction = get("direction", None) or direction_of(self.metric)
        self.aggregation = get("aggregation", "mean") or "mean"
        self.patience = int(get("patience", 10) or 10)
        self.min_delta = float(get("min_delta", 0.0) or 0.0)
        self.best_value = None
        self.best_round = None
        self.stale = 0

    def update(self, rnd: int, evaluated: dict) -> bool:
        """Record this round's control metric; return whether training should stop."""
        if not self.enabled:
            return False
        block = evaluated[self.split]
        vals = [m[self.metric] if m and self.metric in m else None
                for m in block["clients"].values()]
        weights = list(block["sample_counts"].values())
        value = aggregate(vals, weights if self.aggregation == "weighted_mean" else None,
                          self.aggregation)
        if value is None:
            # No validation data skips the update; an unavailable metric is an
            # invalid stopping policy.
            reported = [m for m in block["clients"].values() if m is not None]
            if reported and all(m.get(self.metric) is None for m in reported):
                raise ValueError(
                    f'early stopping is set to "{self.metric}" on the {self.split} '
                    f"split, and this run produces no value for it. "
                    + (unavailable_reason(self.metric)
                       or "No client reported the metric."))
            return False
        improved = (self.best_value is None
                    or (value > self.best_value + self.min_delta
                        if self.direction == "maximize"
                        else value < self.best_value - self.min_delta))
        if improved:
            self.best_value, self.best_round, self.stale = value, rnd, 0
        else:
            self.stale += 1
        return self.stale >= self.patience

    def record(self, reason: str, last_round: int) -> dict:
        if not self.enabled:
            return {"enabled": False, "termination_reason": reason,
                    "stopped_at_round": last_round,
                    "metric": None, "split": None, "direction": None,
                    "aggregation": None, "patience": None, "min_delta": None,
                    "best_round": None, "best_value": None}
        return {
            "enabled": True,
            "termination_reason": reason,
            "stopped_at_round": last_round,
            "metric": self.metric,
            "split": self.split,
            "direction": self.direction,
            "aggregation": self.aggregation,
            "patience": self.patience,
            "min_delta": self.min_delta,
            "best_round": self.best_round,
            "best_value": self.best_value,
        }
