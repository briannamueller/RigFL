"""Algorithm runners and their shared evaluation-history recorder.

Reporting-round selection is performed later from recorded history.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import torch
from torch.utils.data import DataLoader

from rigfl.core.interfaces import IterativeAlgorithm
from rigfl.core.model import ClientModel
from rigfl.eval.metrics import (
    COMPUTED_METRICS,
    direction_of,
    require_computable,
    unavailable_reason,
    uses_pooled_predictions,
)
from rigfl.eval.protocol import evaluate_split
from rigfl.eval.resources import measured, payload_bytes
from rigfl.eval.selection import aggregate


@dataclass
class Client:
    """A client's data, optional model, identity, and persistent algorithm state.

    Behavior lives entirely in the algorithm. ``state`` is an algorithm-defined
    dictionary that persists across this client's operations during one run.
    """

    model: ClientModel | None
    train_loader: DataLoader
    val_loader: DataLoader | None = None
    test_loader: DataLoader | None = None
    client_id: int | None = None
    state: dict = field(default_factory=dict)


def iterative(algorithm: IterativeAlgorithm, clients: list[Client], num_rounds: int,
              device: torch.device, num_classes: int, eval_gap: int = 1,
              verbose: bool = True, tracker=None, early_stopping=None,
              resource_monitor=None, positive_class: int | None = None) -> dict:
    """Train, recording every metric for every client at every evaluation round.

    Returns the canonical history. No round in it is marked selected; use
    :mod:`rigfl.eval.selection` to choose one (accuracy by default).
    """
    _require_operations(algorithm, "iterative",
                        ("init_globals", "local_train", "aggregate", "predict"))
    _start_run(algorithm, clients, device, num_rounds)
    es = _EarlyStopping(early_stopping)
    with measured(resource_monitor, "init_globals", category="algorithm"):
        shared = algorithm.init_globals()

    rounds_evaluated: list[int] = []
    per_client: dict[str, dict[str, dict[str, list]]] = {}
    counts: dict[str, dict[str, list]] = {"validation": {}, "test": {}}
    aggregate_metrics: dict[str, dict[str, list]] = {
        "validation": {},
        "test": {},
    }
    stop_reason = "completed_all_rounds"
    last_round = -1

    for rnd in range(num_rounds):
        algorithm.round_idx = rnd
        shared_size = _payload_size(algorithm, shared, kind="server_to_client")
        uploads = []
        for cid, client in enumerate(clients):
            if resource_monitor is not None:
                resource_monitor.record_transfer(
                    "server_to_client", shared_size, receiver=cid)
            with measured(resource_monitor, "local_train", category="algorithm",
                          client_id=cid):
                upload = algorithm.local_train(client, shared)
            uploads.append(upload)
            if resource_monitor is not None:
                resource_monitor.record_transfer(
                    "client_to_server", _payload_size(
                        algorithm, upload, kind="client_to_server"),
                    sender=cid)
        with measured(resource_monitor, "aggregate", category="algorithm"):
            shared = algorithm.aggregate(uploads, shared)
        last_round = rnd

        if resource_monitor is not None:
            resource_monitor.checkpoint(rnd)

        if rnd % eval_gap == 0 or rnd == num_rounds - 1:
            evaluated = {
                "validation": evaluate_split(algorithm, clients, shared, device, "val",
                                             num_classes,
                                             resource_monitor=resource_monitor,
                                             positive_class=positive_class),
                "test": evaluate_split(algorithm, clients, shared, device, "test",
                                       num_classes,
                                       resource_monitor=resource_monitor,
                                       positive_class=positive_class),
            }
            rounds_evaluated.append(rnd)
            _append(
                per_client,
                counts,
                aggregate_metrics,
                evaluated,
                len(rounds_evaluated),
            )

            if tracker is not None:
                _update_tracker_resources(tracker, resource_monitor)
                tracker.log_round(
                    rnd, evaluated["validation"], evaluated["test"])
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
        "aggregate_metrics": aggregate_metrics,
    }
    _check_alignment(history)

    return {
        "selection_views_supported": ["global", "per-client"],
        "evaluation_history": history,
        "early_stopping": es.record(stop_reason, last_round),
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


def _payload_size(algorithm, payload, *, kind: str) -> int:
    size = getattr(algorithm, "communication_payload_bytes", None)
    return size(payload, kind=kind) if callable(size) else payload_bytes(payload)


def _update_tracker_resources(tracker, monitor) -> None:
    update = getattr(tracker, "update_resources", None)
    if callable(update) and monitor is not None:
        update(monitor.to_dict())


def _append(per_client, counts, aggregate_metrics, evaluated, n_rounds) -> None:
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
        for name in COMPUTED_METRICS:
            series = aggregate_metrics[split].setdefault(name, [])
            while len(series) < n_rounds - 1:
                series.append(None)
            series.append(block["aggregate"].get(name))


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
    for split, metrics in history["aggregate_metrics"].items():
        for name, series in metrics.items():
            if len(series) != n:
                raise RuntimeError(
                    f"evaluation history is misaligned: aggregate {split}.{name} "
                    f"has {len(series)} values for {n} rounds"
                )


def _print_round(rnd: int, evaluated: dict) -> None:
    """Show every available validation metric for this round."""
    from rigfl.eval.protocol import mean_over_clients
    parts = []
    for m in COMPUTED_METRICS:
        v = (
            evaluated["validation"]["aggregate"].get(m)
            if uses_pooled_predictions(m)
            else mean_over_clients(evaluated["validation"], m)
        )
        if v is not None:
            parts.append(f"{m} {v:.4f}")
    line = f"round {rnd:3d} | val " + " ".join(parts) if parts else f"round {rnd:3d}"
    if os.environ.get("RIGFL_LOG_TEST_ROUNDS") == "1":
        tparts = [
            f"{m} {v:.4f}"
            for m in COMPUTED_METRICS
            if (
                v := (
                    evaluated["test"]["aggregate"].get(m)
                    if uses_pooled_predictions(m)
                    else mean_over_clients(evaluated["test"], m)
                )
            ) is not None
        ]
        if tparts:
            line += " | test " + " ".join(tparts)
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
        expected_direction = direction_of(self.metric)
        requested_direction = get("direction", None)
        if requested_direction and requested_direction != expected_direction:
            raise ValueError(
                f'early stopping direction must be "{expected_direction}" for '
                f'metric "{self.metric}", not "{requested_direction}"'
            )
        self.direction = expected_direction
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
        if uses_pooled_predictions(self.metric):
            value = block["aggregate"].get(self.metric)
        else:
            vals = [m[self.metric] if m and self.metric in m else None
                    for m in block["clients"].values()]
            weights = list(block["sample_counts"].values())
            value = aggregate(
                vals,
                weights if self.aggregation == "weighted_mean" else None,
                self.aggregation,
            )
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
            "aggregation": (
                "pooled"
                if uses_pooled_predictions(self.metric)
                else self.aggregation
            ),
            "patience": self.patience,
            "min_delta": self.min_delta,
            "best_round": self.best_round,
            "best_value": self.best_value,
        }
