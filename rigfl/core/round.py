"""Algorithm runners and their shared evaluation-history recorder.

Reporting-round selection is performed later from recorded history.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
from torch.utils.data import DataLoader

from rigfl.core.interfaces import Algorithm
from rigfl.core.model import ClientModel
from rigfl.eval.metrics import (
    COMPUTED_METRICS,
    direction_of,
    unavailable_reason,
)
from rigfl.eval.protocol import evaluate_split
from rigfl.eval.resources import measured, payload_bytes
from rigfl.eval.selection import aggregate

if TYPE_CHECKING:
    from rigfl.experiment.config import EarlyStoppingConfig


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


def iterative(algorithm: Algorithm, clients: list[Client], num_rounds: int,
              device: torch.device, num_classes: int, eval_gap: int = 1,
              verbose: bool = True, tracker=None, early_stopping=None,
              resource_monitor=None) -> dict:
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

    history: dict = {}
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
                                             resource_monitor=resource_monitor),
                "test": evaluate_split(algorithm, clients, shared, device, "test",
                                       num_classes,
                                       resource_monitor=resource_monitor),
            }
            append_evaluation(
                history, rnd, evaluated["validation"], evaluated["test"])

            if tracker is not None:
                _update_tracker_resources(tracker, resource_monitor)
                tracker.log_round(
                    rnd, evaluated["validation"], evaluated["test"])
            if verbose:
                _print_round(rnd, evaluated)

            if es.update(rnd, evaluated):
                stop_reason = "early_stopping"
                if verbose:
                    print(f"  early stop at round {rnd}: validation {es.metric} "
                          f"has not improved by {es.min_delta} for {es.patience} evaluations")
                break

    return {
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


def append_evaluation(history: dict, rnd: int, validation: dict, test: dict) -> None:
    """Add one evaluation round to a run's ``evaluation_history``.

    ``validation`` and ``test`` are ``evaluate_split`` results for that round.
    Start from an empty dict; a runner that evaluates once calls this once.
    Vectors stay aligned with ``evaluation_rounds``: a client with no data
    this round is padded with None rather than skipped.
    """
    rounds = history.setdefault("evaluation_rounds", [])
    per_client = history.setdefault("clients", {})
    counts = history.setdefault("client_sample_counts", {"validation": {}, "test": {}})
    rounds.append(rnd)
    n_rounds = len(rounds)
    for split, block in (("validation", validation), ("test", test)):
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
    _check_alignment(history)


def _check_alignment(history: dict) -> None:
    """Every per-round vector must be as long as evaluation_rounds."""
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
    print(f"round {rnd:3d} | val " + " ".join(parts) if parts else f"round {rnd:3d}")


class _EarlyStopping:
    """Validation-metric early stopping; independent of reporting-round selection."""

    def __init__(self, cfg: EarlyStoppingConfig | None):
        self.enabled = cfg is not None and cfg.enabled
        self.best_value = self.best_round = None
        self.stale = 0
        if not self.enabled:
            self.metric = self.direction = None
            self.aggregation = None
            self.patience = self.min_delta = None
            return

        self.metric = cfg.metric
        self.direction = direction_of(self.metric)
        self.aggregation = cfg.aggregation
        self.patience = cfg.patience
        self.min_delta = cfg.min_delta

    def update(self, rnd: int, evaluated: dict) -> bool:
        """Record this round's control metric; return whether training should stop."""
        if not self.enabled:
            return False
        block = evaluated["validation"]
        vals = [m[self.metric] if m and self.metric in m else None
                for m in block["clients"].values()]
        weights = list(block["sample_counts"].values())
        value = aggregate(
            vals,
            weights if self.aggregation == "sample_count" else None,
            self.aggregation,
        )
        if value is None:
            # No validation data skips the update; an unavailable metric is an
            # invalid stopping policy.
            reported = [m for m in block["clients"].values() if m is not None]
            if reported and all(m.get(self.metric) is None for m in reported):
                raise ValueError(
                    f'early stopping is set to validation "{self.metric}", '
                    f"and this run produces no value for it. "
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
                    "metric": None, "direction": None,
                    "aggregation": None, "patience": None, "min_delta": None,
                    "best_round": None, "best_value": None}
        return {
            "enabled": True,
            "termination_reason": reason,
            "stopped_at_round": last_round,
            "metric": self.metric,
            "direction": self.direction,
            "aggregation": self.aggregation,
            "patience": self.patience,
            "min_delta": self.min_delta,
            "best_round": self.best_round,
            "best_value": self.best_value,
        }
