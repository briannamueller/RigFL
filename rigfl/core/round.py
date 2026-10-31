"""Algorithm runners and their shared evaluation-history recorder.

Reporting-round selection is performed later from recorded history.
"""

from __future__ import annotations

import contextlib
import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch.utils.data import DataLoader

from rigfl.core.interfaces import Algorithm
from rigfl.core.model import ClientModel
from rigfl.eval.metrics import (
    COMPUTED_METRICS,
    compute_all,
    direction_of,
    unavailable_reason,
)
from rigfl.eval.protocol import evaluate_client, evaluate_split
from rigfl.eval.resources import measured, payload_bytes
from rigfl.eval.selection import aggregate
from rigfl.prediction import Predictions

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
    class_weights: torch.Tensor | None = None
    state: dict = field(default_factory=dict)


def client_training_seed(seed: int, client_id: int) -> int:
    """Seed for one client's independent training run."""
    return int(np.random.SeedSequence([seed, client_id]).generate_state(1)[0])


def seed_training(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def independent(algorithm: Algorithm, clients: list[Client], num_rounds: int,
                device: torch.device, num_classes: int, eval_gap: int = 1,
                verbose: bool = True, tracker=None, early_stopping=None,
                resource_monitor=None) -> dict:
    """Train and select one model per client, then evaluate the baseline.

    Local and Global Ensemble share this runner. Their only difference is the
    shared object returned by ``finalize_local_models`` and how ``predict`` uses
    it: Local evaluates each client's own model, while Global Ensemble evaluates
    the same frozen soft-voting pool for every client.

    An algorithm with ``scores_every_epoch`` (Local) is scored on validation and
    test at every evaluation epoch of each client's own training, and those
    epochs make up its history. Clients stop independently, so a client has no
    values after it stops and reporting selects each client's own epoch.
    Otherwise the history holds one evaluation of the finalized models.

    Each client keeps the checkpoint that is best on the algorithm's
    ``member_metric`` when it sets one (Global Ensemble), and otherwise on the
    early-stopping metric. Training still stops on the early-stopping metric.

    Each client trains from its own seed, derived from the run's training seed,
    so its model does not depend on the clients trained before it.
    """
    _require_operations(
        algorithm,
        "independent",
        ("make_optimizer", "train_one_round", "finalize_local_models", "predict"),
    )
    _start_run(algorithm, clients, device, num_rounds)
    enabled = early_stopping is not None and early_stopping.enabled
    metric = early_stopping.metric if enabled else None
    direction = direction_of(metric) if enabled else None
    patience = early_stopping.patience if enabled else None
    min_delta = early_stopping.min_delta if enabled else None
    member_metric = getattr(getattr(algorithm, "config", None), "member_metric", None)
    select = member_metric or metric
    select_direction = direction_of(select) if select else None
    select_delta = min_delta if select == metric else 0.0
    per_epoch = getattr(algorithm, "scores_every_epoch", False)
    selections = {}
    epochs: dict[int, list] = {}
    models = []
    # run_one seeds torch with the training seed before building clients
    run_seed = torch.initial_seed()

    for cid, client in enumerate(clients):
        seed_training(client_training_seed(run_seed, cid))
        model = client.model.to(device)
        optimizer = algorithm.make_optimizer(model)
        best_state = best_value = best_round = stop_value = None
        stale = 0
        last_round = -1
        termination = "completed_all_rounds"
        epochs[cid] = []

        for rnd in range(num_rounds):
            algorithm.round_idx = rnd
            with measured(
                resource_monitor, "local_train", category="algorithm", client_id=cid
            ):
                algorithm.train_one_round(client, optimizer)
            last_round = rnd

            if not (select or per_epoch) or (
                    rnd % eval_gap != 0 and rnd != num_rounds - 1):
                continue
            if per_epoch:
                # scoring that early stopping would not do runs on a forked RNG,
                # so training follows the same random stream either way
                with _forked_rng(not enabled):
                    val = evaluate_client(algorithm, client, cid, None, device,
                                          "val", num_classes, resource_monitor)
                with _forked_rng(True):
                    test = evaluate_client(algorithm, client, cid, None, device,
                                           "test", num_classes, resource_monitor)
                epochs[cid].append((rnd, val, test))
                if not select:
                    continue
                metrics = val[0] or {}
            else:
                with measured(
                    resource_monitor, "validation", category="evaluation",
                    client_id=cid
                ), _forked_rng(not enabled):
                    metrics = _model_validation_metrics(
                        model, client, device, num_classes
                    )
            for name in {metric, select} - {None}:
                if metrics.get(name) is None:
                    raise ValueError(
                        f'independent runner metric "{name}" is unavailable '
                        f"for client {cid}"
                    )
            if _improved(metrics[select], best_value, select_direction, select_delta):
                best_value, best_round = metrics[select], rnd
                best_state = {
                    key: tensor.detach().cpu().clone()
                    for key, tensor in model.state_dict().items()
                }
            if enabled:
                if _improved(metrics[metric], stop_value, direction, min_delta):
                    stop_value, stale = metrics[metric], 0
                else:
                    stale += 1
                    if stale >= patience:
                        termination = "early_stopping"
                        break

        if select:
            if best_state is None:
                raise RuntimeError(
                    f"independent runner selected no checkpoint for client {cid}"
                )
            model.load_state_dict(best_state)
            selected_round = best_round
        else:
            selected_round = last_round
        model.eval()
        models.append(model)
        selections[str(cid)] = {
            "selected_round": selected_round,
            "best_value": best_value,
            "stopped_at_round": last_round,
            "termination_reason": termination,
        }

    shared = algorithm.finalize_local_models(models)
    if resource_monitor is not None and shared is not None:
        for cid, model in enumerate(models):
            resource_monitor.record_transfer(
                "client_to_server",
                _payload_size(algorithm, model, kind="client_to_server"),
                sender=cid,
            )
    if resource_monitor is not None:
        resource_monitor.checkpoint(0)

    history = {}
    if per_epoch:
        for rnd, evaluated in per_client_rounds(epochs):
            append_evaluation(
                history, rnd, evaluated["validation"], evaluated["test"])
            if tracker is not None:
                tracker.log_round(rnd, evaluated["validation"], evaluated["test"])
            if verbose:
                _print_round(rnd, evaluated)
        if tracker is not None:
            _update_tracker_resources(tracker, resource_monitor)
    else:
        evaluated = {
            "validation": evaluate_split(
                algorithm, clients, shared, device, "val", num_classes,
                resource_monitor=resource_monitor,
            ),
            "test": evaluate_split(
                algorithm, clients, shared, device, "test", num_classes,
                resource_monitor=resource_monitor,
            ),
        }
        append_evaluation(history, 0, evaluated["validation"], evaluated["test"])
        if tracker is not None:
            _update_tracker_resources(tracker, resource_monitor)
            tracker.log_round(0, evaluated["validation"], evaluated["test"])
        if verbose:
            _print_round(0, evaluated)

    return {
        "evaluation_history": history,
        **({"round_selection": "client-specific"} if per_epoch else {}),
        "selection_provenance": {
            "stage": "independent_local_training",
            "metric": select,
            "clients": selections,
        },
        "early_stopping": {
            "enabled": enabled,
            "scope": "per_client",
            "termination_reason": (
                "per_client" if enabled else "completed_all_rounds"
            ),
            "stopped_at_round": max(
                (value["stopped_at_round"] for value in selections.values()),
                default=-1,
            ),
            "metric": metric,
            "direction": direction,
            "aggregation": None,
            "patience": patience,
            "min_delta": min_delta,
            "best_round": None,
            "best_value": None,
            "clients": selections,
        },
    }


def per_client_rounds(epochs: dict[int, list]):
    """Yield ``(round, {"validation": block, "test": block})`` in round order.

    ``epochs`` maps each client id to its ``(round, (metrics, count),
    (metrics, count))`` evaluations. Every client appears in every block, with
    ``None`` at rounds it was not evaluated, such as rounds after it stopped.
    """
    by_round: dict[int, dict] = {}
    for cid, evaluations in epochs.items():
        for rnd, val, test in evaluations:
            by_round.setdefault(rnd, {})[cid] = {"validation": val, "test": test}
    for rnd in sorted(by_round):
        scored = by_round[rnd]
        yield rnd, {
            split: {
                "clients": {str(cid): scored[cid][split][0] if cid in scored else None
                            for cid in epochs},
                "sample_counts": {str(cid): scored[cid][split][1] if cid in scored
                                  else None for cid in epochs},
            }
            for split in ("validation", "test")
        }


def _forked_rng(fork: bool):
    return torch.random.fork_rng(devices=[]) if fork else contextlib.nullcontext()


@torch.no_grad()
def _model_validation_metrics(model, client, device, num_classes: int) -> dict:
    if client.val_loader is None:
        raise ValueError("independent early stopping requires validation data")
    was_training = model.training
    model.eval()
    outputs, labels = [], []
    try:
        for x, y in client.val_loader:
            probabilities = Predictions.from_logits(model(x.to(device))).probabilities
            outputs.append(probabilities.cpu())
            labels.append(y.cpu())
    finally:
        model.train(was_training)
    if not labels:
        raise ValueError("independent early stopping received an empty validation split")
    return compute_all(
        Predictions.from_probabilities(torch.cat(outputs)),
        torch.cat(labels),
        num_classes,
        class_weights=client.class_weights,
    )


def _improved(value: float, best: float | None, direction: str,
              min_delta: float) -> bool:
    if best is None:
        return True
    if direction == "minimize":
        return value < best - min_delta
    return value > best + min_delta


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
