"""Per-client evaluation over validation and test splits."""

from __future__ import annotations

import contextlib

import torch
from torch import nn

from rigfl.eval.metrics import canonical, compute_all
from rigfl.eval.resources import measured
from rigfl.prediction import Predictions, check_predictions


def _modules(obj, depth=0):
    """Every nn.Module reachable from a shared object or a client's scratch state."""
    if depth > 2:
        return
    if isinstance(obj, nn.Module):
        yield obj
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _modules(v, depth + 1)
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _modules(v, depth + 1)


@contextlib.contextmanager
def _eval_mode(*objects):
    """Put every reachable module in eval mode, then restore its prior mode.

    Includes auxiliary models in shared/client state, so evaluation doesn't
    update BatchNorm statistics.
    """
    modules = [m for o in objects for m in _modules(o)]
    was = [m.training for m in modules]
    try:
        for m in modules:
            m.eval()
        yield
    finally:
        for m, mode in zip(modules, was):
            m.train(mode)


@torch.no_grad()
def evaluate_split(algorithm, clients, shared, device, split: str, num_classes: int,
                   shared_by_client: list | None = None,
                   resource_monitor=None) -> dict:
    """Every computed metric for every client on one split, keyed by client id.

    A client with no data is recorded as ``None``. Sample counts accompany the
    metrics for weighted aggregation.
    """
    per_client: dict[str, dict[str, float] | None] = {}
    counts: dict[str, int | None] = {}
    for cid, client in enumerate(clients):
        client_shared = (shared_by_client[cid]
                         if shared_by_client is not None else shared)
        per_client[str(cid)], counts[str(cid)] = evaluate_client(
            algorithm, client, cid, client_shared, device, split, num_classes,
            resource_monitor=resource_monitor)
    return {
        "clients": per_client,
        "sample_counts": counts,
    }


def evaluate_client(algorithm, client, cid: int, shared, device, split: str,
                    num_classes: int, resource_monitor=None
                    ) -> tuple[dict[str, float] | None, int | None]:
    """Every computed metric for one client on one split, and its sample count.

    A client with no data on the split gets ``(None, None)``.
    """
    loader = getattr(client, f"{split}_loader")
    if loader is None:
        return None, None
    outputs, labels = [], []
    operation = "validation" if split == "val" else "test"
    with (
        measured(resource_monitor, operation, category="evaluation",
                 client_id=cid),
        _eval_mode(client.model, shared, client.state),
    ):
        for x, y in loader:
            out = check_predictions(algorithm.predict(client, x.to(device), shared))
            outputs.append(_to_cpu(out))
            labels.append(y)
    if not outputs:                      # empty loader -- no data, not zero score
        return None, None
    output, labels = _concat(outputs), torch.cat(labels)
    metrics = compute_all(output, labels, num_classes,
                          class_weights=client.class_weights)
    return metrics, int(labels.numel())


def _to_cpu(out: Predictions) -> Predictions:
    return Predictions(
        labels=out.labels.cpu(),
        probabilities=None if out.probabilities is None else out.probabilities.cpu())


def _concat(outputs: list[Predictions]) -> Predictions:
    """One client's batches into one prediction.

    A field is kept only if every batch has it.
    """
    def joined(name):
        parts = [getattr(o, name) for o in outputs]
        return None if any(p is None for p in parts) else torch.cat(parts)

    return Predictions(labels=torch.cat([o.labels for o in outputs]),
                            probabilities=joined("probabilities"))


def mean_over_clients(evaluated: dict, metric: str) -> float | None:
    """Unweighted mean across available clients, or ``None`` if none report it."""
    name = canonical(metric)
    vals = [v for m in evaluated["clients"].values() if m is not None
            for v in (m.get(name),) if v is not None]
    return sum(vals) / len(vals) if vals else None
