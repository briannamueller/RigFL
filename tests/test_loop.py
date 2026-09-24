"""A tiny end-to-end run of the invariant federated loop.

Mirrors ``examples/smoke.py`` but shrinks everything (fewer classes, tiny
models, 1 local epoch, 2 rounds) so it runs in a fraction of a second on CPU.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from rigfl.core import Client, ClientModel, LearnedProjection, iterative
from rigfl.algorithms.fedproto import FedProto, FedProtoConfig
from rigfl.algorithms.local import Local, LocalConfig

NUM_CLASSES = 3
INPUT_DIM = 8
SHARED_DIM = 4
DEVICE = torch.device("cpu")

# Stable class centers (fixed generator) so the class structure is learnable.
CENTERS = torch.randn(NUM_CLASSES, INPUT_DIM,
                      generator=torch.Generator().manual_seed(12345)) * 3.0


def _loader(n_per_class: int = 12, spread: float = 1.5, batch: int = 16) -> DataLoader:
    xs, ys = [], []
    for c in range(NUM_CLASSES):
        xs.append(CENTERS[c] + spread * torch.randn(n_per_class, INPUT_DIM))
        ys.append(torch.full((n_per_class,), c))
    x, y = torch.cat(xs), torch.cat(ys)
    return DataLoader(TensorDataset(x, y), batch_size=batch, shuffle=True)


def _client(hidden: int) -> Client:
    model = ClientModel(
        nn.Sequential(nn.Linear(INPUT_DIM, hidden), nn.ReLU()),
        LearnedProjection(hidden, SHARED_DIM),
        nn.Linear(SHARED_DIM, NUM_CLASSES),
    )
    return Client(model=model, train_loader=_loader(),
                  val_loader=_loader(6), test_loader=_loader(6))


def _clients():
    return [_client(h) for h in (8, 12)]  # heterogeneous backbones


def _assert_valid_result(result):
    assert isinstance(result, dict)
    # The loop records history and selects nothing; selection is explicit and
    # happens afterwards.
    assert result["selection_views_supported"] == ["global", "per-client"]
    hist = result["evaluation_history"]
    assert hist["evaluation_rounds"] and all(0 <= r < 2 for r in hist["evaluation_rounds"])
    from rigfl.eval.selection import select_global
    sel = select_global(hist, "accuracy")
    assert 0 <= sel["selected_round"] < 2

    # A selection reports per-client values; these assertions are about the
    # aggregate being a sane probability.
    mean = lambda xs: sum(v for v in xs if v is not None) / max(sum(v is not None for v in xs), 1)
    test = {"acc": mean(sel["test"]["accuracy"]),
            "bacc": mean(sel["test"]["balanced_accuracy"])}
    for key in ("acc", "bacc"):
        v = test[key]
        assert isinstance(v, float)
        assert math.isfinite(v)
        assert 0.0 <= v <= 1.0


def test_iterative_local():
    torch.manual_seed(0)
    result = iterative(Local(LocalConfig(local_epochs=1, lr=0.05)), _clients(),
                           num_rounds=2, device=DEVICE, num_classes=NUM_CLASSES,
                           verbose=False)
    _assert_valid_result(result)


def test_iterative_fedproto_aggregating_algorithm():
    torch.manual_seed(0)
    result = iterative(
        FedProto(FedProtoConfig(lamda=1.0, local_epochs=1, lr=0.05)), _clients(),
                           num_rounds=2, device=DEVICE, num_classes=NUM_CLASSES,
                           verbose=False)
    _assert_valid_result(result)
