"""A tiny end-to-end run of the invariant federated loop.

Mirrors ``examples/smoke.py`` but shrinks everything (fewer classes, tiny
models, 1 local epoch, 2 rounds) so it runs in a fraction of a second on CPU.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from rigfl.core import (Client, ClientModel, LearnedProjection, LocalSelection,
                        Predictions, iterative, p2p_one_shot)
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
    assert result["schema_version"] == 3
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


class _OneShot:
    def __init__(self):
        self.events = []

    def prepare(self, model, train_loader, ctx):
        self.events.append(f"prepare:{ctx.client_id}")
        return ctx.client_id

    def one_shot_communication(self, outgoing):
        self.events.append(f"communicate:{outgoing}")
        return [tuple(outgoing) for _ in outgoing]

    def local_computation(self, model, incoming, train_loader, ctx):
        assert incoming == (0, 1)
        self.events.append(f"compute:{ctx.client_id}")
        return LocalSelection(ctx.client_id + 4, "accuracy", 0.8 + ctx.client_id / 10)

    def predict(self, client, x, shared):
        return Predictions.from_logits(client.model(x))


def test_p2p_one_shot_has_one_communication_and_no_federated_round_loop():
    algorithm = _OneShot()
    result = p2p_one_shot(
        algorithm, _clients(), num_rounds=99, device=DEVICE,
        num_classes=NUM_CLASSES, eval_gap=17, verbose=False)

    assert algorithm.events == [
        "prepare:0", "prepare:1", "communicate:[0, 1]", "compute:0", "compute:1"
    ]
    assert result["selection_views_supported"] == ["per-client"]
    assert result["evaluation_history"]["evaluation_rounds"] == [0]
    assert result["selection_provenance"] == {
        "view": "per-client",
        "stage": "local_computation",
        "metric": "accuracy",
        "clients": {
            "0": {"selected_step": 4, "validation_value": 0.8},
            "1": {"selected_step": 5, "validation_value": 0.9},
        },
    }


def test_p2p_one_shot_rejects_round_level_early_stopping():
    with pytest.raises(ValueError, match="cannot be enabled"):
        p2p_one_shot(
            _OneShot(), _clients(), num_rounds=2, device=DEVICE,
            num_classes=NUM_CLASSES, verbose=False,
            early_stopping={"enabled": True, "metric": "accuracy"})
