"""The generic dataset -> clients builder and its train/val partition."""

from __future__ import annotations

import torch
import torch.nn as nn

from rigfl.core import Client
from rigfl.data.builder import _train_val_indices, build_clients

NUM_CLASSES = 3
INPUT_DIM = 8
SHARED_DIM = 4


class TinyBackbone(nn.Module):
    """A minimal backbone exposing ``out_dim`` (required by build_clients)."""

    def __init__(self, hidden: int = 6):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(INPUT_DIM, hidden), nn.ReLU())
        self.out_dim = hidden

    def forward(self, x):
        return self.net(x)


def _synth_source(n_per_split: int = 30):
    """A tiny in-memory Source: (cid, split) -> (X, y, groups)."""

    def source(cid: int, split: str):
        g = torch.Generator().manual_seed(1000 * cid + hash(split) % 1000)
        x = torch.randn(n_per_split, INPUT_DIM, generator=g)
        y = torch.randint(0, NUM_CLASSES, (n_per_split,), generator=g)
        return x, y, None

    return source


def test_train_val_indices_random_partition_is_disjoint_and_complete():
    n = 100
    train, val = _train_val_indices(n, groups=None, val_frac=0.2)
    assert len(val) == 20
    assert len(train) == 80
    assert set(train).isdisjoint(val)
    assert set(train) | set(val) == set(range(n))


def test_train_val_indices_group_split_keeps_whole_groups_on_one_side():
    # 10 groups of 10 samples each; no group may straddle train and val.
    groups = [i // 10 for i in range(100)]
    train, val = _train_val_indices(100, groups=groups, val_frac=0.3)

    assert set(train).isdisjoint(val)
    assert set(train) | set(val) == set(range(100))

    train_groups = {groups[i] for i in train}
    val_groups = {groups[i] for i in val}
    assert train_groups.isdisjoint(val_groups), "a group leaked across the split"
    # every group landed entirely on exactly one side
    assert train_groups | val_groups == set(range(10))


def test_build_clients_yields_clients_with_nonempty_loaders():
    num_clients = 3
    backbones = [lambda: TinyBackbone(6), lambda: TinyBackbone(10)]
    clients = build_clients(
        _synth_source(), num_clients=num_clients, num_classes=NUM_CLASSES,
        backbones=backbones, shared_dim=SHARED_DIM, val_frac=0.2, batch=8,
    )

    assert len(clients) == num_clients
    for c in clients:
        assert isinstance(c, Client)
        # every split has data
        assert len(c.train_loader.dataset) > 0
        assert len(c.val_loader.dataset) > 0
        assert len(c.test_loader.dataset) > 0
        # train/val are disjoint views of the same training array
        assert len(c.train_loader.dataset) + len(c.val_loader.dataset) == 30
        # a batch flows through the model to logits of the right width
        x, y = next(iter(c.train_loader))
        logits = c.model(x)
        assert logits.shape[1] == NUM_CLASSES
        assert x.shape[0] == y.shape[0]


def test_build_clients_gives_each_client_its_own_backbone():
    # Sharing a backbone instance would share weights; factories must not.
    backbones = [lambda: TinyBackbone(6)]
    clients = build_clients(
        _synth_source(), num_clients=2, num_classes=NUM_CLASSES,
        backbones=backbones, shared_dim=SHARED_DIM, batch=8,
    )
    assert clients[0].model.backbone is not clients[1].model.backbone


def test_small_clients_still_get_a_validation_split():
    """int(n * val_frac) truncated to zero, so a small client had no val split
    and was skipped during evaluation -- silently, since the loop just moves on."""
    from rigfl.data.builder import _train_val_indices
    for n in (2, 5, 9, 19):
        train_idx, val_idx = _train_val_indices(n, None, 0.1)
        assert len(val_idx) >= 1, f"n={n} produced an empty validation split"
        assert len(train_idx) >= 1
        assert len(train_idx) + len(val_idx) == n
    assert _train_val_indices(1, None, 0.1) == ([0], [])      # nothing to split
