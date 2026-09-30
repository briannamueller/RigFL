"""The generic dataset -> clients builder and its train/val partition."""

from __future__ import annotations

import torch
import torch.nn as nn

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


def test_build_clients_gives_each_client_its_own_backbone():
    # Sharing a backbone instance would share weights; factories must not.
    backbones = [lambda: TinyBackbone(6)]
    clients = build_clients(
        _synth_source(), num_clients=2, num_classes=NUM_CLASSES,
        backbones=backbones, shared_dim=SHARED_DIM, batch=8,
    )
    assert clients[0].model.backbone is not clients[1].model.backbone


def test_small_clients_still_get_a_validation_split():
    """Small clients receive nonempty training and validation splits."""
    from rigfl.data.builder import _train_val_indices
    for n in (2, 5, 9, 19):
        train_idx, val_idx = _train_val_indices(n, None, 0.1)
        assert len(val_idx) >= 1, f"n={n} produced an empty validation split"
        assert len(train_idx) >= 1
        assert len(train_idx) + len(val_idx) == n
    assert _train_val_indices(1, None, 0.1) == ([0], [])      # nothing to split
    # grouped: one whole group still goes to validation
    train, val = _train_val_indices(4, groups=[0, 0, 1, 1], val_frac=0.1)
    assert len(train) == 2 and len(val) == 2
