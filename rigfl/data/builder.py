"""Generic dataset -> clients builder.

A **source** is a callable ``(cid, split) -> (X, y, groups)`` for one client's
split, where:
  * ``X`` is *array-like* -- a float tensor for small tabular data, or a
    ``np.memmap`` for large image data. RigFL indexes it **lazily, per item**
    (``torch.as_tensor(X[i])``), so a memmapped partition never loads whole into
    RAM.
  * ``y`` is a 1-D int/long array-like.
  * ``groups`` is per-sample group ids (e.g. subject id) or ``None``. When given,
    RigFL carves the validation split **group-wise** (whole groups held out) so
    grouped data (e.g. BraTS's adjacent same-brain slices) cannot leak from train
    into val. When ``None``, val is a plain random split.

CIFAR wraps flwr behind this signature; the biomedical repo exposes the same one.
``build_clients`` is the single place that turns a source into ``Client``s, so
every dataset is handled identically regardless of storage or grouping.
"""

from __future__ import annotations

from collections import Counter
from typing import Callable, Sequence

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from rigfl.core import Client, LearnedProjection, assemble_model

# (cid, split) -> (X array-like, y array-like, groups or None)
Source = Callable[[int, str], tuple]


def _is_multi(X) -> bool:
    """Multi-input X is a namedtuple of arrays (e.g. eICU's ``(ts, static)``);
    single-input X is a bare array."""
    return isinstance(X, tuple) and hasattr(X, "_fields")


class MultiTensor(tuple):
    """A batch of a multi-input sample: a tuple of tensors with ``.to(device)``.

    So an algorithm moves a multi-input batch exactly like a single tensor
    (``x = x.to(device)``), reads ``x.device``, and passes it to ``model(x)``
    unchanged -- the backbone unpacks the fields. Nothing in the algorithms
    needs to know the arity.
    """

    def to(self, device):
        return MultiTensor(t.to(device) for t in self)

    @property
    def device(self):
        return self[0].device


class _ArrayDataset(Dataset):
    """Lazily index array-like ``X`` by a fixed index list, converting to float
    tensors per item so memmaps stay on disk. ``X`` is a bare array (single-input)
    or a namedtuple of arrays (multi-input); a multi-input item is a tuple of
    tensors, one per field."""

    def __init__(self, X, y, indices: Sequence[int]):
        self.fields = [getattr(X, f) for f in X._fields] if _is_multi(X) else [X]
        self.multi = _is_multi(X)
        self.y, self.indices = y, list(indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i):
        j = self.indices[i]
        item = tuple(torch.as_tensor(f[j]).float() for f in self.fields)
        return (item if self.multi else item[0]), int(self.y[j])


def _collate(batch):
    """Stack a batch, keeping single-input as a Tensor and multi-input as a
    ``MultiTensor`` (one stacked tensor per field)."""
    xs, ys = zip(*batch)
    y = torch.tensor(ys)
    if isinstance(xs[0], tuple):                    # multi-input
        return MultiTensor(torch.stack([x[k] for x in xs]) for k in range(len(xs[0]))), y
    return torch.stack(xs), y


def _train_val_indices(n: int, groups, val_frac: float, *,
                       generator: torch.Generator | None = None
                       ) -> tuple[list[int], list[int]]:
    """Split indices into (train, val). Group-aware when ``groups`` is given
    (whole groups held out — no leakage); random otherwise. Deterministic under
    the current torch seed."""
    if groups is None:
        perm = torch.randperm(n, generator=generator).tolist()
        # Reserve at least one validation sample without emptying the train split.
        n_val = min(max(1, round(n * val_frac)), n - 1) if n > 1 else 0
        return perm[n_val:], perm[:n_val]

    groups = list(groups)
    counts = Counter(groups)
    uniq = list(dict.fromkeys(groups))                       # stable unique
    uniq = [uniq[i] for i in torch.randperm(len(uniq), generator=generator).tolist()]
    val_groups, seen, target = set(), 0, int(n * val_frac)
    for g in uniq:
        if seen >= target or len(val_groups) >= len(uniq) - 1:   # keep >=1 group in train
            break
        val_groups.add(g)
        seen += counts[g]
    val_idx = [i for i, g in enumerate(groups) if g in val_groups]
    train_idx = [i for i, g in enumerate(groups) if g not in val_groups]
    return train_idx, val_idx


def build_clients(source: Source, num_clients: int, num_classes: int,
                  backbones: list[Callable[[], nn.Module]], shared_dim: int,
                  val_frac: float = 0.2, batch: int = 32,
                  adapter: Callable[[int, int], nn.Module] | None = None) -> list[Client]:
    """Turn a data source + a pool of backbone *factories* into ``Client``s.

    ``backbones`` are factories (not instances) so every client gets its OWN
    backbone -- sharing an instance would share weights across clients.

    ``adapter`` is a factory ``(native_dim, shared_dim) -> Adapter`` -- the
    representation-alignment component. It defaults to a learned projection; the
    caller supplies a different one (e.g. FedTGP's ``AdaptivePool``) so each
    client model uses the alignment its own paper specifies.
    """
    if adapter is None:
        adapter = lambda native, shared: LearnedProjection(native, shared)
    clients: list[Client] = []
    for cid in range(num_clients):
        x_tr, y_tr, g_tr = source(cid, "train")
        train_idx, val_idx = _train_val_indices(len(y_tr), g_tr, val_frac)
        x_te, y_te, _ = source(cid, "test")

        backbone = backbones[cid % len(backbones)]()          # fresh instance per client
        model = assemble_model(
            backbone, shared_dim=shared_dim, num_classes=num_classes,
            adapter=adapter)
        clients.append(Client(
            model,
            DataLoader(_ArrayDataset(x_tr, y_tr, train_idx), batch_size=batch, shuffle=True, collate_fn=_collate),
            DataLoader(_ArrayDataset(x_tr, y_tr, val_idx), batch_size=batch, collate_fn=_collate),
            DataLoader(_ArrayDataset(x_te, y_te, range(len(y_te))), batch_size=batch, collate_fn=_collate),
        ))
    return clients
