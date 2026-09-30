"""Generic dataset -> clients builder.

A source is a callable ``(cid, split) -> (X, y, groups)`` for one client's split:
``X`` is array-like, ``y`` a 1-D label array, and ``groups`` per-sample group ids
or ``None``. With groups, validation holds out whole groups.
"""

from __future__ import annotations

from collections import Counter
from typing import Callable, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from rigfl.core import Client, assemble_model

# (cid, split) -> (X array-like, y array-like, groups or None)
Source = Callable[[int, str], tuple]


def _is_multi(X) -> bool:
    """Return whether X is a named tuple containing multiple input arrays."""
    return isinstance(X, tuple) and hasattr(X, "_fields")


class MultiTensor(tuple):
    """A batch of a multi-input sample: a tuple of tensors with ``.to(device)``."""

    def to(self, device):
        return MultiTensor(t.to(device) for t in self)

    @property
    def device(self):
        return self[0].device


class _ArrayDataset(Dataset):
    """Lazily index array-like ``X`` by a fixed index list, converting per item
    to ``input_dtype`` so memmaps stay on disk. ``X`` is a bare array (single-input)
    or a namedtuple of arrays (multi-input); a multi-input item is a tuple of
    tensors, one per field."""

    def __init__(
        self,
        X,
        y,
        indices: Sequence[int],
        input_dtype: torch.dtype = torch.float32,
    ):
        self.fields = [getattr(X, f) for f in X._fields] if _is_multi(X) else [X]
        self.multi = _is_multi(X)
        self.y, self.indices = y, list(indices)
        self.input_dtype = input_dtype

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i):
        j = self.indices[i]
        values = [f[j] for f in self.fields]
        values = [
            np.array(value, copy=True)
            if isinstance(value, np.ndarray) and not value.flags.writeable
            else value
            for value in values
        ]
        item = tuple(
            torch.as_tensor(value).to(dtype=self.input_dtype) for value in values
        )
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
    """Split indices into (train, val); whole groups go to val when ``groups`` is given."""
    if groups is None:
        perm = torch.randperm(n, generator=generator).tolist()
        # Reserve at least one validation sample without emptying the train split.
        n_val = min(max(1, round(n * val_frac)), n - 1) if n > 1 else 0
        return perm[n_val:], perm[:n_val]

    groups = list(groups)
    counts = Counter(groups)
    uniq = list(dict.fromkeys(groups))                       # stable unique
    uniq = [uniq[i] for i in torch.randperm(len(uniq), generator=generator).tolist()]
    if len(uniq) < 2:
        return list(range(n)), []
    val_groups, seen, target = set(), 0, max(1, round(n * val_frac))
    for g in uniq:
        if seen >= target or len(val_groups) >= len(uniq) - 1:   # keep >=1 group in train
            break
        val_groups.add(g)
        seen += counts[g]
    val_idx = [i for i, g in enumerate(groups) if g in val_groups]
    train_idx = [i for i, g in enumerate(groups) if g not in val_groups]
    return train_idx, val_idx


def _stream_generator(seed: int, stream: int) -> torch.Generator:
    """One generator per independent stream. Clients share it: they hold different
    data, so drawing the same positions gives them different samples."""
    return torch.Generator().manual_seed(seed + stream)


def _make_client(cid: int, train: Dataset, validation: Dataset, test: Dataset, *,
                 backbones, shared_dim: int, num_classes: int, adapter,
                 batch: int, seed: int, build_models: bool) -> Client:
    model = None
    if build_models:
        backbone = backbones[cid % len(backbones)]()      # fresh instance per client
        model = assemble_model(
            backbone, shared_dim=shared_dim, num_classes=num_classes,
            adapter=adapter)
    return Client(
        model,
        DataLoader(
            train, batch_size=batch, shuffle=True, collate_fn=_collate,
            generator=_stream_generator(seed, 1),
        ),
        DataLoader(
            validation, batch_size=batch, collate_fn=_collate,
            generator=_stream_generator(seed, 2),
        ),
        DataLoader(
            test, batch_size=batch, collate_fn=_collate,
            generator=_stream_generator(seed, 3),
        ),
    )


def build_clients(source: Source, num_clients: int, num_classes: int,
                  backbones: list[Callable[[], nn.Module]], shared_dim: int,
                  val_frac: float = 0.2, batch: int = 32,
                  adapter: Callable[[int, int], nn.Module] | None = None,
                  seed: int = 0, split_seed: int = 0,
                  build_models: bool = True) -> list[Client]:
    """Turn a data source + a pool of backbone *factories* into ``Client``s.

    ``backbones`` are factories so each client gets a fresh instance.

    ``adapter``: factory ``(native_dim, shared_dim) -> module``; without one the
    head sits directly on the backbone's native output.

    ``build_models=False`` is for workflows that construct their own model pool.
    """
    clients: list[Client] = []
    for cid in range(num_clients):
        x_tr, y_tr, g_tr = source(cid, "train")
        train_idx, val_idx = _train_val_indices(
            len(y_tr), g_tr, val_frac,
            generator=_stream_generator(split_seed, 0),
        )
        x_te, y_te, _ = source(cid, "test")
        clients.append(_make_client(
            cid,
            _ArrayDataset(x_tr, y_tr, train_idx),
            _ArrayDataset(x_tr, y_tr, val_idx),
            _ArrayDataset(x_te, y_te, range(len(y_te))),
            backbones=backbones, shared_dim=shared_dim, num_classes=num_classes,
            adapter=adapter, batch=batch, seed=seed, build_models=build_models,
        ))
    return clients
