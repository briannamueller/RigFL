"""BioSilo source: realistically-partitioned biomedical datasets, federated the
way they were collected (by hospital, study, institution).

BioSilo's ``load_partition(dataset, cid, split) -> (X, y, groups)`` is byte-identical
to RigFL's :data:`~rigfl.data.builder.Source`, so the generic ``build_clients``
consumes clinical partitions exactly as it consumes CIFAR -- including group-aware
val carving (no person/subject leaks) and lazy memmap indexing. BioSilo emits no
validation split by design; RigFL carves one from train.

Unlike CIFAR (synthetic Dirichlet partitions built on the fly), a BioSilo dataset
is *generated once* (``biosilo.generate``) into a data root and loaded by id here.
Set ``BIOSILO_DATA_ROOT`` or pass ``root=``.
"""

from __future__ import annotations

from typing import Callable, Optional

import torch.nn as nn

from rigfl.data.builder import build_clients


def biosilo_source(dataset: str, root: Optional[str] = None,
                   partition: Optional[str] = None):
    """A ``(cid, split) -> (X, y, groups)`` source over a generated BioSilo
    partition, plus the partition handle (num_clients / num_classes / inputs)."""
    import biosilo

    handle = biosilo.load(dataset, root=root, partition=partition)

    def source(cid: int, split: str):
        return handle.client(cid, split)

    return source, handle


def temporal_dims(handle) -> tuple[int, int]:
    """``(n_ts, n_static)`` for a two-input (time series, static) partition:
    the series input is ``(T, n_ts)`` and the static input is ``(n_static,)``."""
    ts, static = handle.inputs
    return ts["shape"][-1], static["shape"][0]


def build_biosilo_clients(dataset: str, shared_dim: int,
                          backbones: list[Callable[[], nn.Module]],
                          root: Optional[str] = None, partition: Optional[str] = None,
                          val_frac: float = 0.2, batch: int = 32, adapter=None):
    """Build ``Client``s from a generated BioSilo partition. ``num_clients`` and
    ``num_classes`` come from the partition, not from a config -- a natural
    federation is whatever the data is. ``adapter`` selects the alignment
    component (default: learned projection). Returns ``(clients, handle)``."""
    source, handle = biosilo_source(dataset, root, partition)
    clients = build_clients(source, handle.num_clients, handle.num_classes,
                            backbones, shared_dim, val_frac, batch, adapter)
    return clients, handle
