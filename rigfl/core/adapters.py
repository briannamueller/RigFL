"""Representation-alignment adapters.

A model-heterogeneous algorithm that synchronizes in *representation space*
(interface ``d`` — prototypes, shared representations, an averaged head) needs
every client's heterogeneous backbone to emit the same representation width.

How the original papers reach that shared width:

* **FedProto / FedGH** — a learned ``Linear`` sized to a fixed width (50 / 500).
* **FedKD** — a learned projection ``W_h`` for hidden-state distillation.
* **FedTGP** — parameter-free ``AdaptiveAvgPool1d`` (its own design).

RigFL makes the adapter an explicit, swappable component, so each algorithm uses
the alignment its own paper specifies:

* :class:`LearnedProjection` — the default (a learned linear map).
* :class:`AdaptivePool` — FedTGP's mechanism.
* :class:`Identity` — a no-op when the backbone already emits the shared width.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class Adapter(nn.Module):
    """Maps native backbone features ``(B, native_dim)`` to ``(B, shared_dim)``."""

    shared_dim: int


class LearnedProjection(Adapter):
    """Default adapter: a learned linear map ``native_dim -> shared_dim``.

    Matches how FedProto/FedGH reach a fixed representation width and FedKD's
    ``W_h`` — learnable, and free of the channel-averaging information loss that
    pooling incurs.
    """

    def __init__(self, native_dim: int, shared_dim: int, bias: bool = True) -> None:
        super().__init__()
        self.shared_dim = shared_dim
        self.proj = nn.Linear(native_dim, shared_dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class AdaptivePool(Adapter):
    """Parameter-free ``AdaptiveAvgPool1d`` down to ``shared_dim``.

    FedTGP's mechanism ("we add an average pooling layer after each feature
    extractor"). Lossy: it averages contiguous, arbitrarily ordered native
    channels, so prefer :class:`LearnedProjection` elsewhere.
    """

    def __init__(self, shared_dim: int) -> None:
        super().__init__()
        self.shared_dim = shared_dim
        self.pool = nn.AdaptiveAvgPool1d(shared_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, native_dim) -> (B, 1, native_dim) -> pool last dim -> (B, shared_dim)
        return self.pool(x.unsqueeze(1)).squeeze(1)


class Identity(Adapter):
    """No-op adapter for backbones that already emit ``shared_dim`` features."""

    def __init__(self, shared_dim: int) -> None:
        super().__init__()
        self.shared_dim = shared_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x
