"""Adapters map each client's native backbone features to a shared width ``d``.

Algorithms that exchange or mix representations (prototypes, a shared head)
need every client's backbone to produce the same width. FedProto, FedGH, and
pFedMoE use LearnedProjection; FedTGP uses AdaptivePool, as in its paper. All
other algorithms put the head directly on the backbone's native output.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class Adapter(nn.Module):
    """Maps native backbone features ``(B, native_dim)`` to ``(B, shared_dim)``."""

    shared_dim: int


class LearnedProjection(Adapter):
    """Default adapter: a learned linear map ``native_dim -> shared_dim``.

    Used by FedProto/FedGH in their papers.
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
    extractor"). Averages neighboring feature values to reach ``shared_dim``. The
    feature order is arbitrary, so this discards information that a learned
    projection could keep.
    """

    def __init__(self, shared_dim: int) -> None:
        super().__init__()
        self.shared_dim = shared_dim
        self.pool = nn.AdaptiveAvgPool1d(shared_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, native_dim) -> (B, 1, native_dim) -> pool last dim -> (B, shared_dim)
        return self.pool(x.unsqueeze(1)).squeeze(1)


class Identity(Adapter):
    """No-op adapter; the head reads the backbone's native features."""

    def __init__(self, shared_dim: int) -> None:
        super().__init__()
        self.shared_dim = shared_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x
