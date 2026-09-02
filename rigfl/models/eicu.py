"""Heterogeneous temporal backbones for eICU-style clients.

The input is multi-input: a per-sample time series ``ts`` of shape ``(T, n_ts)``
and a static vector ``static`` of shape ``(n_static,)`` -- carried through RigFL
as a :class:`~rigfl.data.builder.MultiTensor` ``(ts, static)``. Each backbone
encodes the series (GRU / 1D-CNN / LSTM) and the static features (MLP), then
concatenates to native features ``(B, out_dim)``. Widths differ across the pool,
exactly what the client adapters exist to align.

``eicu_pool(n_ts, n_static)`` returns backbone **factories** (one fresh model per
client), the same contract as :func:`rigfl.models.cifar.cifar_pool`. The generic
builder does the adapter + head wiring.
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


class _StaticMLP(nn.Module):
    def __init__(self, n_static: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_static, hidden), nn.ReLU())

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.net(s)


class GRUTabularBackbone(nn.Module):
    """GRU over the series + MLP over static -> concatenated ``out_dim`` features."""

    def __init__(self, n_ts: int, n_static: int, hidden: int = 64, out_dim: int = 128):
        super().__init__()
        self.gru = nn.GRU(n_ts, hidden, batch_first=True)
        self.static = _StaticMLP(n_static, hidden)
        self.head = nn.Linear(hidden * 2, out_dim)
        self.out_dim = out_dim

    def forward(self, x):
        ts, static = x                                   # (B, T, n_ts), (B, n_static)
        _, h = self.gru(ts)                              # h: (1, B, hidden)
        feat = torch.cat([h.squeeze(0), self.static(static)], dim=1)
        return F.relu(self.head(feat))


class ConvTabularBackbone(nn.Module):
    """1D-CNN over the series + MLP over static."""

    def __init__(self, n_ts: int, n_static: int, channels: int = 64, out_dim: int = 128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_ts, channels, 3, padding=1), nn.ReLU(),
            nn.Conv1d(channels, channels, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.static = _StaticMLP(n_static, channels)
        self.head = nn.Linear(channels * 2, out_dim)
        self.out_dim = out_dim

    def forward(self, x):
        ts, static = x
        c = self.conv(ts.transpose(1, 2)).squeeze(-1)    # (B, T, n_ts) -> (B, channels)
        feat = torch.cat([c, self.static(static)], dim=1)
        return F.relu(self.head(feat))


class LSTMTabularBackbone(nn.Module):
    """LSTM over the series + MLP over static (a wider third architecture)."""

    def __init__(self, n_ts: int, n_static: int, hidden: int = 96, out_dim: int = 160):
        super().__init__()
        self.lstm = nn.LSTM(n_ts, hidden, batch_first=True)
        self.static = _StaticMLP(n_static, hidden)
        self.head = nn.Linear(hidden * 2, out_dim)
        self.out_dim = out_dim

    def forward(self, x):
        ts, static = x
        _, (h, _) = self.lstm(ts)                         # h: (1, B, hidden)
        feat = torch.cat([h.squeeze(0), self.static(static)], dim=1)
        return F.relu(self.head(feat))


def eicu_pool(n_ts: int, n_static: int) -> list[Callable[[], nn.Module]]:
    """A heterogeneous temporal backbone pool for ``(ts, static)`` inputs:
    GRU (128) · 1D-CNN (128) · LSTM (160). Factories -- one fresh model per client;
    the differing widths are what the client adapters align."""
    return [
        lambda: GRUTabularBackbone(n_ts, n_static, 64, 128),
        lambda: ConvTabularBackbone(n_ts, n_static, 64, 128),
        lambda: LSTMTabularBackbone(n_ts, n_static, 96, 160),
    ]
