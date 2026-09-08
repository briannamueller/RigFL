"""Backbones for paired time-series and static inputs."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _input_widths(input_spec: dict) -> tuple[int, int]:
    fields = input_spec.get("fields", [])
    shapes = [tuple(field.get("shape", ())) for field in fields]
    if len(shapes) != 2 or len(shapes[0]) != 2 or len(shapes[1]) != 1:
        raise ValueError(
            "temporal models require a [time, features] input followed by a "
            f"[static features] input, got {shapes}"
        )
    return int(shapes[0][1]), int(shapes[1][0])


class _StaticEncoder(nn.Module):
    def __init__(self, width: int, hidden: int):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(width, hidden), nn.ReLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class TemporalGRU(nn.Module):
    """GRU sequence encoder combined with an MLP for static features."""

    def __init__(self, input_spec: dict, hidden: int = 64, out_dim: int = 128):
        super().__init__()
        sequence_width, static_width = _input_widths(input_spec)
        self.sequence = nn.GRU(sequence_width, hidden, batch_first=True)
        self.static = _StaticEncoder(static_width, hidden)
        self.output = nn.Linear(hidden * 2, out_dim)
        self.out_dim = out_dim

    def forward(self, inputs) -> torch.Tensor:
        sequence, static = inputs
        _, hidden = self.sequence(sequence)
        combined = torch.cat([hidden.squeeze(0), self.static(static)], dim=1)
        return F.relu(self.output(combined))


class TemporalCNN(nn.Module):
    """One-dimensional CNN combined with an MLP for static features."""

    def __init__(self, input_spec: dict, channels: int = 64, out_dim: int = 128):
        super().__init__()
        sequence_width, static_width = _input_widths(input_spec)
        self.sequence = nn.Sequential(
            nn.Conv1d(sequence_width, channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(channels, channels, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.static = _StaticEncoder(static_width, channels)
        self.output = nn.Linear(channels * 2, out_dim)
        self.out_dim = out_dim

    def forward(self, inputs) -> torch.Tensor:
        sequence, static = inputs
        encoded = self.sequence(sequence.transpose(1, 2)).squeeze(-1)
        combined = torch.cat([encoded, self.static(static)], dim=1)
        return F.relu(self.output(combined))


class TemporalLSTM(nn.Module):
    """LSTM sequence encoder combined with an MLP for static features."""

    def __init__(self, input_spec: dict, hidden: int = 96, out_dim: int = 160):
        super().__init__()
        sequence_width, static_width = _input_widths(input_spec)
        self.sequence = nn.LSTM(sequence_width, hidden, batch_first=True)
        self.static = _StaticEncoder(static_width, hidden)
        self.output = nn.Linear(hidden * 2, out_dim)
        self.out_dim = out_dim

    def forward(self, inputs) -> torch.Tensor:
        sequence, static = inputs
        _, (hidden, _) = self.sequence(sequence)
        combined = torch.cat([hidden.squeeze(0), self.static(static)], dim=1)
        return F.relu(self.output(combined))
