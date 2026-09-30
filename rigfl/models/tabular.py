"""Backbones for fixed-width tabular inputs."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _input_width(input_spec) -> int:
    shape = tuple(input_spec.get("shape", ())) if input_spec else ()
    if len(shape) != 1 or shape[0] < 1:
        raise ValueError(f"tabular models require a one-dimensional input, got {shape}")
    return int(shape[0])


class TabularLinear(nn.Module):
    """Single-layer tabular feature extractor."""

    def __init__(self, input_spec=None):
        super().__init__()
        self.linear = nn.Linear(_input_width(input_spec), 64)
        self.out_dim = 64

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.linear(x))


class TabularMLP(nn.Module):
    """Two-layer tabular feature extractor."""

    def __init__(self, input_spec=None):
        super().__init__()
        width = _input_width(input_spec)
        self.layers = nn.Sequential(
            nn.Linear(width, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.out_dim = 64

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class TabularResidualMLP(nn.Module):
    """Residual tabular feature extractor."""

    def __init__(self, input_spec=None):
        super().__init__()
        self.input = nn.Linear(_input_width(input_spec), 128)
        self.block = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
        )
        self.output = nn.Linear(128, 64)
        self.out_dim = 64

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = F.relu(self.input(x))
        hidden = F.relu(hidden + self.block(hidden))
        return F.relu(self.output(hidden))
