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


class TabularMLP64(nn.Module):
    """One-hidden-layer MLP with input and hidden-feature normalization."""

    def __init__(self, input_spec=None):
        super().__init__()
        width = _input_width(input_spec)
        self.layers = nn.Sequential(
            nn.BatchNorm1d(width),
            nn.Linear(width, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )
        self.out_dim = 64

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class TabularGLU64(nn.Module):
    """One gated hidden layer: products of paired projections model gene interactions."""

    def __init__(self, input_spec=None):
        super().__init__()
        width = _input_width(input_spec)
        self.normalization = nn.BatchNorm1d(width)
        self.gated = nn.Linear(width, 64)
        self.output = nn.Linear(32, 64)
        self.out_dim = 64

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value, gate = self.gated(self.normalization(x)).chunk(2, dim=1)
        return F.relu(self.output(value * torch.sigmoid(gate)))


class TabularLogisticRegression(nn.Module):
    """Batch-normalized inputs for a directly attached linear head."""

    def __init__(self, input_spec=None):
        super().__init__()
        width = _input_width(input_spec)
        self.normalization = nn.BatchNorm1d(width)
        self.out_dim = width

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.normalization(x)


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


class TabularFlowerMLP(nn.Module):
    """MLP from Flower's PaySim fraud app (``@flwrlabs/fed-fin-fraud``) without
    its output layer."""

    def __init__(self, input_spec=None, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(_input_width(input_spec), 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.out_dim = 64

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class _CrossLayer(nn.Module):
    """DCN-v2 cross layer: ``x0 * (W x + b) + x``."""

    def __init__(self, width: int):
        super().__init__()
        self.linear = nn.Linear(width, width)

    def forward(self, x0: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return x0 * self.linear(x) + x


class TabularDCNv2(nn.Module):
    """DCN-v2 (Wang et al., 2021), stacked: two cross layers, then
    ``TabularFlowerMLP`` as the deep network."""

    def __init__(self, input_spec=None):
        super().__init__()
        width = _input_width(input_spec)
        self.cross = nn.ModuleList(_CrossLayer(width) for _ in range(2))
        self.deep = TabularFlowerMLP(input_spec)
        self.out_dim = 64

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = x
        for layer in self.cross:
            hidden = layer(x, hidden)
        return self.deep(hidden)


class TabularTabNet(nn.Module):
    """TabNet (Arik & Pfister, 2021) at the pytorch-tabnet defaults. Its final
    mapping projects the 8-dim decision output to 64. The mask sparsity loss
    is not added to training. Requires the ``tabnet`` extra."""

    def __init__(self, input_spec=None):
        super().__init__()
        from pytorch_tabnet.tab_network import TabNetNoEmbeddings

        self.tabnet = TabNetNoEmbeddings(_input_width(input_spec), 64)
        # pytorch-tabnet keeps this as a plain tensor, which .to(device) skips
        encoder = self.tabnet.encoder
        group_matrix = encoder.group_attention_matrix
        del encoder.group_attention_matrix
        encoder.register_buffer("group_attention_matrix", group_matrix, persistent=False)
        self.out_dim = 64

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.tabnet(x)[0]
