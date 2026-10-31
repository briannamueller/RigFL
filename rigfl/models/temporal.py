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


class TemporalTPC(nn.Module):
    """Temporal Pointwise Convolution (Rocheteau et al., 2021), light configuration.

    The sequence holds each variable's values followed by their missingness
    masks, so its width is twice the number of variables. Each layer runs a
    causal convolution per variable alongside a pointwise layer that mixes all
    variables and the static features at every time step; both outputs are
    passed on to every later layer.
    """

    def __init__(self, input_spec: dict, layers: int = 3, kernel_size: int = 4,
                 temporal_kernels: int = 4, point_size: int = 8, out_dim: int = 64,
                 dropout: float = 0.45, temporal_dropout: float = 0.05):
        super().__init__()
        sequence_width, static_width = _input_widths(input_spec)
        self.variables = sequence_width // 2
        self.kernel_size = kernel_size
        self.temporal_kernels = temporal_kernels
        self.temporal = nn.ModuleList()
        self.temporal_norms = nn.ModuleList()
        self.pointwise = nn.ModuleList()
        self.point_norms = nn.ModuleList()
        streams, previous_streams = self.variables, 0
        for layer in range(layers):
            # the first layer sees each variable's value and mask; later layers
            # see each stream's skip input plus its temporal kernels
            per_stream = 2 if layer == 0 else 1 + temporal_kernels
            self.temporal.append(nn.Conv1d(
                streams * per_stream, streams * temporal_kernels, kernel_size,
                dilation=self._dilation(layer), groups=streams))
            self.temporal_norms.append(nn.BatchNorm1d(streams * temporal_kernels))
            point_in = sequence_width + static_width
            if layer > 0:
                point_in += previous_streams * temporal_kernels + point_size
            self.pointwise.append(nn.Linear(point_in, point_size))
            self.point_norms.append(nn.BatchNorm1d(point_size))
            previous_streams, streams = streams, streams + point_size
        self.head = nn.Linear(streams * (1 + temporal_kernels) + static_width, out_dim)
        self.head_norm = nn.BatchNorm1d(out_dim)
        self.dropout = nn.Dropout(dropout)
        self.temporal_dropout = nn.Dropout(temporal_dropout)
        self.out_dim = out_dim

    def _dilation(self, layer: int) -> int:
        return 1 if layer == 0 else layer * (self.kernel_size - 1)

    def forward(self, inputs) -> torch.Tensor:
        sequence, static = inputs
        batch, steps, width = sequence.shape
        values = sequence[..., :self.variables]
        masks = sequence[..., self.variables:]
        # interleave so each convolution group gets one variable's value and mask
        x = torch.stack([values, masks], dim=3).reshape(batch, steps, width).transpose(1, 2)
        original = sequence.reshape(batch * steps, width)
        static_steps = static.repeat_interleave(steps, dim=0)
        skip = values.transpose(1, 2)
        temporal_out = point_out = None
        for layer, conv in enumerate(self.temporal):
            padded = F.pad(x, ((self.kernel_size - 1) * self._dilation(layer), 0))
            conv_out = self.temporal_dropout(self.temporal_norms[layer](conv(padded)))
            parts = [p for p in (temporal_out, point_out) if p is not None]
            point_out = self.dropout(self.point_norms[layer](
                self.pointwise[layer](torch.cat([*parts, original, static_steps], dim=1))))
            streams, points = skip.shape[1], point_out.shape[1]
            point_steps = point_out.view(batch, steps, points).transpose(1, 2)
            per_stream = torch.cat([
                skip.unsqueeze(2),
                conv_out.view(batch, streams, self.temporal_kernels, steps),
            ], dim=2)
            pointwise_streams = point_steps.unsqueeze(2).expand(-1, -1, 1 + self.temporal_kernels, -1)
            x = F.relu(torch.cat([per_stream, pointwise_streams], dim=1)).reshape(batch, -1, steps)
            temporal_out = conv_out.transpose(1, 2).reshape(batch * steps, -1)
            skip = torch.cat([skip, point_steps], dim=1)
        combined = torch.cat([x.mean(dim=2), static], dim=1)
        return self.dropout(F.relu(self.head_norm(self.head(combined))))


class TemporalAttention(nn.Module):
    """Attention pooling over time steps combined with an MLP for static features.

    Each time step is projected on its own and the steps are averaged with
    learned weights, so the model has no recurrence or convolution.
    """

    def __init__(self, input_spec: dict, hidden: int = 64, out_dim: int = 64,
                 dropout: float = 0.3):
        super().__init__()
        sequence_width, static_width = _input_widths(input_spec)
        self.step = nn.Linear(sequence_width, hidden)
        self.score = nn.Linear(hidden, 1)
        self.static = _StaticEncoder(static_width, hidden)
        self.output = nn.Sequential(
            nn.Linear(hidden * 2, out_dim), nn.ReLU(), nn.Dropout(dropout))
        self.out_dim = out_dim

    def forward(self, inputs) -> torch.Tensor:
        sequence, static = inputs
        steps = F.relu(self.step(sequence))
        weights = torch.softmax(self.score(steps).squeeze(-1), dim=1)
        pooled = (weights.unsqueeze(-1) * steps).sum(dim=1)
        return self.output(torch.cat([pooled, self.static(static)], dim=1))
