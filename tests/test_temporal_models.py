"""Temporal backbones for paired time-series and static inputs."""

from __future__ import annotations

import pytest
import torch

from rigfl.models.registry import instantiate_backbones

INPUT_SPEC = {"fields": [{"shape": [24, 10]}, {"shape": [7]}]}


@pytest.mark.parametrize("name, dim", [
    ("temporal_gru", 128), ("temporal_cnn", 128), ("temporal_lstm", 160),
    ("temporal_tpc_light", 64), ("temporal_attention", 64),
])
def test_temporal_backbones_encode_sequence_and_static_inputs(name, dim):
    model = instantiate_backbones([name], input_spec=INPUT_SPEC)[0]()
    inputs = (torch.randn(3, 24, 10), torch.randn(3, 7))
    assert model(inputs).shape == (3, dim)
    model.eval()
    assert model((inputs[0][:1], inputs[1][:1])).shape == (1, dim)


def test_tpc_first_convolution_pairs_each_variable_with_its_mask():
    model = instantiate_backbones(["temporal_tpc_light"], input_spec=INPUT_SPEC)[0]().eval()
    captured = []
    model.temporal[0].register_forward_hook(lambda m, i, out: captured.append(out))
    sequence, static = torch.randn(2, 24, 10), torch.randn(2, 7)
    model((sequence, static))
    changed = sequence.clone()
    changed[..., 5] += 1.0  # mask of variable 0
    model((changed, static))
    difference = (captured[1] - captured[0]).abs().view(2, 5, -1, 24).sum(dim=(0, 2, 3))
    assert difference[0] > 0
    assert torch.all(difference[1:] == 0)
