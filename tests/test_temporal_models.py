"""Temporal backbones for paired time-series and static inputs."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from rigfl.models.registry import instantiate_backbones
from rigfl.models.temporal import _causal_conv

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
    # the first batch norm receives the first convolution output
    model.temporal_norms[0].register_forward_pre_hook(lambda m, args: captured.append(args[0]))
    sequence, static = torch.randn(2, 24, 10), torch.randn(2, 7)
    model((sequence, static))
    changed = sequence.clone()
    changed[..., 5] += 1.0  # mask of variable 0
    model((changed, static))
    difference = (captured[1] - captured[0]).abs().view(2, 5, -1, 24).sum(dim=(0, 2, 3))
    assert difference[0] > 0
    assert torch.all(difference[1:] == 0)


def test_tpc_convolutions_match_conv1d():
    model = instantiate_backbones(["temporal_tpc_light"], input_spec=INPUT_SPEC)[0]()
    for conv in model.temporal:
        padded = torch.randn(3, conv.in_channels, 24 + (conv.kernel_size[0] - 1) * conv.dilation[0])
        torch.testing.assert_close(_causal_conv(conv, padded), F.conv1d(
            padded, conv.weight, conv.bias, dilation=conv.dilation, groups=conv.groups))


def test_tpc_trains_on_a_one_sample_batch():
    model = instantiate_backbones(["temporal_tpc_light"], input_spec=INPUT_SPEC)[0]().train()
    model((torch.randn(1, 24, 10), torch.randn(1, 7))).sum().backward()
