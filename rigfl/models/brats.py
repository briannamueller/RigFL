"""Image backbones for BraTS slice classification.

Each model emits features and declares ``out_dim``; RigFL attaches the
task-specific classification head.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _image_channels(input_spec, default: int = 3) -> int:
    if not input_spec:
        return default
    shape = input_spec.get("shape", ())
    if len(shape) != 3:
        raise ValueError(
            f"image models require a [channels, height, width] shape, got {shape}"
        )
    return int(shape[0])


class _Residual(nn.Module):
    def __init__(self, fn: nn.Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fn(x) + x


class ConvMixer256D8(nn.Module):
    """ConvMixer-256/8 with the patch-size-4 stem used for BraTS."""

    def __init__(self, input_spec=None):
        super().__init__()
        dim = 256
        kernel_size = 5
        patch_size = 4
        self.stem = nn.Sequential(
            nn.Conv2d(
                _image_channels(input_spec),
                dim,
                kernel_size=patch_size,
                stride=patch_size,
            ),
            nn.GELU(),
            nn.BatchNorm2d(dim),
        )
        self.blocks = nn.Sequential(*[
            nn.Sequential(
                _Residual(nn.Sequential(
                    nn.Conv2d(
                        dim,
                        dim,
                        kernel_size,
                        groups=dim,
                        padding="same",
                    ),
                    nn.GELU(),
                    nn.BatchNorm2d(dim),
                )),
                nn.Conv2d(dim, dim, kernel_size=1),
                nn.GELU(),
                nn.BatchNorm2d(dim),
            )
            for _ in range(8)
        ])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.out_dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.blocks(self.stem(x))).flatten(1)


def _make_divisible(value, divisor, min_value=None):
    min_value = divisor if min_value is None else min_value
    new_value = max(min_value, int(value + divisor / 2) // divisor * divisor)
    if new_value < 0.9 * value:
        new_value += divisor
    return new_value


class _ConvBNReLU(nn.Sequential):
    def __init__(
        self,
        in_planes,
        out_planes,
        kernel_size=3,
        stride=1,
        groups=1,
        norm_layer=None,
    ):
        norm_layer = nn.BatchNorm2d if norm_layer is None else norm_layer
        padding = (kernel_size - 1) // 2
        super().__init__(
            nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size,
                stride,
                padding,
                groups=groups,
                bias=False,
            ),
            norm_layer(out_planes),
            nn.ReLU6(inplace=True),
        )


class _InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, norm_layer=None):
        super().__init__()
        if stride not in (1, 2):
            raise ValueError(f"stride must be 1 or 2, got {stride}")
        norm_layer = nn.BatchNorm2d if norm_layer is None else norm_layer
        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            layers.append(
                _ConvBNReLU(
                    inp,
                    hidden_dim,
                    kernel_size=1,
                    norm_layer=norm_layer,
                )
            )
        layers.extend([
            _ConvBNReLU(
                hidden_dim,
                hidden_dim,
                stride=stride,
                groups=hidden_dim,
                norm_layer=norm_layer,
            ),
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            norm_layer(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.conv(x)
        return x + result if self.use_res_connect else result


class MobileNetV2(nn.Module):
    """Standard-stride MobileNetV2 backbone."""

    def __init__(
        self,
        input_spec=None,
        width_mult: float = 1.0,
        round_nearest: int = 8,
        norm_layer=None,
    ):
        super().__init__()
        norm_layer = nn.BatchNorm2d if norm_layer is None else norm_layer
        input_channel = _make_divisible(32 * width_mult, round_nearest)
        self.out_dim = _make_divisible(1280 * max(1.0, width_mult), round_nearest)
        setting = [
            (1, 16, 1, 1),
            (6, 24, 2, 2),
            (6, 32, 3, 2),
            (6, 64, 4, 2),
            (6, 96, 3, 1),
            (6, 160, 3, 2),
            (6, 320, 1, 1),
        ]

        features = [
            _ConvBNReLU(
                _image_channels(input_spec),
                input_channel,
                stride=2,
                norm_layer=norm_layer,
            )
        ]
        for expand_ratio, channels, repeats, stride in setting:
            output_channel = _make_divisible(channels * width_mult, round_nearest)
            for index in range(repeats):
                features.append(
                    _InvertedResidual(
                        input_channel,
                        output_channel,
                        stride if index == 0 else 1,
                        expand_ratio,
                        norm_layer,
                    )
                )
                input_channel = output_channel
        features.append(
            _ConvBNReLU(
                input_channel,
                self.out_dim,
                kernel_size=1,
                norm_layer=norm_layer,
            )
        )
        self.features = nn.Sequential(*features)
        self.dropout = nn.Dropout(0.2)
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = F.adaptive_avg_pool2d(x, 1).reshape(x.shape[0], -1)
        return self.dropout(x)
