"""Small CNN backbones for CIFAR-10.

``small_pool()`` returns backbone **factories** (not instances) so every client
gets its own backbone -- sharing an instance across clients would share weights
and break the federated setup. The generic builder (``rigfl.data.builder``) does
the adapter + head wiring.
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


def _image_channels(input_spec, default: int = 3) -> int:
    if not input_spec:
        return default
    shape = input_spec.get("shape", ())
    if len(shape) != 3:
        raise ValueError(f"image models require a [channels, height, width] shape, got {shape}")
    return int(shape[0])


class SmallCNN(nn.Module):
    """Two conv-pool blocks followed by a compact feature projection."""

    def __init__(self, channels: tuple[int, int] = (16, 32), hidden: int = 128,
                 in_ch: int = 3, input_spec=None):
        super().__init__()
        in_ch = _image_channels(input_spec, in_ch)
        c1, c2 = channels
        self.features = nn.Sequential(
            nn.Conv2d(in_ch, c1, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 32 -> 16
            nn.Conv2d(c1, c2, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),      # 16 -> 8
            nn.AdaptiveAvgPool2d((8, 8)),
        )
        self.fc = nn.Linear(c2 * 8 * 8, hidden)
        self.out_dim = hidden

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.fc(self.features(x).flatten(1)))


def small_pool() -> list[Callable[[], nn.Module]]:
    """A small *heterogeneous* backbone pool -- factories, one fresh model per client."""
    return [
        lambda: SmallCNN((16, 32), 128),
        lambda: SmallCNN((32, 64), 256),
        lambda: SmallCNN((16, 32), 64),
    ]


# ── The real heterogeneous CIFAR-10 pool ─────────────────────────────────────
# Three genuinely different architectures, each adapted for 32x32 inputs without
# a torchvision dependency, and each reduced to a feature-extractor backbone
# (forward(x) -> [B, out_dim], classification head stripped):
#   FedAvgCNN     -- the CNN from FedAvg (McMahan et al., AISTATS 2017)
#   CifarResNet18 -- ResNet-18 (He et al., CVPR 2016)
#   CifarMobileNetV2 -- MobileNetV2 (Sandler et al., CVPR 2018)

class FedAvgCNN(nn.Module):
    """FedAvg CNN backbone. Emits 512-dimensional features."""

    def __init__(self, in_features: int = 3, dim: int = 1600, input_spec=None):
        super().__init__()
        in_features = _image_channels(input_spec, in_features)
        self.conv1 = nn.Sequential(nn.Conv2d(in_features, 32, 5), nn.ReLU(inplace=True), nn.MaxPool2d(2))
        self.conv2 = nn.Sequential(nn.Conv2d(32, 64, 5), nn.ReLU(inplace=True), nn.MaxPool2d(2))
        self.pool = nn.AdaptiveAvgPool2d((5, 5))
        self.fc1 = nn.Sequential(nn.Linear(dim, 512), nn.ReLU(inplace=True))
        self.out_dim = 512

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc1(torch.flatten(self.pool(self.conv2(self.conv1(x))), 1))


class _CifarBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride, bias=False), nn.BatchNorm2d(planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))


class CifarResNet18(nn.Module):
    """CIFAR-adapted ResNet-18 backbone (3x3 stride-1 stem, no maxpool). Emits 512-d."""

    def __init__(self, input_spec=None):
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(_image_channels(input_spec), 64, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, 2, 1)
        self.layer2 = self._make_layer(128, 2, 2)
        self.layer3 = self._make_layer(256, 2, 2)
        self.layer4 = self._make_layer(512, 2, 2)
        self.out_dim = 512

    def _make_layer(self, planes, num_blocks, stride):
        layers = []
        for s in [stride] + [1] * (num_blocks - 1):
            layers.append(_CifarBasicBlock(self.in_planes, planes, s))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer4(self.layer3(self.layer2(self.layer1(out))))
        return F.adaptive_avg_pool2d(out, 1).flatten(1)


class _MobileBlock(nn.Module):
    """Inverted residual: expand -> depthwise -> project."""

    def __init__(self, in_planes, out_planes, expansion, stride):
        super().__init__()
        self.stride = stride
        planes = expansion * in_planes
        self.conv1 = nn.Conv2d(in_planes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride, 1, groups=planes, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, out_planes, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)
        self.shortcut = nn.Sequential()
        if stride == 1 and in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, 1, bias=False), nn.BatchNorm2d(out_planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        return out + self.shortcut(x) if self.stride == 1 else out


class CifarMobileNetV2(nn.Module):
    """CIFAR-adapted MobileNetV2 backbone (stem/stride/pool tuned for 32x32). Emits 1280-d."""

    _cfg = [(1, 16, 1, 1), (6, 24, 2, 1), (6, 32, 3, 2), (6, 64, 4, 2),
            (6, 96, 3, 1), (6, 160, 3, 2), (6, 320, 1, 1)]

    def __init__(self, input_spec=None):
        super().__init__()
        self.conv1 = nn.Conv2d(_image_channels(input_spec), 32, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layers = self._make_layers(32)
        self.conv2 = nn.Conv2d(320, 1280, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(1280)
        self.out_dim = 1280

    def _make_layers(self, in_planes):
        layers = []
        for expansion, out_planes, num_blocks, stride in self._cfg:
            for s in [stride] + [1] * (num_blocks - 1):
                layers.append(_MobileBlock(in_planes, out_planes, expansion, s))
                in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(self.layers(out))))
        return F.adaptive_avg_pool2d(out, 1).flatten(1)


def cifar_pool() -> list[Callable[[], nn.Module]]:
    """The heterogeneous CIFAR-10 backbone pool: FedAvgCNN (512) ·
    CifarResNet18 (512) · CifarMobileNetV2 (1280). The differing feature widths
    are exactly what the model adapters exist to align."""
    return [FedAvgCNN, CifarResNet18, CifarMobileNetV2]
