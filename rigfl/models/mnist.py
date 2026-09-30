"""CNN backbones for 28x28 image datasets."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from rigfl.models.cifar import _image_channels


def _validate_image_size(input_spec) -> None:
    if input_spec and tuple(input_spec.get("shape", ())[1:]) != (28, 28):
        raise ValueError("MNIST models require 28x28 images")


class LeNet5(nn.Module):
    """LeNet-5-style backbone adapted for 28x28 images."""

    def __init__(self, input_spec=None):
        super().__init__()
        _validate_image_size(input_spec)
        in_channels = _image_channels(input_spec, default=1)
        self.conv1 = nn.Conv2d(in_channels, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.out_dim = 84

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.avg_pool2d(torch.tanh(self.conv1(x)), 2)
        x = F.avg_pool2d(torch.tanh(self.conv2(x)), 2)
        x = torch.tanh(self.fc1(torch.flatten(x, 1)))
        return torch.tanh(self.fc2(x))


class FedAvgMNISTCNN(nn.Module):
    """CNN backbone used for the MNIST experiments in the FedAvg paper."""

    def __init__(self, input_spec=None):
        super().__init__()
        _validate_image_size(input_spec)
        in_channels = _image_channels(input_spec, default=1)
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.fc = nn.Linear(64 * 7 * 7, 512)
        self.out_dim = 512

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        return F.relu(self.fc(torch.flatten(x, 1)))
