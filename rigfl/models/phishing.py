"""Byte-level convolutional backbone for URL classification."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from rigfl.data.transforms.phishing import PADDING_INDEX, VOCAB_SIZE


class PhishingByteCNN(nn.Module):
    """Feature extractor adapted from Flower Labs' FedPhishGuard model."""

    def __init__(self, input_spec=None):
        super().__init__()
        shape = tuple(input_spec.get("shape", ())) if input_spec else ()
        if len(shape) != 1 or shape[0] < 4:
            raise ValueError(
                "phishing byte CNN requires a one-dimensional token sequence, "
                f"got {shape}"
            )
        embed_dim = 64
        num_filters = 128
        self.embedding = nn.Embedding(
            VOCAB_SIZE, embed_dim, padding_idx=PADDING_INDEX
        )
        self.parallel_convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(
                        embed_dim,
                        num_filters,
                        kernel_size,
                        padding=kernel_size // 2,
                    ),
                    nn.ReLU(),
                    nn.BatchNorm1d(num_filters),
                )
                for kernel_size in (3, 5, 7)
            ]
        )
        self.conv_blocks = nn.Sequential(
            nn.Conv1d(num_filters * 3, 256, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.MaxPool1d(2),
            nn.Conv1d(256, 128, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.MaxPool1d(2),
        )
        self.out_dim = 128

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(x)
        channels_first = embedded.transpose(1, 2)
        parallel = [
            convolution(channels_first) for convolution in self.parallel_convs
        ]
        features = torch.cat(parallel, dim=1)
        convolved = self.conv_blocks(features)
        return F.adaptive_max_pool1d(convolved, 1).squeeze(-1)
