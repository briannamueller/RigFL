"""Byte encoding adapted from Flower Labs' FedPhishGuard application."""

from __future__ import annotations

from urllib.parse import unquote_to_bytes

import numpy as np
import torch

SOURCE_COMMIT = "07b1b6fe3b1d50a252000f1c839497a87f13b144"
SOURCE_PATH = "fed-phish-guard/phishguard/data.py"
SOURCE_URL = (
    "https://github.com/flwrlabs/flower-hub-benchmark/blob/"
    f"{SOURCE_COMMIT}/{SOURCE_PATH}"
)

MAX_LENGTH = 256
PADDING_INDEX = 0
UNKNOWN_INDEX = 1
VOCAB_SIZE = 258
REQUIRED_COLUMNS = ("url", "label", "client_id")


def _url_bytes(url: str) -> bytes:
    return unquote_to_bytes(str(url).lower())


def encode_urls(values) -> torch.Tensor:
    """Encode URLs as fixed-length byte-token sequences."""
    encoded = np.full(
        (len(values), MAX_LENGTH), PADDING_INDEX, dtype=np.int16
    )
    for row, value in enumerate(values):
        byte_values = np.frombuffer(
            _url_bytes(value)[:MAX_LENGTH], dtype=np.uint8
        )
        encoded[row, : len(byte_values)] = byte_values.astype(np.int16) + 2
    return torch.from_numpy(encoded)
