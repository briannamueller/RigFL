"""Shared deterministic test setup using tiny offline inputs."""

from __future__ import annotations

import pytest
import torch


@pytest.fixture(autouse=True)
def _seed():
    """Deterministic RNG for every test."""
    torch.manual_seed(0)

