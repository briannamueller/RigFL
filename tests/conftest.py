"""Shared test setup: make ``rigfl`` importable and seed torch for determinism.

The suite runs entirely offline on tiny synthetic tensors -- no CIFAR download,
no network, no GPU required.
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

# Make the repo root importable regardless of the directory pytest is invoked
# from (rigfl is used from cwd, not installed editable).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


@pytest.fixture(autouse=True)
def _seed():
    """Deterministic RNG for every test."""
    torch.manual_seed(0)
