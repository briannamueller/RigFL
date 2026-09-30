"""Canonical scientific identity for completed runs.

The resolved configuration records everything used by an execution.  Run
identity is the smaller, canonical mapping whose hash names and de-duplicates
scientifically equivalent results.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any


def fingerprint(value: Mapping[str, Any]) -> str:
    """Return RigFL's short stable hash for a canonical identity mapping."""
    encoded = json.dumps(value, sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()[:8]


def hashable(value):
    """A dict value usable in a set or as part of a key."""
    if isinstance(value, dict):
        return tuple(sorted((key, hashable(item)) for key, item in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(hashable(item) for item in value)
    return value


def normalize_early_stopping(value: dict | None) -> dict:
    """Collapse disabled stopping policies to the one behavior they represent."""
    value = value or {}
    if not value.get("enabled"):
        return {"enabled": False}
    return {key: hashable(item) for key, item in value.items() if item is not None}


ENVIRONMENT_ONLY_EXPERIMENT_FIELDS = (
    "device",
    "results_root",
    "quiet",
    "wandb",
    "wandb_project",
    "dataset_config",
    "data_dir",
    "estimate_flops",
)
