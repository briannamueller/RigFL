"""Canonical scientific identity for completed runs.

The resolved configuration records everything used by an execution.  Run
identity is the smaller, canonical mapping whose hash names and de-duplicates
scientifically equivalent results.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from copy import deepcopy
from typing import Any


def fingerprint(value: Mapping[str, Any]) -> str:
    """Return RigFL's short stable hash for a canonical identity mapping."""
    encoded = json.dumps(value, sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()[:8]


def _hashable(value):
    if isinstance(value, dict):
        return tuple(sorted((key, _hashable(item)) for key, item in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_hashable(item) for item in value)
    return value


def normalize_early_stopping(value: dict | None) -> dict:
    """Collapse disabled stopping policies to the one behavior they represent."""
    value = value or {}
    if not value.get("enabled"):
        return {"enabled": False}
    return {key: _hashable(item) for key, item in value.items() if item is not None}


_ENVIRONMENT_ONLY_EXPERIMENT_FIELDS = (
    "device",
    "out_dir",
    "quiet",
    "wandb",
    "wandb_project",
    "dataset_config",
    "data_dir",
)
_ENVIRONMENT_ONLY_ALGORITHM_FIELDS = ("cache_dir",)


def run_identity_input(
    algorithm: str | None,
    experiment: Mapping[str, Any],
    algorithm_config: Mapping[str, Any],
    *,
    ignored_experiment_fields: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Return the exact canonical mapping hashed for one run."""
    exp = deepcopy(dict(experiment))
    alg = deepcopy(dict(algorithm_config))

    for key in _ENVIRONMENT_ONLY_EXPERIMENT_FIELDS:
        exp.pop(key, None)
    for key in ignored_experiment_fields:
        exp.pop(key, None)
    for key in ("partition_seed", "split_seed"):
        if exp.get(key) is None:
            exp.pop(key, None)
    exp.pop("model", None)
    exp.pop("model_family", None)
    exp["early_stopping"] = normalize_early_stopping(exp.get("early_stopping"))
    if not exp.get("estimate_flops"):
        exp.pop("estimate_flops", None)

    for key in _ENVIRONMENT_ONLY_ALGORITHM_FIELDS:
        alg.pop(key, None)

    return {"experiment": exp, "algorithm": alg}
