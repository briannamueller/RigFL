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
from dataclasses import dataclass
from typing import Any, Literal

IDENTITY_SCHEMA_VERSION = 1
READABLE_IDENTITY_SCHEMA_VERSIONS = {1}


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


@dataclass(frozen=True)
class HistoricalAbsence:
    """Compatibility rule for a setting absent from older configurations."""

    introduced_in: int
    section: Literal["experiment", "algorithm"]
    path: str
    equivalent_value: Any
    algorithms: frozenset[str] | None = None
    included_in_pre_schema_fingerprints: bool = False

    def applies_to(self, algorithm: str | None) -> bool:
        return self.algorithms is None or algorithm in self.algorithms


# Keep this list limited to scientific settings whose pre-option behavior is
# unambiguous.  Operational settings belong in the ordinary exclusion lists.
HISTORICAL_ABSENCES = (
    HistoricalAbsence(
        introduced_in=1,
        section="algorithm",
        path="base_models_per_client",
        equivalent_value="all",
        algorithms=frozenset({"feddes"}),
        included_in_pre_schema_fingerprints=True,
    ),
)


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


_MISSING = object()


def _get(mapping: Mapping[str, Any], path: str):
    value: Any = mapping
    for part in path.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return _MISSING
        value = value[part]
    return value


def _remove(mapping: dict[str, Any], path: str) -> None:
    parts = path.split(".")
    parents: list[tuple[dict[str, Any], str]] = []
    current = mapping
    for part in parts[:-1]:
        child = current.get(part)
        if not isinstance(child, dict):
            return
        parents.append((current, part))
        current = child
    current.pop(parts[-1], None)
    for parent, key in reversed(parents):
        if parent.get(key) == {}:
            parent.pop(key)


def _set_missing(mapping: dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    current = mapping
    for part in parts[:-1]:
        child = current.setdefault(part, {})
        if not isinstance(child, dict):
            return
        current = child
    current.setdefault(parts[-1], deepcopy(value))


def expand_historical_absences(
    algorithm: str,
    experiment: Mapping[str, Any],
    algorithm_config: Mapping[str, Any],
    *,
    identity_schema_version: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Expand settings absent from a configuration created by an older schema."""
    resolved_experiment = deepcopy(dict(experiment))
    resolved_algorithm = deepcopy(dict(algorithm_config))
    sections = {
        "experiment": resolved_experiment,
        "algorithm": resolved_algorithm,
    }
    for rule in HISTORICAL_ABSENCES:
        if (
            identity_schema_version < rule.introduced_in
            and rule.applies_to(algorithm)
        ):
            _set_missing(sections[rule.section], rule.path, rule.equivalent_value)
    return resolved_experiment, resolved_algorithm


def run_identity_input(
    algorithm: str | None,
    experiment: Mapping[str, Any],
    algorithm_config: Mapping[str, Any],
    *,
    ignored_experiment_fields: tuple[str, ...] = (),
    apply_historical_equivalence: bool = True,
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

    if apply_historical_equivalence:
        sections = {"experiment": exp, "algorithm": alg}
        for rule in HISTORICAL_ABSENCES:
            if not rule.applies_to(algorithm):
                continue
            section = sections[rule.section]
            if _get(section, rule.path) == rule.equivalent_value:
                _remove(section, rule.path)

    return {"experiment": exp, "algorithm": alg}


def pre_schema_identity_input(
    algorithm: str,
    experiment: Mapping[str, Any],
    algorithm_config: Mapping[str, Any],
    *,
    ignored_experiment_fields: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Return the supported fingerprint format from before identity schema 1."""
    identity = run_identity_input(
        algorithm,
        experiment,
        algorithm_config,
        ignored_experiment_fields=ignored_experiment_fields,
    )
    raw_sections = {"experiment": experiment, "algorithm": algorithm_config}
    identity_sections = {
        "experiment": identity["experiment"],
        "algorithm": identity["algorithm"],
    }
    for rule in HISTORICAL_ABSENCES:
        if not (
            rule.included_in_pre_schema_fingerprints
            and rule.applies_to(algorithm)
        ):
            continue
        value = _get(raw_sections[rule.section], rule.path)
        if value is not _MISSING:
            _set_missing(identity_sections[rule.section], rule.path, value)
    return identity
