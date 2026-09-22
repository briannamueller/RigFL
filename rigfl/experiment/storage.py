"""Locations for completed runs and study artifacts."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

from rigfl.experiment.identity import expand_historical_absences
from rigfl.experiment.paths import deep_merge

RUNS_DIRECTORY = "runs"
SUBMISSION_FILE = "submission.json"


def run_store(results_root: str | Path) -> Path:
    """Return the shared completed-run directory below a results root."""
    root = Path(results_root)
    return root if root.name == RUNS_DIRECTORY else root / RUNS_DIRECTORY


def study_directory(results_root: str | Path, name: str) -> Path:
    """Return the artifact directory for one named sweep or tuning study."""
    return Path(results_root) / name


def _normalized(value):
    if isinstance(value, Mapping):
        return {key: _normalized(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalized(item) for item in value]
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return value
    return value


def _contains(actual, expected) -> bool:
    if isinstance(expected, Mapping):
        return isinstance(actual, Mapping) and all(
            key in actual and _contains(actual[key], value)
            for key, value in expected.items()
        )
    return _normalized(actual) == _normalized(expected)


def record_matches_task(record: dict, task: dict) -> bool:
    """Whether a completed run belongs to one task in a saved sweep grid."""
    if record.get("algorithm") != task.get("algorithm"):
        return False
    config = record.get("config", {})
    return _contains(config.get("experiment", {}), task.get("experiment", {})) and (
        _contains(config.get("algorithm", {}), task.get("algorithm_config", {}))
    )


def read_grid_tasks(grid_path: str | Path) -> list[dict]:
    """Read tasks, expanding a resolved submission's shared defaults."""
    path = Path(grid_path)
    try:
        tasks = [json.loads(line) for line in path.read_text().splitlines() if line]
        metadata_path = path.parent / SUBMISSION_FILE
        if not metadata_path.exists():
            return tasks
        metadata = json.loads(metadata_path.read_text())
        if metadata.get("kind") != "rigfl.sweep_submission":
            raise ValueError(f"unsupported submission metadata in {metadata_path}")
        version = metadata.get("identity_schema_version")
        if not isinstance(version, int):
            raise ValueError(f"identity schema is missing from {metadata_path}")
        experiment_base = metadata.get("experiment_defaults")
        algorithm_bases = metadata.get("algorithm_defaults")
        if not isinstance(experiment_base, dict) or not isinstance(
            algorithm_bases, dict
        ):
            raise ValueError(f"shared defaults are invalid in {metadata_path}")
        expanded = []
        for task in tasks:
            name = task.get("algorithm")
            algorithm_base = algorithm_bases.get(name)
            if not isinstance(algorithm_base, dict):
                raise ValueError(
                    f"submission has no shared algorithm configuration for {name!r}"
                )
            experiment = deep_merge(experiment_base, task.get("experiment", {}))
            algorithm_config = deep_merge(
                algorithm_base, task.get("algorithm_config", {})
            )
            experiment, algorithm_config = expand_historical_absences(
                name,
                experiment,
                algorithm_config,
                identity_schema_version=version,
            )
            expanded.append(
                {
                    **task,
                    "experiment": experiment,
                    "algorithm_config": algorithm_config,
                }
            )
        return expanded
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read sweep grid {path}: {error}") from error


def records_for_grid(records: list[dict], grid_path: str | Path) -> list[dict]:
    """Keep completed runs represented by a grid.jsonl task list."""
    tasks = read_grid_tasks(grid_path)
    return [
        record
        for record in records
        if any(record_matches_task(record, task) for task in tasks)
    ]
