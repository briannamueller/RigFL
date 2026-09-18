"""Locations for completed runs and study artifacts."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

RUNS_DIRECTORY = "runs"


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
    """Read the tasks in a saved sweep grid."""
    path = Path(grid_path)
    try:
        return [json.loads(line) for line in path.read_text().splitlines() if line]
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
