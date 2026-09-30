"""Locations for completed runs and study artifacts."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

RUNS_DIRECTORY = "runs"
SNAPSHOT_FILE = "snapshot.json"
SNAPSHOT_KIND = "rigfl.sweep_snapshot"
SNAPSHOT_SCHEMA_VERSION = 1
GRID_TASK_KIND = "rigfl.sweep_task"
GRID_TASK_SCHEMA_VERSION = 1


def run_store(results_root: str | Path) -> Path:
    """Return the shared completed-run directory below a results root."""
    return Path(results_root) / RUNS_DIRECTORY


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
    experiment = config.get("experiment", {})
    # a setting recorded as null doesn't apply to the algorithm
    expected = {
        key: value for key, value in task.get("experiment", {}).items()
        if not (key in experiment and experiment[key] is None)
    }
    return _contains(experiment, expected) and (
        _contains(config.get("algorithm", {}), task.get("algorithm_config", {}))
    )


def validate_grid_task(task: object, *, source: str) -> dict:
    """Validate one independently persisted JSONL task record."""
    if not isinstance(task, dict):
        raise ValueError(f"{source} is not an object")
    if task.get("kind") != GRID_TASK_KIND:
        raise ValueError(
            f"{source} has kind {task.get('kind')!r}, expected {GRID_TASK_KIND!r}"
        )
    if task.get("schema_version") != GRID_TASK_SCHEMA_VERSION:
        raise ValueError(
            f"{source} has schema_version {task.get('schema_version')!r}, "
            f"expected {GRID_TASK_SCHEMA_VERSION}"
        )
    if not isinstance(task.get("algorithm"), str) or not task["algorithm"]:
        raise ValueError(f"{source} has no algorithm")
    if not isinstance(task.get("experiment"), dict):
        raise ValueError(f"{source} has no experiment configuration")
    if not isinstance(task.get("algorithm_config"), dict):
        raise ValueError(f"{source} has no algorithm configuration")
    return task


def read_grid_tasks(grid_path: str | Path) -> list[dict]:
    """Read tasks, checking snapshot metadata when present."""
    path = Path(grid_path)
    try:
        tasks = [
            validate_grid_task(json.loads(line), source=f"{path} line {index}")
            for index, line in enumerate(path.read_text().splitlines(), 1)
            if line
        ]
        metadata_path = path.parent / SNAPSHOT_FILE
        if not metadata_path.exists():
            return tasks
        metadata = json.loads(metadata_path.read_text())
        if metadata.get("kind") != SNAPSHOT_KIND:
            raise ValueError(f"unsupported snapshot metadata in {metadata_path}")
        if metadata.get("schema_version") != SNAPSHOT_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported snapshot schema in {metadata_path}: "
                f"{metadata.get('schema_version')!r}"
            )
        return tasks
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
