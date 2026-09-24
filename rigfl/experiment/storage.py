"""Locations for completed runs and study artifacts."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

from rigfl.experiment.paths import deep_merge

RUNS_DIRECTORY = "runs"
SUBMISSION_FILE = "submission.json"
SUBMISSION_KIND = "rigfl.sweep_submission"
SUBMISSION_SCHEMA_VERSION = 1
GRID_TASK_KIND = "rigfl.sweep_task"
GRID_TASK_SCHEMA_VERSION = 1


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
    """Read tasks, expanding a resolved submission's shared defaults."""
    path = Path(grid_path)
    try:
        tasks = [
            validate_grid_task(json.loads(line), source=f"{path} line {index}")
            for index, line in enumerate(path.read_text().splitlines(), 1)
            if line
        ]
        metadata_path = path.parent / SUBMISSION_FILE
        if not metadata_path.exists():
            return tasks
        metadata = json.loads(metadata_path.read_text())
        if metadata.get("kind") != SUBMISSION_KIND:
            raise ValueError(f"unsupported submission metadata in {metadata_path}")
        if metadata.get("schema_version") != SUBMISSION_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported submission schema in {metadata_path}: "
                f"{metadata.get('schema_version')!r}"
            )
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
