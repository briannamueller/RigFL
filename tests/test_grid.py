"""Sweep-grid expansion and saved sweep tasks."""

from __future__ import annotations

import json

import pytest

from rigfl.algorithms.local import LocalConfig
from rigfl.experiment import launch as launch_module
from rigfl.experiment.identity import fingerprint
from rigfl.experiment.launch import (
    _write_grid,
    expand,
    stage_task_snapshot,
)
from rigfl.experiment.registry import run_identity
from rigfl.experiment.storage import read_grid_tasks
from tests.helpers import resolved_experiment


def _counts(grid):
    out: dict[str, int] = {}
    for task in grid:
        out[task["algorithm"]] = out.get(task["algorithm"], 0) + 1
    return out


def test_sweep_execution_is_explicit(monkeypatch, tmp_path):
    grid = tmp_path / "sweep" / "grid.jsonl"
    declarations = []
    executions = []
    monkeypatch.setattr(
        launch_module,
        "declare_sweep",
        lambda config, *, results_root: declarations.append(
            (config, results_root)
        ) or grid,
    )
    monkeypatch.setattr(
        launch_module,
        "execute_sweep",
        lambda path: executions.append(path),
    )

    launch_module.main(["sweep.yaml", "--results-root", str(tmp_path)])
    assert executions == []

    launch_module.main([
        "sweep.yaml",
        "--results-root", str(tmp_path),
        "--execute",
    ])
    assert declarations == [
        ("sweep.yaml", str(tmp_path)),
        ("sweep.yaml", str(tmp_path)),
    ]
    assert executions == [grid]


def test_algorithm_axis_multiplies_only_applicable_algorithms():
    grid = expand({
        "sweep": {
            "algorithm": ["local", "fedproto", "fedprox"],
            "mu": [0.1, 0.2, 0.3],
        },
    })
    counts = _counts(grid)
    assert counts["fedprox"] == 3                     # one task per mu value
    assert counts["local"] == 1                       # no FedProx mu
    assert counts["fedproto"] == 1
    keys = {json.dumps(t, sort_keys=True) for t in grid}
    assert len(keys) == len(grid)
    fedprox = next(t for t in grid if t["algorithm"] == "fedprox")
    assert "mu" in fedprox["algorithm_config"] and "mu" not in fedprox["experiment"]

    # experiment axes multiply every algorithm
    grid = expand({
        "sweep": {
            "algorithm": ["local", "fedprox"],
            "training_seed": [0, 1],
            "batch": [16, 32],
        },
    })
    assert _counts(grid) == {"local": 4, "fedprox": 4}

    # axes for different algorithms don't form a cross product
    grid = expand({
        "base": {"model_arch": "fedavg_cnn"},
        "sweep": {
            "algorithm": ["fedprox", "fml"],
            "mu": [0.0, 0.1, 0.2],
            "beta": [0.2, 0.5, 0.8],
        },
    })
    assert _counts(grid) == {"fedprox": 3, "fml": 3}

    # fixed settings go only to the algorithms that use them
    grid = expand({
        "base": {"model_arch": "fedavg_cnn", "mu": 0.2, "beta": 0.7},
        "sweep": {"algorithm": ["fedprox", "fml"]},
    })
    assert [task["algorithm_config"] for task in grid] == [{"mu": 0.2}, {"beta": 0.7}]


def test_misspelt_algorithm_axis_is_refused():
    """A sweep axis unsupported by every selected algorithm is an error."""
    with pytest.raises(SystemExit, match="muu"):
        expand({
            "sweep": {"algorithm": ["fedprox"], "muu": [0.1, 0.2]},
        })


def test_submitted_grid_stays_fixed_when_working_grid_changes(
    tmp_path, monkeypatch
):
    path = tmp_path / "sweep" / "grid.jsonl"
    original = [
        {"algorithm": "local", "experiment": {"training_seed": 0}, "algorithm_config": {}}
    ]
    changed = [
        {"algorithm": "fedavg", "experiment": {"training_seed": 0}, "algorithm_config": {}},
        {"algorithm": "fedavg", "experiment": {"training_seed": 1}, "algorithm_config": {}},
    ]

    def resolve(task, **_kwargs):
        exp = resolved_experiment(training_seed=task["experiment"]["training_seed"])
        return task["algorithm"], exp, LocalConfig(), None

    monkeypatch.setattr(launch_module, "_resolve_task", resolve)
    monkeypatch.setattr(
        launch_module,
        "capture_env",
        lambda: {"packages": {"rigfl": {"source_commit": "one"}}},
    )

    assert _write_grid(path, original) is True
    submitted = stage_task_snapshot(path)
    assert _write_grid(path, changed) is True

    persisted = [json.loads(line) for line in path.read_text().splitlines()]
    assert [
        {key: value for key, value in task.items()
         if key not in {"kind", "schema_version"}}
        for task in persisted
    ] == changed
    submitted_tasks = read_grid_tasks(submitted)
    assert len(submitted_tasks) == 1
    assert submitted_tasks[0]["algorithm"] == "local"
    assert submitted_tasks[0]["experiment"]["training_seed"] == 0
    assert submitted_tasks[0]["run_fingerprint"] == fingerprint(run_identity(
        "local", resolved_experiment(training_seed=0), LocalConfig().model_dump()
    ))
