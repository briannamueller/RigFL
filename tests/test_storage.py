"""Shared completed-run storage and saved sweep membership."""

from __future__ import annotations

import json

from rigfl.experiment.storage import records_for_grid, run_store, study_directory


def test_run_and_study_locations_are_separate(tmp_path):
    assert run_store(tmp_path) == tmp_path / "runs"
    assert run_store(tmp_path / "runs") == tmp_path / "runs"
    assert study_directory(tmp_path, "fedprox_grid") == tmp_path / "fedprox_grid"


def test_saved_grid_selects_only_its_completed_runs(tmp_path):
    task = {
        "algorithm": "fedprox",
        "experiment": {"dataset": "cifar10", "seed": 1},
        "algorithm_config": {"lr": 0.01, "mu": 0.1},
    }
    grid = tmp_path / "grid.jsonl"
    grid.write_text(json.dumps(task) + "\n")
    matching = {
        "algorithm": "fedprox",
        "config": {
            "experiment": {"dataset": "cifar10", "seed": 1, "rounds": 100},
            "algorithm": {"lr": 0.01, "mu": 0.1, "local_epochs": 1},
        },
    }
    other = {
        "algorithm": "fedprox",
        "config": {
            "experiment": {"dataset": "cifar10", "seed": 2, "rounds": 100},
            "algorithm": {"lr": 0.01, "mu": 0.1, "local_epochs": 1},
        },
    }

    assert records_for_grid([matching, other], grid) == [matching]
