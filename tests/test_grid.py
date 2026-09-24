"""Sweep-grid expansion: an algorithm-specific axis multiplies only the algorithms it applies to.

The property that matters for mixed sweeps: ``algorithm.mu``
(a FedProx-only field)
must give FedProx one task per value and every other algorithm exactly one task -- no
duplicate configs that the fingerprint would later skip as "already done".
"""

from __future__ import annotations

import json

from rigfl.algorithms.local import LocalConfig
from rigfl.experiment import launch as launch_module
from rigfl.experiment.launch import _write_grid, build_grid, stage_task_snapshot
from rigfl.experiment.registry import algorithm_run_fingerprint
from rigfl.experiment.storage import read_grid_tasks
from tests.helpers import resolved_experiment


def _counts(grid):
    out: dict[str, int] = {}
    for task in grid:
        out[task["algorithm"]] = out.get(task["algorithm"], 0) + 1
    return out


def test_algorithm_axis_multiplies_only_applicable_algorithms():
    grid = build_grid({
        "algorithms": ["local", "fedproto", "fedprox"],
        "sweep": {"algorithm.mu": [0.1, 0.2, 0.3]},
    })
    counts = _counts(grid)
    assert counts["fedprox"] == 3                     # one task per mu value
    assert counts["local"] == 1                       # no FedProx mu
    assert counts["fedproto"] == 1


def test_no_duplicate_configs():
    grid = build_grid({
        "algorithms": ["local", "fedproto", "fedprox"],
        "sweep": {"algorithm.mu": [0.1, 0.2, 0.3]},
    })
    keys = {json.dumps(t, sort_keys=True) for t in grid}
    assert len(keys) == len(grid)                           # every task config is distinct


def test_relevant_experiment_axis_multiplies_all_algorithms():
    grid = build_grid({
        "algorithms": ["local", "fedprox"],
        "sweep": {
            "experiment.seed": [0, 1],
            "experiment.batch": [16, 32],
        },
    })
    counts = _counts(grid)
    assert counts["local"] == 4 and counts["fedprox"] == 4


def test_algorithm_specific_axis_lands_in_algorithm_config():
    grid = build_grid({
        "algorithms": ["fedprox"],
        "sweep": {"algorithm.mu": [0.7]},
    })
    assert grid[0]["algorithm_config"]["mu"] == 0.7
    assert "mu" not in grid[0]["experiment"]


def test_misspelt_algorithm_axis_is_refused():
    """A sweep axis unsupported by every selected algorithm is an error."""
    import pytest

    from rigfl.experiment.launch import build_grid
    with pytest.raises(SystemExit) as e:
        build_grid({
            "algorithms": ["fedprox"],
            "sweep": {"algorithm.muu": [0.1, 0.2]},
        })
    msg = str(e.value)
    assert "algorithm.muu" in msg
    assert 'Did you mean "mu"?' in msg


def test_misspelt_experiment_axis_is_refused():
    import pytest

    from rigfl.experiment.launch import build_grid
    with pytest.raises(SystemExit) as e:
        build_grid({
            "algorithms": ["fedprox"],
            "sweep": {"experiment.btach": [16, 32]},
        })
    assert 'Did you mean "batch"?' in str(e.value)


def test_misspelt_fixed_algorithm_setting_is_refused():
    """Fixed algorithm settings receive the same validation as sweep axes."""
    import pytest

    from rigfl.experiment.launch import build_grid
    with pytest.raises(SystemExit, match="base.algorithm"):
        build_grid({
            "algorithms": ["fedprox"],
            "base": {"algorithm": {"muu": 0.2}},
        })


def test_fixed_algorithm_settings_are_scoped_across_mixed_algorithms():
    grid = build_grid({
        "algorithms": ["fedprox", "fml"],
        "base": {
            "experiment": {"model": "fedavg_cnn"},
            "algorithm": {"mu": 0.2, "beta": 0.7},
        },
    })

    assert grid == [
        {
            "algorithm": "fedprox",
            "experiment": {"model": "fedavg_cnn"},
            "algorithm_config": {"mu": 0.2},
        },
        {
            "algorithm": "fml",
            "experiment": {"model": "fedavg_cnn"},
            "algorithm_config": {"beta": 0.7},
        },
    ]


def test_two_exclusive_algorithm_axes_do_not_form_a_cross_product():
    grid = build_grid({
        "algorithms": ["fedprox", "fml"],
        "base": {"experiment": {"model": "fedavg_cnn"}},
        "sweep": {
            "algorithm.mu": [0.0, 0.1, 0.2],
            "algorithm.beta": [0.2, 0.5, 0.8],
        },
    })

    assert _counts(grid) == {"fedprox": 3, "fml": 3}
    assert [task["algorithm_config"] for task in grid if task["algorithm"] == "fedprox"] == [
        {"mu": 0.0}, {"mu": 0.1}, {"mu": 0.2}
    ]
    assert [task["algorithm_config"] for task in grid if task["algorithm"] == "fml"] == [
        {"beta": 0.2}, {"beta": 0.5}, {"beta": 0.8},
    ]


def test_fixed_setting_supported_by_no_selected_algorithm_is_an_error():
    import pytest

    with pytest.raises(SystemExit, match="No selected algorithm"):
        build_grid({
            "algorithms": ["fedprox"],
            "base": {"algorithm": {"beta": 0.5}},
        })


def test_submitted_grid_stays_fixed_when_working_grid_changes(
    tmp_path, monkeypatch
):
    path = tmp_path / "sweep" / "grid.jsonl"
    original = [
        {"algorithm": "local", "experiment": {"seed": 0}, "algorithm_config": {}}
    ]
    changed = [
        {"algorithm": "fedavg", "experiment": {"seed": 0}, "algorithm_config": {}},
        {"algorithm": "fedavg", "experiment": {"seed": 1}, "algorithm_config": {}},
    ]

    def resolve(task, **_kwargs):
        exp = resolved_experiment(seed=task["experiment"]["seed"])
        return task["algorithm"], exp, LocalConfig(), None

    monkeypatch.setattr(launch_module, "_resolve_task", resolve)
    monkeypatch.setattr(launch_module, "capture_env", lambda: {"git_commit": "one"})

    assert _write_grid(path, original) is True
    submitted = stage_task_snapshot(path, tmp_path / "results")
    assert _write_grid(path, changed) is True

    persisted = [json.loads(line) for line in path.read_text().splitlines()]
    assert [
        {key: value for key, value in task.items()
         if key not in {"kind", "schema_version"}}
        for task in persisted
    ] == changed
    assert all(task["kind"] == "rigfl.sweep_task" for task in persisted)
    assert all(task["schema_version"] == 1 for task in persisted)
    submitted_tasks = read_grid_tasks(submitted)
    assert len(submitted_tasks) == 1
    assert submitted_tasks[0]["algorithm"] == "local"
    assert submitted_tasks[0]["experiment"]["seed"] == 0
    assert submitted_tasks[0]["run_fingerprint"] == algorithm_run_fingerprint(
        "local", resolved_experiment(seed=0), LocalConfig().model_dump()
    )


def test_grid_reader_rejects_an_unversioned_prototype_grid(tmp_path):
    import pytest

    grid = tmp_path / "grid.jsonl"
    grid.write_text(json.dumps({
        "algorithm": "local",
        "experiment": {},
        "algorithm_config": {},
    }) + "\n")

    with pytest.raises(ValueError, match="kind"):
        read_grid_tasks(grid)


def test_grid_reader_rejects_an_unsupported_submission_schema(tmp_path):
    import pytest

    grid = tmp_path / "grid.jsonl"
    grid.write_text(json.dumps({
        "kind": "rigfl.sweep_task",
        "schema_version": 1,
        "algorithm": "local",
        "experiment": {},
        "algorithm_config": {},
    }) + "\n")
    (tmp_path / "submission.json").write_text(json.dumps({
        "kind": "rigfl.sweep_submission",
        "schema_version": 2,
    }))

    with pytest.raises(ValueError, match="submission schema"):
        read_grid_tasks(grid)
