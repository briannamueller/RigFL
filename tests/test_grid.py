"""Sweep-grid expansion: an algorithm-specific axis multiplies only the algorithms it applies to.

The property that matters for mixed sweeps: ``algorithm.graphroute.graph.k``
(a FedDES-only field)
must give FedDES one task per value and every other algorithm exactly one task -- no
duplicate configs that the fingerprint would later skip as "already done".
"""

from __future__ import annotations

import json

from rigfl.algorithms.local import LocalConfig
from rigfl.experiment import launch as launch_module
from rigfl.experiment.launch import _stage_sge_grid, _write_grid, build_grid
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
        "algorithms": ["local", "fedproto", "feddes"],
        "sweep": {"algorithm.graphroute.graph.k": [3, 5, 10]},
    })
    counts = _counts(grid)
    assert counts["feddes"] == 3                       # one task per k value
    assert counts["local"] == 1                        # no GraphRoute k
    assert counts["fedproto"] == 1


def test_no_duplicate_configs():
    grid = build_grid({
        "algorithms": ["local", "fedproto", "feddes"],
        "sweep": {"algorithm.graphroute.graph.k": [3, 5, 10]},
    })
    keys = {json.dumps(t, sort_keys=True) for t in grid}
    assert len(keys) == len(grid)                           # every task config is distinct


def test_relevant_experiment_axis_multiplies_all_algorithms():
    grid = build_grid({
        "algorithms": ["local", "feddes"],
        "sweep": {
            "experiment.seed": [0, 1],
            "experiment.batch": [16, 32],
        },
    })
    counts = _counts(grid)
    assert counts["local"] == 4 and counts["feddes"] == 4  # 2 seeds x 2 batches each


def test_shared_dimension_does_not_multiply_feddes_runs():
    grid = build_grid({
        "algorithms": ["local", "feddes"],
        "sweep": {"experiment.shared_dim": [64, 128]},
    })

    assert _counts(grid) == {"local": 2, "feddes": 1}
    assert "shared_dim" not in next(
        task["experiment"] for task in grid if task["algorithm"] == "feddes"
    )


def test_shared_dimension_axis_is_rejected_for_feddes_only_sweep():
    import pytest

    with pytest.raises(SystemExit, match="does not apply"):
        build_grid({
            "algorithms": ["feddes"],
            "sweep": {"experiment.shared_dim": [64, 128]},
        })


def test_round_controls_apply_only_to_iterative_methods_in_a_mixed_grid():
    grid = build_grid({
        "algorithms": ["local", "feddes"],
        "base": {
            "experiment": {
                "rounds": 300,
                "eval_gap": 5,
                "early_stopping": {
                    "enabled": True,
                    "metric": "accuracy",
                    "patience": 20,
                },
            },
        },
        "sweep": {"experiment.early_stopping.patience": [10, 20]},
    })

    assert _counts(grid) == {"local": 2, "feddes": 1}
    local = next(task for task in grid if task["algorithm"] == "local")
    assert local["experiment"]["rounds"] == 300
    assert local["experiment"]["eval_gap"] == 5
    assert local["experiment"]["early_stopping"]["enabled"] is True
    feddes = next(task for task in grid if task["algorithm"] == "feddes")
    assert not ({"rounds", "eval_gap", "early_stopping"} & feddes["experiment"].keys())


def test_algorithm_specific_axis_lands_in_algorithm_config():
    grid = build_grid({
        "algorithms": ["feddes"],
        "sweep": {"algorithm.graphroute.graph.k": [7]},
    })
    assert grid[0]["algorithm_config"]["graphroute"]["graph"]["k"] == 7
    assert "graphroute" not in grid[0]["experiment"]


def test_misspelt_algorithm_axis_is_refused():
    """A sweep axis unsupported by every selected algorithm is an error."""
    import pytest

    from rigfl.experiment.launch import build_grid
    with pytest.raises(SystemExit) as e:
        build_grid({
            "algorithms": ["feddes"],
            "sweep": {"algorithm.graphroute.graph.kk": [3, 5]},
        })
    msg = str(e.value)
    assert "algorithm.graphroute.graph.kk" in msg
    assert 'Did you mean "graphroute.graph.k"?' in msg


def test_misspelt_experiment_axis_is_refused():
    import pytest

    from rigfl.experiment.launch import build_grid
    with pytest.raises(SystemExit) as e:
        build_grid({
            "algorithms": ["feddes"],
            "sweep": {"experiment.btach": [16, 32]},
        })
    assert 'Did you mean "batch"?' in str(e.value)


def test_misspelt_fixed_algorithm_setting_is_refused():
    """Fixed algorithm settings receive the same validation as sweep axes."""
    import pytest

    from rigfl.experiment.launch import build_grid
    with pytest.raises(SystemExit, match="base.algorithm"):
        build_grid({
            "algorithms": ["feddes"],
            "base": {"algorithm": {"graphroute": {"graph": {"kk": 3}}}},
        })


def test_fixed_algorithm_settings_are_scoped_across_mixed_algorithms():
    grid = build_grid({
        "algorithms": ["fedprox", "feddes"],
        "base": {
            "experiment": {"model": "fedavg_cnn"},
            "algorithm": {"mu": 0.2, "graphroute": {"graph": {"k": 7}}},
        },
    })

    assert grid == [
        {
            "algorithm": "fedprox",
            "experiment": {"model": "fedavg_cnn"},
            "algorithm_config": {"mu": 0.2},
        },
        {
            "algorithm": "feddes",
            "experiment": {"model": "fedavg_cnn"},
            "algorithm_config": {"graphroute": {"graph": {"k": 7}}},
        },
    ]


def test_two_exclusive_algorithm_axes_do_not_form_a_cross_product():
    grid = build_grid({
        "algorithms": ["fedprox", "feddes"],
        "base": {"experiment": {"model": "fedavg_cnn"}},
        "sweep": {
            "algorithm.mu": [0.0, 0.1, 0.2],
            "algorithm.graphroute.graph.k": [3, 5, 7],
        },
    })

    assert _counts(grid) == {"fedprox": 3, "feddes": 3}
    assert [task["algorithm_config"] for task in grid if task["algorithm"] == "fedprox"] == [
        {"mu": 0.0}, {"mu": 0.1}, {"mu": 0.2}
    ]
    assert [task["algorithm_config"] for task in grid if task["algorithm"] == "feddes"] == [
        {"graphroute": {"graph": {"k": 3}}},
        {"graphroute": {"graph": {"k": 5}}},
        {"graphroute": {"graph": {"k": 7}}},
    ]


def test_fixed_setting_supported_by_no_selected_algorithm_is_an_error():
    import pytest

    with pytest.raises(SystemExit, match="No selected algorithm"):
        build_grid({
            "algorithms": ["feddes"],
            "base": {"algorithm": {"local_epochs": 2}},
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
    submitted = _stage_sge_grid(path, tmp_path / "results")
    assert _write_grid(path, changed) is True

    assert [json.loads(line) for line in path.read_text().splitlines()] == changed
    submitted_tasks = read_grid_tasks(submitted)
    assert len(submitted_tasks) == 1
    assert submitted_tasks[0]["algorithm"] == "local"
    assert submitted_tasks[0]["experiment"]["seed"] == 0
    assert submitted_tasks[0]["run_fingerprint"] == algorithm_run_fingerprint(
        "local", resolved_experiment(seed=0), LocalConfig().model_dump()
    )
