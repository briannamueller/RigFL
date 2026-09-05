"""Sweep-grid expansion: an algorithm-specific axis multiplies only the algorithms it applies to.

The property that matters for mixed sweeps: ``algorithm.graph_k`` (a FedDES-only field)
must give FedDES one task per value and every other algorithm exactly one task -- no
duplicate configs that the fingerprint would later skip as "already done".
"""

from __future__ import annotations

import json

from rigfl.experiment.launch import build_grid


def _counts(grid):
    out: dict[str, int] = {}
    for task in grid:
        out[task["algorithm"]] = out.get(task["algorithm"], 0) + 1
    return out


def test_algorithm_axis_multiplies_only_applicable_algorithms():
    grid = build_grid({
        "algorithms": ["local", "fedproto", "feddes"],
        "sweep": {"algorithm.graph_k": [3, 5, 10]},          # feddes-only field
    })
    counts = _counts(grid)
    assert counts["feddes"] == 3                            # one task per graph_k
    assert counts["local"] == 1                             # no graph_k -> not multiplied
    assert counts["fedproto"] == 1


def test_no_duplicate_configs():
    grid = build_grid({
        "algorithms": ["local", "fedproto", "feddes"],
        "sweep": {"algorithm.graph_k": [3, 5, 10]},
    })
    keys = {json.dumps(t, sort_keys=True) for t in grid}
    assert len(keys) == len(grid)                           # every task config is distinct


def test_experiment_axis_multiplies_all_algorithms():
    grid = build_grid({
        "algorithms": ["local", "feddes"],
        "sweep": {"seed": [0, 1], "batch": [16, 32]},    # experiment axes apply to all
    })
    counts = _counts(grid)
    assert counts["local"] == 4 and counts["feddes"] == 4  # 2 seeds x 2 batches each


def test_algorithm_specific_axis_lands_in_algorithm_config():
    grid = build_grid({"algorithms": ["feddes"], "sweep": {"algorithm.graph_k": [7]}})
    assert grid[0]["algorithm_config"]["graph_k"] == 7
    assert "graph_k" not in grid[0]["experiment"]


def test_misspelt_algorithm_axis_is_refused():
    """Auto-scoping collapses an algorithm axis for algorithms that lack the field,
    which is right when some algorithm has it and indistinguishable from a typo
    when none does. The axis used to vanish here, before the task runner could
    reject it, leaving one default task and a sweep that reported the setting it
    was asked for."""
    import pytest

    from rigfl.experiment.launch import build_grid
    with pytest.raises(SystemExit) as e:
        build_grid({"algorithms": ["feddes"], "sweep": {"algorithm.graf_k": [3, 5]}})
    msg = str(e.value)
    assert "algorithm.graf_k" in msg
    assert 'Did you mean "graph_k"?' in msg


def test_misspelt_experiment_axis_is_refused():
    import pytest

    from rigfl.experiment.launch import build_grid
    with pytest.raises(SystemExit) as e:
        build_grid({"algorithms": ["feddes"], "sweep": {"exp.btach": [16, 32]}})
    assert 'Did you mean "batch"?' in str(e.value)


def test_misspelt_fixed_algorithm_setting_is_refused():
    """base.algorithm entries are as easy to misspell as an axis, and were dropped
    just as quietly."""
    import pytest

    from rigfl.experiment.launch import build_grid
    with pytest.raises(SystemExit, match="base.algorithm"):
        build_grid({"algorithms": ["feddes"], "base": {"algorithm": {"graf_k": 3}}})


def test_valid_algorithm_axis_expands():
    from rigfl.experiment.launch import build_grid
    grid = build_grid({"algorithms": ["feddes"], "sweep": {"algorithm.graph_k": [3, 5]}})
    assert [t["algorithm_config"]["graph_k"] for t in grid] == [3, 5]


def test_algorithm_axis_does_not_multiply_algorithms_without_the_field():
    """graph_k belongs to FedDES, not Local. Sweeping it must give FedDES its
    variants and Local exactly one task -- not a copy per value."""
    from rigfl.experiment.launch import build_grid
    grid = build_grid({"algorithms": ["feddes", "local"],
                       "sweep": {"algorithm.graph_k": [3, 5, 10]}})
    counts = {m: sum(t["algorithm"] == m for t in grid) for m in ("feddes", "local")}
    assert counts == {"feddes": 3, "local": 1}


def test_axis_valid_for_only_one_selected_algorithm_is_kept():
    """An axis no *selected* algorithm has is an error; an axis some have is scoped."""
    from rigfl.experiment.launch import build_grid
    grid = build_grid({"algorithms": ["feddes", "local"],
                       "sweep": {"algorithm.graph_k": [3, 5]}})
    assert {t["algorithm"] for t in grid} == {"feddes", "local"}


def test_fixed_algorithm_settings_are_scoped_across_mixed_algorithms():
    grid = build_grid({
        "algorithms": ["fedprox", "feddes"],
        "base": {
            "experiment": {"model_architectures": ["fedavg_cnn"]},
            "algorithm": {"mu": 0.2, "graph_k": 7},
        },
    })

    assert grid == [
        {
            "algorithm": "fedprox",
            "experiment": {"model_architectures": ["fedavg_cnn"]},
            "algorithm_config": {"mu": 0.2},
        },
        {
            "algorithm": "feddes",
            "experiment": {"model_architectures": ["fedavg_cnn"]},
            "algorithm_config": {"graph_k": 7},
        },
    ]


def test_two_exclusive_algorithm_axes_do_not_form_a_cross_product():
    grid = build_grid({
        "algorithms": ["fedprox", "feddes"],
        "base": {"experiment": {"model_architectures": ["fedavg_cnn"]}},
        "sweep": {
            "algorithm.mu": [0.0, 0.1, 0.2],
            "algorithm.graph_k": [3, 5, 7],
        },
    })

    assert _counts(grid) == {"fedprox": 3, "feddes": 3}
    assert [task["algorithm_config"] for task in grid if task["algorithm"] == "fedprox"] == [
        {"mu": 0.0}, {"mu": 0.1}, {"mu": 0.2}
    ]
    assert [task["algorithm_config"] for task in grid if task["algorithm"] == "feddes"] == [
        {"graph_k": 3}, {"graph_k": 5}, {"graph_k": 7}
    ]


def test_fixed_setting_supported_by_no_selected_algorithm_is_an_error():
    import pytest

    with pytest.raises(SystemExit, match="No selected algorithm"):
        build_grid({
            "algorithms": ["feddes"],
            "base": {"algorithm": {"local_epochs": 2}},
        })
