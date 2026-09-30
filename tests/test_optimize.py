"""Adaptive-search configuration and manifest handling."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from rigfl.experiment.optimize import (
    _manifest,
    _sampler,
    _tasks,
    parse_optimization,
    run_study,
)
from tests.helpers import STARTER_CONFIGS, STARTER_DATASETS


def _spec():
    return {
        "name": "adaptive",
        "algorithm": "fedtgp",
        "base": {
            "dataset": "cifar10",
            "dataset_config": STARTER_DATASETS,
            "rounds": 1,
            "local_epochs": 1,
        },
        "replicates": [
            {
                "partition_seed": seed,
                "split_seed": seed,
                "training_seed": seed,
            }
            for seed in (0, 1, 2)
        ],
        "tuning": {
            "trials": 20,
            "sampler": {
                "class": "optuna.samplers.TPESampler",
                "options": {"seed": 9},
            },
            "metric": "accuracy",
            "search_space": {
                "lr": {
                    "type": "float",
                    "low": 0.0001,
                    "high": 0.1,
                    "log": True,
                },
                "server_epochs": {
                    "type": "int",
                    "low": 3,
                    "high": 15,
                    "step": 2,
                },
                "lamda": {
                    "type": "categorical",
                    "values": [0.1, 0.5],
                },
                "margin_cap": {
                    "type": "categorical",
                    "values": [64, 128],
                },
            },
        },
    }


def test_included_hpo_example_validates():
    import yaml

    path = STARTER_CONFIGS / "experiments" / "mnist_hpo.yaml"
    spec = parse_optimization(yaml.safe_load(path.read_text()))

    assert spec.algorithm == "fedprox"
    assert spec.trials == 4


def test_optuna_uses_paired_data_and_training_seed_conditions():
    raw = _spec()
    raw["replicates"] = [
        {"partition_seed": 1, "split_seed": 2, "training_seed": 3},
        {"partition_seed": 4, "split_seed": 5, "training_seed": 6},
    ]

    spec = parse_optimization(raw)
    tasks = _tasks(spec, {"algorithm.server_epochs": 7})

    assert [
        (
            task["experiment"]["partition_seed"],
            task["experiment"]["split_seed"],
            task["experiment"]["training_seed"],
        )
        for task in tasks
    ] == [(1, 2, 3), (4, 5, 6)]
    # each replicate runs the same candidate
    assert all(task["algorithm_config"]["server_epochs"] == 7 for task in tasks)

    raw["replicates"] = 3
    assert parse_optimization(raw).replicate_conditions == [
        {"partition_seed": seed, "split_seed": seed, "training_seed": seed}
        for seed in (0, 1, 2)
    ]


def test_optuna_cannot_optimize_the_split_seed():
    raw = _spec()
    raw["tuning"]["search_space"] = {
        "split_seed": {"type": "categorical", "values": [0, 1]}
    }

    with pytest.raises(SystemExit, match="cannot be optimized"):
        parse_optimization(raw)


def test_default_sampler_is_seeded():
    default = _spec()
    del default["tuning"]["sampler"]

    assert parse_optimization(default).sampler_options == {"seed": 0}


def test_manifest_deduplicates_repeated_optuna_suggestions():
    raw = _spec()
    raw["tuning"]["search_space"] = {
        "server_epochs": {"type": "int", "low": 3, "high": 9}
    }
    spec = parse_optimization(raw)
    complete = SimpleNamespace(name="COMPLETE")
    failed = SimpleNamespace(name="FAIL")
    study = SimpleNamespace(
        study_name="adaptive",
        trials=[
            SimpleNamespace(
                number=0, state=complete, params={"algorithm.server_epochs": 5},
                user_attrs={},
            ),
            SimpleNamespace(
                number=1, state=complete, params={"algorithm.server_epochs": 5},
                user_attrs={},
            ),
            SimpleNamespace(
                number=2, state=complete, params={"algorithm.server_epochs": 7},
                user_attrs={},
            ),
            SimpleNamespace(
                number=3, state=failed, params={"algorithm.server_epochs": 9},
                user_attrs={},
            ),
        ],
    )
    manifest = _manifest(spec, study)
    assert len(manifest["candidates"]) == 2
    assert manifest["candidates"][0]["optuna_trial_numbers"] == [0, 1]


def test_resuming_a_seeded_sampler_does_not_replay_its_initial_suggestions(
    tmp_path, monkeypatch
):
    pytest.importorskip("optuna")
    raw = _spec()
    raw["tuning"]["sampler"] = {
        "class": "optuna.samplers.TPESampler",
        "options": {"seed": 9, "n_startup_trials": 10},
    }
    raw["tuning"]["search_space"] = {
        "server_epochs": {
            "type": "int",
            "low": 3,
            "high": 1_000_003,
        }
    }
    spec = parse_optimization(raw)

    def fake_objective(_spec, _runs_dir, _force):
        return lambda trial: trial.suggest_int(
            "algorithm.server_epochs", 3, 1_000_003
        )

    monkeypatch.setattr("rigfl.experiment.optimize._objective", fake_objective)
    first, _ = run_study(
        spec, tmp_path, study_name=None, target_trials=3
    )
    resumed, _ = run_study(
        spec, tmp_path, study_name=None, target_trials=6
    )

    # the target is a total, not a number of new trials
    assert len(resumed.trials) == 6
    first_values = [
        trial.params["algorithm.server_epochs"] for trial in first.trials
    ]
    resumed_values = [
        trial.params["algorithm.server_epochs"] for trial in resumed.trials[3:]
    ]
    assert resumed_values != first_values


def test_study_refuses_to_mix_trials_from_a_changed_configuration(
    tmp_path, monkeypatch
):
    pytest.importorskip("optuna")
    raw = _spec()
    raw["tuning"]["search_space"] = {
        "server_epochs": {"type": "int", "low": 3, "high": 9}
    }
    original = parse_optimization(raw)

    def fake_objective(_spec, _runs_dir, _force):
        return lambda trial: trial.suggest_int("algorithm.server_epochs", 3, 9)

    monkeypatch.setattr("rigfl.experiment.optimize._objective", fake_objective)
    run_study(original, tmp_path, study_name=None, target_trials=1)

    raw["tuning"]["search_space"]["server_epochs"]["high"] = 11
    changed = parse_optimization(raw)
    with pytest.raises(SystemExit, match="different RigFL configuration"):
        run_study(changed, tmp_path, study_name=None, target_trials=2)


def test_grid_sampler_derives_its_grid_from_the_search_space():
    optuna = pytest.importorskip("optuna")
    raw = _spec()
    raw["tuning"].pop("trials")
    raw["tuning"]["sampler"] = {
        "class": "optuna.samplers.GridSampler",
        "options": {"seed": 7},
    }
    raw["tuning"]["search_space"] = {
        "server_epochs": {
            "type": "categorical",
            "values": [3, 5, 7],
        },
        "margin_cap": {
            "type": "categorical",
            "values": [64, 128],
        },
    }

    spec = parse_optimization(raw)
    sampler = _sampler(spec)

    assert spec.trials == 6
    assert isinstance(sampler, optuna.samplers.GridSampler)
