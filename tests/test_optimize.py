"""Adaptive-search configuration and manifest handling."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from rigfl.experiment.launch import expand
from rigfl.experiment.optimize import (
    _manifest,
    _sampler,
    _suggest,
    _tasks,
    parse_optimization,
    run_study,
    write_study_selection,
)


def _spec():
    return {
        "name": "adaptive",
        "algorithms": ["fedtgp"],
        "base": {
            "experiment": {"dataset": "cifar10", "rounds": 1},
            "algorithm": {"local_epochs": 1},
        },
        "replicates": [
            {
                "partition_seed": seed,
                "split_seed": seed,
                "experiment_seed": seed,
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
                "algorithm.lr": {
                    "type": "float",
                    "low": 0.0001,
                    "high": 0.1,
                    "log": True,
                },
                "algorithm.server_epochs": {
                    "type": "int",
                    "low": 3,
                    "high": 15,
                    "step": 2,
                },
                "algorithm.lamda": {
                    "type": "categorical",
                    "values": [0.1, 0.5],
                },
                "algorithm.margin_cap": {
                    "type": "categorical",
                    "values": [64, 128],
                },
            },
        },
    }


def test_included_hpo_example_validates():
    import yaml

    path = Path(__file__).parents[1] / "configs/experiments/cifar10_hpo.yaml"
    spec = parse_optimization(yaml.safe_load(path.read_text()))

    assert spec.algorithm == "fedprox"
    assert spec.trials == 4


def _with_zipped_replicates(raw: dict) -> dict:
    return raw


def test_optimization_parses_a_joint_mixed_search_space():
    raw = _with_zipped_replicates(_spec())
    raw["tuning"]["search_space"]["algorithm.server_epochs"] = {
        "type": "int",
        "low": 3,
        "high": 15,
        "step": 2,
    }
    spec = parse_optimization(raw)
    assert spec.algorithm == "fedtgp"
    assert spec.replicates == [0, 1, 2]
    assert spec.trials == 20
    assert set(spec.search_space) == {
        "algorithm.lr",
        "algorithm.server_epochs",
        "algorithm.lamda",
        "algorithm.margin_cap",
    }
    tasks = _tasks(
        spec,
        {
            "algorithm.lr": 0.01,
            "algorithm.server_epochs": 7,
            "algorithm.lamda": 0.1,
            "algorithm.margin_cap": 64,
        },
    )
    assert [task["experiment"]["seed"] for task in tasks] == [0, 1, 2]
    assert all(
        task["algorithm_config"]["server_epochs"] == 7 for task in tasks
    )


def test_optuna_uses_paired_data_and_training_seed_conditions():
    raw = _spec()
    raw["replicates"] = [
        {"partition_seed": 1, "split_seed": 2, "experiment_seed": 3},
        {"partition_seed": 4, "split_seed": 5, "experiment_seed": 6},
    ]

    spec = parse_optimization(raw)
    tasks = _tasks(spec, {"algorithm.server_epochs": 7})

    assert spec.replicates == [3, 6]
    assert [
        (
            task["experiment"]["partition_seed"],
            task["experiment"]["split_seed"],
            task["experiment"]["seed"],
        )
        for task in tasks
    ] == [(1, 2, 3), (4, 5, 6)]


def test_optuna_parameters_are_not_cartesian_sweep_axes():
    raw = _spec()
    raw["tuning"]["search_space"]["algorithm.server_epochs"] = {
        "type": "int",
        "low": 3,
        "high": 15,
    }
    with pytest.raises(SystemExit, match="ordinary sweeps do not perform"):
        expand(raw)


@pytest.mark.parametrize("field", ["dataset", "partition_seed", "split_seed"])
def test_optuna_cannot_optimize_data_or_replicate_identity(field):
    raw = _with_zipped_replicates(_spec())
    raw["tuning"]["search_space"] = {
        f"experiment.{field}": {"type": "categorical", "values": [0, 1]}
    }

    with pytest.raises(SystemExit, match="cannot be optimized"):
        parse_optimization(raw)


@pytest.mark.parametrize(
    "change, message",
    [
        (("algorithms", ["local", "fedavg"]), "exactly one algorithm"),
        (("sweep", {"experiment.batch": [16, 32]}), "does not use sweep"),
    ],
)
def test_optimization_rejects_ambiguous_study_scopes(change, message):
    raw = _spec()
    raw["tuning"]["search_space"]["algorithm.server_epochs"] = {
        "type": "int",
        "low": 3,
        "high": 15,
    }
    raw[change[0]] = change[1]
    with pytest.raises(SystemExit, match=message):
        parse_optimization(raw)


def test_distribution_validation_rejects_unknown_paths_and_invalid_ranges():
    raw = _spec()
    raw["tuning"]["search_space"] = {
        "algorithm.not_a_field": {"type": "int", "low": 1, "high": 2}
    }
    with pytest.raises(SystemExit, match="unknown fedtgp search parameter"):
        parse_optimization(raw)

    raw = _spec()
    raw["tuning"]["search_space"] = {
        "algorithm.server_lr": {
            "type": "float",
            "low": 0.1,
            "high": 0.01,
        }
    }
    with pytest.raises(SystemExit, match="low < high"):
        parse_optimization(raw)


def test_suggest_uses_the_declared_distribution_type():
    calls = []

    class Trial:
        def suggest_categorical(self, name, values):
            calls.append(("categorical", name, values))
            return values[0]

        def suggest_int(self, name, low, high, **options):
            calls.append(("int", name, low, high, options))
            return low

        def suggest_float(self, name, low, high, **options):
            calls.append(("float", name, low, high, options))
            return low

    trial = Trial()
    assert _suggest(trial, "a", {"type": "categorical", "values": ["x", "y"]}) == "x"
    assert _suggest(trial, "b", {"type": "int", "low": 1, "high": 5, "step": 2}) == 1
    assert (
        _suggest(trial, "c", {"type": "float", "low": 0.001, "high": 0.1, "log": True})
        == 0.001
    )
    assert [call[0] for call in calls] == ["categorical", "int", "float"]


def test_default_sampler_options_do_not_leak_into_an_explicit_sampler():
    default = _spec()
    del default["tuning"]["sampler"]
    explicit = _spec()
    explicit["tuning"]["sampler"] = {
        "class": "optuna.samplers.RandomSampler"
    }

    assert parse_optimization(default).sampler_options == {"seed": 0}
    assert parse_optimization(explicit).sampler_options == {}


def test_manifest_deduplicates_repeated_optuna_suggestions():
    raw = _spec()
    raw["tuning"]["search_space"] = {
        "algorithm.server_epochs": {"type": "int", "low": 3, "high": 9}
    }
    spec = parse_optimization(raw)
    complete = SimpleNamespace(name="COMPLETE")
    failed = SimpleNamespace(name="FAIL")
    study = SimpleNamespace(
        study_name="adaptive",
        trials=[
            SimpleNamespace(
                number=0, state=complete, params={"algorithm.server_epochs": 5}
            ),
            SimpleNamespace(
                number=1, state=complete, params={"algorithm.server_epochs": 5}
            ),
            SimpleNamespace(
                number=2, state=complete, params={"algorithm.server_epochs": 7}
            ),
            SimpleNamespace(
                number=3,
                state=failed,
                params={"algorithm.server_epochs": 9},
                user_attrs={
                    "validation_scores": [0.5],
                    "source_results": ["seed0.json"],
                    "failure_reason": "RuntimeError: training failed",
                },
            ),
        ],
    )
    manifest = _manifest(spec, study)
    assert len(manifest["candidates"]) == 2
    assert manifest["candidates"][0]["optuna_trial_numbers"] == [0, 1]
    assert manifest["engine"] == "optuna"
    assert manifest["search_space"] == spec.search_space
    assert [entry["replicates"] for entry in manifest["evaluations"]] == [
        [0, 1, 2],
        [0, 1, 2],
    ]
    assert manifest["optimization"]["trial_counts"] == {"complete": 3, "fail": 1}
    failed_trial = manifest["optimization"]["trials"][3]
    assert failed_trial["validation_scores"] == [0.5]
    assert failed_trial["source_results"] == ["seed0.json"]
    assert failed_trial["failure_reason"] == "RuntimeError: training failed"


def test_study_resumes_to_a_target_total_without_adding_trials(tmp_path, monkeypatch):
    pytest.importorskip("optuna")
    raw = _with_zipped_replicates(_spec())
    raw["tuning"]["search_space"] = {
        "algorithm.server_epochs": {"type": "int", "low": 3, "high": 9}
    }
    spec = parse_optimization(raw)

    def fake_objective(_spec, _results_dir, _force):
        return lambda trial: trial.suggest_int("algorithm.server_epochs", 3, 9)

    monkeypatch.setattr("rigfl.experiment.optimize._objective", fake_objective)
    first, _ = run_study(spec, tmp_path, storage=None, study_name=None, target_trials=3)
    second, manifest = run_study(
        spec, tmp_path, storage=None, study_name=None, target_trials=3
    )
    assert len(first.trials) == 3
    assert len(second.trials) == 3
    assert manifest["optimization"]["target_trials"] == 3


@pytest.mark.parametrize(
    ("sampler_class", "sampler_options"),
    [
        ("optuna.samplers.RandomSampler", {"seed": 9}),
        ("optuna.samplers.TPESampler", {"seed": 9, "n_startup_trials": 10}),
    ],
)
def test_resuming_a_seeded_sampler_does_not_replay_its_initial_suggestions(
    tmp_path, monkeypatch, sampler_class, sampler_options
):
    pytest.importorskip("optuna")
    raw = _with_zipped_replicates(_spec())
    raw["tuning"]["sampler"] = {
        "class": sampler_class,
        "options": sampler_options,
    }
    raw["tuning"]["search_space"] = {
        "algorithm.server_epochs": {
            "type": "int",
            "low": 3,
            "high": 1_000_003,
        }
    }
    spec = parse_optimization(raw)

    def fake_objective(_spec, _results_dir, _force):
        return lambda trial: trial.suggest_int(
            "algorithm.server_epochs", 3, 1_000_003
        )

    monkeypatch.setattr("rigfl.experiment.optimize._objective", fake_objective)
    first, _ = run_study(
        spec, tmp_path, storage=None, study_name=None, target_trials=3
    )
    resumed, _ = run_study(
        spec, tmp_path, storage=None, study_name=None, target_trials=6
    )

    first_values = [
        trial.params["algorithm.server_epochs"] for trial in first.trials
    ]
    resumed_values = [
        trial.params["algorithm.server_epochs"] for trial in resumed.trials[3:]
    ]
    assert resumed_values != first_values


def test_selection_refuses_a_study_without_completed_trials(tmp_path):
    spec = parse_optimization(_with_zipped_replicates(_spec()))
    with pytest.raises(SystemExit, match="no trial completed successfully"):
        write_study_selection(spec, tmp_path, {"candidates": []})


def test_study_refuses_to_mix_trials_from_a_changed_configuration(
    tmp_path, monkeypatch
):
    pytest.importorskip("optuna")
    raw = _spec()
    raw["tuning"]["search_space"] = {
        "algorithm.server_epochs": {"type": "int", "low": 3, "high": 9}
    }
    original = parse_optimization(raw)

    def fake_objective(_spec, _results_dir, _force):
        return lambda trial: trial.suggest_int("algorithm.server_epochs", 3, 9)

    monkeypatch.setattr("rigfl.experiment.optimize._objective", fake_objective)
    run_study(original, tmp_path, storage=None, study_name=None, target_trials=1)

    raw["tuning"]["search_space"]["algorithm.server_epochs"]["high"] = 11
    changed = parse_optimization(raw)
    with pytest.raises(SystemExit, match="different RigFL configuration"):
        run_study(changed, tmp_path, storage=None, study_name=None, target_trials=2)


def test_sampler_options_reproduce_suggestions(tmp_path, monkeypatch):
    pytest.importorskip("optuna")
    raw = _spec()
    raw["tuning"]["sampler"] = {
        "class": "RandomSampler",
        "options": {"seed": 9},
    }
    raw["tuning"]["search_space"] = {
        "algorithm.server_epochs": {"type": "int", "low": 3, "high": 99}
    }
    spec = parse_optimization(raw)

    def fake_objective(_spec, _results_dir, _force):
        return lambda trial: trial.suggest_int("algorithm.server_epochs", 3, 99)

    monkeypatch.setattr("rigfl.experiment.optimize._objective", fake_objective)
    first, _ = run_study(
        spec, tmp_path / "first", storage=None, study_name=None, target_trials=8
    )
    second, _ = run_study(
        spec, tmp_path / "second", storage=None, study_name=None, target_trials=8
    )

    assert [trial.params for trial in first.trials] == [
        trial.params for trial in second.trials
    ]


def test_sampler_class_accepts_short_and_full_optuna_paths():
    optuna = pytest.importorskip("optuna")
    for class_path in ("RandomSampler", "optuna.samplers.RandomSampler"):
        raw = _spec()
        raw["tuning"]["sampler"] = {
            "class": class_path,
            "options": {"seed": 4},
        }
        sampler = _sampler(parse_optimization(raw))
        assert isinstance(sampler, optuna.samplers.RandomSampler)


def test_sampler_class_must_implement_the_optuna_sampler_interface():
    raw = _spec()
    raw["tuning"]["sampler"] = {"class": "builtins.dict"}
    with pytest.raises(SystemExit, match="must inherit"):
        _sampler(parse_optimization(raw))


def test_sampler_options_are_passed_to_the_constructor():
    raw = _spec()
    raw["tuning"]["sampler"] = {
        "class": "RandomSampler",
        "options": {"not_an_option": True},
    }
    with pytest.raises(SystemExit, match="cannot construct"):
        _sampler(parse_optimization(raw))


def test_grid_sampler_derives_its_grid_from_the_search_space():
    optuna = pytest.importorskip("optuna")
    raw = _with_zipped_replicates(_spec())
    raw["tuning"].pop("trials")
    raw["tuning"]["sampler"] = {
        "class": "optuna.samplers.GridSampler",
        "options": {"seed": 7},
    }
    raw["tuning"]["search_space"] = {
        "algorithm.server_epochs": {
            "type": "categorical",
            "values": [3, 5, 7],
        },
        "algorithm.margin_cap": {
            "type": "categorical",
            "values": [64, 128],
        },
    }

    spec = parse_optimization(raw)
    sampler = _sampler(spec)

    assert spec.trials == 6
    assert isinstance(sampler, optuna.samplers.GridSampler)
    study = optuna.create_study(sampler=sampler, direction="maximize")

    def objective(trial):
        k = trial.suggest_categorical(
            "algorithm.server_epochs", [3, 5, 7]
        )
        hidden = trial.suggest_categorical(
            "algorithm.margin_cap", [64, 128]
        )
        return float(k + hidden)

    study.optimize(objective, n_trials=spec.trials)
    assert {
        (
            trial.params["algorithm.server_epochs"],
            trial.params["algorithm.margin_cap"],
        )
        for trial in study.trials
    } == {
        (k, hidden) for k in (3, 5, 7) for hidden in (64, 128)
    }


def test_grid_sampler_rejects_a_second_grid_declaration():
    raw = _with_zipped_replicates(_spec())
    raw["tuning"].pop("trials")
    raw["tuning"]["sampler"] = {
        "class": "GridSampler",
        "options": {"search_space": {"x": [1, 2]}},
    }
    raw["tuning"]["search_space"] = {
        "algorithm.server_epochs": {
            "type": "categorical",
            "values": [3, 5],
        }
    }

    with pytest.raises(SystemExit, match="do not repeat the grid"):
        _sampler(parse_optimization(raw))


def test_grid_sampler_requires_finite_categorical_values():
    raw = _with_zipped_replicates(_spec())
    raw["tuning"].pop("trials")
    raw["tuning"]["sampler"] = {"class": "GridSampler"}

    with pytest.raises(SystemExit, match="requires categorical values"):
        parse_optimization(raw)


def test_replicate_counts_expand_to_matched_seeds():
    raw = _spec()
    raw["replicates"] = 3

    spec = parse_optimization(raw)

    assert spec.replicate_conditions == [
        {"partition_seed": seed, "split_seed": seed, "experiment_seed": seed}
        for seed in (0, 1, 2)
    ]
