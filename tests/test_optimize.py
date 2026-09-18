"""Adaptive-search configuration and manifest handling."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from rigfl.experiment.collect import _resolve_selection
from rigfl.experiment.intensification import finalize_intensification
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
from rigfl.experiment.tuning import rank, write_ranking


def _spec():
    return {
        "name": "adaptive",
        "algorithms": ["feddes"],
        "base": {
            "experiment": {"dataset": "cifar10", "rounds": 1},
            "algorithm": {"graphroute": {"base": {"epochs": 1}}},
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
                "algorithm.graphroute.base.lr": {
                    "type": "float",
                    "low": 0.0001,
                    "high": 0.1,
                    "log": True,
                },
                "algorithm.graphroute.graph.k": {
                    "type": "int",
                    "low": 3,
                    "high": 15,
                    "step": 2,
                },
                "algorithm.graphroute.gnn.arch": {
                    "type": "categorical",
                    "values": ["gat", "graph_gps"],
                },
                "algorithm.graphroute.gnn.hidden_dim": {
                    "type": "categorical",
                    "values": [64, 128],
                },
            },
        },
    }


@pytest.mark.parametrize(
    "filename", ["cifar10_tune.yaml", "cifar10_optuna.yaml"]
)
def test_included_tuning_examples_validate(filename):
    import yaml

    path = Path(__file__).parents[1] / "experiments" / filename
    spec = parse_optimization(yaml.safe_load(path.read_text()))

    assert spec.algorithm == "fedprox"


def _with_zipped_replicates(raw: dict) -> dict:
    return raw


def _intensification_replicates() -> list[dict[str, int]]:
    return [
        {
            "partition_seed": seed,
            "split_seed": seed,
            "experiment_seed": seed,
        }
        for seed in (10, 11, 12, 13, 14)
    ]


def test_optimization_parses_a_joint_mixed_search_space():
    raw = _with_zipped_replicates(_spec())
    raw["tuning"]["search_space"]["algorithm.graphroute.graph.k"] = {
        "type": "int",
        "low": 3,
        "high": 15,
        "step": 2,
    }
    spec = parse_optimization(raw)
    assert spec.algorithm == "feddes"
    assert spec.replicates == [0, 1, 2]
    assert spec.trials == 20
    assert set(spec.search_space) == {
        "algorithm.graphroute.base.lr",
        "algorithm.graphroute.graph.k",
        "algorithm.graphroute.gnn.arch",
        "algorithm.graphroute.gnn.hidden_dim",
    }
    tasks = _tasks(
        spec,
        {
            "algorithm.graphroute.base.lr": 0.01,
            "algorithm.graphroute.graph.k": 7,
            "algorithm.graphroute.gnn.arch": "gat",
            "algorithm.graphroute.gnn.hidden_dim": 64,
        },
    )
    assert [task["experiment"]["seed"] for task in tasks] == [0, 1, 2]
    assert all(
        task["algorithm_config"]["graphroute"]["graph"]["k"] == 7 for task in tasks
    )


def test_optuna_uses_paired_data_and_training_seed_conditions():
    raw = _spec()
    raw["replicates"] = [
        {"partition_seed": 1, "split_seed": 2, "experiment_seed": 3},
        {"partition_seed": 4, "split_seed": 5, "experiment_seed": 6},
    ]

    spec = parse_optimization(raw)
    tasks = _tasks(spec, {"algorithm.graphroute.graph.k": 7})

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
    raw["tuning"]["search_space"]["algorithm.graphroute.graph.k"] = {
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
    raw["tuning"]["search_space"]["algorithm.graphroute.graph.k"] = {
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
    with pytest.raises(SystemExit, match="unknown feddes search parameter"):
        parse_optimization(raw)

    raw = _spec()
    raw["tuning"]["search_space"] = {
        "algorithm.graphroute.gnn.lr": {
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
        "algorithm.graphroute.graph.k": {"type": "int", "low": 3, "high": 9}
    }
    spec = parse_optimization(raw)
    complete = SimpleNamespace(name="COMPLETE")
    failed = SimpleNamespace(name="FAIL")
    study = SimpleNamespace(
        study_name="adaptive",
        trials=[
            SimpleNamespace(
                number=0, state=complete, params={"algorithm.graphroute.graph.k": 5}
            ),
            SimpleNamespace(
                number=1, state=complete, params={"algorithm.graphroute.graph.k": 5}
            ),
            SimpleNamespace(
                number=2, state=complete, params={"algorithm.graphroute.graph.k": 7}
            ),
            SimpleNamespace(
                number=3,
                state=failed,
                params={"algorithm.graphroute.graph.k": 9},
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
        "algorithm.graphroute.graph.k": {"type": "int", "low": 3, "high": 9}
    }
    spec = parse_optimization(raw)

    def fake_objective(_spec, _results_dir, _force):
        return lambda trial: trial.suggest_int("algorithm.graphroute.graph.k", 3, 9)

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
        "algorithm.graphroute.graph.k": {
            "type": "int",
            "low": 3,
            "high": 1_000_003,
        }
    }
    spec = parse_optimization(raw)

    def fake_objective(_spec, _results_dir, _force):
        return lambda trial: trial.suggest_int(
            "algorithm.graphroute.graph.k", 3, 1_000_003
        )

    monkeypatch.setattr("rigfl.experiment.optimize._objective", fake_objective)
    first, _ = run_study(
        spec, tmp_path, storage=None, study_name=None, target_trials=3
    )
    resumed, _ = run_study(
        spec, tmp_path, storage=None, study_name=None, target_trials=6
    )

    first_values = [
        trial.params["algorithm.graphroute.graph.k"] for trial in first.trials
    ]
    resumed_values = [
        trial.params["algorithm.graphroute.graph.k"] for trial in resumed.trials[3:]
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
        "algorithm.graphroute.graph.k": {"type": "int", "low": 3, "high": 9}
    }
    original = parse_optimization(raw)

    def fake_objective(_spec, _results_dir, _force):
        return lambda trial: trial.suggest_int("algorithm.graphroute.graph.k", 3, 9)

    monkeypatch.setattr("rigfl.experiment.optimize._objective", fake_objective)
    run_study(original, tmp_path, storage=None, study_name=None, target_trials=1)

    raw["tuning"]["search_space"]["algorithm.graphroute.graph.k"]["high"] = 11
    changed = parse_optimization(raw)
    with pytest.raises(SystemExit, match="different RigFL configuration"):
        run_study(changed, tmp_path, storage=None, study_name=None, target_trials=2)


def test_intensification_policy_can_change_without_repeating_the_search(
    tmp_path, monkeypatch
):
    pytest.importorskip("optuna")
    raw = _with_zipped_replicates(_spec())
    raw["tuning"]["search_space"] = {
        "algorithm.graphroute.graph.k": {"type": "int", "low": 3, "high": 9}
    }
    original = parse_optimization(raw)

    def fake_objective(_spec, _results_dir, _force):
        return lambda trial: trial.suggest_int("algorithm.graphroute.graph.k", 3, 9)

    monkeypatch.setattr("rigfl.experiment.optimize._objective", fake_objective)
    run_study(original, tmp_path, storage=None, study_name=None, target_trials=1)

    raw["tuning"]["intensification"] = {
        "top_k": 2,
        "replicates": _intensification_replicates()[:2],
        "practical_threshold": 0.01,
    }
    changed_intensification = parse_optimization(raw)
    study, _ = run_study(
        changed_intensification,
        tmp_path,
        storage=None,
        study_name=None,
        target_trials=1,
        export_only=True,
    )

    assert len(study.trials) == 1


def test_sampler_options_reproduce_suggestions(tmp_path, monkeypatch):
    pytest.importorskip("optuna")
    raw = _spec()
    raw["tuning"]["sampler"] = {
        "class": "RandomSampler",
        "options": {"seed": 9},
    }
    raw["tuning"]["search_space"] = {
        "algorithm.graphroute.graph.k": {"type": "int", "low": 3, "high": 99}
    }
    spec = parse_optimization(raw)

    def fake_objective(_spec, _results_dir, _force):
        return lambda trial: trial.suggest_int("algorithm.graphroute.graph.k", 3, 99)

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
        "algorithm.graphroute.graph.k": {
            "type": "categorical",
            "values": [3, 5, 7],
        },
        "algorithm.graphroute.gnn.hidden_dim": {
            "type": "categorical",
            "values": [64, 128],
        },
    }

    spec = parse_optimization(raw)
    sampler = _sampler(spec)

    assert spec.trials == 6
    assert isinstance(sampler, optuna.samplers.GridSampler)
    assert sampler._search_space == {
        "algorithm.graphroute.graph.k": [3, 5, 7],
        "algorithm.graphroute.gnn.hidden_dim": [64, 128],
    }


def test_grid_sampler_rejects_a_second_grid_declaration():
    raw = _with_zipped_replicates(_spec())
    raw["tuning"].pop("trials")
    raw["tuning"]["sampler"] = {
        "class": "GridSampler",
        "options": {"search_space": {"x": [1, 2]}},
    }
    raw["tuning"]["search_space"] = {
        "algorithm.graphroute.graph.k": {
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


def test_intensification_requires_new_seeds_and_a_practical_threshold():
    raw = _with_zipped_replicates(_spec())
    raw["tuning"]["intensification"] = {
        "top_k": 3,
        "replicates": [
            {"partition_seed": 0, "split_seed": 0, "experiment_seed": 10},
            {"partition_seed": 10, "split_seed": 10, "experiment_seed": 11},
        ],
        "practical_threshold": 0.01,
    }
    with pytest.raises(SystemExit, match="data-seed pairs"):
        parse_optimization(raw)

    raw["tuning"]["intensification"]["replicates"] = (
        _intensification_replicates()[:2]
    )
    del raw["tuning"]["intensification"]["practical_threshold"]
    with pytest.raises(SystemExit, match="practical_threshold"):
        parse_optimization(raw)


def _run_record(
    task: dict, *, value: float, test_value: float, communication: int = 100
) -> dict:
    """One completed run, as the launcher would have recorded it."""
    k = task["algorithm_config"]["graphroute"]["graph"]["k"]
    seed = task["experiment"]["seed"]
    clients = {
        str(client): {
            "validation": {"accuracy": [value]},
            "test": {"accuracy": [test_value]},
        }
        for client in range(2)
    }
    counts = {
        split: {str(client): [10] for client in range(2)}
        for split in ("validation", "test")
    }
    return {
        "algorithm": "feddes",
        "config": {
            "experiment": {
                "dataset": "cifar10",
                "partition_id": "partition-a",
                "partition_scheme": "dirichlet",
                "data_backend": "flower",
                "num_clients": 2,
                "num_classes": 10,
                "validation_fraction": 0.2,
                "input_kind": "image",
                "input_spec": {"shape": [3, 32, 32]},
                "resolved_models": ["cnn", "cnn"],
                "rounds": 1,
                "partition_seed": task["experiment"].get("partition_seed", 0),
                "split_seed": task["experiment"].get("split_seed", 0),
                "seed": seed,
            },
            "algorithm": task["algorithm_config"],
        },
        "result": {
            "selection_views_supported": ["global", "per-client"],
            "evaluation_history": {
                "evaluation_rounds": [0],
                "clients": clients,
                "client_sample_counts": counts,
            },
        },
        "resources": {
            "observed": {"communication_bytes": {"total": communication}},
            "attributed_training": {
                "flops": 1000,
                "wall_seconds": 1.0,
                "wall_seconds_comparable": True,
            },
            "measurement": {"timing": {"hardware": {"device": "cpu"}}},
        },
        "_source_file": f"k{k}_seed{seed}.json",
    }


def _write_run_store(root: Path, records: list[dict]) -> Path:
    """Put screening results where a study's run store would hold them."""
    runs = root / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    for record in records:
        (runs / Path(record["_source_file"]).name).write_text(json.dumps(record))
    return runs


def test_intensification_shortlists_compares_and_selects_on_validation(
    tmp_path, monkeypatch
):
    raw = _with_zipped_replicates(_spec())
    raw["tuning"]["search_space"] = {
        "algorithm.graphroute.graph.k": {
            "type": "categorical",
            "values": [3, 5, 7],
        }
    }
    raw["tuning"]["intensification"] = {
        "top_k": 3,
        "replicates": _intensification_replicates(),
        "practical_threshold": 0.01,
        "prefer": "communication",
    }
    spec = parse_optimization(raw)
    complete = SimpleNamespace(name="COMPLETE")
    study = SimpleNamespace(
        study_name="adaptive",
        trials=[
            SimpleNamespace(
                number=0,
                state=complete,
                params={"algorithm.graphroute.graph.k": 3},
                value=0.80,
                user_attrs={},
            ),
            SimpleNamespace(
                number=1,
                state=complete,
                params={"algorithm.graphroute.graph.k": 5},
                value=0.79,
                user_attrs={},
            ),
            SimpleNamespace(
                number=2,
                state=complete,
                params={"algorithm.graphroute.graph.k": 7},
                value=0.70,
                user_attrs={},
            ),
        ],
    )

    def fake_run(task, out_dir, **_):
        k = task["algorithm_config"]["graphroute"]["graph"]["k"]
        return _run_record(
            task,
            value={3: 0.80, 5: 0.80, 7: 0.70}[k],
            test_value=0.99 if k == 7 else 0.01,
            communication={3: 300, 5: 200, 7: 100}[k],
        )

    manifest = _manifest(spec, study)
    screening_records = [
        fake_run(task, tmp_path)
        for trial in study.trials
        for task in _tasks(spec, trial.params)
    ]
    _write_run_store(tmp_path, screening_records)
    screening = rank(
        screening_records,
        manifest,
        metric="accuracy",
        views=["global"],
    )
    study_dir = tmp_path / "study"
    write_ranking(screening, screening_records, manifest, study_dir)

    monkeypatch.setattr("rigfl.experiment.intensification.run_config", fake_run)
    artifact = finalize_intensification(
        study_dir / "ranking.json", run_missing=True
    )
    group = artifact["groups"][0]

    assert group["validation_leader"] == 0
    assert group["practically_equivalent_candidates"] == [0, 1]
    assert group["selected_candidate"] == 1
    for candidate in group["candidates"]:
        per_replicate = candidate["intensification_validation"]["per_replicate"]
        assert len(per_replicate) == 5
        assert {
            (
                item["partition_seed"],
                item["split_seed"],
                item["experiment_seed"],
            )
            for item in per_replicate
        } == {(seed, seed, seed) for seed in [10, 11, 12, 13, 14]}
    assert artifact["selection_protocol"]["intensification_replicates"] == (
        _intensification_replicates()
    )
    assert all(
        comparison["protocol"]["evaluation_split"] == "validation"
        for comparison in group["comparisons_to_validation_leader"]
    )
    assert (study_dir / "intensification" / "evaluation.json").exists()
    assert group["selected_result_files"] == [
        f"../../runs/k5_seed{seed}.json" for seed in [10, 11, 12, 13, 14]
    ]

    import yaml

    selected = yaml.safe_load(
        (study_dir / "selected.yaml").read_text()
    )
    assert selected["replicates"] == _intensification_replicates()
    assert selected["base"]["algorithm"]["graphroute"]["graph"]["k"] == 5
    expanded, manifest = expand(selected)
    assert len(expanded) == 5
    assert manifest is None


@pytest.mark.parametrize(
    "ranking, winning_k", [("intensification", 5), ("pooled", 3)]
)
def test_ranking_basis_decides_the_shortlist_winner(
    tmp_path, monkeypatch, ranking, winning_k
):
    raw = _with_zipped_replicates(_spec())
    raw["tuning"]["search_space"] = {
        "algorithm.graphroute.graph.k": {"type": "categorical", "values": [3, 5]}
    }
    raw["tuning"]["intensification"] = {
        "top_k": 2,
        "replicates": _intensification_replicates(),
        "practical_threshold": 0.01,
        "ranking": ranking,
    }
    spec = parse_optimization(raw)
    complete = SimpleNamespace(name="COMPLETE")
    study = SimpleNamespace(
        study_name="adaptive",
        trials=[
            SimpleNamespace(
                number=index,
                state=complete,
                params={"algorithm.graphroute.graph.k": k},
                value=value,
                user_attrs={},
            )
            for index, (k, value) in enumerate(((3, 0.90), (5, 0.80)))
        ],
    )

    # k=3 screens better and intensifies worse, so the two bases disagree.
    screened = {3: 0.90, 5: 0.80}
    intensified = {3: 0.70, 5: 0.75}

    def fake_run(task, out_dir, **_):
        k = task["algorithm_config"]["graphroute"]["graph"]["k"]
        seed = task["experiment"]["seed"]
        return _run_record(
            task,
            value=intensified[k] if seed >= 10 else screened[k],
            test_value=0.5,
        )

    manifest = _manifest(spec, study)
    screening_records = [
        fake_run(task, tmp_path)
        for trial in study.trials
        for task in _tasks(spec, trial.params)
    ]
    _write_run_store(tmp_path, screening_records)
    ranked = rank(screening_records, manifest, metric="accuracy", views=["global"])
    study_dir = tmp_path / "study"
    write_ranking(ranked, screening_records, manifest, study_dir)

    monkeypatch.setattr("rigfl.experiment.intensification.run_config", fake_run)
    artifact = finalize_intensification(study_dir / "ranking.json", run_missing=True)
    group = artifact["groups"][0]
    selected = next(
        candidate
        for candidate in group["candidates"]
        if candidate["candidate_id"] == group["selected_candidate"]
    )

    assert artifact["selection_protocol"]["ranking_basis"] == ranking
    assert selected["candidate_parameters"]["algorithm.graphroute.graph.k"] == winning_k
    for candidate in group["candidates"]:
        if ranking == "pooled":
            assert len(candidate["pooled_validation"]["per_replicate"]) == 8
        else:
            assert "pooled_validation" not in candidate


def test_collection_uses_and_enforces_the_adaptive_objective_protocol():
    manifest = {
        "optimization": {
            "metric": "balanced_accuracy",
            "selection_view": "per-client",
            "selection_aggregation": "weighted_mean",
            "round_tie_break": "latest",
        }
    }
    defaults = SimpleNamespace(
        selection_metric=None,
        selection_view=None,
        selection_aggregation=None,
        tie_break=None,
    )
    assert _resolve_selection(defaults, [], manifest) == (
        "balanced_accuracy",
        "per-client",
        "weighted_mean",
        "latest",
    )
    changed = SimpleNamespace(
        selection_metric="accuracy",
        selection_view=None,
        selection_aggregation=None,
        tie_break=None,
    )
    with pytest.raises(SystemExit, match="objective protocol"):
        _resolve_selection(changed, [], manifest)
