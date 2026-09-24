"""Validation-based ranking for candidates produced by an Optuna study."""

from __future__ import annotations

import itertools
import json
from types import SimpleNamespace

import pytest

from rigfl.experiment.launch import expand
from rigfl.experiment.optimize import _manifest, _tasks, parse_optimization
from rigfl.experiment.tuning import (
    TuningError,
    load_manifest,
    place_records,
    rank,
    write_manifest,
    write_ranking,
)


def _raw() -> dict:
    tuning = {
        "sampler": {"class": "GridSampler", "options": {"seed": 4}},
        "metric": "accuracy",
        "search_space": {
            "algorithm.lr": {
                "type": "categorical",
                "values": [0.01, 0.1],
            },
            "algorithm.mu": {
                "type": "categorical",
                "values": [0.0, 1.0],
            },
        },
    }
    return {
        "name": "fedprox_grid",
        "algorithms": ["fedprox"],
        "base": {
            "experiment": {
                "dataset": "cifar10",
                "model": "fedavg_cnn",
                "rounds": 1,
                "eval_gap": 1,
            },
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
        "tuning": tuning,
    }


def _study(raw: dict | None = None):
    spec = parse_optimization(raw or _raw())
    values = [
        distribution["values"] for distribution in spec.search_space.values()
    ]
    complete = SimpleNamespace(name="COMPLETE")
    trials = []
    for number, combination in enumerate(itertools.product(*values)):
        parameters = dict(zip(spec.search_space, combination))
        sources = [f"candidate{number}_seed{seed}.json" for seed in spec.replicates]
        trials.append(
            SimpleNamespace(
                number=number,
                state=complete,
                params=parameters,
                value=0.0,
                user_attrs={"source_results": sources},
            )
        )
    study = SimpleNamespace(study_name=spec.name, trials=trials)
    manifest = _manifest(spec, study)
    manifest["_path"] = "/tmp/study.json"
    return spec, manifest


def _history(
    validation: list[list[float]],
    test: list[list[float]],
    *,
    metric: str = "accuracy",
    counts: list[int] | None = None,
) -> dict:
    clients = {
        str(client): {
            "validation": {metric: list(validation[client])},
            "test": {metric: list(test[client])},
        }
        for client in range(len(validation))
    }
    sample_counts = counts or [10] * len(validation)
    return {
        "schema_version": 3,
        "selection_views_supported": ["global", "per-client"],
        "evaluation_history": {
            "evaluation_rounds": list(range(len(validation[0]))),
            "clients": clients,
            "client_sample_counts": {
                split: {
                    str(client): [sample_counts[client]] * len(validation[client])
                    for client in range(len(validation))
                }
                for split in ("validation", "test")
            },
        },
        "early_stopping": {
            "enabled": False,
            "termination_reason": "completed_all_rounds",
            "stopped_at_round": len(validation[0]) - 1,
            "metric": None,
            "split": None,
            "direction": None,
            "aggregation": None,
            "patience": None,
            "min_delta": None,
            "best_round": None,
            "best_value": None,
        },
    }


def _flat(validation: float, test: float, *, metric: str = "accuracy") -> dict:
    return _history(
        [[validation], [validation]],
        [[test], [test]],
        metric=metric,
    )


def _records(spec, manifest, score, *, omit=()) -> list[dict]:
    records = []
    for candidate in manifest["candidates"]:
        for replicate_index, task in enumerate(_tasks(spec, candidate["parameters"])):
            if (candidate["id"], replicate_index) in omit:
                continue
            records.append(
                {
                    "algorithm": task["algorithm"],
                    "config": {
                        "experiment": task["experiment"],
                        "algorithm": task["algorithm_config"],
                    },
                    "result": score(candidate["id"], replicate_index),
                    "_source_file": (
                        f"candidate{candidate['id']}_seed"
                        f"{spec.replicates[replicate_index]}.json"
                    ),
                }
            )
    return records


def _rank(records, manifest, **options):
    return rank(
        records,
        manifest,
        metric=options.pop("metric", "accuracy"),
        views=options.pop("views", ["global"]),
        **options,
    )


def _selected(artifact: dict, view: str = "global") -> int:
    return artifact["groups"][0]["rankings"][view]["selected_candidate"]


def test_ordinary_sweeps_execute_without_creating_a_tuning_study():
    grid, study = expand(
        {
            "algorithms": ["local", "fedavg"],
            "sweep": {
                "experiment.dataset": ["cifar10", "mnist"],
                "experiment.seed": [0, 1],
                "algorithm.local_epochs": [1, 2],
            },
        }
    )

    assert study is None
    assert len(grid) == 16


def test_ordinary_sweeps_reject_a_tuning_section():
    with pytest.raises(SystemExit, match="ordinary sweeps do not perform"):
        expand(_raw())


def test_grid_search_candidates_are_complete_joint_assignments():
    spec, manifest = _study()

    assert spec.trials == 4
    assert len(manifest["candidates"]) == 4
    assert [candidate["id"] for candidate in manifest["candidates"]] == list(
        range(4)
    )
    assert all(
        set(candidate["parameters"]) == {"algorithm.lr", "algorithm.mu"}
        for candidate in manifest["candidates"]
    )


def test_each_candidate_is_evaluated_on_every_replicate():
    spec, manifest = _study()
    records = _records(spec, manifest, lambda candidate, _: _flat(candidate, 0.0))

    artifact = _rank(records, manifest)

    assert len(artifact["groups"]) == 1
    assert all(
        candidate["views"]["global"]["validation"]["n"] == 3
        for candidate in artifact["groups"][0]["candidates"]
    )


def test_selection_uses_the_best_complete_configuration():
    spec, manifest = _study()
    scores = {0: 0.50, 1: 0.95, 2: 0.80, 3: 0.75}
    records = _records(
        spec, manifest, lambda candidate, _: _flat(scores[candidate], 0.0)
    )

    artifact = _rank(records, manifest)

    assert _selected(artifact) == 1
    assert manifest["candidates"][1]["parameters"] == {
        "algorithm.lr": 0.01,
        "algorithm.mu": 1.0,
    }


def test_test_scores_cannot_change_the_ranking():
    spec, manifest = _study()
    validation = {0: 0.9, 1: 0.8, 2: 0.7, 3: 0.6}
    records = _records(
        spec,
        manifest,
        lambda candidate, _: _flat(validation[candidate], 1.0 - validation[candidate]),
    )
    first = _rank(records, manifest)
    for record in records:
        record["result"]["evaluation_history"]["clients"]["0"]["test"][
            "accuracy"
        ] = [1000.0]
    second = _rank(records, manifest)

    assert _selected(first) == _selected(second) == 0
    assert "test" not in json.dumps(first).lower()


def test_loss_is_minimized():
    spec, manifest = _study()
    records = _records(
        spec,
        manifest,
        lambda candidate, _: _flat(0.4 + candidate / 10, 0.0, metric="loss"),
    )

    artifact = _rank(records, manifest, metric="loss")

    assert _selected(artifact) == 0
    assert artifact["selection_protocol"]["direction"] == "minimize"


def test_weighted_and_unweighted_client_aggregation_can_select_differently():
    spec, manifest = _study()

    def score(candidate, _):
        if candidate == 0:
            return _history([[0.9], [0.1]], [[0.0], [0.0]], counts=[1, 9])
        return _history([[0.4], [0.4]], [[0.0], [0.0]], counts=[1, 9])

    records = _records(spec, manifest, score)
    mean = _rank(records, manifest, aggregation="mean")
    weighted = _rank(records, manifest, aggregation="weighted_mean")

    assert _selected(mean) == 0
    assert _selected(weighted) == 1


def test_global_and_per_client_round_selection_can_rank_differently():
    spec, manifest = _study()

    def score(candidate, _):
        if candidate == 0:
            return _history(
                [[0.9, 0.0], [0.0, 0.9]],
                [[0.0, 0.0], [0.0, 0.0]],
            )
        return _history(
            [[0.6, 0.6], [0.6, 0.6]],
            [[0.0, 0.0], [0.0, 0.0]],
        )

    records = _records(spec, manifest, score)
    artifact = _rank(records, manifest, views=["global", "per-client"])

    assert _selected(artifact, "global") == 1
    assert _selected(artifact, "per-client") == 0


def test_missing_replicate_makes_a_candidate_ineligible():
    spec, manifest = _study()
    records = _records(
        spec,
        manifest,
        lambda candidate, _: _flat(1.0 if candidate == 0 else 0.5, 0.0),
        omit={(0, 2)},
    )

    artifact = _rank(records, manifest)
    candidate = artifact["groups"][0]["candidates"][0]

    assert candidate["views"]["global"]["eligible"] is False
    assert candidate["views"]["global"]["missing_seeds"] == [2]
    assert _selected(artifact) == 1


def test_result_outside_the_declared_replicates_is_not_averaged():
    spec, manifest = _study()
    records = _records(spec, manifest, lambda *_: _flat(0.5, 0.0))
    extra = dict(records[0])
    extra["config"] = {
        "experiment": dict(records[0]["config"]["experiment"], seed=99),
        "algorithm": records[0]["config"]["algorithm"],
    }
    extra["_source_file"] = "extra.json"

    artifact = _rank(records + [extra], manifest)

    assert len(artifact["unassigned_records"]) == 1
    assert all(
        candidate["views"]["global"]["validation"]["n"] == 3
        for candidate in artifact["groups"][0]["candidates"]
    )


def test_duplicate_result_for_one_candidate_and_replicate_is_rejected():
    spec, manifest = _study()
    records = _records(spec, manifest, lambda *_: _flat(0.5, 0.0))
    duplicate = dict(records[0], _source_file="duplicate.json")

    with pytest.raises(TuningError, match="Duplicate results"):
        place_records(records + [duplicate], manifest)


def test_ranking_writes_a_final_selection(tmp_path):
    spec, manifest = _study()
    records = _records(
        spec,
        manifest,
        lambda candidate, _: _flat(1.0 - candidate / 10, 0.0),
    )
    artifact = _rank(records, manifest)

    written = write_ranking(artifact, records, manifest, tmp_path / "study")

    assert {path.name for path in written} == {
        "ranking.json",
        "selection.json",
        "selected.yaml",
    }
    selection = json.loads((tmp_path / "study" / "selection.json").read_text())
    assert selection["kind"] == "rigfl.tuning_selection"
    assert selection["selected_from"] == "initial_ranking"
    assert all(
        source.startswith("../runs/")
        for source in selection["groups"][0]["selected_result_files"]
    )


def test_study_document_round_trips(tmp_path):
    _, manifest = _study()
    manifest.pop("_path")

    path = write_manifest(manifest, tmp_path)
    loaded = load_manifest(tmp_path)

    assert path.name == "study.json"
    assert loaded["engine"] == "optuna"
    assert loaded["search_space"] == manifest["search_space"]


def test_future_study_schema_is_rejected(tmp_path):
    _, manifest = _study()
    manifest.pop("_path")
    manifest["schema_version"] += 1
    (tmp_path / "study.json").write_text(json.dumps(manifest))

    with pytest.raises(TuningError, match="schema"):
        load_manifest(tmp_path)
