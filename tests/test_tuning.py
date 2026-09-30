"""Validation-based ranking for candidates produced by an Optuna study."""

from __future__ import annotations

import itertools
import json
from types import SimpleNamespace

import pytest

from rigfl.experiment.launch import expand
from rigfl.experiment.optimize import _manifest, _tasks, parse_optimization
from rigfl.experiment.tuning import rank, write_ranking
from tests.helpers import STARTER_DATASETS


def _raw() -> dict:
    tuning = {
        "sampler": {"class": "GridSampler", "options": {"seed": 4}},
        "metric": "accuracy",
        "search_space": {
            "lr": {
                "type": "categorical",
                "values": [0.01, 0.1],
            },
            "mu": {
                "type": "categorical",
                "values": [0.0, 1.0],
            },
        },
    }
    return {
        "name": "fedprox_grid",
        "algorithm": "fedprox",
        "base": {
            "dataset": "cifar10",
            "dataset_config": STARTER_DATASETS,
            "model_arch": "fedavg_cnn",
            "rounds": 1,
            "eval_gap": 1,
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
        sources = [
            f"candidate{number}_seed{condition['training_seed']}.json"
            for condition in spec.replicate_conditions
        ]
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
            seed = task["experiment"]["training_seed"]
            records.append(
                {
                    "algorithm": task["algorithm"],
                    "config": {
                        "experiment": task["experiment"],
                        "algorithm": task["algorithm_config"],
                    },
                    "result": score(candidate["id"], replicate_index),
                    "_source_file": f"candidate{candidate['id']}_seed{seed}.json",
                }
            )
    return records


def _rank(records, manifest, **options):
    return rank(
        records,
        manifest,
        metric=options.pop("metric", "accuracy"),
        view=options.pop("view", "shared"),
        **options,
    )


def _selected(artifact: dict) -> int:
    return artifact["ranking"]["selected_candidate_id"]


def test_ordinary_sweeps_reject_a_tuning_section():
    with pytest.raises(SystemExit, match="ordinary sweeps do not perform"):
        expand(_raw())


def test_selection_uses_the_best_complete_configuration():
    spec, manifest = _study()
    scores = {0: 0.50, 1: 0.95, 2: 0.80, 3: 0.75}
    records = _records(
        spec, manifest, lambda candidate, _: _flat(scores[candidate], 0.0)
    )

    artifact = _rank(records, manifest)

    assert all(
        candidate["validation"]["n"] == 3 for candidate in artifact["candidates"]
    )
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


def test_weighted_and_unweighted_client_aggregation_can_select_differently():
    spec, manifest = _study()

    def score(candidate, _):
        if candidate == 0:
            return _history([[0.9], [0.1]], [[0.0], [0.0]], counts=[1, 9])
        value = 0.4 if candidate == 1 else 0.3
        return _history(
            [[value], [value]], [[0.0], [0.0]], counts=[1, 9]
        )

    records = _records(spec, manifest, score)
    mean = _rank(records, manifest, aggregation="uniform")
    weighted = _rank(records, manifest, aggregation="sample_count")

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
        value = 0.6 if candidate == 1 else 0.5
        return _history(
            [[value, value], [value, value]],
            [[0.0, 0.0], [0.0, 0.0]],
        )

    records = _records(spec, manifest, score)
    assert _selected(_rank(records, manifest, view="shared")) == 1
    assert _selected(_rank(records, manifest, view="client-specific")) == 0


def test_missing_replicate_makes_a_candidate_ineligible():
    spec, manifest = _study()
    records = _records(
        spec,
        manifest,
        lambda candidate, _: _flat(
            1.0 if candidate == 0 else 0.5 - candidate / 100, 0.0
        ),
        omit={(0, 2)},
    )

    artifact = _rank(records, manifest)
    candidate = artifact["candidates"][0]

    assert candidate["eligible"] is False
    assert candidate["missing_seeds"] == [2]
    assert _selected(artifact) == 1


def test_tied_best_candidates_remain_unselected(tmp_path):
    spec, manifest = _study()
    scores = {0: 0.95, 1: 0.95, 2: 0.80, 3: 0.75}
    records = _records(
        spec, manifest, lambda candidate, _: _flat(scores[candidate], 0.0)
    )
    artifact = _rank(records, manifest)
    study_dir = tmp_path / "study"
    study_dir.mkdir()
    (study_dir / "selected.yaml").write_text("stale")

    write_ranking(artifact, records, manifest, study_dir)

    assert not (study_dir / "selected.yaml").exists()
    ranking = json.loads((study_dir / "ranking.json").read_text())
    outcome = ranking["ranking"]
    assert outcome["best_candidate_ids"] == [0, 1]
    assert outcome["selected_candidate_id"] is None
