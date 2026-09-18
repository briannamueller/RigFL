"""Paired comparisons of frozen configurations."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rigfl.eval.comparison import (
    ConfigurationComparisonError,
    apply_holm,
    compare_configurations,
    format_comparisons,
)
from rigfl.eval.metrics import register, unregister
from rigfl.experiment import compare as compare_module
from rigfl.experiment.compare import (
    _comparison_results,
    _contrasts,
    _group_results,
    _load_selected_results,
)
from rigfl.experiment.config import result_data_configuration_id


def _record(
    algorithm,
    seed,
    values,
    *,
    validation=None,
    setting=1,
    partition_seed=None,
    split_seed=None,
    data_variant="a",
):
    validation = validation or [[0.5] for _ in values]
    partition_seed = seed if partition_seed is None else partition_seed
    split_seed = seed if split_seed is None else split_seed
    partition_id = f"partition-{data_variant}-{partition_seed}-{split_seed}"
    rounds = list(range(len(values[0])))
    clients = {
        str(index): {
            "validation": {"accuracy": list(validation[index])},
            "test": {"accuracy": list(client_values)},
        }
        for index, client_values in enumerate(values)
    }
    counts = {
        split: {str(index): [10] * len(rounds) for index in range(len(values))}
        for split in ("validation", "test")
    }
    return {
        "algorithm": algorithm,
        "config": {
            "experiment": {
                "dataset": "cifar10",
                "partition_id": partition_id,
                "partition_scheme": "dirichlet",
                "data_backend": "flower",
                "num_clients": len(values),
                "num_classes": 2,
                "validation_fraction": 0.2,
                "input_kind": "image",
                "input_spec": {"shape": [3, 32, 32]},
                "partition_seed": partition_seed,
                "split_seed": split_seed,
                "seed": seed,
                "rounds": len(rounds),
            },
            "algorithm": {"setting": setting},
        },
        "partition": {
            "generated": {
                "partition_id": partition_id,
                "settings": {
                    "source_dataset": "uoft-cs/cifar10",
                    "partition": {
                        "scheme": "dirichlet",
                        "num_clients": len(values),
                        "alpha": 0.5 if data_variant == "a" else 1.0,
                        "partition_seed": partition_seed,
                        "split_seed": split_seed,
                    },
                },
            }
        },
        "result": {
            "selection_views_supported": ["global", "per-client"],
            "evaluation_history": {
                "evaluation_rounds": rounds,
                "clients": clients,
                "client_sample_counts": counts,
            },
        },
    }


def _estimate(summary, name):
    return summary["effects"][name]["estimate"]


def test_comparison_pairs_clients_and_orients_gain():
    left = [
        _record("a", 0, [[0.8], [0.4]], setting=1),
        _record("a", 1, [[0.7], [0.6]], setting=1),
    ]
    right = [
        _record("b", 0, [[0.6], [0.5]], setting=2),
        _record("b", 1, [[0.6], [0.6]], setting=2),
    ]
    result = compare_configurations(
        left,
        right,
        "accuracy",
        left_label="a",
        right_label="b",
        practical_threshold=0.05,
    )

    assert result["coverage"] == {
        "pair_count": 4,
        "run_count": 2,
        "replicate_count": 2,
        "client_count": 2,
    }
    assert result["data_condition"]["partition_ids"] == [
        "partition-a-0-0",
        "partition-a-1-1",
    ]
    assert [pair["gain"] for pair in result["paired_differences"]] == pytest.approx(
        [0.2, -0.1, 0.1, 0.0]
    )
    assert _estimate(result, "mean_gain") == pytest.approx(0.05)
    assert _estimate(result, "benefit_rate") == 0.5
    assert _estimate(result, "harm_rate") == 0.25
    assert result["uncertainty"]["available"] is True
    assert result["effects"]["mean_gain"]["ci_low"] is not None
    assert result["randomization_test"]["available"] is False
    assert result["randomization_test"]["run_count"] == 2


def test_comparison_is_reproducible_and_changes_sign_when_swapped():
    left = [_record("a", seed, [[0.8], [0.7]]) for seed in (0, 1)]
    right = [_record("b", seed, [[0.5], [0.6]], setting=2) for seed in (0, 1)]
    first = compare_configurations(
        left, right, "accuracy", left_label="a", right_label="b"
    )
    again = compare_configurations(
        left, right, "accuracy", left_label="a", right_label="b"
    )
    swapped = compare_configurations(
        right, left, "accuracy", left_label="b", right_label="a"
    )
    assert first["effects"] == again["effects"]
    assert _estimate(first, "mean_gain") == -_estimate(swapped, "mean_gain")


def test_comparison_table_includes_local_contrasts_without_client_analysis():
    local = [_record("local", 0, [[0.5], [0.7]], setting=1)]
    feddes = [_record("feddes", 0, [[0.6], [0.5]], setting=2)]
    fml = [_record("fml", 0, [[0.7], [0.68]], setting=3)]

    method_comparison = compare_configurations(
        feddes, fml, "accuracy", left_label="feddes", right_label="fml",
        practical_threshold=0.05,
    )
    comparisons = [
        method_comparison,
        compare_configurations(
            feddes, local, "accuracy", left_label="feddes", right_label="local",
            practical_threshold=0.05,
        ),
        compare_configurations(
            local, fml, "accuracy", left_label="local", right_label="fml",
            practical_threshold=0.05,
        ),
    ]
    table = format_comparisons(comparisons)
    assert "feddes − fml" in table
    assert "feddes − local" in table
    assert "local − fml" in table
    assert "Negative transfer analysis" not in table
    assert "| benefit rate | NTR | NTM | NTB |" not in table


def test_lower_is_better_metric_is_oriented_as_left_gain():
    register("comparison_error", "minimize", fn=lambda p, y, c: 0.0)
    try:
        left = _record("a", 0, [[0.2], [0.6]])
        right = _record("b", 0, [[0.5], [0.5]], setting=2)
        for record in (left, right):
            for client in record["result"]["evaluation_history"]["clients"].values():
                client["validation"]["comparison_error"] = client["validation"][
                    "accuracy"
                ]
                client["test"]["comparison_error"] = client["test"]["accuracy"]
        result = compare_configurations(
            [left], [right], "comparison_error", left_label="a", right_label="b"
        )
        assert [pair["gain"] for pair in result["paired_differences"]] == pytest.approx(
            [0.3, -0.1]
        )
    finally:
        unregister("comparison_error")


def test_comparison_rejects_unmatched_or_unfrozen_results():
    with pytest.raises(ConfigurationComparisonError, match="identical data"):
        compare_configurations(
            [_record("a", 0, [[0.6]])],
            [_record("b", 1, [[0.5]], setting=2)],
            "accuracy",
            left_label="a",
            right_label="b",
        )
    left = [
        _record("a", 0, [[0.6]], setting=1),
        _record("a", 1, [[0.6]], setting=2),
    ]
    right = [
        _record("b", 0, [[0.5]], setting=3),
        _record("b", 1, [[0.5]], setting=3),
    ]
    with pytest.raises(ConfigurationComparisonError, match="not frozen"):
        compare_configurations(left, right, "accuracy", left_label="a", right_label="b")


def test_comparison_matches_the_complete_replicate_condition():
    left = [_record("a", 0, [[0.6]], partition_seed=0, split_seed=0)]
    right = [
        _record(
            "b",
            0,
            [[0.5]],
            setting=2,
            partition_seed=1,
            split_seed=0,
        )
    ]

    with pytest.raises(ConfigurationComparisonError, match="replicate conditions"):
        compare_configurations(
            left, right, "accuracy", left_label="a", right_label="b"
        )


def test_unmatched_mixed_seed_values_produce_the_intended_error():
    left = [
        _record("a", 0, [[0.6]], setting=1),
        _record("a", 1, [[0.6]], setting=1),
    ]
    left[0]["config"]["experiment"]["partition_seed"] = None
    right = [_record("b", 1, [[0.5]], setting=2)]

    with pytest.raises(ConfigurationComparisonError, match="identical data"):
        compare_configurations(
            left, right, "accuracy", left_label="a", right_label="b"
        )


def test_legacy_results_without_partition_metadata_keep_partition_identity():
    left = _record("a", 0, [[0.6]], setting=1)
    right = _record("a", 0, [[0.6]], setting=1)
    left.pop("partition")
    right.pop("partition")
    for record in (left, right):
        experiment = record["config"]["experiment"]
        experiment["partition_seed"] = None
        experiment["split_seed"] = None
    left["config"]["experiment"]["partition_id"] = "partition-a"
    right["config"]["experiment"]["partition_id"] = "partition-b"

    assert result_data_configuration_id(left) != result_data_configuration_id(right)


def test_one_shot_fallback_is_recorded_for_each_side():
    left = [_record("feddes", 0, [[0.6]])]
    left[0]["result"]["selection_views_supported"] = ["per-client"]
    right = [_record("local", 0, [[0.5]], setting=2)]
    result = compare_configurations(
        left, right, "accuracy", left_label="feddes", right_label="local"
    )
    assert result["protocol"]["selection_views"] == {
        "feddes": "per-client",
        "local": "global",
    }


def test_holm_adjustment_is_monotone_in_p_value_order():
    comparisons = [
        {"randomization_test": {"available": True, "p_value": 0.03}},
        {"randomization_test": {"available": True, "p_value": 0.01}},
        {"randomization_test": {"available": False}},
    ]
    apply_holm(comparisons)
    assert comparisons[1]["randomization_test"]["adjusted_p_value"] == 0.02
    assert comparisons[0]["randomization_test"]["adjusted_p_value"] == 0.03
    assert "adjusted_p_value" not in comparisons[2]["randomization_test"]


def test_randomization_test_requires_six_runs():
    left = [_record("a", seed, [[0.8], [0.7]]) for seed in range(6)]
    right = [_record("b", seed, [[0.5], [0.6]], setting=2) for seed in range(6)]
    result = compare_configurations(
        left, right, "accuracy", left_label="a", right_label="b"
    )
    assert result["randomization_test"]["available"] is True
    assert result["randomization_test"]["experimental_unit"] == (
        "replicate_condition"
    )


def test_final_comparison_rejects_crossed_seed_conditions():
    left = []
    right = []
    for partition_seed in (0, 1):
        for split_seed in (0, 1):
            for experiment_seed in (0, 1):
                left.append(
                    _record(
                        "a",
                        experiment_seed,
                        [[0.6], [0.7]],
                        partition_seed=partition_seed,
                        split_seed=split_seed,
                    )
                )
                right.append(
                    _record(
                        "b",
                        experiment_seed,
                        [[0.5], [0.6]],
                        setting=2,
                        partition_seed=partition_seed,
                        split_seed=split_seed,
                    )
                )

    with pytest.raises(ConfigurationComparisonError, match="distinct experiment_seed"):
        compare_configurations(
            left,
            right,
            "accuracy",
            left_label="a",
            right_label="b",
        )


def test_result_discovery_separates_data_conditions_and_finds_reference():
    records = []
    for data_variant in ("a", "b"):
        for algorithm, setting in (("local", 1), ("fedavg", 2)):
            for seed in (0, 1):
                records.append(
                    _record(
                        algorithm,
                        seed,
                        [[0.5]],
                        setting=setting,
                        data_variant=data_variant,
                    )
                )

    grouped = _group_results(records)

    assert len(grouped) == 2
    assert {context[0] for context in grouped} == {"cifar10"}
    assert all(
        set(configurations) == {"fedavg", "local"}
        for configurations in grouped.values()
    )
    first_context = next(iter(grouped))
    assert _contrasts(
        grouped[first_context],
        reference="local",
        declared=[],
        all_pairs=False,
    ) == [("fedavg", "local")]


def test_duplicate_reversed_contrasts_are_rejected():
    records = {"a": [], "b": []}
    with pytest.raises(ConfigurationComparisonError, match="declared once"):
        _contrasts(
            records,
            reference=None,
            declared=[("a", "b"), ("b", "a")],
            all_pairs=False,
        )


def test_result_discovery_labels_multiple_variants_of_one_algorithm():
    first_context = [
        _record("fedavg", 0, [[0.5]], setting=1),
        _record("fedavg", 0, [[0.6]], setting=2),
    ]
    second_context = []
    for record in first_context:
        copied = _record(
            "fedavg",
            0,
            [[0.5]],
            setting=record["config"]["algorithm"]["setting"],
            data_variant="b",
        )
        second_context.append(copied)
    grouped = _group_results([*first_context, *second_context])
    contexts = list(grouped)
    first_labels = set(grouped[contexts[0]])
    second_labels = set(grouped[contexts[1]])
    assert len(first_labels) == 2
    assert all(label.startswith("fedavg@") for label in first_labels)
    assert first_labels == second_labels


@pytest.mark.parametrize(
    "gain, threshold, conclusion",
    [
        (0.0, 0.01, "practically_equivalent"),
        (0.02, 0.01, "left_better"),
        (-0.02, 0.01, "right_better"),
    ],
)
def test_cluster_bootstrap_recovers_constant_effects(gain, threshold, conclusion):
    left = [
        _record("left", seed, [[0.5 + gain], [0.6 + gain]], setting=1)
        for seed in range(5)
    ]
    right = [_record("right", seed, [[0.5], [0.6]], setting=2) for seed in range(5)]
    result = compare_configurations(
        left,
        right,
        "accuracy",
        left_label="left",
        right_label="right",
        practical_threshold=threshold,
    )

    effect = result["effects"]["mean_gain"]
    assert effect["estimate"] == pytest.approx(gain)
    assert effect["ci_low"] == pytest.approx(gain)
    assert effect["ci_high"] == pytest.approx(gain)
    assert result["practical_conclusion"] == conclusion


def test_comparison_rejects_client_coverage_that_changes_between_seeds():
    left = [
        _record("left", 0, [[0.5], [0.6]], setting=1),
        _record("left", 1, [[0.5]], setting=1),
    ]
    right = [
        _record("right", 0, [[0.5], [0.6]], setting=2),
        _record("right", 1, [[0.5]], setting=2),
    ]
    left[1]["config"]["experiment"]["num_clients"] = 2
    right[1]["config"]["experiment"]["num_clients"] = 2
    left[1]["partition"]["generated"]["settings"]["partition"]["num_clients"] = 2
    right[1]["partition"]["generated"]["settings"]["partition"]["num_clients"] = 2
    with pytest.raises(ConfigurationComparisonError, match="expected 2"):
        compare_configurations(
            left, right, "accuracy", left_label="left", right_label="right"
        )


def test_wall_time_is_unavailable_for_different_hardware():
    left = [_record("left", 0, [[0.5]], setting=1)]
    right = [_record("right", 0, [[0.5]], setting=2)]
    for records, device in ((left, "gpu-a"), (right, "gpu-b")):
        records[0]["resources"] = {
            "observed": {"communication_bytes": {"total": 10}},
            "attributed_training": {
                "flops": 100,
                "wall_seconds": 1.0,
                "wall_seconds_comparable": True,
            },
            "measurement": {"timing": {"hardware": {"device": device}}},
        }

    result = compare_configurations(
        left, right, "accuracy", left_label="left", right_label="right"
    )

    assert result["resources"]["communication_bytes"]["available"] is True
    assert result["resources"]["flops"]["available"] is True
    assert result["resources"]["wall_seconds"]["available"] is False


def test_general_comparison_allows_different_client_model_assignments():
    left = [_record("left", 0, [[0.5], [0.6]], setting=1)]
    right = [_record("right", 0, [[0.5], [0.6]], setting=2)]
    left[0]["config"]["experiment"]["resolved_models"] = ["cnn_a", "cnn_b"]
    right[0]["config"]["experiment"]["resolved_models"] = ["cnn_b", "cnn_a"]

    result = compare_configurations(
        left, right, "accuracy", left_label="left", right_label="right"
    )

    assert result["effects"]["mean_gain"]["estimate"] == pytest.approx(0.0)


def test_run_level_effect_uses_the_declared_client_aggregation():
    left = [_record("left", 0, [[0.6], [0.8]], setting=1)]
    right = [_record("right", 0, [[0.5], [0.5]], setting=2)]
    for record in (*left, *right):
        for split in ("validation", "test"):
            record["result"]["evaluation_history"]["client_sample_counts"][split] = {
                "0": [1],
                "1": [3],
            }

    result = compare_configurations(
        left,
        right,
        "accuracy",
        left_label="left",
        right_label="right",
        aggregation="weighted_mean",
    )

    assert result["run_level_differences"][0]["gain"] == pytest.approx(0.25)
    assert result["effects"]["mean_gain"]["estimate"] == pytest.approx(0.25)


def test_compare_command_discovers_configurations_in_one_directory(
    tmp_path, monkeypatch, capsys
):
    records = []
    for algorithm, setting, score in (("local", 1, 0.5), ("fedavg", 2, 0.6)):
        for seed in range(6):
            record = _record(algorithm, seed, [[score], [score]], setting=setting)
            record["_source_file"] = str(tmp_path / f"{algorithm}_{seed}.json")
            records.append(record)
    monkeypatch.setattr(
        compare_module, "_load_results", lambda _path, **_kwargs: records
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "compare",
            "--results-dir",
            str(tmp_path),
            "--reference",
            "local",
            "--practical-threshold",
            "0.01",
        ],
    )

    compare_module.main()

    output = capsys.readouterr().out
    assert "fedavg − local" in output
    assert "cifar10 /" in output


def test_selection_discovery_loads_only_selected_results(tmp_path, monkeypatch):
    study_dir = tmp_path / "search"
    runs_dir = tmp_path / "runs"
    study_dir.mkdir()
    runs_dir.mkdir()
    (study_dir / "selection.json").write_text(
        json.dumps(
            {
                "kind": "rigfl.tuning_selection",
                "groups": [
                    {"selected_result_files": ["../runs/a.json", "../runs/b.json"]}
                ],
            }
        )
    )
    loaded = {
        "a.json": _record("fedavg", 0, [[0.6]], setting=1),
        "b.json": _record("fedavg", 1, [[0.6]], setting=1),
    }
    seen = []

    def fake_load(path):
        seen.append(path.name)
        return [loaded[path.name]]

    monkeypatch.setattr(compare_module, "_load_results", fake_load)

    records = _load_selected_results(tmp_path)

    assert seen == ["a.json", "b.json"]
    assert records == [loaded["a.json"], loaded["b.json"]]


def test_selection_file_is_the_only_final_tuning_artifact(tmp_path, monkeypatch):
    selection_dir = tmp_path / "search"
    selection_dir.mkdir()
    selection_path = selection_dir / "selection.json"
    selection_path.write_text(
        json.dumps(
            {
                "kind": "rigfl.tuning_selection",
                "groups": [{"selected_result_files": ["../runs/selected.json"]}],
            }
        )
    )
    selected = dict(_record("fedavg", 0, [[0.6]], setting=1))
    selected["_source_file"] = str(tmp_path / "runs" / "selected.json")
    monkeypatch.setattr(
        compare_module, "_load_results", lambda _path, **_kwargs: [selected]
    )

    records = _load_selected_results(selection_path)

    assert records == [selected]


def test_comparison_combines_selected_artifacts_and_ordinary_results(
    tmp_path, monkeypatch
):
    study_dir = tmp_path / "search"
    study_dir.mkdir()
    (study_dir / "selection.json").write_text(
        json.dumps(
            {
                "kind": "rigfl.tuning_selection",
                "groups": [{"selected_result_files": ["../runs/fedavg.json"]}],
            }
        )
    )
    fedavg = dict(_record("fedavg", 0, [[0.6]], setting=1))
    fedavg["_source_file"] = str(tmp_path / "runs" / "fedavg.json")
    local = dict(_record("local", 0, [[0.5]], setting=1))
    local["_source_file"] = str(tmp_path / "local.json")

    def fake_load(path, *, required=True):
        return [local] if Path(path).is_dir() else [fedavg]

    monkeypatch.setattr(compare_module, "_load_results", fake_load)

    records = _comparison_results(tmp_path)

    assert {record["algorithm"] for record in records} == {"local", "fedavg"}


def test_comparison_discovers_ordinary_results_in_shared_run_store(
    tmp_path, monkeypatch
):
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    local = dict(_record("local", 0, [[0.5]], setting=1))
    local["_source_file"] = str(runs_dir / "local.json")
    seen = []

    def fake_load(path, *, required=True):
        seen.append(Path(path))
        return [local] if Path(path) == runs_dir else []

    monkeypatch.setattr(compare_module, "_load_results", fake_load)

    records = _comparison_results(tmp_path)

    assert seen == [tmp_path, runs_dir]
    assert records == [local]


def test_comparison_accepts_one_run_result_file(tmp_path, monkeypatch):
    result_path = tmp_path / "local.json"
    local = dict(_record("local", 0, [[0.5]], setting=1))
    local["_source_file"] = str(result_path)
    monkeypatch.setattr(
        compare_module,
        "_load_results",
        lambda path, **kwargs: [local] if Path(path) == result_path else [],
    )

    assert _comparison_results(result_path) == [local]


def test_comparison_excludes_unselected_optuna_trials_from_shared_store(
    tmp_path, monkeypatch
):
    study_dir = tmp_path / "search"
    runs_dir = tmp_path / "runs"
    study_dir.mkdir()
    runs_dir.mkdir()
    (study_dir / "study.json").write_text(
        json.dumps(
            {
                "kind": "rigfl.tuning_study",
                "evaluations": [
                    {
                        "source_results": [
                            "selected.json",
                            "unselected.json",
                        ]
                    }
                ],
            }
        )
    )
    (study_dir / "selection.json").write_text(
        json.dumps(
            {
                "kind": "rigfl.tuning_selection",
                "groups": [
                    {"selected_result_files": ["../runs/selected.json"]}
                ],
            }
        )
    )
    selected = dict(_record("fedavg", 0, [[0.6]], setting=1))
    selected["_source_file"] = str(runs_dir / "selected.json")
    unselected = dict(_record("fedavg", 0, [[0.4]], setting=2))
    unselected["_source_file"] = str(runs_dir / "unselected.json")
    local = dict(_record("local", 0, [[0.5]], setting=1))
    local["_source_file"] = str(runs_dir / "local.json")

    def fake_load(path, *, required=True):
        path = Path(path)
        if path.name == "selected.json":
            return [selected]
        if path == runs_dir:
            return [selected, unselected, local]
        return []

    monkeypatch.setattr(compare_module, "_load_results", fake_load)

    records = _comparison_results(tmp_path)

    assert {record["_source_file"] for record in records} == {
        selected["_source_file"],
        local["_source_file"],
    }


def test_results_root_and_shared_run_store_have_same_comparison_scope(
    tmp_path, monkeypatch
):
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    local = dict(_record("local", 0, [[0.5]], setting=1))
    local["_source_file"] = str(runs_dir / "local.json")
    monkeypatch.setattr(
        compare_module,
        "_load_results",
        lambda path, **kwargs: [local] if Path(path) == runs_dir else [],
    )

    assert _comparison_results(tmp_path) == _comparison_results(runs_dir)


def test_selection_loads_its_existing_selected_results(tmp_path, monkeypatch):
    selection_dir = tmp_path / "search"
    selection_dir.mkdir()
    selection_path = selection_dir / "selection.json"
    selection_path.write_text(
        json.dumps(
            {
                "kind": "rigfl.tuning_selection",
                "groups": [
                    {
                        "selected_result_files": [
                            "../runs/seed0.json",
                            "../runs/seed1.json",
                        ]
                    }
                ],
            }
        )
    )
    seen = []

    def fake_load(path, *, required=True):
        seen.append(Path(path).name)
        record = dict(_record("fedavg", len(seen), [[0.6]], setting=1))
        record["_source_file"] = str(path)
        return [record]

    monkeypatch.setattr(compare_module, "_load_results", fake_load)

    records = _load_selected_results(selection_path)

    assert seen == ["seed0.json", "seed1.json"]
    assert len(records) == 2
