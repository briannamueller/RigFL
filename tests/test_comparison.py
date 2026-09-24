"""Paired comparisons of frozen configurations."""

from __future__ import annotations

import pytest

from rigfl.eval.comparison import (
    ConfigurationComparisonError,
    compare_configurations,
    format_comparisons,
)
from rigfl.eval.metrics import register, unregister
from rigfl.experiment import compare as compare_module
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
    assert set(result["effects"]) == {"mean_gain"}
    assert result["effects"]["mean_gain"]["n"] == 2
    assert result["effects"]["mean_gain"]["df"] == 1
    assert result["effects"]["mean_gain"][
        "pooled_within_replicate_client_sd"
    ] == pytest.approx(0.025 ** 0.5)
    assert result["uncertainty"]["available"] is True
    assert result["uncertainty"]["method"] == "replicate_level_t_interval"
    assert result["effects"]["mean_gain"]["ci_low"] is not None


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
    fedprox = [_record("fedprox", 0, [[0.6], [0.5]], setting=2)]
    fml = [_record("fml", 0, [[0.7], [0.68]], setting=3)]

    method_comparison = compare_configurations(
        fedprox, fml, "accuracy", left_label="fedprox", right_label="fml",
        practical_threshold=0.05,
    )
    comparisons = [
        method_comparison,
        compare_configurations(
            fedprox, local, "accuracy", left_label="fedprox", right_label="local",
            practical_threshold=0.05,
        ),
        compare_configurations(
            local, fml, "accuracy", left_label="local", right_label="fml",
            practical_threshold=0.05,
        ),
    ]
    table = format_comparisons(comparisons)
    assert "fedprox − fml" in table
    assert "fedprox − local" in table
    assert "local − fml" in table


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


def test_partition_id_preserves_data_identity_without_an_optional_summary():
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
    left = [_record("fedprox", 0, [[0.6]])]
    left[0]["result"]["selection_views_supported"] = ["per-client"]
    right = [_record("local", 0, [[0.5]], setting=2)]
    result = compare_configurations(
        left, right, "accuracy", left_label="fedprox", right_label="local"
    )
    assert result["protocol"]["selection_views"] == {
        "fedprox": "per-client",
        "local": "global",
    }


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


@pytest.mark.parametrize(
    "gain, threshold, conclusion",
    [
        (0.0, 0.01, "practically_equivalent"),
        (0.02, 0.01, "left_better"),
        (-0.02, 0.01, "right_better"),
    ],
)
def test_replicate_t_interval_recovers_constant_effects(gain, threshold, conclusion):
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
    config = tmp_path / "reporting.yaml"
    config.write_text(
        "version: 1\n"
        "filters:\n"
        "  main_results:\n"
        "    algorithm: [local, fedavg]\n"
        "comparisons:\n"
        "  fedavg_vs_local:\n"
        "    filter: main_results\n"
        "    field: algorithm\n"
        "    values: [fedavg, local]\n"
        "    select: exact\n"
        "    practical_margin: 0.01\n"
    )
    saved = tmp_path / "comparison.csv"

    compare_module.main([
        str(tmp_path), "--config", str(config),
        "--comparison", "fedavg_vs_local", "--save", str(saved),
    ])

    output = capsys.readouterr().out
    assert "algorithm=fedavg − algorithm=local" in output
    assert "selected algorithm=fedavg" in output
    assert "cifar10 /" in output
    assert saved.exists()
    assert "mean_difference" in saved.read_text()


def test_named_comparison_selects_each_side_by_validation(tmp_path, monkeypatch, capsys):
    records = []
    specifications = (
        ("a", 1, 0.7, 0.2),
        ("a", 2, 0.9, 0.8),
        ("b", 3, 0.6, 0.3),
        ("b", 4, 0.8, 0.7),
    )
    for algorithm, setting, validation, test in specifications:
        for seed in (0, 1):
            records.append(
                _record(
                    algorithm,
                    seed,
                    [[test], [test]],
                    validation=[[validation], [validation]],
                    setting=setting,
                )
            )
    monkeypatch.setattr(
        compare_module, "_load_results", lambda _path, **_kwargs: records
    )
    config = tmp_path / "reporting.yaml"
    config.write_text(
        "version: 1\n"
        "filters:\n  candidates:\n    algorithm: [a, b]\n"
        "comparisons:\n  selected:\n"
        "    filter: candidates\n"
        "    field: algorithm\n"
        "    values: [a, b]\n"
        "    select: best_validation\n"
        "    practical_margin: 0.0\n"
    )

    compare_module.main([
        str(tmp_path), "--config", str(config), "--comparison", "selected"
    ])

    output = capsys.readouterr().out
    assert "config.algorithm.setting=2" in output
    assert "config.algorithm.setting=4" in output
    assert "algorithm=a − algorithm=b" in output


def test_exact_hyperparameter_comparison_rejects_other_changed_settings(
    tmp_path, monkeypatch
):
    left = _record("a", 0, [[0.6]], setting=1)
    right = _record("a", 0, [[0.7]], setting=2)
    left["config"]["algorithm"]["other"] = "left"
    right["config"]["algorithm"]["other"] = "right"
    monkeypatch.setattr(
        compare_module, "_load_results", lambda _path, **_kwargs: [left, right]
    )
    config = tmp_path / "reporting.yaml"
    config.write_text(
        "version: 1\n"
        "filters:\n  candidates:\n    algorithm: [a]\n"
        "comparisons:\n  settings:\n"
        "    filter: candidates\n"
        "    field: config.algorithm.setting\n"
        "    values: [1, 2]\n"
        "    select: exact\n"
        "    practical_margin: 0.0\n"
    )

    with pytest.raises(SystemExit, match="confounded.*config.algorithm.other"):
        compare_module.main([
            str(tmp_path), "--config", str(config), "--comparison", "settings"
        ])
