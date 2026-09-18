"""Grouping for the results table: automatic labels and --group-by overrides.

The property that matters: on a sweep of an algorithm-specific field, each setting gets
its OWN row (its own mean ± CI), rather than being averaged together as extra seeds.
"""

from __future__ import annotations

import json

import pytest

from rigfl.eval.transfer import (
    format_negative_transfer_profile,
    format_negative_transfer_table,
)
from rigfl.experiment import collect as collect_module
from rigfl.experiment.collect import (
    _field,
    _records_supporting,
    _rows_by_algorithm,
    _rows_by_group,
    _sort_rows_by_validation,
)
from rigfl.experiment.collect import main as collect_main
from tests.helpers import resolved_experiment

# Selection policy is fixed here on purpose: these tests are about grouping and
# pairing, not about which metric selects.
_SEL = {
    "metric": "accuracy",
    "view": "global",
    "aggregation": "mean",
    "tie_break": "earliest",
}


def _history(*client_values):
    """A one-round schema-2 result whose clients hold the given values.

    These fixtures are about grouping and pairing, so one round is enough.
    """
    clients = {
        str(i): {"validation": {"accuracy": [v], "balanced_accuracy": [v]},
                 "test": {"accuracy": [v], "balanced_accuracy": [v]}}
        for i, v in enumerate(client_values)
    }
    counts = {s: {str(i): [10] for i in range(len(client_values))}
              for s in ("validation", "test")}
    return {"schema_version": 3,
            "selection_views_supported": ["global", "per-client"],
            "evaluation_history": {"evaluation_rounds": [0], "clients": clients,
                                   "client_sample_counts": counts}}


def _rows(by_algorithm, group_by=None):
    if group_by:
        return _rows_by_group(by_algorithm, group_by, _SEL["metric"],
                              view=_SEL["view"], aggregation=_SEL["aggregation"],
                              tie_break=_SEL["tie_break"])
    return _rows_by_algorithm(by_algorithm, _SEL["metric"], view=_SEL["view"],
                           aggregation=_SEL["aggregation"], tie_break=_SEL["tie_break"])


def _rec(algorithm, seed, result_acc, **algorithm_cfg):
    return {
        "algorithm": algorithm,
        "config": {"algorithm": algorithm_cfg,
                   "experiment": {"batch": 32, "seed": seed}},
        "result": _history(result_acc),
    }


def _feddes_config(k):
    return {"graphroute": {"graph": {"k": k}}}


def test_field_reads_algorithm_and_experiment_and_name():
    rec = _rec("feddes", 0, 0.7, **_feddes_config(5))
    assert _field(rec, "algorithm") == "feddes"
    assert _field(rec, "algorithm.graphroute.graph.k") == 5
    assert _field(rec, "experiment.batch") == 32


def test_missing_flop_setting_matches_the_disabled_default():
    from rigfl.experiment.collect import experiment_condition

    old = _rec("local", 0, 0.7)
    current = _rec("local", 0, 0.7)
    current["config"]["experiment"]["estimate_flops"] = False
    assert experiment_condition(old) == experiment_condition(current)


def test_both_omits_an_unsupported_view_instead_of_duplicating_fallback():
    feddes = _rec("feddes", 0, 0.7)
    feddes["result"]["selection_views_supported"] = ["per-client"]
    local = _rec("local", 0, 0.5)
    by_algorithm = {"feddes": [feddes], "local": [local]}

    assert set(_records_supporting(by_algorithm, "global")) == {"local"}
    assert set(_records_supporting(by_algorithm, "per-client")) == {
        "feddes", "local"
    }


def test_group_by_hyperparameter_makes_one_row_per_setting():
    by_algorithm = {"feddes": [
        _rec("feddes", 0, 0.60, **_feddes_config(3)),
        _rec("feddes", 1, 0.62, **_feddes_config(3)),
        _rec("feddes", 0, 0.80, **_feddes_config(5)),
        _rec("feddes", 1, 0.82, **_feddes_config(5)),
    ]}
    rows = _rows(by_algorithm, ["algorithm.graphroute.graph.k"])
    assert set(rows) == {"feddes k=3", "feddes k=5"}
    assert rows["feddes k=3"]["seeds"] == 2
    assert abs(rows["feddes k=3"]["test_mean"] - 0.61) < 1e-9
    assert abs(rows["feddes k=5"]["test_mean"] - 0.81) < 1e-9


def test_group_by_rejects_unlabelled_algorithm_variants():
    records = {"feddes": [
        _rec("feddes", 0, 0.60, **_feddes_config(3)),
        _rec("feddes", 1, 0.80, **_feddes_config(5)),
    ]}

    with pytest.raises(ValueError, match="multiple algorithm configurations"):
        _rows(records, ["algorithm"])


def test_group_by_keeps_algorithms_separate():
    by_algorithm = {
        "feddes": [_rec("feddes", 0, 0.8, **_feddes_config(5))],
        "local": [_rec("local", 0, 0.5)],
    }
    rows = _rows(by_algorithm, ["algorithm.graphroute.graph.k"])
    # Local has no GraphRoute k, so it gets its own row labelled with None.
    assert "feddes k=5" in rows
    assert "local k=None" in rows


def _cond_rec(algorithm, seed, *, dataset="cifar10", partition_id="partition-a",
              accs=(0.8, 0.7),
              algorithm_cfg=None):
    """Use each algorithm's validated default configuration when unspecified."""
    if algorithm_cfg is None:
        from rigfl.experiment.registry import config_class
        algorithm_cfg = config_class(algorithm)().model_dump()
    return {"algorithm": algorithm,
            "config": {"experiment": {
                "dataset": dataset, "partition_id": partition_id, "seed": seed,
                "partition_seed": 0, "split_seed": 0,
                "data_backend": "flower", "partition_scheme": "dirichlet",
                "num_clients": len(accs), "num_classes": 10,
                "validation_fraction": 0.2, "input_kind": "image",
            },
                       "algorithm": algorithm_cfg},
            "partition": {
                "generated": {
                    "partition_id": partition_id,
                    "settings": {
                        "source_dataset": dataset,
                        "partition": {
                            "scheme": "dirichlet",
                            "num_clients": len(accs),
                            "configuration": partition_id,
                            "partition_seed": 0,
                            "split_seed": 0,
                        },
                    },
                }
            },
            "result": _history(*accs)}


def test_algorithms_with_different_configs_share_an_experiment():
    """Local and FedDES configure differently by nature; that must not separate
    them, or Local-relative reporting has nothing to pair against."""
    from rigfl.experiment.collect import experiment_condition
    assert experiment_condition(_cond_rec("local", 0)) == experiment_condition(_cond_rec("feddes", 0))


def test_negative_transfer_is_computed_across_algorithms():
    rows = _rows({
        "local":  [_cond_rec("local", 0, accs=(0.9, 0.9))],
        "feddes": [_cond_rec("feddes", 0, accs=(0.5, 0.5))],
    })
    feddes = next(k for k in rows if k.startswith("feddes"))
    transfer = rows[feddes]["negative_transfer"]
    assert transfer["negative_transfer_rate"]["estimate"] == 1.0
    assert transfer["negative_transfer_magnitude"]["estimate"] == pytest.approx(0.4)


def test_client_model_pool_is_part_of_the_experiment_condition():
    from rigfl.experiment.collect import experiment_condition

    local = _cond_rec("local", 0)
    feddes = _cond_rec("feddes", 0)
    local["config"]["experiment"]["model"] = "fedavg_cnn"
    feddes["config"]["experiment"]["model"] = "cifar_resnet18"

    assert experiment_condition(local) != experiment_condition(feddes)
    rows = _rows({"local": [local], "feddes": [feddes]})
    feddes_row = next(value for key, value in rows.items()
                       if key.startswith("feddes"))
    assert "negative_transfer" not in feddes_row


def test_mixed_model_capabilities_share_the_requested_experiment_condition():
    from rigfl.experiment.collect import experiment_condition
    from rigfl.experiment.registry import resolve_algorithm_models

    exp = resolved_experiment(
        model="fedavg_cnn", model_family="image_heterogeneous_3"
    )
    homogeneous = resolve_algorithm_models("fedavg", exp)
    heterogeneous = resolve_algorithm_models("feddes", exp)
    a = _cond_rec("local", 0)
    b = _cond_rec("feddes", 0)
    a["config"]["experiment"] = homogeneous.model_dump(mode="json")
    b["config"]["experiment"] = heterogeneous.model_dump(mode="json")

    assert experiment_condition(a) == experiment_condition(b)
    rows = _rows({"local": [a], "feddes": [b]})
    comparison = rows["feddes"]["negative_transfer"]
    assert comparison["available"] is False
    assert "resolved client-model assignment" in comparison["reason"]


def test_experiments_are_not_averaged_together():
    """Two datasets in one directory are two experiments, not extra seeds."""
    from rigfl.experiment.collect import experiment_condition
    recs = {"local": [_cond_rec("local", 0), _cond_rec("local", 0, dataset="mnist")],
            "feddes": [_cond_rec("feddes", 0), _cond_rec("feddes", 0, dataset="mnist")]}
    assert len({experiment_condition(r) for rs in recs.values() for r in rs}) == 2
    rows = _rows(recs)
    assert len(rows) == 4
    assert all(s["seeds"] == 1 for s in rows.values())


def test_local_is_paired_within_its_own_experiment():
    rows = _rows({
        "local":  [_cond_rec("local", 0, accs=(0.9, 0.9))],                  # cifar only
        "feddes": [_cond_rec("feddes", 0, accs=(0.5, 0.5)),
                   _cond_rec("feddes", 0, dataset="mnist", accs=(0.5, 0.5))],
    })
    cifar = next(k for k in rows if k.startswith("feddes") and "cifar10" in k)
    mnist = next(k for k in rows if k.startswith("feddes") and "mnist" in k)
    assert rows[cifar]["negative_transfer"]["negative_transfer_rate"]["estimate"] == 1.0
    assert "negative_transfer" not in rows[mnist]  # no Local run in that experiment


def test_one_algorithm_swept_over_its_own_settings_gets_separate_rows():
    rows = _rows({"feddes": [
        _cond_rec("feddes", 0, algorithm_cfg=_feddes_config(3)),
        _cond_rec("feddes", 0, algorithm_cfg=_feddes_config(9)),
    ]})
    assert set(rows) == {
        "feddes graphroute.graph.k=3",
        "feddes graphroute.graph.k=9",
    }


def test_default_labels_show_each_varying_algorithm_setting():
    rows = _rows({"fedavg": [
        _rec("fedavg", 0, 0.60, lr=0.01, local_epochs=1),
        _rec("fedavg", 1, 0.62, lr=0.01, local_epochs=1),
        _rec("fedavg", 0, 0.80, lr=0.03, local_epochs=2),
    ]})
    assert set(rows) == {
        "fedavg local_epochs=1 lr=0.01",
        "fedavg local_epochs=2 lr=0.03",
    }
    assert rows["fedavg local_epochs=1 lr=0.01"]["seeds"] == 2


def test_default_rows_order_by_validation_within_a_data_configuration():
    lower = _cond_rec("feddes", 0, accs=(0.6, 0.6), algorithm_cfg=_feddes_config(3))
    higher = _cond_rec("feddes", 0, accs=(0.8, 0.8), algorithm_cfg=_feddes_config(9))
    lower["config"]["experiment"]["rounds"] = 50
    higher["config"]["experiment"]["rounds"] = 100

    rows = _rows({"feddes": [lower, higher]})

    assert "rounds=100" in next(iter(rows))
    assert [row["val_mean"] for row in rows.values()] == [0.8, 0.6]


def test_validation_order_keeps_different_reporting_views_separate():
    rows = {
        "per-client": {"dataset": "cifar10", "data_configuration_id": "same",
                       "selection_view": "per-client", "val_mean": 0.9},
        "global": {"dataset": "cifar10", "data_configuration_id": "same",
                   "selection_view": "global", "val_mean": 0.6},
    }
    assert list(_sort_rows_by_validation(rows, "accuracy")) == ["global", "per-client"]


def test_validation_order_respects_metric_direction_and_data_setup():
    rows = {
        "higher loss": {"dataset": "cifar10", "data_configuration_id": "a",
                        "selection_view": "global", "val_mean": 0.8},
        "other data": {"dataset": "cifar10", "data_configuration_id": "b",
                       "selection_view": "global", "val_mean": 0.1},
        "lower loss": {"dataset": "cifar10", "data_configuration_id": "a",
                       "selection_view": "global", "val_mean": 0.2},
    }
    assert list(_sort_rows_by_validation(rows, "loss")) == [
        "lower loss", "higher loss", "other data"
    ]


def test_saved_grid_shows_completed_and_missing_seed_combinations():
    from rigfl.eval.report import format_replicate_details, format_table

    record = _cond_rec("feddes", 0, algorithm_cfg=_feddes_config(3))
    tasks = [
        {"algorithm": "feddes",
         "experiment": {"dataset": "cifar10", "partition_seed": 0,
                        "split_seed": 0, "seed": seed},
         "algorithm_config": _feddes_config(3)}
        for seed in range(3)
    ]

    rows = _rows_by_algorithm(
        {"feddes": [record]}, "accuracy", view="global", aggregation="mean",
        tie_break="earliest", grid_tasks=tasks,
    )

    summary = next(iter(rows.values()))
    assert summary["runs"] == 1
    assert summary["expected_runs"] == 3
    assert summary["replicate_conditions"] == [
        {"partition_seed": 0, "split_seed": 0, "experiment_seed": 0}
    ]
    assert [item["experiment_seed"] for item in summary["missing_replicates"]] == [1, 2]
    assert "| 1/3 |" in format_table(rows, "accuracy")
    assert "missing: (0, 0, 1), (0, 0, 2)" in format_replicate_details(rows)


def test_saved_grid_counts_each_configuration_separately():
    records = [
        _rec("fedavg", 0, 0.6, lr=0.01),
        _rec("fedavg", 0, 0.8, lr=0.03),
        _rec("fedavg", 1, 0.8, lr=0.03),
    ]
    tasks = [
        {"algorithm": "fedavg", "experiment": {"seed": seed},
         "algorithm_config": {"lr": lr}}
        for lr in (0.01, 0.03) for seed in range(3)
    ]
    rows = _rows_by_algorithm(
        {"fedavg": records}, "accuracy", view="global", aggregation="mean",
        tie_break="earliest", grid_tasks=tasks,
    )
    assert rows["fedavg lr=0.01"]["expected_runs"] == 3
    assert rows["fedavg lr=0.01"]["runs"] == 1
    assert rows["fedavg lr=0.03"]["expected_runs"] == 3
    assert rows["fedavg lr=0.03"]["runs"] == 2


def test_collect_help_has_no_separate_rank_flag(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["collect", "--help"])
    with pytest.raises(SystemExit) as exit_info:
        collect_main()
    assert exit_info.value.code == 0
    help_text = capsys.readouterr().out
    assert "--rank" not in help_text
    assert "--ignore-invalid" not in help_text
    assert "--strict-results" in help_text
    assert "--performance-margin" in help_text
    assert "--negative-transfer-threshold" not in help_text


def test_collect_reports_invalid_files_and_keeps_valid_runs(tmp_path, monkeypatch, capsys):
    def load(_directory, _dataset, *, ignore_invalid, invalid):
        assert ignore_invalid is True
        invalid.append(("broken.json", "missing evaluation history"))
        return {"feddes": [_cond_rec("feddes", 0)]}

    monkeypatch.setattr(collect_module, "load_results", load)
    report = tmp_path / "summary.md"
    artifact = tmp_path / "summary.json"
    monkeypatch.setattr("sys.argv", [
        "collect", "--out", str(report), "--out-json", str(artifact)
    ])

    collect_main()

    assert "WARNING: excluded 1 invalid result file" in capsys.readouterr().out
    assert report.read_text().startswith("**Warning:** Excluded 1 invalid result file")
    assert "broken.json" in report.read_text()
    assert json.loads(artifact.read_text())["ignored_invalid_results"] == [
        {"file": "broken.json", "reason": "missing evaluation history"}
    ]


def test_collect_labels_client_level_performance_analysis(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(
        collect_module,
        "load_results",
        lambda *_args, **_kwargs: {
            "local": [_cond_rec("local", 0)],
            "fedavg": [_cond_rec("fedavg", 0)],
        },
    )
    report = tmp_path / "summary.md"
    monkeypatch.setattr("sys.argv", ["collect", "--out", str(report)])

    collect_main()

    assert "### Client-level performance analysis" in capsys.readouterr().out
    assert "### Client-level performance analysis: global" in report.read_text()


def test_collect_strict_results_stops_on_invalid_file(monkeypatch):
    def load(_directory, _dataset, *, ignore_invalid, invalid):
        assert ignore_invalid is False
        raise SystemExit("invalid result")

    monkeypatch.setattr(collect_module, "load_results", load)
    monkeypatch.setattr("sys.argv", ["collect", "--strict-results"])
    with pytest.raises(SystemExit, match="invalid result"):
        collect_main()


def test_loader_can_skip_invalid_file_without_losing_valid_runs(tmp_path, monkeypatch):
    (tmp_path / "broken.json").write_text("{")
    (tmp_path / "valid.json").write_text(json.dumps({
        "algorithm": "fedavg", "config": {"experiment": {"dataset": "cifar10"}}
    }))
    monkeypatch.setattr(collect_module, "is_run_result", lambda _record: True)
    monkeypatch.setattr(collect_module, "validate_run_record", lambda _record, path: None)

    with pytest.raises(SystemExit, match="strict mode stopped"):
        collect_module.load_results(tmp_path, None)

    invalid = []
    rows = collect_module.load_results(tmp_path, None, ignore_invalid=True, invalid=invalid)
    assert len(rows["fedavg"]) == 1
    assert invalid[0][0] == "broken.json"


def test_sweep_task_rejects_an_unknown_setting(tmp_path):
    """The array-task path rejects unknown algorithm settings."""
    import json

    from rigfl.experiment.launch import run_task
    grid = tmp_path / "grid.jsonl"

    def write(algorithm_config):
        grid.write_text(json.dumps({"algorithm": "feddes",
                                    "experiment": {"dataset": "cifar10", "seed": 0},
                                    "algorithm_config": algorithm_config}) + "\n")

    write({"graphroute": {"graph": {"kk": 7}}})
    with pytest.raises(SystemExit, match="graphroute.graph.kk"):
        run_task(str(grid), 1, tmp_path, dry_run=True)

    write(_feddes_config(7))
    run_task(str(grid), 1, tmp_path, dry_run=True)


def test_group_by_still_separates_experiments():
    """--group-by says how to label rows, not that different datasets may be
    averaged together as extra seeds."""
    recs = {"feddes": [
        _cond_rec("feddes", 0, partition_id="partition-a", algorithm_cfg=_feddes_config(5)),
        _cond_rec("feddes", 1, partition_id="partition-a", algorithm_cfg=_feddes_config(5)),
        _cond_rec("feddes", 0, partition_id="partition-b", algorithm_cfg=_feddes_config(5)),
    ]}
    rows = _rows(recs, ["algorithm.graphroute.graph.k"])
    assert len(rows) == 2, rows                  # distinct partitions stay apart
    assert sorted(s["seeds"] for s in rows.values()) == [1, 2]


def test_zipped_data_and_training_replicates_form_one_result_row():
    records = []
    for seed in range(3):
        record = _cond_rec(
            "feddes",
            seed,
            partition_id=f"partition-{seed}",
            algorithm_cfg=_feddes_config(5),
        )
        experiment = record["config"]["experiment"]
        experiment["partition_seed"] = seed
        experiment["split_seed"] = seed
        settings = record["partition"]["generated"]["settings"]["partition"]
        settings["partition_seed"] = seed
        settings["split_seed"] = seed
        settings.pop("configuration")
        records.append(record)

    rows = _rows({"feddes": records}, ["algorithm.graphroute.graph.k"])

    assert len(rows) == 1
    summary = next(iter(rows.values()))
    assert summary["seeds"] == 3
    assert summary["independent_replicates"] is True
    assert summary["test_ci"] is not None


@pytest.mark.parametrize("drop_one", [False, True])
def test_crossed_collection_omits_performance_and_transfer_intervals(drop_one):
    by_algorithm = {"local": [], "feddes": []}
    for partition_seed in range(2):
        for split_seed in range(2):
            for experiment_seed in range(2):
                partition_id = (
                    f"partition-{partition_seed}-{split_seed}"
                )
                for algorithm, accuracy in (("local", 0.5), ("feddes", 0.6)):
                    record = _cond_rec(
                        algorithm,
                        experiment_seed,
                        partition_id=partition_id,
                        accs=(accuracy, accuracy),
                    )
                    experiment = record["config"]["experiment"]
                    experiment["partition_seed"] = partition_seed
                    experiment["split_seed"] = split_seed
                    settings = record["partition"]["generated"]["settings"][
                        "partition"
                    ]
                    settings["partition_seed"] = partition_seed
                    settings["split_seed"] = split_seed
                    settings.pop("configuration")
                    by_algorithm[algorithm].append(record)

    if drop_one:
        by_algorithm["local"].pop()
        by_algorithm["feddes"].pop()

    rows = _rows_by_algorithm(
        by_algorithm,
        "accuracy",
        view="global",
        aggregation="mean",
        tie_break="earliest",
        transfer_profile=(0.05,),
    )
    feddes = rows["feddes"]

    assert feddes["runs"] == (7 if drop_one else 8)
    assert feddes["seeds"] == 2
    assert feddes["test_ci"] is None
    transfer = feddes["negative_transfer"]
    assert transfer["negative_transfer_rate"]["estimate"] == 0.0
    assert transfer["negative_transfer_rate"]["ci_low"] is None
    assert transfer["uncertainty"]["reason"] == (
        "experiment seeds are reused across run conditions"
    )
    assert "feddes §" in format_negative_transfer_table(rows)
    assert "feddes §" in format_negative_transfer_profile(rows)


def test_zipped_collection_keeps_transfer_intervals():
    by_algorithm = {
        "local": [
            _cond_rec("local", seed, accs=(0.5, 0.5)) for seed in range(3)
        ],
        "feddes": [
            _cond_rec("feddes", seed, accs=(0.6, 0.4)) for seed in range(3)
        ],
    }
    rows = _rows_by_algorithm(
        by_algorithm,
        "accuracy",
        view="global",
        aggregation="mean",
        tie_break="earliest",
        transfer_profile=(0.05,),
    )
    transfer = rows["feddes"]["negative_transfer"]

    assert transfer["uncertainty"]["available"] is True
    assert transfer["negative_transfer_rate"]["ci_low"] is not None
    assert "[" in format_negative_transfer_table(rows)
    assert "§" not in format_negative_transfer_table(rows)
    assert "NTR (δ=0.05)" in format_negative_transfer_profile(rows)


def test_experiments_differing_only_in_an_unlabelled_field_stay_apart():
    """Every configuration difference produces a distinct result row."""

    def rec(seed, batch):
        return {"algorithm": "feddes",
                "config": {"experiment": {
                    "dataset": "cifar10", "partition_id": "partition-a",
                    "data_backend": "flower", "partition_scheme": "dirichlet",
                    "num_clients": 20, "num_classes": 10,
                    "validation_fraction": 0.2, "input_kind": "image",
                    "seed": seed, "batch": batch},
                           "algorithm": _feddes_config(5)},
                "result": _history(.7, .6)}

    recs = {"feddes": [rec(0, 32), rec(1, 32), rec(0, 64)]}
    for rows in (_rows(recs), _rows(recs, ["algorithm.graphroute.graph.k"])):
        assert len(rows) == 2, rows
        assert sorted(s["seeds"] for s in rows.values()) == [1, 2]
        assert any("batch=64" in k for k in rows)      # labelled by what differs


def test_early_stopping_settings_separate_experiments():
    """Stopping changes where training ends, so runs that stopped differently are
    different experiments -- not extra seeds. (Selection stays out: it is
    post-hoc over an identical history.)"""
    from rigfl.experiment.collect import experiment_condition

    def rec(**es):
        return {"config": {"experiment": {"dataset": "cifar10", "early_stopping": es}}}

    assert experiment_condition(rec(enabled=True, patience=5)) != \
           experiment_condition(rec(enabled=True, patience=20))
    assert experiment_condition(rec(enabled=True, metric="accuracy")) != \
           experiment_condition(rec(enabled=True, metric="balanced_accuracy"))
    assert experiment_condition(rec(enabled=False)) == experiment_condition(rec(enabled=False))


def _es_rec(seed, **early_stopping):
    rec = _cond_rec("local", seed)
    rec["config"]["experiment"]["early_stopping"] = early_stopping
    return rec


def test_rows_are_uniquely_labelled_when_only_early_stopping_differs():
    """Early-stopping differences appear in unique result-row labels."""
    recs = {"local": [_es_rec(0, enabled=True, metric="accuracy", patience=5),
                      _es_rec(0, enabled=True, metric="accuracy", patience=20)]}
    for rows in (_rows(recs), _rows(recs, ["algorithm"])):
        assert len(rows) == 2, rows
        assert len(set(rows)) == 2, "row labels must be unique"
        assert any("patience=5" in k for k in rows)
        assert any("patience=20" in k for k in rows)


def test_the_three_condition_helpers_agree_on_their_fields():
    """Condition grouping and display labels use the same fields."""
    from rigfl.experiment.collect import (
        condition_fields,
        describe_condition,
        experiment_condition,
        varying_fields,
    )
    a = _es_rec(0, enabled=True, metric="accuracy", patience=5)
    b = _es_rec(0, enabled=True, metric="accuracy", patience=20)
    assert experiment_condition(a) != experiment_condition(b)
    varying = varying_fields([a, b])
    assert varying == ["early_stopping.patience"]
    assert set(varying) <= set(condition_fields(a))          # labels use real fields
    assert describe_condition(a, varying) != describe_condition(b, varying)


def test_disabled_early_stopping_ignores_its_inactive_settings():
    """A patience that never applies must not mint a second experiment."""
    from rigfl.experiment.collect import experiment_condition
    assert experiment_condition(_es_rec(0, enabled=False, patience=5)) == \
           experiment_condition(_es_rec(0, enabled=False, patience=20))
    assert len(_rows({"local": [_es_rec(0, enabled=False, patience=5),
                                _es_rec(1, enabled=False, patience=20)]})) == 1


def test_disabled_early_stopping_ignores_inactive_settings_for_run_identity():
    from rigfl.experiment.config import run_fingerprint
    off5 = resolved_experiment(early_stopping={"enabled": False, "patience": 5})
    off20 = resolved_experiment(early_stopping={"enabled": False, "patience": 20})
    assert run_fingerprint(off5, {}) == run_fingerprint(off20, {})
    on5 = resolved_experiment(early_stopping={"enabled": True, "metric": "accuracy", "patience": 5})
    on20 = resolved_experiment(early_stopping={"enabled": True, "metric": "accuracy", "patience": 20})
    assert run_fingerprint(on5, {}) != run_fingerprint(on20, {})
