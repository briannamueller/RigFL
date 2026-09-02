"""Grouping for the results table: default (per algorithm) vs --group-by hyperparameter.

The property that matters: on a sweep of an algorithm-specific field, each setting gets
its OWN row (its own mean ± CI), rather than being averaged together as extra seeds.
"""

from __future__ import annotations

from rigfl.experiment.collect import (_field, _records_supporting,
                                      _rows_by_algorithm, _rows_by_group)

# Selection policy is fixed here on purpose: these tests are about grouping and
# pairing, not about which metric selects.
_SEL = dict(metric="accuracy", view="global", aggregation="mean", tie_break="earliest")


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
        "config": {"algorithm": {"seed_note": seed, **algorithm_cfg}, "experiment": {"alpha": 0.1, "seed": seed}},
        "result": _history(result_acc),
    }


def test_field_reads_algorithm_and_experiment_and_name():
    rec = _rec("feddes", 0, 0.7, graph_k=5)
    assert _field(rec, "algorithm") == "feddes"
    assert _field(rec, "algorithm.graph_k") == 5
    assert _field(rec, "exp.alpha") == 0.1
    assert _field(rec, "alpha") == 0.1          # bare -> experiment field


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
        _rec("feddes", 0, 0.60, graph_k=3), _rec("feddes", 1, 0.62, graph_k=3),
        _rec("feddes", 0, 0.80, graph_k=5), _rec("feddes", 1, 0.82, graph_k=5),
    ]}
    rows = _rows(by_algorithm, ["algorithm.graph_k"])
    assert set(rows) == {"feddes graph_k=3", "feddes graph_k=5"}
    assert rows["feddes graph_k=3"]["seeds"] == 2                 # 2 seeds per setting
    assert abs(rows["feddes graph_k=3"]["test_mean"] - 0.61) < 1e-9
    assert abs(rows["feddes graph_k=5"]["test_mean"] - 0.81) < 1e-9


def test_group_by_keeps_algorithms_separate():
    by_algorithm = {
        "feddes": [_rec("feddes", 0, 0.8, graph_k=5)],
        "local": [_rec("local", 0, 0.5)],
    }
    rows = _rows(by_algorithm, ["algorithm.graph_k"])
    # local has no graph_k -> its own row labelled with None; feddes distinct
    assert "feddes graph_k=5" in rows
    assert "local graph_k=None" in rows


def _cond_rec(algorithm, seed, *, dataset="cifar10", alpha=0.1, accs=(0.8, 0.7),
              algorithm_cfg=None):
    """Uses each algorithm's REAL config by default.

    The earlier version of these tests gave Local and FedDES the same synthetic
    algorithm config, which hid the fact that folding algorithm settings into the
    grouping key stopped the two from ever being paired.
    """
    if algorithm_cfg is None:
        from rigfl.experiment.registry import config_class
        algorithm_cfg = config_class(algorithm)().model_dump()
    return {"algorithm": algorithm,
            "config": {"experiment": {"dataset": dataset, "alpha": alpha, "seed": seed,
                                      "num_clients": 20},
                       "algorithm": algorithm_cfg},
            "result": _history(*accs)}


def test_algorithms_with_different_configs_share_an_experiment():
    """Local and FedDES configure differently by nature; that must not separate
    them, or win-rate has nothing to pair against."""
    from rigfl.experiment.collect import experiment_condition
    assert experiment_condition(_cond_rec("local", 0)) == experiment_condition(_cond_rec("feddes", 0))


def test_win_rate_is_computed_across_algorithms():
    rows = _rows({
        "local":  [_cond_rec("local", 0, accs=(0.9, 0.9))],
        "feddes": [_cond_rec("feddes", 0, accs=(0.5, 0.5))],
    })
    feddes = [k for k in rows if k.startswith("feddes")][0]
    assert "win" in rows[feddes] and rows[feddes]["win"] == 0.0     # paired, and loses


def test_client_model_pool_is_part_of_the_experiment_condition():
    from rigfl.experiment.collect import experiment_condition

    local = _cond_rec("local", 0)
    feddes = _cond_rec("feddes", 0)
    local["config"]["experiment"]["model_architectures"] = ["fedavg_cnn"]
    feddes["config"]["experiment"]["model_architectures"] = ["cifar_resnet18"]

    assert experiment_condition(local) != experiment_condition(feddes)
    rows = _rows({"local": [local], "feddes": [feddes]})
    feddes_row = next(value for key, value in rows.items()
                       if key.startswith("feddes"))
    assert "win" not in feddes_row


def test_equivalent_client_family_and_list_share_an_experiment_condition():
    from rigfl.experiment.collect import experiment_condition
    from rigfl.experiment.config import ExperimentConfig
    from rigfl.experiment.run import resolve_experiment_architectures

    family = resolve_experiment_architectures(
        ExperimentConfig(model_architecture_family="image_heterogeneous_3"),
        input_kind="image")
    explicit = resolve_experiment_architectures(
        ExperimentConfig(model_architectures=list(family.model_architectures)),
        input_kind="image")
    a = _cond_rec("local", 0)
    b = _cond_rec("feddes", 0)
    a["config"]["experiment"] = family.model_dump(mode="json")
    b["config"]["experiment"] = explicit.model_dump(mode="json")

    assert experiment_condition(a) == experiment_condition(b)


def test_experiments_are_not_averaged_together():
    """Two datasets in one directory are two experiments, not extra seeds."""
    from rigfl.experiment.collect import experiment_condition
    recs = {"local": [_cond_rec("local", 0), _cond_rec("local", 0, dataset="eICU")],
            "feddes": [_cond_rec("feddes", 0), _cond_rec("feddes", 0, dataset="eICU")]}
    assert len({experiment_condition(r) for rs in recs.values() for r in rs}) == 2
    rows = _rows(recs)
    assert len(rows) == 4
    assert all(s["seeds"] == 1 for s in rows.values())


def test_local_is_paired_within_its_own_experiment():
    rows = _rows({
        "local":  [_cond_rec("local", 0, accs=(0.9, 0.9))],                  # cifar only
        "feddes": [_cond_rec("feddes", 0, accs=(0.5, 0.5)),
                   _cond_rec("feddes", 0, dataset="eICU", accs=(0.5, 0.5))],
    })
    cifar = [k for k in rows if k.startswith("feddes") and "cifar10" in k][0]
    eicu = [k for k in rows if k.startswith("feddes") and "eICU" in k][0]
    assert rows[cifar]["win"] == 0.0        # paired with cifar's local
    assert "win" not in rows[eicu]          # no local ran in that experiment


def test_one_algorithm_swept_over_its_own_settings_gets_separate_rows():
    rows = _rows({"feddes": [
        _cond_rec("feddes", 0, algorithm_cfg={"graph_k": 3}),
        _cond_rec("feddes", 0, algorithm_cfg={"graph_k": 9}),
    ]})
    assert len(rows) == 2 and all("variant" in k for k in rows)


def test_sweep_task_rejects_an_unknown_setting(tmp_path):
    """Actually run the array-task path.

    The earlier version of this test only asserted that "graf_k" is not a model
    field, which was true before the repair too -- so it would have passed while
    the bug was live. This one calls run_task and requires it to refuse.
    """
    import json

    import pytest

    from rigfl.experiment.launch import run_task
    grid = tmp_path / "grid.jsonl"

    def write(algorithm_config):
        grid.write_text(json.dumps({"algorithm": "feddes",
                                    "experiment": {"dataset": "synthetic", "seed": 0},
                                    "algorithm_config": algorithm_config}) + "\n")

    write({"graf_k": 7})
    with pytest.raises(SystemExit, match="graf_k"):
        run_task(str(grid), 1, tmp_path, dry_run=True)

    write({"graph_k": 7})                      # the correct spelling still runs
    run_task(str(grid), 1, tmp_path, dry_run=True)


def test_group_by_still_separates_experiments():
    """--group-by says how to label rows, not that different datasets may be
    averaged together as extra seeds."""
    recs = {"feddes": [
        _cond_rec("feddes", 0, alpha=0.1, algorithm_cfg={"graph_k": 5}),
        _cond_rec("feddes", 1, alpha=0.1, algorithm_cfg={"graph_k": 5}),
        _cond_rec("feddes", 0, alpha=0.5, algorithm_cfg={"graph_k": 5}),
    ]}
    rows = _rows(recs, ["algorithm.graph_k"])
    assert len(rows) == 2, rows                       # alpha 0.1 and 0.5 stay apart
    assert sorted(s["seeds"] for s in rows.values()) == [1, 2]


def test_experiments_differing_only_in_an_unlabelled_field_stay_apart():
    """Rows were keyed by a display label built from a fixed field list, so two
    experiments differing only in batch size rendered identically and one
    overwrote the other -- while the collector announced two."""

    def rec(seed, batch):
        return {"algorithm": "feddes",
                "config": {"experiment": {"dataset": "cifar10", "alpha": 0.1,
                                          "num_clients": 20, "seed": seed, "batch": batch},
                           "algorithm": {"graph_k": 5}},
                "result": _history(.7, .6)}

    recs = {"feddes": [rec(0, 32), rec(1, 32), rec(0, 64)]}
    for rows in (_rows(recs), _rows(recs, ["algorithm.graph_k"])):
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
    """experiment_condition separated these, but the label was built from a
    different field list -- so both rows rendered identically and one
    overwrote the other, while the collector reported two experiments."""
    recs = {"local": [_es_rec(0, enabled=True, metric="accuracy", patience=5),
                      _es_rec(0, enabled=True, metric="accuracy", patience=20)]}
    for rows in (_rows(recs), _rows(recs, ["algorithm"])):
        assert len(rows) == 2, rows
        assert len(set(rows)) == 2, "row labels must be unique"
        assert any("patience=5" in k for k in rows)
        assert any("patience=20" in k for k in rows)


def test_the_three_condition_helpers_agree_on_their_fields():
    """They used to derive from different lists, which is how a field could
    separate two experiments while being invisible in their labels."""
    from rigfl.experiment.collect import (condition_fields, describe_condition,
                                          experiment_condition, varying_fields)
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
    from rigfl.experiment.config import ExperimentConfig, run_fingerprint
    off5 = ExperimentConfig(early_stopping={"enabled": False, "patience": 5})
    off20 = ExperimentConfig(early_stopping={"enabled": False, "patience": 20})
    assert run_fingerprint(off5, {}) == run_fingerprint(off20, {})
    on5 = ExperimentConfig(early_stopping={"enabled": True, "metric": "accuracy", "patience": 5})
    on20 = ExperimentConfig(early_stopping={"enabled": True, "metric": "accuracy", "patience": 20})
    assert run_fingerprint(on5, {}) != run_fingerprint(on20, {})
