"""Grouping and Local pairing for result reports."""

from __future__ import annotations

import csv
import json

import pytest

from rigfl.experiment import collect as collect_module
from rigfl.experiment.collect import (
    _rows_by_algorithm,
)
from rigfl.experiment.collect import main as collect_main
from tests.helpers import resolved_experiment

# Selection policy is fixed here on purpose: these tests are about grouping and
# pairing, not about which metric selects.
_SEL = {
    "metric": "accuracy",
    "view": "shared",
    "aggregation": "uniform",
}


def _history(*client_values):
    """A one-round result whose clients hold the given values."""
    clients = {
        str(i): {"validation": {"accuracy": [v], "balanced_accuracy": [v]},
                 "test": {"accuracy": [v], "balanced_accuracy": [v]}}
        for i, v in enumerate(client_values)
    }
    counts = {s: {str(i): [10] for i in range(len(client_values))}
              for s in ("validation", "test")}
    return {"evaluation_history": {"evaluation_rounds": [0], "clients": clients,
                                   "client_sample_counts": counts}}


def _rows(by_algorithm, **options):
    return _rows_by_algorithm(by_algorithm, _SEL["metric"], view=_SEL["view"],
                           aggregation=_SEL["aggregation"], **options)


def _rec(algorithm, seed, result_acc, **algorithm_cfg):
    return {
        "algorithm": algorithm,
        "config": {"algorithm": algorithm_cfg,
                   "experiment": {"batch": 32, "training_seed": seed}},
        "result": _history(result_acc),
    }


def _fedprox_config(mu):
    return {"mu": mu}


def test_hyperparameter_sweep_makes_one_row_per_setting():
    by_algorithm = {"fedprox": [
        _rec("fedprox", 0, 0.60, **_fedprox_config(3)),
        _rec("fedprox", 1, 0.62, **_fedprox_config(3)),
        _rec("fedprox", 0, 0.80, **_fedprox_config(5)),
        _rec("fedprox", 1, 0.82, **_fedprox_config(5)),
    ]}
    rows = _rows(by_algorithm)
    assert set(rows) == {"fedprox mu=3", "fedprox mu=5"}
    assert rows["fedprox mu=3"]["seeds"] == 2
    assert abs(rows["fedprox mu=3"]["test_mean"] - 0.61) < 1e-9
    assert abs(rows["fedprox mu=5"]["test_mean"] - 0.81) < 1e-9


def _cond_rec(algorithm, seed, *, dataset="cifar10", partition_id="partition-a",
              accs=(0.8, 0.7),
              algorithm_cfg=None):
    """A run record for one experiment condition; algorithm config defaults to the validated defaults."""
    if algorithm_cfg is None:
        from rigfl.experiment.registry import config_class
        algorithm_cfg = config_class(algorithm)().model_dump()
    return {"algorithm": algorithm,
            "config": {"experiment": {
                "dataset": dataset, "partition_id": partition_id, "training_seed": seed,
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


def test_client_model_pool_is_part_of_the_experiment_condition():
    from rigfl.experiment.collect import experiment_condition

    local = _cond_rec("local", 0)
    fedprox = _cond_rec("fedprox", 0)
    local["config"]["experiment"]["model_arch"] = "fedavg_cnn"
    fedprox["config"]["experiment"]["model_arch"] = "cifar_resnet18"

    assert experiment_condition(local) != experiment_condition(fedprox)
    rows = _rows({"local": [local], "fedprox": [fedprox]})
    fedprox_row = next(value for key, value in rows.items()
                       if key.startswith("fedprox"))
    assert "negative_transfer" not in fedprox_row


def test_experiments_are_not_averaged_together():
    """Two datasets in one directory are two experiments, not extra seeds."""
    from rigfl.experiment.collect import experiment_condition
    recs = {"local": [_cond_rec("local", 0), _cond_rec("local", 0, dataset="mnist")],
            "fedprox": [_cond_rec("fedprox", 0), _cond_rec("fedprox", 0, dataset="mnist")]}
    assert len({experiment_condition(r) for rs in recs.values() for r in rs}) == 2
    rows = _rows(recs)
    assert len(rows) == 4
    assert all(s["seeds"] == 1 for s in rows.values())

    # an unlabelled field like batch size still separates experiments
    batch64 = _cond_rec("fedprox", 0)
    batch64["config"]["experiment"]["batch"] = 64
    assert experiment_condition(batch64) != experiment_condition(_cond_rec("fedprox", 0))


def test_local_is_paired_within_its_own_experiment():
    # Local and FedProx have different algorithm configs but still pair
    rows = _rows({
        "local":  [_cond_rec("local", 0, accs=(0.9, 0.9))],                  # cifar only
        "fedprox": [_cond_rec("fedprox", 0, accs=(0.5, 0.5)),
                   _cond_rec("fedprox", 0, dataset="mnist", accs=(0.5, 0.5))],
    })
    cifar = next(k for k in rows if k.startswith("fedprox") and "cifar10" in k)
    mnist = next(k for k in rows if k.startswith("fedprox") and "mnist" in k)
    transfer = rows[cifar]["negative_transfer"]
    assert transfer["negative_transfer_rate"]["estimate"] == 1.0
    assert transfer["negative_transfer_magnitude"]["estimate"] == pytest.approx(0.4)
    assert "negative_transfer" not in rows[mnist]  # no Local run in that experiment


def test_gain_analysis_pairs_best_validation_configurations():
    def rec(algorithm, lr, validation, test):
        record = _cond_rec(algorithm, 0, algorithm_cfg={"local_epochs": 1, "lr": lr})
        for client in record["result"]["evaluation_history"]["clients"].values():
            client["validation"]["accuracy"] = [validation]
            client["test"]["accuracy"] = [test]
        return record

    def chosen(test_values, best_only=False):
        rows = _rows({
            "fedavg": [rec("fedavg", 0.01, 0.8, test_values[2]),
                       rec("fedavg", 0.05, 0.5, test_values[3])],
            "local": [rec("local", 0.01, 0.6, test_values[0]),
                      rec("local", 0.05, 0.7, test_values[1])],
        }, best_only=best_only)
        compared = [row["negative_transfer"] for row in rows.values()
                    if "negative_transfer" in row]
        assert len(compared) == 1
        shown = [(row["algorithm"], row["configuration"]) for row in rows.values()]
        return shown, (compared[0]["configuration"], compared[0]["local_configuration"])

    for test_values in ((0.5, 0.5, 0.5, 0.5), (0.9, 0.1, 0.1, 0.9)):
        # Local first, then each method's configurations best validation first
        shown, pair = chosen(test_values)
        assert shown == [("local", "lr=0.05"), ("local", "lr=0.01"),
                         ("fedavg", "lr=0.01"), ("fedavg", "lr=0.05")]
        assert pair == ("lr=0.01", "lr=0.05")
        # --best keeps exactly the configurations the gain analysis pairs
        shown, pair = chosen(test_values, best_only=True)
        assert shown == [("local", "lr=0.05"), ("fedavg", "lr=0.01")]
        assert pair == ("lr=0.01", "lr=0.05")


def test_collect_labels_client_level_gain_analysis(tmp_path, monkeypatch):
    monkeypatch.setattr(
        collect_module,
        "load_results",
        lambda *_args, **_kwargs: {
            "local": [_cond_rec("local", 0)],
            "fedavg": [_cond_rec("fedavg", 0)],
        },
    )
    config = tmp_path / "reporting.yaml"
    config.write_text(
        "version: 1\nfilters:\n  main_results:\n"
        "    algorithm: [local, fedavg]\n"
    )

    report = tmp_path / "main.csv"
    collect_main([
        "--results-root", str(tmp_path), "--config", str(config),
        "--filter", "main_results", "--save", str(report),
    ])

    assert report.exists()
    transfer = tmp_path / "main_client_level_gain_analysis.csv"
    assert "pooled_client_difference_sd" in transfer.read_text().splitlines()[0]


def test_null_setting_joins_every_condition_it_fits(tmp_path, monkeypatch, capsys):
    def rec(algorithm, rounds, acc):
        record = _cond_rec(algorithm, 0, accs=(acc, acc))
        record["config"]["experiment"]["rounds"] = rounds
        return record

    records = {
        "local": [rec("local", 3, 0.5), rec("local", 5, 0.7)],
        "fedavg": [rec("fedavg", None, 0.6)],
    }
    rows = _rows(records)
    gains = {
        row["condition"]: row["negative_transfer"]["mean_gain"]["estimate"]
        for row in rows.values() if row["algorithm"] == "fedavg"
    }
    assert gains == pytest.approx({"rounds=3": 0.1, "rounds=5": -0.1})

    monkeypatch.setattr(collect_module, "load_results", lambda *_a, **_k: records)
    config = tmp_path / "reporting.yaml"
    config.write_text("version: 1\n")
    for records_ in records.values():
        for index, record in enumerate(records_):
            record["_source_file"] = f"{record['algorithm']}{index}.json"
    collect_main(["--results-root", str(tmp_path), "--config", str(config),
                  "--save", str(tmp_path / "s.csv")])
    assert "3 configuration(s), 3 completed replicate(s)" in capsys.readouterr().out

    summary = list(csv.DictReader((tmp_path / "s.csv").open()))
    assert [row["rounds"] for row in summary if row["algorithm"] == "fedavg"] == [""]
    gain = csv.DictReader((tmp_path / "s_client_level_gain_analysis.csv").open())
    assert sorted(row["rounds"] for row in gain) == ["3", "5"]


def test_study_selects_only_its_evaluated_runs(tmp_path):
    (tmp_path / "study.json").write_text(json.dumps({
        "kind": "rigfl.tuning_study",
        "evaluations": [{"candidate_id": 0, "source_results": ["hpo.json"]}],
    }))
    hpo = {"_source_file": "hpo.json"}
    sweep = {"_source_file": "sweep.json"}

    assert collect_module.records_for_study([hpo, sweep], tmp_path) == [hpo]


def test_loader_can_skip_invalid_file_without_losing_valid_runs(tmp_path, monkeypatch):
    (tmp_path / "broken.json").write_text("{")
    (tmp_path / "valid.json").write_text(json.dumps({
        "algorithm": "fedavg", "config": {"experiment": {"dataset": "cifar10"}}
    }))
    monkeypatch.setattr(collect_module, "is_run_result", lambda _record: True)
    monkeypatch.setattr(collect_module, "validate_run_record", lambda _record, path: None)

    with pytest.raises(SystemExit, match="invalid result file"):
        collect_module.load_results(tmp_path, None)

    invalid = []
    rows = collect_module.load_results(tmp_path, None, ignore_invalid=True, invalid=invalid)
    assert len(rows["fedavg"]) == 1
    assert invalid[0][0] == "broken.json"


def test_zipped_data_and_training_replicates_form_one_result_row():
    by_algorithm = {"local": [], "fedprox": []}
    for seed in range(3):
        for algorithm, accs, cfg in (("local", (0.5, 0.5), None),
                                     ("fedprox", (0.6, 0.4), _fedprox_config(5))):
            record = _cond_rec(
                algorithm,
                seed,
                partition_id=f"partition-{seed}",
                accs=accs,
                algorithm_cfg=cfg,
            )
            experiment = record["config"]["experiment"]
            experiment["partition_seed"] = seed
            experiment["split_seed"] = seed
            settings = record["partition"]["generated"]["settings"]["partition"]
            settings["partition_seed"] = seed
            settings["split_seed"] = seed
            settings.pop("configuration")
            by_algorithm[algorithm].append(record)

    rows = _rows(by_algorithm)

    assert len(rows) == 2
    summary = rows["fedprox"]
    assert summary["seeds"] == 3
    assert summary["independent_replicates"] is True
    assert summary["test_ci"] is not None
    assert summary["negative_transfer"]["negative_transfer_rate"]["ci_low"] is not None


@pytest.mark.parametrize("drop_one", [False, True])
def test_crossed_collection_omits_performance_and_transfer_intervals(drop_one):
    by_algorithm = {"local": [], "fedprox": []}
    for partition_seed in range(2):
        for split_seed in range(2):
            for training_seed in range(2):
                partition_id = (
                    f"partition-{partition_seed}-{split_seed}"
                )
                for algorithm, accuracy in (("local", 0.5), ("fedprox", 0.6)):
                    record = _cond_rec(
                        algorithm,
                        training_seed,
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
        by_algorithm["fedprox"].pop()

    fedprox = _rows(by_algorithm)["fedprox"]

    assert fedprox["runs"] == (7 if drop_one else 8)
    assert fedprox["seeds"] == 2
    assert fedprox["test_ci"] is None
    transfer = fedprox["negative_transfer"]
    assert transfer["negative_transfer_rate"]["estimate"] == 0.0
    assert transfer["negative_transfer_rate"]["ci_low"] is None


def _es_rec(seed, **early_stopping):
    rec = _cond_rec("local", seed)
    rec["config"]["experiment"]["early_stopping"] = early_stopping
    return rec


def test_rows_are_uniquely_labelled_when_only_early_stopping_differs():
    """Early-stopping differences appear in unique result-row labels."""
    recs = {"local": [_es_rec(0, enabled=True, metric="accuracy", patience=5),
                      _es_rec(0, enabled=True, metric="accuracy", patience=20)]}
    rows = _rows(recs)
    assert len(rows) == 2, rows
    assert len(set(rows)) == 2, "row labels must be unique"
    assert any("patience=5" in k for k in rows)
    assert any("patience=20" in k for k in rows)

    from rigfl.experiment.collect import experiment_condition
    assert experiment_condition(_es_rec(0, enabled=True, metric="accuracy")) != \
           experiment_condition(_es_rec(0, enabled=True, metric="balanced_accuracy"))


def test_disabled_early_stopping_ignores_inactive_settings_for_run_identity():
    from rigfl.experiment.identity import fingerprint
    from rigfl.experiment.registry import run_identity

    def _fingerprint(name, exp, algorithm_config):
        return fingerprint(run_identity(name, exp, algorithm_config))

    off5 = resolved_experiment(early_stopping={"enabled": False, "patience": 5})
    off20 = resolved_experiment(early_stopping={"enabled": False, "patience": 20})
    assert _fingerprint("local", off5, {}) == _fingerprint("local", off20, {})
    assert len(_rows({"local": [_es_rec(0, enabled=False, patience=5),
                                _es_rec(1, enabled=False, patience=20)]})) == 1
    on5 = resolved_experiment(early_stopping={"enabled": True, "metric": "accuracy", "patience": 5})
    on20 = resolved_experiment(early_stopping={"enabled": True, "metric": "accuracy", "patience": 20})
    assert _fingerprint("local", on5, {}) != _fingerprint("local", on20, {})
