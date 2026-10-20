"""Run identity is based on resolved partition facts, not storage locations."""

from __future__ import annotations

from rigfl.experiment.config import result_data_configuration_id, result_filename
from rigfl.experiment.identity import fingerprint
from rigfl.experiment.registry import run_identity
from tests.helpers import resolved_experiment


def _fingerprint(name, exp, algorithm_config):
    return fingerprint(run_identity(name, exp, algorithm_config))


def test_run_identity_tracks_partition_but_not_its_storage_location():
    a = resolved_experiment(
        dataset="cifar10", partition_id="partition-abc", data_dir="/data/a"
    )
    b = resolved_experiment(
        dataset="cifar10", partition_id="partition-abc", data_dir="/data/b",
        dataset_config="/configs/elsewhere.yaml",
    )
    assert _fingerprint("local", a, {}) == _fingerprint("local", b, {})

    other = resolved_experiment(dataset="cifar10", partition_id="partition-other")
    assert _fingerprint("local", a, {}) != _fingerprint("local", other, {})

    # the validation split is part of identity even within one partition
    first = resolved_experiment(partition_id="same", split_seed=3)
    second = resolved_experiment(partition_id="same", split_seed=4)
    assert _fingerprint("local", first, {}) != _fingerprint("local", second, {})


def test_environment_flags_do_not_change_identity():
    a = resolved_experiment()
    b = resolved_experiment(
        wandb=True, device="cpu", results_root="/tmp/x", quiet=False,
        estimate_flops=True,
    )
    assert _fingerprint("local", a, {}) == _fingerprint("local", b, {})


def test_algorithm_name_is_outside_fingerprint_and_inside_result_identity():
    exp = resolved_experiment()
    algorithm_config = {"local_epochs": 1, "lr": 0.01}

    fp = _fingerprint("local", exp, algorithm_config)
    assert result_filename(exp, "local", fp) != result_filename(exp, "global_ensemble", fp)
    assert result_filename(exp, "local", fp).endswith(f"local_{fp}_seed0.json")
    assert result_filename(exp, "global_ensemble", fp).endswith(f"global_ensemble_{fp}_seed0.json")


def _biosilo_record(*, seed: int, static_width: int, schema_version: int = 3):
    return {
        "config": {
            "experiment": {
                "dataset": "eicu_mortality_24h",
                "data_backend": "biosilo",
                "num_clients": 10,
                "num_classes": 2,
                "validation_fraction": 0.2,
                "input_kind": "temporal",
                "input_spec": {
                    "input_kind": "temporal",
                    "fields": [
                        {"name": "ts", "shape": [24, 54], "dtype": "float32"},
                        {
                            "name": "static",
                            "shape": [static_width],
                            "dtype": "float32",
                        },
                    ],
                },
            }
        },
        "partition": {
            "biosilo": {
                "dataset": "eICU",
                "source_manifest_version": schema_version,
                "settings": {
                    "task": "mortality_24h",
                    "num_clients": 10,
                    "seed": seed,
                },
            }
        },
    }


def test_biosilo_data_configuration_ignores_seed_derived_input_width():
    first = _biosilo_record(seed=0, static_width=120)
    second = _biosilo_record(seed=1, static_width=117)

    assert result_data_configuration_id(first) == result_data_configuration_id(second)


def test_biosilo_data_configuration_tracks_settings_and_schema():
    baseline = _biosilo_record(seed=0, static_width=120)
    different_settings = _biosilo_record(seed=1, static_width=117)
    different_settings["partition"]["biosilo"]["settings"]["task"] = "shock_4h"
    different_schema = _biosilo_record(
        seed=1, static_width=117, schema_version=4
    )

    baseline_id = result_data_configuration_id(baseline)
    assert baseline_id != result_data_configuration_id(different_settings)
    assert baseline_id != result_data_configuration_id(different_schema)
