"""Run identity is based on resolved partition facts, not storage locations."""

from __future__ import annotations

from rigfl.experiment.config import result_filename
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
