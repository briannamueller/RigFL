"""Run identity is based on resolved partition facts, not storage locations."""

from __future__ import annotations

from rigfl.experiment.config import result_filename, run_fingerprint
from tests.helpers import resolved_experiment


def test_filename_uses_the_resolved_partition_identity():
    exp = resolved_experiment(dataset="cifar10", partition_id="partition-abc")
    name = result_filename(exp, "fedproto", "deadbeef")
    assert name == "cifar10_partition-abc_fedproto_seed0_deadbeef.json"


def test_run_identity_tracks_partition_but_not_its_storage_location():
    a = resolved_experiment(
        dataset="cifar10", partition_id="partition-abc", data_dir="/data/a"
    )
    b = resolved_experiment(
        dataset="cifar10", partition_id="partition-abc", data_dir="/data/b",
        dataset_config="/configs/elsewhere.yaml",
    )
    assert run_fingerprint(a, {}) == run_fingerprint(b, {})

    other = resolved_experiment(dataset="cifar10", partition_id="partition-other")
    assert run_fingerprint(a, {}) != run_fingerprint(other, {})


def test_backend_and_validation_fraction_are_part_of_identity():
    base = resolved_experiment(partition_id="same")
    other_backend = resolved_experiment(
        partition_id="same", data_backend="biosilo", partition_scheme=None,
        input_kind="temporal",
        input_spec={"input_kind": "temporal", "n_ts": 3, "n_static": 2,
                    "seq_len": 8},
    )
    other_validation = resolved_experiment(
        partition_id="same", validation_fraction=0.3
    )
    assert run_fingerprint(base, {}) != run_fingerprint(other_backend, {})
    assert run_fingerprint(base, {}) != run_fingerprint(other_validation, {})


def test_environment_flags_do_not_change_identity():
    a = resolved_experiment()
    b = resolved_experiment(
        wandb=True, device="cpu", out_dir="/tmp/x", quiet=False
    )
    assert run_fingerprint(a, {}) == run_fingerprint(b, {})


def test_algorithm_name_is_outside_fingerprint_and_inside_result_identity():
    exp = resolved_experiment()
    algorithm_config = {"local_epochs": 1, "lr": 0.01}

    fp = run_fingerprint(exp, algorithm_config)
    assert result_filename(exp, "local", fp) != result_filename(exp, "global", fp)
    assert result_filename(exp, "local", fp).endswith(f"local_seed0_{fp}.json")
    assert result_filename(exp, "global", fp).endswith(f"global_seed0_{fp}.json")
