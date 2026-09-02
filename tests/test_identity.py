"""Scheme-aware run identity: the fingerprint + result filename ignore fields
irrelevant to the partition scheme, so a natural (BioSilo) run is never tagged
with a meaningless alpha or a stale default num_clients.

Imports only rigfl.experiment.config (no torch), so it runs anywhere.
"""

from __future__ import annotations

from rigfl.experiment.config import ExperimentConfig, result_filename, run_fingerprint

NAT = dict(scheme="natural", dataset="eICU", partition="mortality_24h_n0_size_s1_abc")
DIR = dict(scheme="dirichlet", dataset="cifar10", alpha=0.1, num_clients=20)
GENERATED = dict(
    scheme="generated", dataset="cifar10", partition="partition-abc",
    num_clients=3, num_classes=10, alpha=0.5,
)


def test_generated_filename_uses_the_derived_partition_identity():
    name = result_filename(ExperimentConfig(**GENERATED), "fedproto", "deadbeef")
    assert name == "cifar10_partition-abc_fedproto_seed0_deadbeef.json"


def test_generated_run_identity_tracks_partition_but_not_its_storage_location():
    a = ExperimentConfig(**GENERATED, data_dir="/data/a")
    b = ExperimentConfig(**GENERATED, data_dir="/data/b",
                         dataset_config="/configs/elsewhere.yaml")
    assert run_fingerprint(a, {}) == run_fingerprint(b, {})

    other = ExperimentConfig(**{**GENERATED, "partition": "partition-other"})
    assert run_fingerprint(a, {}) != run_fingerprint(other, {})


def test_natural_filename_uses_partition_not_alpha():
    name = result_filename(ExperimentConfig(**NAT), "fedproto", "deadbeef")
    assert "a0.1" not in name
    assert name == "eICU_mortality_24h_n0_size_s1_abc_fedproto_seed0_deadbeef.json"


def test_dirichlet_filename_keeps_alpha():
    name = result_filename(ExperimentConfig(**DIR), "fedproto", "deadbeef")
    assert name == "cifar10_a0.1_fedproto_seed0_deadbeef.json"


def test_natural_fp_ignores_alpha_and_num_clients():
    a = ExperimentConfig(**NAT)
    b = ExperimentConfig(**{**NAT, "alpha": 0.9, "num_clients": 999})   # irrelevant to natural
    assert run_fingerprint(a, {}) == run_fingerprint(b, {})
    c = ExperimentConfig(**{**NAT, "partition": "other_partition"})     # identity IS the partition
    assert run_fingerprint(a, {}) != run_fingerprint(c, {})


def test_dirichlet_fp_ignores_partition_but_tracks_alpha():
    a = ExperimentConfig(**DIR)
    b = ExperimentConfig(**{**DIR, "partition": "irrelevant", "data_root": "/x"})
    assert run_fingerprint(a, {}) == run_fingerprint(b, {})
    c = ExperimentConfig(**{**DIR, "alpha": 0.5})
    assert run_fingerprint(a, {}) != run_fingerprint(c, {})


def test_env_flags_dont_change_identity():
    # device / out_dir / quiet / wandb don't affect the numbers, so a re-run that
    # only changes them must keep the same fingerprint (skip-done still matches).
    a = ExperimentConfig(**DIR)
    b = ExperimentConfig(**{**DIR, "wandb": True, "device": "cpu", "out_dir": "/tmp/x", "quiet": False})
    assert run_fingerprint(a, {}) == run_fingerprint(b, {})


def test_algorithm_name_is_outside_the_fingerprint_and_inside_result_identity():
    exp = ExperimentConfig(**DIR)
    algorithm_config = {"local_epochs": 1, "lr": 0.01}

    fp = run_fingerprint(exp, algorithm_config)
    assert result_filename(exp, "local", fp) != result_filename(exp, "global", fp)
    assert result_filename(exp, "local", fp).endswith(f"local_seed0_{fp}.json")
    assert result_filename(exp, "global", fp).endswith(f"global_seed0_{fp}.json")
