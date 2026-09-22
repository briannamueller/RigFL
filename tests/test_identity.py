"""Run identity is based on resolved partition facts, not storage locations."""

from __future__ import annotations

from rigfl.algorithms.feddes import FedDESConfig
from rigfl.experiment.config import result_filename, run_fingerprint
from rigfl.experiment.identity import fingerprint, run_identity_input
from rigfl.experiment.registry import (
    algorithm_run_fingerprint,
    ignored_experiment_fields,
    legacy_algorithm_run_fingerprint,
)
from tests.helpers import resolved_experiment


def test_filename_uses_the_resolved_partition_identity():
    exp = resolved_experiment(dataset="cifar10", partition_id="partition-abc")
    name = result_filename(exp, "fedproto", "deadbeef")
    assert name == "cifar10_partition-abc_fedproto_deadbeef_seed0.json"


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


def test_validation_fraction_is_part_of_identity():
    base = resolved_experiment(partition_id="same")
    other_validation = resolved_experiment(
        partition_id="same", validation_fraction=0.3
    )
    assert run_fingerprint(base, {}) != run_fingerprint(other_validation, {})


def test_split_seed_is_part_of_identity():
    first = resolved_experiment(partition_id="same", split_seed=3)
    second = resolved_experiment(partition_id="same", split_seed=4)

    assert run_fingerprint(first, {}) != run_fingerprint(second, {})


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
    assert result_filename(exp, "local", fp).endswith(f"local_{fp}_seed0.json")
    assert result_filename(exp, "global", fp).endswith(f"global_{fp}_seed0.json")


def test_historical_equivalent_value_keeps_identity_but_other_values_do_not():
    exp = resolved_experiment()
    historical = FedDESConfig(base_models_per_client="all").model_dump()
    historical.pop("base_models_per_client")
    old_input = run_identity_input(
        "feddes",
        exp.model_dump(),
        historical,
        ignored_experiment_fields=ignored_experiment_fields("feddes"),
        apply_historical_equivalence=False,
    )

    old_fingerprint = fingerprint(old_input)
    assert algorithm_run_fingerprint(
        "feddes",
        exp,
        FedDESConfig(base_models_per_client="all").model_dump(),
    ) == old_fingerprint
    assert algorithm_run_fingerprint(
        "feddes", exp, FedDESConfig().model_dump()
    ) == algorithm_run_fingerprint(
        "feddes",
        exp,
        FedDESConfig(base_models_per_client="assigned").model_dump(),
    )
    assert algorithm_run_fingerprint(
        "feddes", exp, FedDESConfig().model_dump()
    ) != old_fingerprint
    assert legacy_algorithm_run_fingerprint(
        "feddes",
        exp,
        FedDESConfig(base_models_per_client="all").model_dump(),
    ) != old_fingerprint
