"""Integration with generated BioSilo partitions."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import yaml

biosilo = pytest.importorskip("biosilo")

from rigfl.data.biosilo import (
    biosilo_input_spec,
    generate_biosilo_partition,
)
from rigfl.data.builder import build_clients
from rigfl.data.config import dataset_settings
from rigfl.experiment.artifacts import validate_run_record
from rigfl.experiment.config import ExperimentConfig
from rigfl.experiment.launch import _validate_biosilo_partitions
from rigfl.experiment.registry import config_class
from rigfl.experiment.run import resolve_experiment_data
from rigfl.models.registry import instantiate_backbones
from tests.helpers import run_resolved


def _generate(root, *, n_inputs=1, with_groups=False):
    path = biosilo.generate(
        "synthetic",
        root=root,
        n_clients=3,
        n_per_client=32,
        n_classes=3,
        n_features=5,
        n_inputs=n_inputs,
        with_groups=with_groups,
        group_size=4,
        seed=7,
    )
    return biosilo.load("synthetic", root=root, partition=path.name)


def _parameters(*, n_inputs=1, with_groups=False):
    return {
        "n_clients": 3,
        "n_per_client": 32,
        "n_classes": 3,
        "n_features": 5,
        "n_inputs": n_inputs,
        "with_groups": with_groups,
        "group_size": 4,
        "seed": 7,
    }


def _dataset_config(
    tmp_path, *, n_inputs=1, with_groups=False, validation_fraction=0.25,
    split_seed=0,
):
    path = tmp_path / "datasets.yaml"
    config = {
        "datasets": {
            "biomedical": {
                "backend": "biosilo",
                "source_dataset": "synthetic",
                "parameters": _parameters(
                    n_inputs=n_inputs, with_groups=with_groups
                ),
                "validation_fraction": validation_fraction,
                "split_seed": split_seed,
            }
        }
    }
    path.write_text(yaml.safe_dump(config))
    return path


def _generate_configured(tmp_path, *, n_inputs=1, with_groups=False):
    config_path = _dataset_config(
        tmp_path, n_inputs=n_inputs, with_groups=with_groups
    )
    settings = dataset_settings("biomedical", config_path)
    handle, created = generate_biosilo_partition(settings, data_dir=tmp_path)
    return config_path, handle, created


def test_validation_split_preserves_biosilo_groups(tmp_path):
    handle = _generate(tmp_path, n_inputs=2, with_groups=True)
    _, input_spec = biosilo_input_spec(handle)
    backbones = instantiate_backbones(["temporal_gru"], input_spec=input_spec)
    clients = build_clients(
        handle.client,
        handle.num_clients,
        handle.num_classes,
        backbones,
        8,
        val_frac=0.25,
        batch=8,
        seed=11,
    )
    _, _, groups = handle.client(0, "train")
    train_indices = clients[0].train_loader.dataset.indices
    validation_indices = clients[0].val_loader.dataset.indices

    train_groups = set(np.asarray(groups)[train_indices].tolist())
    validation_groups = set(np.asarray(groups)[validation_indices].tolist())

    assert train_groups.isdisjoint(validation_groups)


def test_biosilo_split_seed_is_independent_of_training_seed(tmp_path):
    handle = _generate(tmp_path)
    _, input_spec = biosilo_input_spec(handle)
    backbones = instantiate_backbones(["tabular_mlp"], input_spec=input_spec)

    def indices(*, training_seed, split_seed):
        clients = build_clients(
            handle.client,
            handle.num_clients,
            handle.num_classes,
            backbones,
            8,
            val_frac=0.25,
            batch=8,
            seed=training_seed,
            split_seed=split_seed,
        )
        return (
            list(clients[0].train_loader.dataset.indices),
            list(clients[0].val_loader.dataset.indices),
            list(clients[0].test_loader.dataset.indices),
        )

    first = indices(training_seed=3, split_seed=11)
    other_training_seed = indices(training_seed=4, split_seed=11)
    other_split_seed = indices(training_seed=3, split_seed=12)

    assert first == other_training_seed
    assert first[2] == other_split_seed[2]
    assert first[:2] != other_split_seed[:2]


def test_biosilo_seed_overrides_resolve_the_matching_partition(
    tmp_path, monkeypatch
):
    config = _dataset_config(tmp_path, split_seed=2)
    experiment = ExperimentConfig(
        dataset="biomedical",
        dataset_config=str(config),
        data_dir=str(tmp_path),
        rounds=1,
        model_arch="tabular_mlp",
        partition_seed=11,
        split_seed=13,
        training_seed=17,
    )

    with pytest.raises(FileNotFoundError, match="--partition-seed 11"):
        resolve_experiment_data(experiment)

    from rigfl.data import generate

    monkeypatch.setattr(
        "sys.argv",
        [
            "generate",
            "--dataset",
            "biomedical",
            "--config",
            str(config),
            "--data-dir",
            str(tmp_path),
            "--partition-seed",
            "11",
            "--split-seed",
            "13",
        ],
    )
    generate.main()
    resolved, data = resolve_experiment_data(experiment)

    assert resolved.partition_seed == 11
    assert resolved.split_seed == 13
    assert resolved.training_seed == 17
    assert data.handle.path.is_dir()


def test_biosilo_sweep_preflight_checks_every_partition(tmp_path):
    config = _dataset_config(tmp_path, split_seed=2)
    task = {
        "algorithm": "local",
        "experiment": {
            "dataset": "biomedical",
            "dataset_config": str(config),
            "data_dir": str(tmp_path),
            "partition_seed": 11,
            "rounds": 1,
        },
        "algorithm_config": {},
    }

    with pytest.raises(SystemExit, match="BioSilo partition preflight failed"):
        _validate_biosilo_partitions([task])

    settings = dataset_settings("biomedical", config)
    settings = settings.model_copy(
        update={"parameters": {**settings.parameters, "seed": 11}}
    )
    generate_biosilo_partition(settings, data_dir=tmp_path)

    _validate_biosilo_partitions([task])


def test_biosilo_memmap_inputs_remain_lazy(tmp_path):
    path = biosilo.generate(
        "synthetic_memmap",
        root=tmp_path,
        n_clients=2,
        n_per_client=24,
        n_features=5,
        seed=13,
    )
    handle = biosilo.load(
        "synthetic_memmap", root=tmp_path, partition=path.name
    )
    _, input_spec = biosilo_input_spec(handle)
    clients = build_clients(
        handle.client,
        handle.num_clients,
        handle.num_classes,
        instantiate_backbones(["tabular_mlp"], input_spec=input_spec),
        8,
        val_frac=0.25,
        batch=4,
        seed=13,
    )

    assert isinstance(clients[0].train_loader.dataset.fields[0], np.memmap)
    inputs, _ = next(iter(clients[0].train_loader))
    assert inputs.shape == (4, 5)


@pytest.mark.parametrize(
    ("n_inputs", "architectures"),
    [
        (1, ["tabular_mlp"]),
        (2, ["temporal_gru"]),
    ],
)
def test_biosilo_runs_through_experiment_infrastructure(
    tmp_path, n_inputs, architectures
):
    config_path, handle, _ = _generate_configured(
        tmp_path, n_inputs=n_inputs, with_groups=n_inputs == 2
    )
    experiment = ExperimentConfig(
        dataset="biomedical",
        dataset_config=str(config_path),
        data_dir=str(tmp_path),
        model_arch=architectures[0],
        rounds=1,
        shared_dim=8,
        batch=8,
        quiet=True,
    )

    resolved, data = resolve_experiment_data(experiment)
    record = run_resolved(
        "local",
        resolved,
        config_class("local")(local_epochs=1),
        torch.device("cpu"),
        data,
    )

    assert resolved.data_backend == "biosilo"
    assert resolved.partition_scheme is None
    assert resolved.partition_id == handle.partition_id
    assert resolved.partition_seed == 7
    assert resolved.split_seed == 0
    assert record["partition"]["biosilo"]["dataset"] == "synthetic"
    assert [row["source_client_id"] for row in record["partition"]["per_client"]] == [
        "site-0", "site-1", "site-2"
    ]
    validate_run_record(record)
