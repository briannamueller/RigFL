"""Integration with generated BioSilo partitions."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

biosilo = pytest.importorskip("biosilo")

from rigfl.data.biosilo import (biosilo_input_spec, build_biosilo_clients,
                                load_biosilo_partition)
from rigfl.data.builder import MultiTensor
from rigfl.data.config import BioSiloDatasetSettings
from rigfl.experiment.artifacts import validate_run_record
from rigfl.experiment.config import ExperimentConfig
from rigfl.experiment.registry import config_class
from rigfl.experiment.run import resolve_experiment_data, run_one
from rigfl.models.registry import instantiate_backbones, resolve_model_architectures


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


def _dataset_config(tmp_path, handle, validation_fraction=0.25):
    path = tmp_path / "datasets.yaml"
    path.write_text(
        "datasets:\n"
        "  biomedical:\n"
        "    backend: biosilo\n"
        "    source_dataset: synthetic\n"
        f"    partition: {handle.partition_id}\n"
        f"    data_root: {tmp_path}\n"
        f"    validation_fraction: {validation_fraction}\n"
    )
    return path


def test_biosilo_configuration_loads_an_exact_partition(tmp_path):
    handle = _generate(tmp_path)
    settings = BioSiloDatasetSettings(
        source_dataset="synthetic",
        partition=handle.partition_id,
        data_root=str(tmp_path),
    )

    loaded = load_biosilo_partition(settings)

    assert loaded.partition_id == handle.partition_id
    assert loaded.client_ids == ["site-0", "site-1", "site-2"]


def test_biosilo_input_forms_map_to_model_families(tmp_path):
    numeric = _generate(tmp_path)
    temporal = _generate(tmp_path, n_inputs=2)
    image = SimpleNamespace(inputs=[{
        "name": "x", "shape": [4, 240, 240], "dtype": "float32"
    }])

    assert biosilo_input_spec(numeric)[0] == "numeric"
    assert biosilo_input_spec(temporal)[0] == "temporal"
    image_kind, image_spec = biosilo_input_spec(image)
    image_model = instantiate_backbones(
        ["fedavg_cnn"], input_spec=image_spec
    )[0]()

    assert image_kind == "image"
    assert image_model(torch.randn(1, 4, 240, 240)).shape == (1, 512)


def test_temporal_models_accept_biosilo_multi_input_batches(tmp_path):
    handle = _generate(tmp_path, n_inputs=2, with_groups=True)
    input_kind, input_spec = biosilo_input_spec(handle)
    names = resolve_model_architectures(
        architecture_family=None,
        architectures=None,
        input_kind=input_kind,
    )
    backbones = instantiate_backbones(names, input_spec=input_spec)
    clients = build_biosilo_clients(
        handle,
        shared_dim=16,
        batch=8,
        seed=3,
        backbones=backbones,
        validation_fraction=0.25,
    )

    inputs, targets = next(iter(clients[0].train_loader))

    assert isinstance(inputs, MultiTensor)
    assert [model()(inputs).shape for model in backbones] == [
        (len(targets), 128),
        (len(targets), 128),
        (len(targets), 160),
    ]


def test_validation_split_preserves_biosilo_groups(tmp_path):
    handle = _generate(tmp_path, n_inputs=2, with_groups=True)
    _, input_spec = biosilo_input_spec(handle)
    backbones = instantiate_backbones(["temporal_gru"], input_spec=input_spec)
    clients = build_biosilo_clients(
        handle,
        shared_dim=8,
        batch=8,
        seed=11,
        backbones=backbones,
        validation_fraction=0.25,
    )
    _, _, groups = handle.client(0, "train")
    train_indices = clients[0].train_loader.dataset.indices
    validation_indices = clients[0].val_loader.dataset.indices

    train_groups = set(np.asarray(groups)[train_indices].tolist())
    validation_groups = set(np.asarray(groups)[validation_indices].tolist())

    assert train_groups.isdisjoint(validation_groups)


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
    clients = build_biosilo_clients(
        handle,
        shared_dim=8,
        batch=4,
        seed=13,
        backbones=instantiate_backbones(["tabular_mlp"], input_spec=input_spec),
        validation_fraction=0.25,
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
    handle = _generate(tmp_path, n_inputs=n_inputs, with_groups=n_inputs == 2)
    config_path = _dataset_config(tmp_path, handle)
    experiment = ExperimentConfig(
        dataset="biomedical",
        dataset_config=str(config_path),
        model_architectures=architectures,
        rounds=1,
        shared_dim=8,
        batch=8,
        quiet=True,
    )

    resolved, data = resolve_experiment_data(experiment)
    record = run_one(
        "local",
        resolved,
        config_class("local")(local_epochs=1),
        torch.device("cpu"),
        data=data,
    )

    assert resolved.data_backend == "biosilo"
    assert resolved.partition_scheme is None
    assert resolved.partition_id == handle.partition_id
    assert record["partition"]["biosilo"]["dataset"] == "synthetic"
    assert [row["source_client_id"] for row in record["partition"]["per_client"]] == [
        "site-0", "site-1", "site-2"
    ]
    validate_run_record(record)


def test_feddes_accepts_a_biosilo_multi_input_partition(tmp_path):
    pytest.importorskip("graphroute")
    handle = _generate(tmp_path, n_inputs=2, with_groups=True)
    experiment = ExperimentConfig(
        dataset="biomedical",
        dataset_config=str(_dataset_config(tmp_path, handle)),
        model_architectures=["temporal_gru"],
        rounds=1,
        shared_dim=8,
        batch=8,
        quiet=True,
    )
    resolved, data = resolve_experiment_data(experiment)

    record = run_one(
        "feddes",
        resolved,
        config_class("feddes")(
            base_epochs=1,
            base_split_mode="in_sample",
            gnn_arch="mlp",
            gnn_epochs=2,
            gnn_patience=1,
            calibrate=False,
            cache_dir="",
        ),
        torch.device("cpu"),
        data=data,
    )

    assert record["result"]["selection_views_supported"] == ["per-client"]
    validate_run_record(record)
