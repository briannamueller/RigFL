"""Generated partition configuration, identity, persistence, and resolution."""

from __future__ import annotations

import json

import pytest
import torch
import torch.nn as nn

from rigfl.data import partitions
from rigfl.data.partitions import (
    build_partition_clients,
    generate_partition,
    load_partition,
    partition_fingerprint,
)
from rigfl.data.config import (
    FlowerDatasetSettings,
    dataset_settings,
    load_dataset_registry,
)
from rigfl.experiment.config import ExperimentConfig, run_fingerprint
from rigfl.experiment.run import resolve_experiment_data
from rigfl.algorithms.local import Local, LocalConfig
from rigfl.core.round import iterative


DATASET = "my_images"


def test_bundled_dataset_config_includes_the_starter_datasets():
    registry = load_dataset_registry()

    assert {
        "mnist",
        "fashion_mnist",
        "cifar10",
        "cifar100",
        "tiny_imagenet",
        "femnist",
        "paysim_fraud",
        "phishing_urls",
    } <= set(registry.datasets)
    assert registry.datasets["mnist"].source_dataset == "ylecun/mnist"
    assert (
        registry.datasets["fashion_mnist"].source_dataset
        == "zalando-datasets/fashion_mnist"
    )
    assert registry.datasets["cifar100"].data_transform == "cifar100"
    assert registry.datasets["tiny_imagenet"].data_transform == "tiny_imagenet"
    assert registry.datasets["femnist"].partition.client_limit == 20
    assert registry.datasets["paysim_fraud"].partition.partition_by == "BankID"
    assert registry.datasets["phishing_urls"].partition.partition_by == "client_id"
    assert registry.datasets["phishing_urls"].partition.client_limit == 5
    assert all(
        registry.datasets[name].partition.num_clients == 5
        for name in (
            "mnist",
            "fashion_mnist",
            "cifar10",
            "cifar100",
            "tiny_imagenet",
        )
    )


def test_synthetic_partitioner_defaults_to_five_clients():
    settings = FlowerDatasetSettings(source_dataset="organization/source-data")

    assert settings.partition.num_clients == 5


def _config(path, *, alpha=0.3):
    path.write_text(
        "datasets:\n"
        f"  {DATASET}:\n"
        "    backend: flower\n"
        "    source_dataset: organization/source-data\n"
        "    source_subset: one-configuration\n"
        "    partition:\n"
        "      scheme: dirichlet\n"
        "      num_clients: 2\n"
        f"      alpha: {alpha}\n"
        "      partition_seed: 7\n"
        "      train_per_client: 12\n"
        "      test_per_client: 6\n"
        "      val_frac: 0.25\n"
    )
    return path


def _fake_flower_backend(settings, output_directory):
    num_clients = getattr(settings.partition, "num_clients", None)
    if num_clients is None:
        partition_sizes = getattr(settings.partition, "partition_sizes", None)
        num_clients = len(partition_sizes) if partition_sizes is not None else 2
    clients = []
    for cid in range(num_clients):
        directory = output_directory / "clients" / f"client_{cid}"
        directory.mkdir(parents=True)
        summary = {"client_id": cid}
        sizes = {}
        label_counts = {}
        for split, n in (("train", 9), ("validation", 3), ("test", 6)):
            inputs = torch.rand(n, 3, 32, 32)
            targets = (torch.arange(n) + cid) % 3
            torch.save((inputs, targets), directory / f"{split}.pt")
            sizes[split] = n
            label_counts[split] = torch.bincount(targets, minlength=3).tolist()
        summary["sizes"] = sizes
        summary["label_counts"] = label_counts
        clients.append(summary)
    return {
        "backend": "flower",
        "source": {
            "dataset": settings.source_dataset,
            "subset": settings.source_subset,
            "splits": {"train": "train", "test": "test", "validation": None},
            "input_column": "image",
            "target_column": "label",
        },
        "task": "classification",
        "num_clients": num_clients,
        "input_spec": {"kind": "image", "shape": [3, 32, 32]},
        "target_spec": {
            "dtype": "int64",
            "shape": [],
            "num_classes": 3,
            "class_names": ["a", "b", "c"],
        },
        "clients": clients,
    }


def _generate(monkeypatch, config, data_dir):
    monkeypatch.setitem(partitions.BACKEND_GENERATORS, "flower", _fake_flower_backend)
    return generate_partition(DATASET, config_path=config, data_dir=data_dir)


def test_partition_fingerprint_is_stable_and_tracks_generation_settings(tmp_path):
    config = _config(tmp_path / "datasets.yaml", alpha=0.3)
    first = dataset_settings(DATASET, config)
    assert partition_fingerprint(DATASET, first) == partition_fingerprint(DATASET, first)

    _config(config, alpha=0.8)
    second = dataset_settings(DATASET, config)
    assert partition_fingerprint(DATASET, first) != partition_fingerprint(DATASET, second)


def test_partition_fingerprint_tracks_pipeline_version(monkeypatch, tmp_path):
    settings = dataset_settings(DATASET, _config(tmp_path / "datasets.yaml"))
    baseline = partition_fingerprint(DATASET, settings)

    monkeypatch.setattr(
        partitions,
        "PARTITION_PIPELINE_VERSION",
        partitions.PARTITION_PIPELINE_VERSION + 1,
    )

    assert baseline != partition_fingerprint(DATASET, settings)


def test_partition_fingerprint_tracks_merged_and_client_split_settings():
    def settings(
        merge_splits, validation_fraction=0.1, test_fraction=0.2, stratify=True
    ):
        return FlowerDatasetSettings(
            source_dataset="organization/source-data",
            source_splits={"merge_splits": merge_splits},
            client_split={
                "validation_fraction": validation_fraction,
                "test_fraction": test_fraction,
                "stratify": stratify,
            },
            partition={"scheme": "iid", "num_clients": 2},
        )

    baseline = partition_fingerprint(DATASET, settings(["train", "test"]))

    assert baseline == partition_fingerprint(DATASET, settings(["train", "test"]))
    assert baseline != partition_fingerprint(DATASET, settings(["test", "train"]))
    assert baseline != partition_fingerprint(
        DATASET, settings(["train", "test"], validation_fraction=0.15)
    )
    assert baseline != partition_fingerprint(
        DATASET, settings(["train", "test"], test_fraction=0.25)
    )
    assert baseline != partition_fingerprint(
        DATASET, settings(["train", "test"], stratify=False)
    )


def test_partition_fingerprint_tracks_source_and_transform_settings():
    def settings(*, revision="revision-1", client_limit=2, transform="image"):
        return FlowerDatasetSettings(
            source_dataset="organization/source-data",
            source_revision=revision,
            source_splits={"merge_splits": ["train"]},
            client_split={
                "validation_fraction": 0.1,
                "test_fraction": 0.2,
                "stratify": False,
            },
            input_column="image",
            target_column="label",
            data_transform=transform,
            partition={
                "scheme": "natural_id",
                "partition_by": "writer",
                "client_limit": client_limit,
            },
        )

    baseline = partition_fingerprint(DATASET, settings())

    assert baseline != partition_fingerprint(DATASET, settings(revision="revision-2"))
    assert baseline != partition_fingerprint(DATASET, settings(client_limit=3))
    assert baseline != partition_fingerprint(DATASET, settings(transform="auto"))


def test_partition_count_can_be_derived_from_partition_sizes(monkeypatch, tmp_path):
    config = tmp_path / "datasets.yaml"
    config.write_text(
        "datasets:\n"
        f"  {DATASET}:\n"
        "    backend: flower\n"
        "    source_dataset: organization/source-data\n"
        "    partition:\n"
        "      scheme: size\n"
        "      partition_sizes: [10, 20, 30]\n"
    )

    artifact, _ = _generate(monkeypatch, config, tmp_path / "data")

    assert artifact.manifest["num_clients"] == 3
    assert artifact.settings.partition.partition_sizes == [10, 20, 30]


def test_partition_count_can_be_derived_from_natural_ids(monkeypatch, tmp_path):
    config = tmp_path / "datasets.yaml"
    config.write_text(
        "datasets:\n"
        f"  {DATASET}:\n"
        "    backend: flower\n"
        "    source_dataset: organization/source-data\n"
        "    partition:\n"
        "      scheme: natural_id\n"
        "      partition_by: patient_id\n"
    )

    artifact, _ = _generate(monkeypatch, config, tmp_path / "data")

    assert artifact.manifest["num_clients"] == 2
    assert artifact.settings.partition.partition_by == "patient_id"


def test_generation_dispatches_by_backend_and_reuses_partition(monkeypatch, tmp_path):
    config = _config(tmp_path / "datasets.yaml")
    artifact, created = _generate(monkeypatch, config, tmp_path / "data")
    assert created is True
    assert artifact.dataset == DATASET
    assert artifact.path.name == f"partition_{artifact.partition_id}"
    assert (artifact.path / "clients" / "client_0" / "train.pt").exists()
    manifest = json.loads((artifact.path / "manifest.json").read_text())
    assert manifest["pipeline_version"] == partitions.PARTITION_PIPELINE_VERSION
    assert manifest["source"]["dataset"] == "organization/source-data"
    assert "validation" not in manifest["source"]["splits"]
    assert manifest["num_clients"] == 2
    assert manifest["clients"][0]["sizes"]["validation"] == 3
    assert manifest["clients"][0]["label_counts"]["train"] == [3, 3, 3]
    manifest_text = (artifact.path / "manifest.json").read_text()
    assert manifest_text.index('"dataset"') < manifest_text.index('"clients"')
    assert '"sizes": {"train": 9, "validation": 3, "test": 6}' in manifest_text
    assert '"shape": [3, 32, 32]' in manifest_text

    reused, created = generate_partition(
        DATASET, config_path=config, data_dir=tmp_path / "data"
    )
    assert created is False
    assert reused.path == artifact.path


def test_experiment_uses_alias_to_resolve_partition(monkeypatch, tmp_path):
    config = _config(tmp_path / "datasets.yaml")
    generated, _ = _generate(monkeypatch, config, tmp_path / "data")
    exp = ExperimentConfig(
        dataset=DATASET,
        dataset_config=str(config),
        data_dir=str(tmp_path / "data"),
        rounds=2,
        seed=11,
    )
    resolved, data = resolve_experiment_data(exp)
    assert data.artifact.path == generated.path
    assert resolved.data_backend == "flower"
    assert resolved.partition_scheme == "dirichlet"
    assert resolved.partition_id == generated.partition_id
    assert resolved.num_clients == 2
    assert resolved.num_classes == 3
    assert resolved.validation_fraction == 0.25

    other_seed, _ = resolve_experiment_data(exp.model_copy(update={"seed": 12}))
    assert other_seed.partition_id == resolved.partition_id
    assert run_fingerprint(other_seed, LocalConfig().model_dump()) != run_fingerprint(
        resolved, LocalConfig().model_dump()
    )


def test_experiment_uses_merged_client_validation_fraction(monkeypatch, tmp_path):
    config = tmp_path / "datasets.yaml"
    config.write_text(
        "datasets:\n"
        f"  {DATASET}:\n"
        "    backend: flower\n"
        "    source_dataset: organization/source-data\n"
        "    source_splits:\n"
        "      merge_splits: [train, test]\n"
        "    client_split:\n"
        "      validation_fraction: 0.15\n"
        "      test_fraction: 0.2\n"
        "      stratify: true\n"
        "    partition:\n"
        "      scheme: iid\n"
        "      num_clients: 2\n"
    )
    _generate(monkeypatch, config, tmp_path / "data")

    resolved, _ = resolve_experiment_data(
        ExperimentConfig(
            dataset=DATASET,
            dataset_config=str(config),
            data_dir=str(tmp_path / "data"),
        )
    )

    assert resolved.validation_fraction == 0.15


def test_partition_settings_are_not_accepted_in_experiment():
    with pytest.raises(Exception, match="alpha"):
        ExperimentConfig(alpha=0.9)
    with pytest.raises(Exception, match="num_clients"):
        ExperimentConfig(num_clients=5)


class _Backbone(nn.Module):
    out_dim = 4

    def forward(self, x):
        return x.flatten(1)[:, :4]


def test_generated_partition_builds_clients_without_repartitioning(monkeypatch, tmp_path):
    config = _config(tmp_path / "datasets.yaml")
    _generate(monkeypatch, config, tmp_path / "data")
    artifact = load_partition(DATASET, config_path=config, data_dir=tmp_path / "data")
    clients = build_partition_clients(
        artifact, shared_dim=4, batch=4, backbones=[_Backbone]
    )
    assert len(clients) == 2
    assert len(clients[0].train_loader.dataset) == 9
    assert len(clients[0].val_loader.dataset) == 3
    assert len(clients[0].test_loader.dataset) == 6


def test_generated_partition_can_skip_unused_client_models(monkeypatch, tmp_path):
    config = _config(tmp_path / "datasets.yaml")
    artifact, _ = _generate(monkeypatch, config, tmp_path / "data")

    clients = build_partition_clients(
        artifact, shared_dim=4, batch=4, build_models=False
    )

    assert all(client.model is None for client in clients)


def test_evaluation_frequency_does_not_change_training(monkeypatch, tmp_path):
    config = _config(tmp_path / "datasets.yaml")
    artifact, _ = _generate(monkeypatch, config, tmp_path / "data")

    def trained_state(eval_gap):
        torch.manual_seed(19)
        clients = build_partition_clients(
            artifact, shared_dim=4, batch=4, seed=19, backbones=[_Backbone]
        )
        iterative(
            Local(LocalConfig(local_epochs=1, lr=0.05)),
            clients,
            num_rounds=3,
            device=torch.device("cpu"),
            num_classes=3,
            eval_gap=eval_gap,
            verbose=False,
        )
        return [
            {
                name: value.detach().clone()
                for name, value in client.model.state_dict().items()
            }
            for client in clients
        ]

    every_round = trained_state(1)
    final_round = trained_state(3)

    assert all(
        torch.equal(every_round[cid][name], final_round[cid][name])
        for cid in range(len(every_round))
        for name in every_round[cid]
    )


def test_generated_partition_runs_through_experiment_infrastructure(monkeypatch, tmp_path):
    from rigfl.experiment.artifacts import validate_run_record
    from rigfl.experiment.registry import config_class
    from rigfl.experiment.run import run_one

    config = _config(tmp_path / "datasets.yaml")
    generated, _ = _generate(monkeypatch, config, tmp_path / "data")
    exp = ExperimentConfig(
        dataset=DATASET,
        dataset_config=str(config),
        data_dir=str(tmp_path / "data"),
        model="fedavg_cnn",
        rounds=1,
        shared_dim=8,
        batch=4,
        quiet=True,
    )
    record = run_one("local", exp, config_class("local")(), torch.device("cpu"))
    assert record["config"]["experiment"]["partition_id"] == generated.partition_id
    assert record["config"]["experiment"]["partition_scheme"] == "dirichlet"
    assert set(record["result"]["evaluation_history"]["clients"]) == {"0", "1"}
    validate_run_record(record)


@pytest.mark.parametrize("scheme", [
    "continuous", "dirichlet", "distribution", "exponential",
    "grouped_natural_id", "iid", "inner_dirichlet", "linear",
    "natural_id", "pathological", "shard", "size", "square",
])
def test_every_flower_partitioner_resolves_for_an_experiment(
    scheme, monkeypatch, tmp_path
):
    """Generating a supported partitioner must also make it runnable."""
    import yaml
    from tests.test_flower_data import PARTITIONER_CASES

    config = tmp_path / "datasets.yaml"
    config.write_text(yaml.safe_dump({
        "datasets": {
            DATASET: {
                "backend": "flower",
                "source_dataset": "organization/source-data",
                "partition": {
                    "scheme": scheme,
                    **PARTITIONER_CASES[scheme],
                },
            },
        },
    }))
    _generate(monkeypatch, config, tmp_path / "data")
    resolved, data = resolve_experiment_data(ExperimentConfig(
        dataset=DATASET,
        dataset_config=str(config),
        data_dir=str(tmp_path / "data"),
    ))

    assert resolved.data_backend == "flower"
    assert resolved.partition_scheme == scheme
    assert resolved.partition_id == data.artifact.partition_id
