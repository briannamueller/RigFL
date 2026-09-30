"""Generated partition configuration, identity, persistence, and resolution."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from rigfl.algorithms.local import Local, LocalConfig
from rigfl.core.round import iterative
from rigfl.data import partitions
from rigfl.data.config import (
    FlowerDatasetSettings,
    dataset_settings,
    inactive_replicate_seed_fields,
    load_dataset_registry,
)
from rigfl.data.partitions import (
    build_partition_clients,
    generate_partition,
    load_partition,
    partition_fingerprint,
)
from rigfl.experiment.config import ExperimentConfig
from rigfl.experiment.identity import fingerprint
from rigfl.experiment.launch import expand
from rigfl.experiment.registry import run_identity
from rigfl.experiment.run import resolve_experiment_data
from tests.helpers import STARTER_DATASETS, run_resolved

DATASET = "my_images"


def _fingerprint(name, exp, algorithm_config):
    return fingerprint(run_identity(name, exp, algorithm_config))


def test_bundled_dataset_config_includes_the_starter_datasets():
    registry = load_dataset_registry(STARTER_DATASETS)

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


def test_sweep_rejects_varied_data_seeds_that_cannot_change_the_data(tmp_path):
    config = tmp_path / "datasets.yaml"
    config.write_text(
        "datasets:\n"
        "  stable:\n"
        "    backend: flower\n"
        "    source_dataset: organization/source-data\n"
        "    source_splits:\n"
        "      train: train\n"
        "      validation: validation\n"
        "      test: test\n"
        "    partition:\n"
        "      scheme: natural_id\n"
        "      partition_by: client_id\n"
        "      shuffle: false\n"
        "      train_per_client: null\n"
        "      test_per_client: null\n"
    )
    spec = {
        "base": {
            "dataset": "stable",
            "dataset_config": str(config),
            "rounds": 1,
        },
        "sweep": {"algorithm": ["local"], "partition_seed": [0, 1]},
    }

    assert inactive_replicate_seed_fields(dataset_settings("stable", config)) == {
        "partition_seed",
        "split_seed",
    }
    with pytest.raises(SystemExit, match="does not use the varied replicate"):
        expand(spec)


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
        "      split_seed: 13\n"
        "      train_per_client: 12\n"
        "      test_per_client: 6\n"
        "      val_frac: 0.25\n"
    )
    return path


def _fake_flower_backend(settings, output_directory):
    num_clients = settings.partition.num_clients
    clients = []
    for cid in range(num_clients):
        directory = output_directory / "clients" / f"client_{cid}"
        directory.mkdir(parents=True)
        summary = {"client_id": cid}
        sizes = {}
        label_counts = {}
        for split, n in (("train", 12), ("test", 6)):
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
    monkeypatch.setattr(partitions, "generate_flower_partition", _fake_flower_backend)
    return generate_partition(DATASET, config_path=config, data_dir=data_dir)


def test_partition_fingerprint_is_stable_and_tracks_generation_settings(
    monkeypatch, tmp_path
):
    config = _config(tmp_path / "datasets.yaml", alpha=0.3)
    first = dataset_settings(DATASET, config)
    baseline = partition_fingerprint(DATASET, first)
    assert baseline == partition_fingerprint(DATASET, first)

    _config(config, alpha=0.8)
    second = dataset_settings(DATASET, config)
    assert baseline != partition_fingerprint(DATASET, second)

    monkeypatch.setattr(
        partitions,
        "PARTITION_PIPELINE_VERSION",
        partitions.PARTITION_PIPELINE_VERSION + 1,
    )
    assert baseline != partition_fingerprint(DATASET, first)


def test_partition_fingerprint_ignores_the_validation_split_settings():
    first = FlowerDatasetSettings(
        source_dataset="organization/source-data",
        partition={"scheme": "iid", "num_clients": 2, "split_seed": 3},
    )
    second = first.model_copy(
        update={
            "partition": first.partition.model_copy(
                update={"split_seed": 4, "val_frac": 0.3}
            )
        }
    )

    assert partition_fingerprint(DATASET, first) == partition_fingerprint(
        DATASET, second
    )

    merged = FlowerDatasetSettings(
        source_dataset="organization/source-data",
        source_splits={"merge_splits": ["train", "test"]},
        client_split={
            "validation_fraction": 0.1,
            "test_fraction": 0.2,
            "stratify": True,
        },
        partition={"scheme": "iid", "num_clients": 2},
    )
    more_validation = merged.model_copy(
        update={
            "client_split": merged.client_split.model_copy(
                update={"validation_fraction": 0.15}
            )
        }
    )
    assert partition_fingerprint(DATASET, merged) == partition_fingerprint(
        DATASET, more_validation
    )


def test_generation_dispatches_by_backend_and_reuses_partition(monkeypatch, tmp_path):
    config = _config(tmp_path / "datasets.yaml")
    artifact, created = _generate(monkeypatch, config, tmp_path / "data")
    assert created is True
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
        training_seed=11,
    )
    resolved, data = resolve_experiment_data(exp)
    assert data.artifact.path == generated.path

    other_seed, _ = resolve_experiment_data(exp.model_copy(update={"training_seed": 12}))
    assert other_seed.partition_id == resolved.partition_id
    assert _fingerprint("local", other_seed, LocalConfig().model_dump()) != _fingerprint("local", 
        resolved, LocalConfig().model_dump()
    )


def test_experiment_requires_explicit_flower_partition_generation(
    monkeypatch, tmp_path
):
    config = _config(tmp_path / "datasets.yaml")

    def unexpected_generation(*args, **kwargs):
        raise AssertionError("experiment startup must not generate data")

    monkeypatch.setattr(partitions, "generate_flower_partition", unexpected_generation)

    with pytest.raises(FileNotFoundError):
        resolve_experiment_data(
            ExperimentConfig(
                dataset=DATASET,
                dataset_config=str(config),
                data_dir=str(tmp_path / "data"),
            )
        )


def test_seed_overrides_generate_and_resolve_the_matching_partition(
    monkeypatch, tmp_path
):
    config = _config(tmp_path / "datasets.yaml")
    monkeypatch.setattr(partitions, "generate_flower_partition", _fake_flower_backend)
    base = ExperimentConfig(
        dataset=DATASET,
        dataset_config=str(config),
        data_dir=str(tmp_path / "data"),
        rounds=1,
        partition_seed=17,
        split_seed=23,
        training_seed=31,
    )
    settings = dataset_settings(DATASET, config)
    settings = settings.model_copy(
        update={
            "partition": settings.partition.model_copy(
                update={"partition_seed": 17, "split_seed": 23}
            )
        }
    )
    generate_partition(
        DATASET,
        config_path=config,
        data_dir=tmp_path / "data",
        settings=settings,
    )

    resolved, data = resolve_experiment_data(base)
    other_split_seed, split_data = resolve_experiment_data(
        base.model_copy(update={"split_seed": 24})
    )

    assert resolved.partition_seed == 17
    assert resolved.split_seed == 23
    assert resolved.training_seed == 31
    assert data.artifact.path.is_dir()
    assert other_split_seed.partition_id == resolved.partition_id
    assert split_data.artifact.path.is_dir()


class _Backbone(nn.Module):
    out_dim = 4

    def forward(self, x):
        return x.flatten(1)[:, :4]


def test_merged_validation_fraction_is_a_share_of_the_whole_client(tmp_path):
    # 20 samples per client: generation carved off 4 test, so train.pt holds 16.
    directory = tmp_path / "clients" / "client_0"
    directory.mkdir(parents=True)
    for split, n in (("train", 16), ("test", 4)):
        torch.save(
            (torch.rand(n, 3, 32, 32), torch.zeros(n, dtype=torch.long)),
            directory / f"{split}.pt",
        )
    artifact = partitions.PartitionArtifact(
        dataset=DATASET,
        partition_id="merged",
        path=tmp_path,
        settings=None,
        manifest={
            "task": "classification",
            "num_clients": 1,
            "target_spec": {"num_classes": 2},
            "input_spec": {"kind": "image"},
            "source": {
                "client_split": {"validation_fraction": 0.15, "test_fraction": 0.2}
            },
        },
    )

    clients = build_partition_clients(
        artifact,
        shared_dim=4,
        batch=4,
        validation_fraction=0.15,
        backbones=[_Backbone],
    )

    # 15% of the 20-sample client is 3, not 15% of the 16 stored training samples.
    assert len(clients[0].val_loader.dataset) == 3
    assert len(clients[0].train_loader.dataset) == 13


def test_split_seed_changes_validation_only(monkeypatch, tmp_path):
    config = _config(tmp_path / "datasets.yaml")
    _generate(monkeypatch, config, tmp_path / "data")
    artifact = load_partition(DATASET, config_path=config, data_dir=tmp_path / "data")

    def build(split_seed):
        return build_partition_clients(
            artifact,
            shared_dim=4,
            batch=4,
            split_seed=split_seed,
            validation_fraction=0.25,
            backbones=[_Backbone],
        )

    first = build(11)
    second = build(12)
    repeated = build(11)

    def indices(clients, loader):
        return [set(getattr(client, loader).dataset.indices) for client in clients]

    assert indices(first, "val_loader") != indices(second, "val_loader")
    assert indices(first, "train_loader") != indices(second, "train_loader")
    assert indices(first, "val_loader") == indices(repeated, "val_loader")
    assert indices(first, "test_loader") == indices(second, "test_loader")
    assert len(first[0].val_loader.dataset) == 3
    assert len(first[0].train_loader.dataset) == 9


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

    config = _config(tmp_path / "datasets.yaml")
    generated, _ = _generate(monkeypatch, config, tmp_path / "data")
    exp = ExperimentConfig(
        dataset=DATASET,
        dataset_config=str(config),
        data_dir=str(tmp_path / "data"),
        model_arch="fedavg_cnn",
        rounds=1,
        shared_dim=8,
        batch=4,
        quiet=True,
    )
    resolved, data = resolve_experiment_data(exp)
    record = run_resolved(
        "local", resolved, config_class("local")(), torch.device("cpu"), data
    )
    repeated = run_resolved(
        "local", resolved, config_class("local")(), torch.device("cpu"), data
    )
    assert record["config"]["experiment"]["partition_id"] == generated.partition_id
    assert set(record["result"]["evaluation_history"]["clients"]) == {"0", "1"}
    assert repeated["result"] == record["result"]
    validate_run_record(record)
