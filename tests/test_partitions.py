"""Generated partition configuration, identity, persistence, and resolution."""

from __future__ import annotations

import json

import torch
import torch.nn as nn

from rigfl.data import partitions
from rigfl.data.partitions import (
    build_partition_clients,
    generate_partition,
    load_partition,
    partition_fingerprint,
)
from rigfl.data.config import dataset_settings
from rigfl.experiment.config import ExperimentConfig, run_fingerprint
from rigfl.experiment.run import resolve_experiment_data
from rigfl.algorithms.local import LocalConfig


DATASET = "my_images"


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
        for split, n in (("train", 9), ("validation", 3), ("test", 6)):
            inputs = torch.rand(n, 3, 32, 32)
            targets = (torch.arange(n) + cid) % 3
            torch.save((inputs, targets), directory / f"{split}.pt")
            summary[split] = n
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
    assert manifest["source"]["dataset"] == "organization/source-data"
    assert manifest["num_clients"] == 2
    assert manifest["clients"][0]["validation"] == 3

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
    resolved, artifact = resolve_experiment_data(exp)
    assert artifact.path == generated.path
    assert resolved.scheme == "generated"
    assert resolved.partition == generated.partition_id
    assert resolved.num_clients == 2
    assert resolved.num_classes == 3
    assert resolved.val_frac == 0.25

    other_seed, _ = resolve_experiment_data(exp.model_copy(update={"seed": 12}))
    assert other_seed.partition == resolved.partition
    assert run_fingerprint(other_seed, LocalConfig().model_dump()) != run_fingerprint(
        resolved, LocalConfig().model_dump()
    )


def test_partition_settings_are_not_accepted_in_experiment(monkeypatch, tmp_path):
    config = _config(tmp_path / "datasets.yaml")
    _generate(monkeypatch, config, tmp_path / "data")
    exp = ExperimentConfig(
        dataset=DATASET,
        dataset_config=str(config),
        data_dir=str(tmp_path / "data"),
        alpha=0.9,
    )
    try:
        resolve_experiment_data(exp)
    except ValueError as exc:
        assert "dataset configuration" in str(exc) and "alpha" in str(exc)
    else:
        raise AssertionError("experiment-level alpha should have been rejected")


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
        model_architectures=["fedavg_cnn"],
        rounds=1,
        shared_dim=8,
        batch=4,
        quiet=True,
    )
    record = run_one("local", exp, config_class("local")(), torch.device("cpu"))
    assert record["config"]["experiment"]["partition"] == generated.partition_id
    assert record["config"]["experiment"]["scheme"] == "generated"
    assert set(record["result"]["evaluation_history"]["clients"]) == {"0", "1"}
    validate_run_record(record)
