"""Configuration, identity, persistence, and loading for generated partitions."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from rigfl.core import Client, LearnedProjection, assemble_model
from rigfl.data.builder import _ArrayDataset, _collate
from rigfl.data.config import (
    DEFAULT_DATASET_CONFIG,
    DEFAULT_DATA_DIR,
    DatasetSettings,
    dataset_settings,
    FlowerDatasetSettings,
)
from rigfl.data.flower import generate_flower_partition


MANIFEST_SCHEMA_VERSION = 2


@dataclass(frozen=True)
class PartitionArtifact:
    dataset: str
    partition_id: str
    path: Path
    settings: DatasetSettings
    manifest: dict


def partition_fingerprint(dataset: str, settings: DatasetSettings) -> str:
    """Stable identity derived only from settings that determine partition data."""
    payload = {"dataset": dataset, **settings.model_dump(mode="json")}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:12]


def partition_path(data_dir: str | Path, dataset: str, partition_id: str) -> Path:
    return Path(data_dir) / dataset / f"partition_{partition_id}"


def expected_partition(
    dataset: str,
    *,
    config_path: str | Path = DEFAULT_DATASET_CONFIG,
    data_dir: str | Path = DEFAULT_DATA_DIR,
) -> tuple[DatasetSettings, str, Path]:
    settings = dataset_settings(dataset, config_path)
    if not isinstance(settings, FlowerDatasetSettings):
        raise ValueError(
            f"dataset {dataset!r} uses the {settings.backend!r} backend and does "
            "not produce a RigFL-generated partition"
        )
    partition_id = partition_fingerprint(dataset, settings)
    return settings, partition_id, partition_path(data_dir, dataset, partition_id)


BACKEND_GENERATORS = {"flower": generate_flower_partition}


def generate_partition(
    dataset: str,
    *,
    config_path: str | Path = DEFAULT_DATASET_CONFIG,
    data_dir: str | Path = DEFAULT_DATA_DIR,
) -> tuple[PartitionArtifact, bool]:
    """Generate and atomically publish the configured partition.

    Returns ``(artifact, created)``. A complete existing artifact is reused.
    """
    settings, partition_id, target = expected_partition(
        dataset, config_path=config_path, data_dir=data_dir
    )
    if target.exists():
        return load_partition(dataset, config_path=config_path, data_dir=data_dir), False
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    try:
        try:
            generator = BACKEND_GENERATORS[settings.backend]
        except KeyError as exc:
            raise ValueError(f"unsupported dataset backend: {settings.backend!r}") from exc
        backend_metadata = generator(settings, temporary)
        manifest = {
            **backend_metadata,
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "dataset": dataset,
            "partition_id": partition_id,
            "settings": settings.model_dump(mode="json"),
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        os.replace(temporary, target)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return load_partition(dataset, config_path=config_path, data_dir=data_dir), True


def _read_manifest(path: Path) -> dict:
    manifest_path = path / "manifest.json"
    if not manifest_path.exists():
        raise ValueError(f"generated partition is incomplete: missing {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read generated partition manifest: {manifest_path}") from exc
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported partition manifest schema in {manifest_path}: "
            f"{manifest.get('schema_version')!r}"
        )
    return manifest


def load_partition(
    dataset: str,
    *,
    config_path: str | Path = DEFAULT_DATASET_CONFIG,
    data_dir: str | Path = DEFAULT_DATA_DIR,
) -> PartitionArtifact:
    """Resolve the configured fingerprint and load its generated manifest."""
    settings, partition_id, path = expected_partition(
        dataset, config_path=config_path, data_dir=data_dir
    )
    if not path.exists():
        raise FileNotFoundError(
            f"generated partition for dataset {dataset!r} was not found at {path}. "
            f"Run: python -m rigfl.data.generate --dataset {dataset}"
        )
    manifest = _read_manifest(path)
    expected = {
        "dataset": dataset,
        "partition_id": partition_id,
        "settings": settings.model_dump(mode="json"),
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(
                f"generated partition manifest {path / 'manifest.json'} has a "
                f"different {key}; regenerate the configured partition"
            )
    num_clients = manifest.get("num_clients")
    if (
        not isinstance(num_clients, int)
        or isinstance(num_clients, bool)
        or num_clients < 1
    ):
        raise ValueError(
            f"generated partition manifest {path / 'manifest.json'} has an invalid "
            "num_clients value; regenerate the configured partition"
        )
    configured_num_clients = getattr(settings.partition, "num_clients", None)
    partition_sizes = getattr(settings.partition, "partition_sizes", None)
    if partition_sizes is not None:
        configured_num_clients = len(partition_sizes)
    if configured_num_clients is not None and num_clients != configured_num_clients:
        raise ValueError(
            f"generated partition manifest {path / 'manifest.json'} has a different "
            "num_clients value than the partition configuration; regenerate the "
            "configured partition"
        )
    if manifest.get("task") not in {"classification", "regression"}:
        raise ValueError(f"generated partition manifest {path / 'manifest.json'} has no valid task")
    if not isinstance(manifest.get("input_spec"), dict) or not isinstance(
        manifest.get("target_spec"), dict
    ):
        raise ValueError(f"generated partition manifest {path / 'manifest.json'} lacks data specs")
    for cid in range(num_clients):
        client_dir = path / "clients" / f"client_{cid}"
        for split in ("train", "validation", "test"):
            split_path = client_dir / f"{split}.pt"
            if not split_path.exists():
                raise ValueError(
                    f"generated partition is incomplete: missing {split_path}"
                )
    return PartitionArtifact(dataset, partition_id, path, settings, manifest)


def _load_split(path: Path):
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # PyTorch 2.0 has no weights_only argument.
        value = torch.load(path, map_location="cpu")
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError(f"partition split {path} must contain (inputs, targets)")
    inputs, targets = value
    if not isinstance(inputs, torch.Tensor) or not isinstance(targets, torch.Tensor):
        raise ValueError(f"partition split {path} must contain tensors")
    if len(inputs) != len(targets):
        raise ValueError(f"partition split {path} has mismatched inputs and targets")
    return inputs, targets


def build_partition_clients(
    artifact: PartitionArtifact,
    *,
    shared_dim: int,
    batch: int,
    adapter=None,
    backbones=None,
) -> list[Client]:
    """Construct federated clients from one previously generated partition."""
    if artifact.manifest["task"] != "classification":
        raise ValueError(
            "RigFL's current experiment algorithms support classification only; "
            f"partition {artifact.dataset!r} contains {artifact.manifest['task']} targets"
        )
    if adapter is None:
        adapter = lambda native, shared: LearnedProjection(native, shared)
    if backbones is None:
        raise ValueError("backbones must be selected independently of the dataset")
    if not backbones:
        raise ValueError("backbones must contain at least one model factory")
    num_clients = int(artifact.manifest["num_clients"])
    num_classes = int(artifact.manifest["target_spec"]["num_classes"])
    clients = []
    for cid in range(num_clients):
        directory = artifact.path / "clients" / f"client_{cid}"
        x_train, y_train = _load_split(directory / "train.pt")
        x_validation, y_validation = _load_split(directory / "validation.pt")
        x_test, y_test = _load_split(directory / "test.pt")
        backbone = backbones[cid % len(backbones)]()
        model = assemble_model(
            backbone, shared_dim=shared_dim, num_classes=num_classes,
            adapter=adapter)
        clients.append(
            Client(
                model,
                DataLoader(
                    _ArrayDataset(x_train, y_train, range(len(y_train))),
                    batch_size=batch,
                    shuffle=True,
                    collate_fn=_collate,
                ),
                DataLoader(
                    _ArrayDataset(
                        x_validation, y_validation, range(len(y_validation))
                    ),
                    batch_size=batch,
                    collate_fn=_collate,
                ),
                DataLoader(
                    _ArrayDataset(x_test, y_test, range(len(y_test))),
                    batch_size=batch,
                    collate_fn=_collate,
                ),
            )
        )
    return clients
