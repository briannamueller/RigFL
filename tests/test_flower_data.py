"""Metadata-driven source resolution for the generic Flower backend."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from datasets import ClassLabel, Dataset, DatasetDict, Features, Image, Value
from PIL import Image as PILImage

from rigfl.data import flower
from rigfl.data.config import FlowerDatasetSettings


def _metadata(monkeypatch, *, features, splits=("train", "test"), supervised=None):
    builder = SimpleNamespace(
        config=SimpleNamespace(name="resolved-subset"),
        info=SimpleNamespace(features=features, supervised_keys=supervised),
    )
    monkeypatch.setattr(flower, "load_dataset_builder", lambda *args, **kwargs: builder)
    monkeypatch.setattr(
        flower, "get_dataset_config_names", lambda *args, **kwargs: ["resolved-subset"]
    )
    monkeypatch.setattr(
        flower, "get_dataset_split_names", lambda *args, **kwargs: list(splits)
    )


def test_metadata_infers_standard_splits_and_unambiguous_columns(monkeypatch):
    _metadata(
        monkeypatch,
        features=Features({
            "img": Image(),
            "label": ClassLabel(names=["zero", "one"]),
        }),
    )
    resolved = flower.inspect_flower_source(
        FlowerDatasetSettings(source_dataset="organization/data")
    )
    assert resolved.splits.train == "train"
    assert resolved.splits.test == "test"
    assert resolved.input_column == "img"
    assert resolved.target_column == "label"
    assert resolved.task == "classification"
    assert resolved.class_names == ["zero", "one"]


def test_ambiguous_targets_require_an_override(monkeypatch):
    _metadata(
        monkeypatch,
        features=Features({
            "img": Image(),
            "fine_label": ClassLabel(num_classes=10),
            "coarse_label": ClassLabel(num_classes=2),
        }),
    )
    with pytest.raises(ValueError, match="target_column"):
        flower.inspect_flower_source(
            FlowerDatasetSettings(source_dataset="organization/data")
        )

    # a named transform settles the choice
    resolved = flower.inspect_flower_source(
        FlowerDatasetSettings(
            source_dataset="organization/data", data_transform="cifar100"
        )
    )
    assert resolved.target_column == "fine_label"


def test_generation_rejects_automatically_detected_regression(monkeypatch, tmp_path):
    _metadata(
        monkeypatch,
        features=Features({"features": Value("float32"), "target": Value("float32")}),
        supervised=("features", "target"),
    )

    with pytest.raises(ValueError, match="supports classification datasets only"):
        flower.generate_flower_partition(
            FlowerDatasetSettings(
                source_dataset="organization/data",
                partition={"scheme": "iid", "num_clients": 2},
            ),
            tmp_path,
        )


def test_merged_partition_is_split_into_requested_client_fractions():
    features = Features({
        "sample_id": Value("int64"),
        "label": ClassLabel(num_classes=2),
    })
    dataset = Dataset.from_dict(
        {
            "sample_id": list(range(100)),
            "label": [index % 2 for index in range(100)],
        },
        features=features,
    )
    settings = SimpleNamespace(
        validation_fraction=0.1,
        test_fraction=0.2,
        stratify=True,
    )

    splits = flower._split_client_partition(
        dataset,
        settings,
        "label",
        test_seed=4,
    )

    assert {name: len(split) for name, split in splits.items()} == {
        "train": 80,
        "test": 20,
    }
    ids = [set(split["sample_id"]) for split in splits.values()]
    assert set.union(*ids) == set(range(100))
    assert not ids[0] & ids[1]
    assert all(set(split["label"]) == {0, 1} for split in splits.values())

    repeated = flower._split_client_partition(dataset, settings, "label", test_seed=4)
    other = flower._split_client_partition(dataset, settings, "label", test_seed=5)
    assert set(repeated["test"]["sample_id"]) == set(splits["test"]["sample_id"])
    assert set(other["test"]["sample_id"]) != set(splits["test"]["sample_id"])


def test_image_mode_coerces_mixed_images_to_one_shape():
    grayscale = PILImage.fromarray(np.zeros((4, 4), dtype=np.uint8), mode="L")
    rgb = PILImage.fromarray(np.zeros((4, 4, 3), dtype=np.uint8), mode="RGB")

    inputs = flower._image_tensor(
        [grayscale, rgb], mean=None, std=None, image_mode="rgb"
    )

    assert inputs.shape == (2, 3, 4, 4)


class _FakePartitioner:
    def __init__(self, count, identity_map=None):
        self.num_partitions = count
        self.partition_id_to_natural_id = identity_map
        self.partition_id_to_natural_ids = identity_map


class _FakeFederatedDataset:
    def __init__(self, counts, identity_maps=None):
        identity_maps = identity_maps or {}
        self.partitioners = {
            split: _FakePartitioner(count, identity_maps.get(split))
            for split, count in counts.items()
        }

    def load_partition(self, partition_id, split):
        assert partition_id == 0
        return f"first-{split}"


def test_natural_client_identities_must_match_across_splits():
    fds = _FakeFederatedDataset(
        {"train": 2, "test": 2},
        {"train": {0: "a", 1: "b"}, "test": {0: "a", 1: "c"}},
    )
    with pytest.raises(ValueError, match="same natural IDs"):
        flower._initialize_partitions(
            fds, {"train": "train", "test": "test"}, "natural_id"
        )


def test_merged_generation_partitions_once_then_splits_each_client(
    monkeypatch, tmp_path
):
    features = Features({
        "sample_id": Value("int64"),
        "label": ClassLabel(num_classes=2),
    })
    _metadata(monkeypatch, features=features)
    source = DatasetDict({
        "train": Dataset.from_dict(
            {
                "sample_id": list(range(30)),
                "label": [index % 2 for index in range(30)],
            },
            features=features,
        ),
        "test": Dataset.from_dict(
            {
                "sample_id": list(range(30, 40)),
                "label": [index % 2 for index in range(30, 40)],
            },
            features=features,
        ),
    })

    class FakeFederatedDataset:
        def __init__(
            self, *, preprocessor, partitioners, shuffle, seed, **_kwargs
        ):
            dataset = source.shuffle(seed=seed) if shuffle else source
            dataset = preprocessor(dataset) if preprocessor else dataset
            self.partitioners = partitioners
            for split, partitioner in partitioners.items():
                partitioner.dataset = dataset[split]

        def load_partition(self, partition_id, split):
            return self.partitioners[split].load_partition(partition_id)

    monkeypatch.setattr(
        "flwr_datasets.FederatedDataset", FakeFederatedDataset
    )
    settings = FlowerDatasetSettings(
        source_dataset="organization/data",
        source_splits={"merge_splits": ["train", "test"]},
        client_split={
            "validation_fraction": 0.1,
            "test_fraction": 0.2,
            "stratify": True,
        },
        input_column="sample_id",
        target_column="label",
        partition={
            "scheme": "iid",
            "num_clients": 2,
            "partition_seed": 7,
            "train_per_client": None,
            "test_per_client": None,
        },
    )

    manifest = flower.generate_flower_partition(settings, tmp_path)

    assert all(
        (client["sizes"]["train"], client["sizes"]["test"]) == (16, 4)
        for client in manifest["clients"]
    )
    observed = []
    for client_id in range(2):
        for split in ("train", "test"):
            inputs, _ = torch.load(
                tmp_path / "clients" / f"client_{client_id}" / f"{split}.pt",
                weights_only=True,
            )
            observed.extend(inputs.flatten().int().tolist())
    assert sorted(observed) == list(range(40))


def test_natural_client_limit_preserves_selected_source_ids(monkeypatch, tmp_path):
    features = Features({
        "sample": Value("int64"),
        "writer": Value("string"),
        "label": ClassLabel(num_classes=2),
    })
    _metadata(monkeypatch, features=features, splits=("train",))
    source = DatasetDict({
        "train": Dataset.from_dict(
            {
                "sample": list(range(120)),
                "writer": [f"writer-{index // 30}" for index in range(120)],
                "label": [index % 2 for index in range(120)],
            },
            features=features,
        )
    })

    class FakeFederatedDataset:
        def __init__(
            self, *, preprocessor, partitioners, shuffle, seed, **_kwargs
        ):
            dataset = source.shuffle(seed=seed) if shuffle else source
            dataset = preprocessor(dataset) if preprocessor else dataset
            self.partitioners = partitioners
            for split, partitioner in partitioners.items():
                partitioner.dataset = dataset[split]

        def load_partition(self, partition_id, split):
            return self.partitioners[split].load_partition(partition_id)

    monkeypatch.setattr("flwr_datasets.FederatedDataset", FakeFederatedDataset)
    settings = FlowerDatasetSettings(
        source_dataset="organization/data",
        source_splits={"merge_splits": ["train"]},
        client_split={
            "validation_fraction": 0.1,
            "test_fraction": 0.2,
            "stratify": False,
        },
        input_column="sample",
        target_column="label",
        partition={
            "scheme": "natural_id",
            "partition_by": "writer",
            "client_limit": 2,
            "partition_seed": 5,
            "train_per_client": None,
            "test_per_client": None,
        },
    )

    first = flower.generate_flower_partition(settings, tmp_path / "first")
    second = flower.generate_flower_partition(settings, tmp_path / "second")

    first_ids = [client["source_client_id"] for client in first["clients"]]
    second_ids = [client["source_client_id"] for client in second["clients"]]
    assert first["num_clients"] == 2
    assert first_ids == second_ids
    assert len(set(first_ids)) == 2
    assert set(first_ids) <= {f"writer-{index}" for index in range(4)}
