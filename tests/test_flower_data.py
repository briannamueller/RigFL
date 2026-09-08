"""Metadata-driven source resolution for the generic Flower backend."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from datasets import ClassLabel, Dataset, DatasetDict, Features, Image, Value
from PIL import Image as PILImage

from rigfl.data import flower
from rigfl.data.config import FlowerDatasetSettings


PARTITIONER_CASES = {
    "continuous": {
        "num_clients": 3,
        "partition_by": "score",
        "strictness": 0.6,
    },
    "dirichlet": {
        "num_clients": 3,
        "alpha": [0.2, 0.4, 0.6],
        "min_partition_size": 2,
        "self_balancing": True,
    },
    "distribution": {
        "distribution_array": [[1], [2], [3]],
        "num_clients": 3,
        "num_unique_labels_per_partition": 1,
        "preassigned_num_samples_per_label": 2,
        "rescale": True,
    },
    "exponential": {"num_clients": 3},
    "grouped_natural_id": {
        "partition_by": "user_id",
        "group_size": 2,
        "mode": "strict",
        "sort_unique_ids": True,
    },
    "iid": {"num_clients": 3},
    "inner_dirichlet": {
        "partition_sizes": [20, 20, 20],
        "alpha": 0.4,
    },
    "linear": {"num_clients": 3},
    "natural_id": {"partition_by": "user_id"},
    "pathological": {
        "num_clients": 3,
        "num_classes_per_partition": 2,
        "class_assignment_mode": "deterministic",
    },
    "shard": {
        "num_clients": 3,
        "num_shards_per_partition": 2,
        "keep_incomplete_shard": True,
    },
    "size": {"partition_sizes": [20, 20, 20]},
    "square": {"num_clients": 3},
}


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


def test_nonstandard_splits_require_mapping_and_list_available_names(monkeypatch):
    _metadata(
        monkeypatch,
        features=Features({"image": Image(), "label": ClassLabel(num_classes=2)}),
        splits=("training", "holdout"),
    )
    with pytest.raises(ValueError, match="training") as error:
        flower.inspect_flower_source(
            FlowerDatasetSettings(source_dataset="organization/data")
        )
    assert "holdout" in str(error.value)


def test_merged_source_splits_are_resolved(monkeypatch):
    _metadata(
        monkeypatch,
        features=Features({"image": Image(), "label": ClassLabel(num_classes=2)}),
        splits=("train", "validation", "test"),
    )
    settings = FlowerDatasetSettings(
        source_dataset="organization/data",
        source_splits={"merge_splits": ["train", "validation", "test"]},
        client_split={
            "validation_fraction": 0.1,
            "test_fraction": 0.2,
            "stratify": True,
        },
    )

    resolved = flower.inspect_flower_source(settings)

    assert resolved.splits.merge_splits == ["train", "validation", "test"]


def test_invalid_merged_split_lists_available_names(monkeypatch):
    _metadata(
        monkeypatch,
        features=Features({"image": Image(), "label": ClassLabel(num_classes=2)}),
    )
    settings = FlowerDatasetSettings(
        source_dataset="organization/data",
        source_splits={"merge_splits": ["train", "holdout"]},
        client_split={
            "validation_fraction": 0.1,
            "test_fraction": 0.2,
            "stratify": True,
        },
    )

    with pytest.raises(ValueError, match="holdout") as error:
        flower.inspect_flower_source(settings)

    assert "test" in str(error.value)


def test_client_split_requires_merged_source_splits():
    client_split = {
        "validation_fraction": 0.1,
        "test_fraction": 0.2,
        "stratify": True,
    }
    with pytest.raises(ValueError, match="client_split is required"):
        FlowerDatasetSettings(
            source_dataset="organization/data",
            source_splits={"merge_splits": ["train", "test"]},
        )
    with pytest.raises(ValueError, match="requires source_splits.merge_splits"):
        FlowerDatasetSettings(
            source_dataset="organization/data",
            client_split=client_split,
        )


def test_merged_source_split_names_must_be_unique():
    with pytest.raises(ValueError, match="unique names"):
        FlowerDatasetSettings(
            source_dataset="organization/data",
            source_splits={"merge_splits": ["train", "train"]},
            client_split={
                "validation_fraction": 0.1,
                "test_fraction": 0.2,
                "stratify": True,
            },
        )


def test_client_split_fractions_leave_training_data():
    with pytest.raises(ValueError, match="sum to less than one"):
        FlowerDatasetSettings(
            source_dataset="organization/data",
            source_splits={"merge_splits": ["train", "test"]},
            client_split={
                "validation_fraction": 0.4,
                "test_fraction": 0.6,
                "stratify": True,
            },
        )


def test_invalid_subset_lists_available_configurations(monkeypatch):
    _metadata(
        monkeypatch,
        features=Features({"image": Image(), "label": ClassLabel(num_classes=2)}),
    )
    with pytest.raises(ValueError, match="resolved-subset"):
        flower.inspect_flower_source(
            FlowerDatasetSettings(
                source_dataset="organization/data", source_subset="not-a-subset"
            )
        )


def test_ambiguous_targets_require_override_and_list_candidates(monkeypatch):
    _metadata(
        monkeypatch,
        features=Features({
            "image": Image(),
            "fine_label": ClassLabel(num_classes=10),
            "coarse_label": ClassLabel(num_classes=2),
        }),
    )
    with pytest.raises(ValueError, match="target_column") as error:
        flower.inspect_flower_source(
            FlowerDatasetSettings(source_dataset="organization/data")
        )
    assert "fine_label" in str(error.value)
    assert "coarse_label" in str(error.value)


def test_supervised_keys_take_priority_over_ambiguous_columns(monkeypatch):
    _metadata(
        monkeypatch,
        features=Features({
            "pixels": Image(),
            "mask": Image(),
            "label": ClassLabel(num_classes=2),
            "other_label": ClassLabel(num_classes=4),
        }),
        supervised=("pixels", "label"),
    )
    resolved = flower.inspect_flower_source(
        FlowerDatasetSettings(source_dataset="organization/data")
    )
    assert resolved.input_column == "pixels"
    assert resolved.target_column == "label"


def test_unknown_data_transform_lists_registered_names():
    with pytest.raises(ValueError, match="cifar10"):
        FlowerDatasetSettings(
            source_dataset="organization/data",
            data_transform="not_registered",
        )


def test_named_transform_resolves_ambiguous_columns(monkeypatch):
    _metadata(
        monkeypatch,
        features=Features({
            "img": Image(),
            "fine_label": ClassLabel(num_classes=10),
            "coarse_label": ClassLabel(num_classes=2),
        }),
    )
    resolved = flower.inspect_flower_source(
        FlowerDatasetSettings(
            source_dataset="organization/data", data_transform="cifar100"
        )
    )

    assert resolved.input_column == "img"
    assert resolved.target_column == "fine_label"


def test_dirichlet_rejects_regression_targets(monkeypatch):
    _metadata(
        monkeypatch,
        features=Features({"features": Value("float32"), "target": Value("float32")}),
        supervised=("features", "target"),
    )
    with pytest.raises(ValueError, match="not supported for regression"):
        flower.inspect_flower_source(
            FlowerDatasetSettings(source_dataset="organization/data")
        )


def test_merged_split_rejects_stratified_regression(monkeypatch):
    _metadata(
        monkeypatch,
        features=Features({"features": Value("float32"), "target": Value("float32")}),
        supervised=("features", "target"),
    )
    settings = FlowerDatasetSettings(
        source_dataset="organization/data",
        source_splits={"merge_splits": ["train", "test"]},
        client_split={
            "validation_fraction": 0.1,
            "test_fraction": 0.2,
            "stratify": True,
        },
        partition={"scheme": "iid", "num_clients": 2},
    )

    with pytest.raises(ValueError, match="only for classification"):
        flower.inspect_flower_source(settings)


@pytest.mark.parametrize(("scheme", "arguments"), PARTITIONER_CASES.items())
def test_every_horizontal_flower_partitioner_is_configurable(scheme, arguments):
    expected_classes = {
        "continuous": "ContinuousPartitioner",
        "dirichlet": "DirichletPartitioner",
        "distribution": "DistributionPartitioner",
        "exponential": "ExponentialPartitioner",
        "grouped_natural_id": "GroupedNaturalIdPartitioner",
        "iid": "IidPartitioner",
        "inner_dirichlet": "InnerDirichletPartitioner",
        "linear": "LinearPartitioner",
        "natural_id": "NaturalIdPartitioner",
        "pathological": "PathologicalPartitioner",
        "shard": "ShardPartitioner",
        "size": "SizePartitioner",
        "square": "SquarePartitioner",
    }
    settings = FlowerDatasetSettings(
        source_dataset="organization/data",
        partition={
            "scheme": scheme,
            "partition_seed": 9,
            "shuffle": False,
            **arguments,
        },
    )

    partitioner = flower.FLOWER_PARTITIONERS[scheme](settings.partition, "label")

    assert partitioner.__class__.__name__ == expected_classes[scheme]
    assert settings.partition.partition_seed == 9
    assert settings.partition.shuffle is False
    configured_count = getattr(settings.partition, "num_clients", None)
    if configured_count is not None:
        assert partitioner._num_partitions == configured_count
    configured_sizes = getattr(settings.partition, "partition_sizes", None)
    if configured_sizes is not None:
        assert list(partitioner._partition_sizes) == configured_sizes

    configured_column = getattr(settings.partition, "partition_by", None)
    if hasattr(partitioner, "_partition_by"):
        assert partitioner._partition_by == (configured_column or "label")

    forwarded_fields = {
        "continuous": ("strictness",),
        "dirichlet": ("min_partition_size", "self_balancing"),
        "distribution": (
            "num_unique_labels_per_partition",
            "preassigned_num_samples_per_label",
            "rescale",
        ),
        "grouped_natural_id": ("group_size", "mode", "sort_unique_ids"),
        "pathological": ("num_classes_per_partition", "class_assignment_mode"),
        "shard": (
            "num_shards_per_partition",
            "shard_size",
            "keep_incomplete_shard",
        ),
    }
    for field in forwarded_fields.get(scheme, ()):
        assert getattr(partitioner, f"_{field}") == getattr(settings.partition, field)
    if scheme == "distribution":
        assert isinstance(partitioner._distribution_array, np.ndarray)


def test_partitioner_registry_covers_every_configured_scheme():
    assert set(flower.FLOWER_PARTITIONERS) == set(PARTITIONER_CASES)


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

    splits = flower._split_client_partition(dataset, settings, "label", seed=4)

    assert {name: len(split) for name, split in splits.items()} == {
        "train": 70,
        "validation": 10,
        "test": 20,
    }
    ids = [set(split["sample_id"]) for split in splits.values()]
    assert set.union(*ids) == set(range(100))
    assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
    assert all(set(split["label"]) == {0, 1} for split in splits.values())


def test_merge_source_splits_uses_every_sample_once():
    source = DatasetDict({
        "train": Dataset.from_dict({"sample_id": [0, 1], "label": [0, 1]}),
        "test": Dataset.from_dict({"sample_id": [2, 3], "label": [0, 1]}),
    })

    merged = flower._merge_source_splits(
        source,
        split_names=("train", "test"),
        shuffle=False,
        seed=0,
    )

    assert list(merged) == ["merged"]
    assert merged["merged"]["sample_id"] == [0, 1, 2, 3]


def test_image_mode_coerces_mixed_images_to_one_shape():
    grayscale = PILImage.fromarray(np.zeros((4, 4), dtype=np.uint8), mode="L")
    rgb = PILImage.fromarray(np.zeros((4, 4, 3), dtype=np.uint8), mode="RGB")

    inputs = flower._image_tensor(
        [grayscale, rgb], mean=None, std=None, image_mode="rgb"
    )

    assert inputs.shape == (2, 3, 4, 4)


@pytest.mark.parametrize(("scheme", "arguments"), PARTITIONER_CASES.items())
def test_every_horizontal_flower_partitioner_creates_partitions(scheme, arguments):
    settings = FlowerDatasetSettings(
        source_dataset="organization/data",
        partition={
            "scheme": scheme,
            "partition_seed": 9,
            "shuffle": False,
            **arguments,
        },
    )
    dataset = Dataset.from_dict({
        "feature": list(range(60)),
        "label": [index % 3 for index in range(60)],
        "score": [float(index) for index in range(60)],
        "user_id": [f"user-{index % 6}" for index in range(60)],
    })
    partitioner = flower.FLOWER_PARTITIONERS[scheme](settings.partition, "label")
    partitioner.dataset = dataset

    partitions = [
        partitioner.load_partition(partition_id)
        for partition_id in range(partitioner.num_partitions)
    ]

    assert all(len(partition) > 0 for partition in partitions)
    assert sum(map(len, partitions)) == len(dataset)


def test_configured_partition_column_must_exist(monkeypatch):
    _metadata(
        monkeypatch,
        features=Features({
            "image": Image(),
            "label": ClassLabel(num_classes=2),
        }),
    )
    settings = FlowerDatasetSettings(
        source_dataset="organization/data",
        partition={
            "scheme": "continuous",
            "partition_by": "missing_score",
            "strictness": 0.5,
        },
    )
    with pytest.raises(ValueError, match="missing_score") as error:
        flower.inspect_flower_source(settings)
    assert "image" in str(error.value) and "label" in str(error.value)


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


def test_split_partition_counts_must_match():
    fds = _FakeFederatedDataset({"train": 3, "test": 2})
    with pytest.raises(ValueError, match="train=3, test=2"):
        flower._initialize_partitions(
            fds, {"train": "train", "test": "test"}, "iid"
        )


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
            "validation_per_client": None,
            "test_per_client": None,
        },
    )

    manifest = flower.generate_flower_partition(settings, tmp_path)

    assert manifest["source"]["splits"] == {"merge_splits": ["train", "test"]}
    assert manifest["source"]["data_transform"]["name"] == "auto"
    assert manifest["source"]["client_split"] == {
        "validation_fraction": 0.1,
        "test_fraction": 0.2,
        "stratify": True,
    }
    assert all(
        (
            client["sizes"]["train"],
            client["sizes"]["validation"],
            client["sizes"]["test"],
        )
        == (14, 2, 4)
        for client in manifest["clients"]
    )
    observed = []
    for client_id in range(2):
        for split in ("train", "validation", "test"):
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
            "validation_per_client": None,
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
