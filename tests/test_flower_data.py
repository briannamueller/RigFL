"""Metadata-driven source resolution for the generic Flower backend."""

from types import SimpleNamespace

import numpy as np
import pytest
from datasets import ClassLabel, Features, Image, Value

from rigfl.data.config import FlowerDatasetSettings
from rigfl.data import flower


PARTITIONER_CASES = {
    "continuous": {
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
        "rescale": False,
    },
    "exponential": {},
    "grouped_natural_id": {
        "partition_by": "user_id",
        "group_size": 2,
        "mode": "strict",
        "sort_unique_ids": True,
    },
    "iid": {},
    "inner_dirichlet": {
        "partition_sizes": [20, 20, 20],
        "alpha": 0.4,
    },
    "linear": {},
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
    "square": {},
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
