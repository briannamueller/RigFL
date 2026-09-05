"""Validated configuration for named dataset sources and client partitions."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


DEFAULT_DATASET_CONFIG = "configs/datasets.yaml"
DEFAULT_DATA_DIR = "data"


class SourceSplits(BaseModel):
    """Map RigFL's split roles to names used by the source dataset."""

    model_config = ConfigDict(extra="forbid")

    train: str
    test: str
    validation: str | None = None


class PreprocessingSettings(BaseModel):
    """Dataset-independent conversion of one source feature to a tensor."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["auto", "image", "numeric"] = "auto"
    mean: list[float] | None = None
    std: list[float] | None = None

    @model_validator(mode="after")
    def _normalization_is_complete(self):
        if (self.mean is None) != (self.std is None):
            raise ValueError("preprocessing.mean and preprocessing.std must be set together")
        if self.mean is not None:
            if not self.mean or len(self.mean) != len(self.std):
                raise ValueError("preprocessing.mean and preprocessing.std must have equal nonzero lengths")
            if any(value <= 0 for value in self.std):
                raise ValueError("preprocessing.std values must be greater than zero")
        return self


class PartitionSettingsBase(BaseModel):
    """Settings shared by every horizontal Flower partitioner."""

    model_config = ConfigDict(extra="forbid")

    partition_seed: int = Field(0, ge=0)
    shuffle: bool = True
    train_per_client: int | None = Field(2000, ge=1)
    validation_per_client: int | None = Field(None, ge=1)
    test_per_client: int | None = Field(500, ge=1)
    val_frac: float = Field(0.2, gt=0, lt=1)


PositiveFloat = Annotated[float, Field(gt=0)]
PositiveInt = Annotated[int, Field(ge=1)]
Alpha = PositiveFloat | list[PositiveFloat]


class ContinuousSettings(PartitionSettingsBase):
    """Arguments passed to Flower's ContinuousPartitioner."""

    scheme: Literal["continuous"] = "continuous"
    num_clients: int = Field(20, ge=1)
    partition_by: str
    strictness: float = Field(ge=0, le=1)


class DirichletSettings(PartitionSettingsBase):
    """Arguments passed to Flower's DirichletPartitioner."""

    scheme: Literal["dirichlet"] = "dirichlet"
    num_clients: int = Field(20, ge=1)
    partition_by: str | None = None
    alpha: Alpha = 0.1
    min_partition_size: int = Field(10, ge=1)
    self_balancing: bool = False

    @model_validator(mode="after")
    def _alpha_matches_clients(self):
        if isinstance(self.alpha, list) and len(self.alpha) != self.num_clients:
            raise ValueError(
                "a list-valued alpha must contain one value per client partition"
            )
        return self


class DistributionSettings(PartitionSettingsBase):
    """Arguments passed to Flower's DistributionPartitioner."""

    scheme: Literal["distribution"] = "distribution"
    distribution_array: list[list[float]]
    num_clients: int = Field(20, ge=1)
    num_unique_labels_per_partition: int = Field(ge=1)
    partition_by: str | None = None
    preassigned_num_samples_per_label: int = Field(ge=0)
    rescale: bool = True


class ExponentialSettings(PartitionSettingsBase):
    """Arguments passed to Flower's ExponentialPartitioner."""

    scheme: Literal["exponential"] = "exponential"
    num_clients: int = Field(20, ge=1)


class GroupedNaturalIdSettings(PartitionSettingsBase):
    """Arguments passed to Flower's GroupedNaturalIdPartitioner."""

    scheme: Literal["grouped_natural_id"] = "grouped_natural_id"
    partition_by: str
    group_size: int = Field(ge=1)
    mode: Literal[
        "allow-smaller", "allow-bigger", "drop-reminder", "strict"
    ] = "allow-smaller"
    sort_unique_ids: bool = True


class IidSettings(PartitionSettingsBase):
    """Arguments passed to Flower's IidPartitioner."""

    scheme: Literal["iid"] = "iid"
    num_clients: int = Field(20, ge=1)


class InnerDirichletSettings(PartitionSettingsBase):
    """Arguments passed to Flower's InnerDirichletPartitioner."""

    scheme: Literal["inner_dirichlet"] = "inner_dirichlet"
    partition_sizes: list[PositiveInt] = Field(min_length=1)
    partition_by: str | None = None
    alpha: Alpha = 0.1


class LinearSettings(PartitionSettingsBase):
    """Arguments passed to Flower's LinearPartitioner."""

    scheme: Literal["linear"] = "linear"
    num_clients: int = Field(20, ge=1)


class NaturalIdSettings(PartitionSettingsBase):
    """Arguments passed to Flower's NaturalIdPartitioner."""

    scheme: Literal["natural_id"] = "natural_id"
    partition_by: str


class PathologicalSettings(PartitionSettingsBase):
    """Arguments passed to Flower's PathologicalPartitioner."""

    scheme: Literal["pathological"] = "pathological"
    num_clients: int = Field(20, ge=1)
    partition_by: str | None = None
    num_classes_per_partition: int = Field(ge=1)
    class_assignment_mode: Literal[
        "random", "deterministic", "first-deterministic"
    ] = "random"


class ShardSettings(PartitionSettingsBase):
    """Arguments passed to Flower's ShardPartitioner."""

    scheme: Literal["shard"] = "shard"
    num_clients: int = Field(20, ge=1)
    partition_by: str | None = None
    num_shards_per_partition: int | None = Field(None, ge=1)
    shard_size: int | None = Field(None, ge=1)
    keep_incomplete_shard: bool = False

    @model_validator(mode="after")
    def _shard_definition_is_present(self):
        if self.num_shards_per_partition is None and self.shard_size is None:
            raise ValueError(
                "shard partitioning requires num_shards_per_partition or shard_size"
            )
        return self


class SizeSettings(PartitionSettingsBase):
    """Arguments passed to Flower's SizePartitioner."""

    scheme: Literal["size"] = "size"
    partition_sizes: list[PositiveInt] = Field(min_length=1)


class SquareSettings(PartitionSettingsBase):
    """Arguments passed to Flower's SquarePartitioner."""

    scheme: Literal["square"] = "square"
    num_clients: int = Field(20, ge=1)


PartitionSettings = Annotated[
    ContinuousSettings
    | DirichletSettings
    | DistributionSettings
    | ExponentialSettings
    | GroupedNaturalIdSettings
    | IidSettings
    | InnerDirichletSettings
    | LinearSettings
    | NaturalIdSettings
    | PathologicalSettings
    | ShardSettings
    | SizeSettings
    | SquareSettings,
    Field(discriminator="scheme"),
]


class FlowerDatasetSettings(BaseModel):
    """One user-named Hugging Face dataset consumed through Flower."""

    model_config = ConfigDict(extra="forbid")

    backend: Literal["flower"] = "flower"
    source_dataset: str
    source_subset: str | None = None
    source_splits: SourceSplits | None = None
    input_column: str | None = None
    target_column: str | None = None
    task: Literal["auto", "classification", "regression"] = "auto"
    preprocessing: PreprocessingSettings = Field(default_factory=PreprocessingSettings)
    partition: PartitionSettings = Field(default_factory=DirichletSettings)


class BioSiloDatasetSettings(BaseModel):
    """One existing BioSilo partition consumed without copying its data."""

    model_config = ConfigDict(extra="forbid")

    backend: Literal["biosilo"] = "biosilo"
    source_dataset: str
    partition: str
    data_root: str | None = None
    validation_fraction: float = Field(0.2, gt=0, lt=1)

    @model_validator(mode="after")
    def _partition_is_named(self):
        if not self.partition.strip():
            raise ValueError("a BioSilo dataset entry requires a partition id")
        return self


DatasetSettings = Annotated[
    FlowerDatasetSettings | BioSiloDatasetSettings,
    Field(discriminator="backend"),
]


class DatasetRegistry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    datasets: dict[str, DatasetSettings]


def load_dataset_registry(path: str | Path = DEFAULT_DATASET_CONFIG) -> DatasetRegistry:
    """Load the shared registry containing every user-named dataset source."""
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(
            f"dataset configuration not found: {config_path}. "
            "Set experiment.dataset_config or run from the RigFL repository root."
        )
    loaded = yaml.safe_load(config_path.read_text()) or {}
    return DatasetRegistry.model_validate(loaded)


def dataset_settings(
    dataset: str, path: str | Path = DEFAULT_DATASET_CONFIG
) -> DatasetSettings:
    registry = load_dataset_registry(path)
    try:
        return registry.datasets[dataset]
    except KeyError as exc:
        known = ", ".join(sorted(registry.datasets)) or "(none)"
        raise KeyError(
            f"dataset {dataset!r} is not defined in {path}; known datasets: {known}"
        ) from exc
