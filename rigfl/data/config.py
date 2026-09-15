"""Validated configuration for named dataset sources and client partitions."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from rigfl.data.transforms import get_data_transform

DEFAULT_DATASET_CONFIG = "configs/datasets.yaml"
DEFAULT_DATA_DIR = "data"


class SourceSplits(BaseModel):
    """Map RigFL's split roles to names used by the source dataset."""

    model_config = ConfigDict(extra="forbid")

    train: str = Field(description="Source split used for client training data.")
    test: str = Field(description="Source split used for client test data.")
    validation: str | None = Field(
        None, description="Optional source split used for client validation data."
    )


class MergedSourceSplits(BaseModel):
    """Source splits merged before creating client partitions."""

    model_config = ConfigDict(extra="forbid")

    merge_splits: list[str] = Field(
        min_length=1, description="Source splits combined before client partitioning."
    )

    @model_validator(mode="after")
    def _split_names_are_unique(self):
        if len(set(self.merge_splits)) != len(self.merge_splits):
            raise ValueError("source_splits.merge_splits must contain unique names")
        return self


class ClientSplitSettings(BaseModel):
    """Fractions used to divide each merged client partition."""

    model_config = ConfigDict(extra="forbid")

    validation_fraction: float = Field(
        gt=0,
        lt=1,
        description="Fraction of each client partition reserved for validation.",
    )
    test_fraction: float = Field(
        gt=0,
        lt=1,
        description="Fraction of each client partition reserved for testing.",
    )
    stratify: bool = Field(
        description="Preserve target proportions when creating client splits."
    )

    @model_validator(mode="after")
    def _leave_training_data(self):
        if self.validation_fraction + self.test_fraction >= 1:
            raise ValueError(
                "client_split.validation_fraction and test_fraction must sum to "
                "less than one"
            )
        return self


class PartitionSettingsBase(BaseModel):
    """Settings shared by every horizontal Flower partitioner."""

    model_config = ConfigDict(extra="forbid")

    partition_seed: int = Field(
        0, ge=0, description="Seed used to assign data to clients."
    )
    split_seed: int = Field(
        0, ge=0, description="Seed used to create client validation splits."
    )
    shuffle: bool = Field(True, description="Shuffle source rows before partitioning.")
    train_per_client: int | None = Field(
        2000,
        ge=1,
        description="Maximum training samples saved per client; null keeps all samples.",
    )
    validation_per_client: int | None = Field(
        None,
        ge=1,
        description="Maximum validation samples saved per client; null keeps all samples.",
    )
    test_per_client: int | None = Field(
        500,
        ge=1,
        description="Maximum test samples saved per client; null keeps all samples.",
    )
    val_frac: float = Field(
        0.2,
        gt=0,
        lt=1,
        description="Fraction of client training data reserved for validation.",
    )


PositiveFloat = Annotated[float, Field(gt=0)]
PositiveInt = Annotated[int, Field(ge=1)]
Alpha = PositiveFloat | list[PositiveFloat]


class ContinuousSettings(PartitionSettingsBase):
    """Arguments passed to Flower's ContinuousPartitioner."""

    scheme: Literal["continuous"] = "continuous"
    num_clients: int = Field(5, ge=1, description="Number of client partitions.")
    partition_by: str = Field(description="Column used to order samples.")
    strictness: float = Field(
        ge=0, le=1, description="Strength of the continuous ordering constraint."
    )


class DirichletSettings(PartitionSettingsBase):
    """Arguments passed to Flower's DirichletPartitioner."""

    scheme: Literal["dirichlet"] = "dirichlet"
    num_clients: int = Field(5, ge=1, description="Number of client partitions.")
    partition_by: str | None = Field(
        None, description="Column whose values are distributed across clients."
    )
    alpha: Alpha = Field(
        0.1, description="Dirichlet concentration for client distributions."
    )
    min_partition_size: int = Field(
        10, ge=1, description="Minimum samples assigned to each client."
    )
    self_balancing: bool = Field(
        False,
        description="Reduce assignments to clients that already exceed the average size.",
    )

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
    distribution_array: list[list[float]] = Field(
        description="Requested label distribution for each client."
    )
    num_clients: int = Field(5, ge=1, description="Number of client partitions.")
    num_unique_labels_per_partition: int = Field(
        ge=1, description="Labels represented in each client partition."
    )
    partition_by: str | None = Field(
        None, description="Column whose values are distributed across clients."
    )
    preassigned_num_samples_per_label: int = Field(
        ge=0, description="Samples reserved per label before distribution."
    )
    rescale: bool = Field(
        True, description="Rescale requested distributions to available samples."
    )


class ExponentialSettings(PartitionSettingsBase):
    """Arguments passed to Flower's ExponentialPartitioner."""

    scheme: Literal["exponential"] = "exponential"
    num_clients: int = Field(5, ge=1, description="Number of client partitions.")


class GroupedNaturalIdSettings(PartitionSettingsBase):
    """Arguments passed to Flower's GroupedNaturalIdPartitioner."""

    scheme: Literal["grouped_natural_id"] = "grouped_natural_id"
    partition_by: str = Field(
        description="Column containing natural client identifiers."
    )
    group_size: int = Field(
        ge=1, description="Natural identifiers grouped into each client."
    )
    mode: Literal["allow-smaller", "allow-bigger", "drop-reminder", "strict"] = Field(
        "allow-smaller", description="Handling of a final incomplete group."
    )
    sort_unique_ids: bool = Field(
        True, description="Sort natural identifiers before grouping."
    )
    client_limit: int | None = Field(
        None, ge=1, description="Maximum number of generated clients."
    )


class IidSettings(PartitionSettingsBase):
    """Arguments passed to Flower's IidPartitioner."""

    scheme: Literal["iid"] = "iid"
    num_clients: int = Field(5, ge=1, description="Number of client partitions.")


class InnerDirichletSettings(PartitionSettingsBase):
    """Arguments passed to Flower's InnerDirichletPartitioner."""

    scheme: Literal["inner_dirichlet"] = "inner_dirichlet"
    partition_sizes: list[PositiveInt] = Field(
        min_length=1, description="Requested size of each client partition."
    )
    partition_by: str | None = Field(
        None, description="Column whose values are distributed across clients."
    )
    alpha: Alpha = Field(
        0.1, description="Dirichlet concentration within each partition."
    )


class LinearSettings(PartitionSettingsBase):
    """Arguments passed to Flower's LinearPartitioner."""

    scheme: Literal["linear"] = "linear"
    num_clients: int = Field(5, ge=1, description="Number of client partitions.")


class NaturalIdSettings(PartitionSettingsBase):
    """Arguments passed to Flower's NaturalIdPartitioner."""

    scheme: Literal["natural_id"] = "natural_id"
    partition_by: str = Field(
        description="Column containing natural client identifiers."
    )
    client_limit: int | None = Field(
        None, ge=1, description="Maximum number of generated clients."
    )


class PathologicalSettings(PartitionSettingsBase):
    """Arguments passed to Flower's PathologicalPartitioner."""

    scheme: Literal["pathological"] = "pathological"
    num_clients: int = Field(5, ge=1, description="Number of client partitions.")
    partition_by: str | None = Field(
        None, description="Column whose values are distributed across clients."
    )
    num_classes_per_partition: int = Field(
        ge=1, description="Classes assigned to each client."
    )
    class_assignment_mode: Literal["random", "deterministic", "first-deterministic"] = (
        Field("random", description="How classes are assigned to clients.")
    )


class ShardSettings(PartitionSettingsBase):
    """Arguments passed to Flower's ShardPartitioner."""

    scheme: Literal["shard"] = "shard"
    num_clients: int = Field(5, ge=1, description="Number of client partitions.")
    partition_by: str | None = Field(
        None, description="Column used to form label-sorted shards."
    )
    num_shards_per_partition: int | None = Field(
        None, ge=1, description="Shards assigned to each client."
    )
    shard_size: int | None = Field(
        None, ge=1, description="Number of samples in each shard."
    )
    keep_incomplete_shard: bool = Field(
        False, description="Retain a final shard smaller than the requested size."
    )

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
    partition_sizes: list[PositiveInt] = Field(
        min_length=1, description="Requested size of each client partition."
    )


class SquareSettings(PartitionSettingsBase):
    """Arguments passed to Flower's SquarePartitioner."""

    scheme: Literal["square"] = "square"
    num_clients: int = Field(5, ge=1, description="Number of client partitions.")


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
    source_dataset: str = Field(description="Hugging Face dataset identifier.")
    source_subset: str | None = Field(
        None, description="Optional Hugging Face dataset subset."
    )
    source_revision: str | None = Field(
        None, description="Optional Hugging Face dataset revision."
    )
    source_splits: SourceSplits | MergedSourceSplits | None = Field(
        None, description="Source splits used to build client data."
    )
    client_split: ClientSplitSettings | None = Field(
        None, description="Client split fractions used after merging source splits."
    )
    input_column: str | None = Field(
        None, description="Source column containing model inputs."
    )
    target_column: str | None = Field(
        None, description="Source column containing prediction targets."
    )
    task: Literal["auto", "classification"] = Field(
        "auto",
        description="Use source metadata or explicitly treat the target as class labels.",
    )
    data_transform: str = Field(
        "auto",
        min_length=1,
        description="Registered transform applied to source examples.",
    )
    partition: PartitionSettings = Field(
        default_factory=DirichletSettings,
        description="Client partitioning strategy and limits.",
    )

    @field_validator("data_transform")
    @classmethod
    def _data_transform_is_registered(cls, value):
        get_data_transform(value)
        return value

    @model_validator(mode="after")
    def _client_split_matches_source_splits(self):
        merged = isinstance(self.source_splits, MergedSourceSplits)
        if merged and self.client_split is None:
            raise ValueError(
                "client_split is required when source_splits.merge_splits is set"
            )
        if not merged and self.client_split is not None:
            raise ValueError("client_split requires source_splits.merge_splits")
        return self


class BioSiloDatasetSettings(BaseModel):
    """One BioSilo dataset generated and consumed through RigFL."""

    model_config = ConfigDict(extra="forbid")

    backend: Literal["biosilo"] = "biosilo"
    source_dataset: str = Field(min_length=1, description="BioSilo dataset name.")
    data_root: str | None = Field(None, description="Optional BioSilo storage root.")
    parameters: dict[str, Any] = Field(
        default_factory=dict, description="Parameters passed to the BioSilo dataset."
    )
    validation_fraction: float = Field(
        0.2,
        gt=0,
        lt=1,
        description="Fraction of client training data reserved for validation.",
    )
    split_seed: int = Field(
        0, ge=0, description="Seed used to create client validation splits."
    )

    @field_validator("source_dataset")
    @classmethod
    def _value_is_not_blank(cls, value):
        if not value.strip():
            raise ValueError("must not be blank")
        return value

    @field_validator("parameters")
    @classmethod
    def _parameters_do_not_override_rigfl_controls(cls, value):
        reserved = {"dataset", "params", "root", "overwrite", "version"}
        overlap = sorted(reserved & set(value))
        if overlap:
            raise ValueError(
                "parameters cannot contain RigFL-controlled BioSilo arguments: "
                + ", ".join(overlap)
            )
        return value


DatasetSettings = Annotated[
    FlowerDatasetSettings | BioSiloDatasetSettings,
    Field(discriminator="backend"),
]


_SEEDED_PARTITIONERS = {
    "continuous",
    "dirichlet",
    "distribution",
    "inner_dirichlet",
    "pathological",
    "shard",
}


def inactive_replicate_seed_fields(settings: DatasetSettings) -> set[str]:
    """Return data-seed fields that cannot affect this dataset configuration."""
    if isinstance(settings, BioSiloDatasetSettings):
        return set()

    inactive = set()
    if (
        isinstance(settings.source_splits, SourceSplits)
        and settings.source_splits.validation is not None
    ):
        inactive.add("split_seed")

    partition = settings.partition
    caps_samples = any(
        getattr(partition, name) is not None
        for name in (
            "train_per_client",
            "validation_per_client",
            "test_per_client",
        )
    )
    selects_clients = getattr(partition, "client_limit", None) is not None
    merges_splits = isinstance(settings.source_splits, MergedSourceSplits)
    if not any(
        (
            partition.shuffle,
            caps_samples,
            selects_clients,
            merges_splits,
            partition.scheme in _SEEDED_PARTITIONERS,
        )
    ):
        inactive.add("partition_seed")
    return inactive


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
