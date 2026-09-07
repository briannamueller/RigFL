"""Generic Hugging Face dataset partitioning through Flower Datasets."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from pathlib import Path

import numpy as np
import torch
from datasets import (
    ClassLabel,
    DatasetDict,
    Image,
    concatenate_datasets,
    get_dataset_config_names,
    get_dataset_split_names,
    load_dataset_builder,
)

from rigfl.data.builder import _train_val_indices
from rigfl.data.config import (
    FlowerDatasetSettings,
    MergedSourceSplits,
    SourceSplits,
)
from rigfl.data.transforms import DataTransform, get_data_transform
from rigfl.data.transforms.image import image_tensor as _image_tensor


_MERGED_SPLIT = "merged"


@dataclass(frozen=True)
class ResolvedFlowerSource:
    """Source choices resolved from Hugging Face metadata before partitioning."""

    subset: str | None
    splits: SourceSplits | MergedSourceSplits
    input_column: str
    target_column: str
    task: str
    features: object
    class_names: list[str] | None
    data_transform: DataTransform


def _available(values) -> str:
    return "\n".join(f"- {value}" for value in values) or "- (none)"


def inspect_flower_source(settings: FlowerDatasetSettings) -> ResolvedFlowerSource:
    """Resolve split and feature roles without creating a client partition."""
    data_transform = get_data_transform(settings.data_transform)
    subsets = list(
        get_dataset_config_names(
            settings.source_dataset, revision=settings.source_revision
        )
    )
    requested_subset = settings.source_subset
    if requested_subset is not None and requested_subset not in subsets:
        raise ValueError(
            f"source_subset={requested_subset!r} does not exist.\n"
            f"Available source subsets:\n{_available(subsets)}"
        )
    if requested_subset is None and len(subsets) == 1:
        requested_subset = subsets[0]
    try:
        builder = load_dataset_builder(
            settings.source_dataset,
            name=requested_subset,
            revision=settings.source_revision,
        )
    except ValueError as exc:
        if requested_subset is None:
            raise ValueError(
                "The source has multiple configurations and no unambiguous default.\n"
                f"Available source subsets:\n{_available(subsets)}\n"
                "Set source_subset in the dataset configuration."
            ) from exc
        raise
    subset = getattr(builder.config, "name", requested_subset)
    splits = list(
        get_dataset_split_names(
            settings.source_dataset,
            config_name=subset,
            revision=settings.source_revision,
        )
    )
    features = builder.info.features
    if features is None:
        raise ValueError(
            f"{settings.source_dataset!r} does not publish a feature schema; "
            "RigFL cannot infer its model input and target columns"
        )
    columns = list(features)
    missing_columns = [
        column for column in data_transform.required_columns if column not in features
    ]
    if missing_columns:
        missing = ", ".join(repr(column) for column in missing_columns)
        raise ValueError(
            f"data_transform={settings.data_transform!r} requires missing columns: "
            f"{missing}.\nAvailable columns:\n{_available(columns)}"
        )

    if settings.source_splits is None:
        missing = [name for name in ("train", "test") if name not in splits]
        if missing:
            raise ValueError(
                "RigFL could not infer the required train/test source splits.\n"
                f"Available source splits:\n{_available(splits)}\n"
                "Set source_splits in the dataset configuration."
            )
        source_splits = SourceSplits(train="train", test="test")
    elif isinstance(settings.source_splits, MergedSourceSplits):
        source_splits = settings.source_splits
        invalid = [name for name in source_splits.merge_splits if name not in splits]
        if invalid:
            given = ", ".join(repr(name) for name in invalid)
            raise ValueError(
                f"Invalid split name in source_splits.merge_splits: {given}.\n"
                f"Available source splits:\n{_available(splits)}"
            )
    else:
        source_splits = settings.source_splits
        requested = {
            "train": source_splits.train,
            "test": source_splits.test,
            **(
                {"validation": source_splits.validation}
                if source_splits.validation is not None
                else {}
            ),
        }
        invalid = {role: name for role, name in requested.items() if name not in splits}
        if invalid:
            given = ", ".join(f"{role}={name!r}" for role, name in invalid.items())
            raise ValueError(
                f"Invalid source split mapping: {given}.\n"
                f"Available source splits:\n{_available(splits)}"
            )
        if len(set(requested.values())) != len(requested):
            raise ValueError("source_splits must map each RigFL role to a different source split")

    supervised = builder.info.supervised_keys
    supervised_input = getattr(supervised, "input", None)
    supervised_target = getattr(supervised, "output", None)
    if isinstance(supervised, (tuple, list)) and len(supervised) == 2:
        supervised_input, supervised_target = supervised

    if (
        settings.target_column is not None
        and data_transform.target_column is not None
        and settings.target_column != data_transform.target_column
    ):
        raise ValueError(
            f"data_transform={settings.data_transform!r} uses target column "
            f"{data_transform.target_column!r}, not {settings.target_column!r}"
        )
    configured_target = data_transform.target_column or settings.target_column
    if configured_target is not None:
        if configured_target not in features:
            raise ValueError(
                f"target_column={configured_target!r} does not exist.\n"
                f"Available columns:\n{_available(columns)}"
            )
        target_column = configured_target
    elif supervised_target in features:
        target_column = supervised_target
    else:
        candidates = [name for name, feature in features.items() if isinstance(feature, ClassLabel)]
        if len(candidates) != 1:
            raise ValueError(
                "RigFL could not infer one target column.\n"
                f"ClassLabel columns:\n{_available(candidates)}\n"
                "Set target_column in the dataset configuration."
            )
        target_column = candidates[0]

    if (
        settings.input_column is not None
        and data_transform.input_column is not None
        and settings.input_column != data_transform.input_column
    ):
        raise ValueError(
            f"data_transform={settings.data_transform!r} uses input column "
            f"{data_transform.input_column!r}, not {settings.input_column!r}"
        )
    configured_input = data_transform.input_column or settings.input_column
    if configured_input is not None:
        if configured_input not in features and data_transform.prepare is None:
            raise ValueError(
                f"input_column={configured_input!r} does not exist.\n"
                f"Available columns:\n{_available(columns)}"
            )
        input_column = configured_input
    elif supervised_input in features and supervised_input != target_column:
        input_column = supervised_input
    else:
        image_candidates = [
            name for name, feature in features.items()
            if name != target_column and isinstance(feature, Image)
        ]
        remaining = [
            name for name, feature in features.items()
            if name != target_column and not isinstance(feature, ClassLabel)
        ]
        candidates = image_candidates if len(image_candidates) == 1 else remaining
        if len(candidates) != 1:
            raise ValueError(
                "RigFL could not infer one input column.\n"
                f"Candidate input columns:\n{_available(candidates)}\n"
                "Set input_column in the dataset configuration."
            )
        input_column = candidates[0]

    if input_column == target_column:
        raise ValueError("input_column and target_column must be different")

    configured_partition_column = getattr(settings.partition, "partition_by", None)
    if (
        configured_partition_column is not None
        and configured_partition_column not in features
    ):
        raise ValueError(
            f"partition.partition_by={configured_partition_column!r} does not exist.\n"
            f"Available columns:\n{_available(columns)}"
        )

    target_feature = features[target_column]
    inferred_task = "classification" if isinstance(target_feature, ClassLabel) else "regression"
    if (
        settings.task != "auto"
        and data_transform.task is not None
        and settings.task != data_transform.task
    ):
        raise ValueError(
            f"data_transform={settings.data_transform!r} defines a "
            f"{data_transform.task} task, not {settings.task}"
        )
    task = data_transform.task or (
        inferred_task if settings.task == "auto" else settings.task
    )
    if (
        isinstance(source_splits, MergedSourceSplits)
        and settings.client_split is not None
        and settings.client_split.stratify
        and task != "classification"
    ):
        raise ValueError(
            "client_split.stratify is supported only for classification targets"
        )
    label_partitioners = {
        "dirichlet",
        "distribution",
        "inner_dirichlet",
        "pathological",
        "shard",
    }
    if (
        task == "regression"
        and settings.partition.scheme in label_partitioners
        and configured_partition_column is None
    ):
        raise ValueError(
            f"{settings.partition.scheme!r} partitioning groups samples by discrete "
            "values and is not supported for regression targets by default. Set "
            "partition.partition_by to a categorical column or choose a compatible "
            "partitioner"
        )
    class_names = (
        list(data_transform.class_names)
        if data_transform.class_names is not None
        else list(target_feature.names)
        if isinstance(target_feature, ClassLabel)
        else None
    )
    return ResolvedFlowerSource(
        subset=subset,
        splits=source_splits,
        input_column=input_column,
        target_column=target_column,
        task=task,
        features=features,
        class_names=class_names,
        data_transform=data_transform,
    )


def _partition_column(settings, target_column: str) -> str:
    """Use an explicitly configured column or default to the prediction target."""
    return getattr(settings, "partition_by", None) or target_column


def _continuous(settings, target_column: str):
    from flwr_datasets.partitioner import ContinuousPartitioner

    return ContinuousPartitioner(
        num_partitions=settings.num_clients,
        partition_by=_partition_column(settings, target_column),
        strictness=settings.strictness,
        shuffle=settings.shuffle,
        seed=settings.partition_seed,
    )


def _dirichlet(settings, target_column: str):
    from flwr_datasets.partitioner import DirichletPartitioner

    return DirichletPartitioner(
        num_partitions=settings.num_clients,
        partition_by=_partition_column(settings, target_column),
        alpha=settings.alpha,
        min_partition_size=settings.min_partition_size,
        self_balancing=settings.self_balancing,
        shuffle=settings.shuffle,
        seed=settings.partition_seed,
    )


def _distribution(settings, target_column: str):
    from flwr_datasets.partitioner import DistributionPartitioner

    return DistributionPartitioner(
        distribution_array=np.asarray(settings.distribution_array),
        num_partitions=settings.num_clients,
        num_unique_labels_per_partition=settings.num_unique_labels_per_partition,
        partition_by=_partition_column(settings, target_column),
        preassigned_num_samples_per_label=(
            settings.preassigned_num_samples_per_label
        ),
        rescale=settings.rescale,
        shuffle=settings.shuffle,
        seed=settings.partition_seed,
    )


def _exponential(settings, _target_column: str):
    from flwr_datasets.partitioner import ExponentialPartitioner

    return ExponentialPartitioner(num_partitions=settings.num_clients)


def _grouped_natural_id(settings, _target_column: str):
    from flwr_datasets.partitioner import GroupedNaturalIdPartitioner

    return GroupedNaturalIdPartitioner(
        partition_by=settings.partition_by,
        group_size=settings.group_size,
        mode=settings.mode,
        sort_unique_ids=settings.sort_unique_ids,
    )


def _iid(settings, _target_column: str):
    from flwr_datasets.partitioner import IidPartitioner

    return IidPartitioner(num_partitions=settings.num_clients)


def _inner_dirichlet(settings, target_column: str):
    from flwr_datasets.partitioner import InnerDirichletPartitioner

    return InnerDirichletPartitioner(
        partition_sizes=settings.partition_sizes,
        partition_by=_partition_column(settings, target_column),
        alpha=settings.alpha,
        shuffle=settings.shuffle,
        seed=settings.partition_seed,
    )


def _linear(settings, _target_column: str):
    from flwr_datasets.partitioner import LinearPartitioner

    return LinearPartitioner(num_partitions=settings.num_clients)


def _natural_id(settings, _target_column: str):
    from flwr_datasets.partitioner import NaturalIdPartitioner

    return NaturalIdPartitioner(partition_by=settings.partition_by)


def _pathological(settings, target_column: str):
    from flwr_datasets.partitioner import PathologicalPartitioner

    return PathologicalPartitioner(
        num_partitions=settings.num_clients,
        partition_by=_partition_column(settings, target_column),
        num_classes_per_partition=settings.num_classes_per_partition,
        class_assignment_mode=settings.class_assignment_mode,
        shuffle=settings.shuffle,
        seed=settings.partition_seed,
    )


def _shard(settings, target_column: str):
    from flwr_datasets.partitioner import ShardPartitioner

    return ShardPartitioner(
        num_partitions=settings.num_clients,
        partition_by=_partition_column(settings, target_column),
        num_shards_per_partition=settings.num_shards_per_partition,
        shard_size=settings.shard_size,
        keep_incomplete_shard=settings.keep_incomplete_shard,
        shuffle=settings.shuffle,
        seed=settings.partition_seed,
    )


def _size(settings, _target_column: str):
    from flwr_datasets.partitioner import SizePartitioner

    return SizePartitioner(partition_sizes=settings.partition_sizes)


def _square(settings, _target_column: str):
    from flwr_datasets.partitioner import SquarePartitioner

    return SquarePartitioner(num_partitions=settings.num_clients)


FLOWER_PARTITIONERS = {
    "continuous": _continuous,
    "dirichlet": _dirichlet,
    "distribution": _distribution,
    "exponential": _exponential,
    "grouped_natural_id": _grouped_natural_id,
    "iid": _iid,
    "inner_dirichlet": _inner_dirichlet,
    "linear": _linear,
    "natural_id": _natural_id,
    "pathological": _pathological,
    "shard": _shard,
    "size": _size,
    "square": _square,
}


def _cap(partition, limit: int | None, seed: int):
    if limit is not None and limit < len(partition):
        return partition.shuffle(seed=seed).select(range(limit))
    return partition


def _merge_source_splits(
    dataset: DatasetDict,
    *,
    split_names: tuple[str, ...],
    shuffle: bool,
    seed: int,
) -> DatasetDict:
    parts = [dataset[name] for name in split_names]
    merged = parts[0] if len(parts) == 1 else concatenate_datasets(parts)
    if shuffle:
        merged = merged.shuffle(seed=seed)
    return DatasetDict({_MERGED_SPLIT: merged})


def _split_client_partition(partition, settings, target_column: str, seed: int):
    stratify_by = target_column if settings.stratify else None
    try:
        train_validation = partition.train_test_split(
            test_size=settings.test_fraction,
            seed=seed,
            stratify_by_column=stratify_by,
        )
        validation_fraction = settings.validation_fraction / (
            1 - settings.test_fraction
        )
        train_validation_split = train_validation["train"].train_test_split(
            test_size=validation_fraction,
            seed=seed + 1,
            stratify_by_column=stratify_by,
        )
    except ValueError as exc:
        suffix = (
            " Set client_split.stratify to false to split without stratification."
            if settings.stratify
            else ""
        )
        raise ValueError(
            "Could not divide a client partition into the requested train, validation, "
            f"and test fractions.{suffix}"
        ) from exc
    return {
        "train": train_validation_split["train"],
        "validation": train_validation_split["test"],
        "test": train_validation["test"],
    }


def _convert_partition(partition, resolved: ResolvedFlowerSource):
    feature = partition.features[resolved.input_column]
    transform = resolved.data_transform
    conversion = transform.input_kind
    if conversion == "auto":
        conversion = "image" if isinstance(feature, Image) else "numeric"
    values = partition[resolved.input_column]
    if conversion == "image":
        inputs = _image_tensor(
            values,
            mean=transform.mean,
            std=transform.std,
            image_mode=transform.image_mode,
        )
        input_kind = "image"
    else:
        try:
            inputs = torch.as_tensor(np.asarray(values)).float()
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"input column {resolved.input_column!r} cannot be converted to one "
                "numeric tensor"
            ) from exc
        if inputs.ndim < 2:
            inputs = inputs.unsqueeze(1)
        input_kind = "numeric"

    target_values = partition[resolved.target_column]
    try:
        targets = torch.as_tensor(np.asarray(target_values))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"target column {resolved.target_column!r} cannot be converted to a tensor"
        ) from exc
    targets = targets.long() if resolved.task == "classification" else targets.float()
    return inputs, targets, input_kind


def _prepare_source(
    dataset: DatasetDict,
    *,
    data_transform: DataTransform,
    training_split: str,
    merge_splits: tuple[str, ...] | None,
    shuffle: bool,
    seed: int,
    fitted_parameters: dict,
) -> DatasetDict:
    if data_transform.prepare is not None:
        dataset, parameters = data_transform.prepare(dataset, training_split)
        fitted_parameters.update(parameters)
    if merge_splits is not None:
        dataset = _merge_source_splits(
            dataset, split_names=merge_splits, shuffle=shuffle, seed=seed
        )
    return dataset


def _save(path: Path, inputs: torch.Tensor, targets: torch.Tensor, indices) -> None:
    index = torch.as_tensor(list(indices), dtype=torch.long)
    torch.save((inputs.index_select(0, index), targets.index_select(0, index)), path)


def _initialize_partitions(fds, role_to_source: dict[str, str], scheme: str):
    """Assign every source split and establish one consistent client count."""
    initial = {}
    counts = {}
    identity_maps = {}
    for role, source_split in role_to_source.items():
        initial[role] = fds.load_partition(0, source_split)
        partitioner = fds.partitioners[source_split]
        counts[role] = partitioner.num_partitions
        if scheme == "natural_id":
            identity_maps[role] = partitioner.partition_id_to_natural_id
        elif scheme == "grouped_natural_id":
            identity_maps[role] = partitioner.partition_id_to_natural_ids

    if len(set(counts.values())) != 1:
        details = ", ".join(f"{role}={count}" for role, count in counts.items())
        raise ValueError(
            "Each dataset split must produce the same number of client partitions; "
            f"got {details}"
        )
    if identity_maps:
        mappings = list(identity_maps.values())
        if any(mapping != mappings[0] for mapping in mappings[1:]):
            raise ValueError(
                f"{scheme!r} partitioning must assign the same natural IDs to each "
                "client across the training, validation, and test splits"
            )
    return initial, next(iter(counts.values()))


def _client_raw_partitions(
    fds,
    *,
    partition_id: int,
    client_id: int,
    merged_source: bool,
    initial_merged,
    initial_partitions,
    role_to_source,
    settings,
    target_column: str,
):
    p = settings.partition
    if merged_source:
        merged_partition = (
            initial_merged
            if partition_id == 0
            else fds.load_partition(partition_id, _MERGED_SPLIT)
        )
        raw = _split_client_partition(
            merged_partition,
            settings.client_split,
            target_column,
            p.partition_seed + client_id * 17,
        )
    else:
        raw = {
            role: (
                initial_partitions[role]
                if partition_id == 0
                else fds.load_partition(partition_id, source_split)
            )
            for role, source_split in role_to_source.items()
        }

    limits = {
        "train": p.train_per_client,
        "validation": p.validation_per_client,
        "test": p.test_per_client,
    }
    raw = {
        role: _cap(
            partition,
            limits[role],
            p.partition_seed + client_id * 17 + offset,
        )
        for offset, (role, partition) in enumerate(raw.items())
    }
    if not merged_source and "validation" not in raw:
        generator = torch.Generator().manual_seed(p.partition_seed + client_id)
        train_indices, validation_indices = _train_val_indices(
            len(raw["train"]), None, p.val_frac, generator=generator
        )
        training = raw["train"]
        raw["train"] = training.select(train_indices)
        raw["validation"] = training.select(validation_indices)
    return raw


def generate_flower_partition(
    settings: FlowerDatasetSettings,
    output_directory: Path,
) -> dict:
    """Generate client tensor files and return metadata for the artifact manifest."""
    from flwr_datasets import FederatedDataset

    resolved = inspect_flower_source(settings)
    partition_factory = FLOWER_PARTITIONERS[settings.partition.scheme]
    merged_source = isinstance(resolved.splits, MergedSourceSplits)
    if merged_source:
        role_to_source = None
        partitioners = {
            _MERGED_SPLIT: partition_factory(
                settings.partition, resolved.target_column
            )
        }
        merge_splits = tuple(resolved.splits.merge_splits)
        training_split = merge_splits[0]
    else:
        role_to_source = {
            "train": resolved.splits.train,
            "test": resolved.splits.test,
            **(
                {"validation": resolved.splits.validation}
                if resolved.splits.validation is not None
                else {}
            ),
        }
        partitioners = {
            source_name: partition_factory(settings.partition, resolved.target_column)
            for source_name in role_to_source.values()
        }
        merge_splits = None
        training_split = resolved.splits.train
    fitted_parameters = {}
    preprocessor = partial(
        _prepare_source,
        data_transform=resolved.data_transform,
        training_split=training_split,
        merge_splits=merge_splits,
        shuffle=settings.partition.shuffle,
        seed=settings.partition.partition_seed,
        fitted_parameters=fitted_parameters,
    )
    fds = FederatedDataset(
        dataset=settings.source_dataset,
        subset=resolved.subset,
        preprocessor=preprocessor,
        partitioners=partitioners,
        shuffle=settings.partition.shuffle,
        seed=settings.partition.partition_seed,
        revision=settings.source_revision,
    )

    p = settings.partition
    if merged_source:
        initial_merged = fds.load_partition(0, _MERGED_SPLIT)
        total_clients = fds.partitioners[_MERGED_SPLIT].num_partitions
        initial_partitions = None
    else:
        initial_partitions, total_clients = _initialize_partitions(
            fds, role_to_source, p.scheme
        )
        initial_merged = None

    client_limit = getattr(p, "client_limit", None)
    if client_limit is not None:
        if client_limit > total_clients:
            raise ValueError(
                f"partition.client_limit={client_limit} exceeds the "
                f"{total_clients} available natural clients"
            )
        selected_partition_ids = sorted(
            np.random.default_rng(p.partition_seed).choice(
                total_clients, size=client_limit, replace=False
            ).tolist()
        )
    else:
        selected_partition_ids = list(range(total_clients))
    num_clients = len(selected_partition_ids)

    identity_map = None
    if p.scheme in {"natural_id", "grouped_natural_id"}:
        split = _MERGED_SPLIT if merged_source else next(iter(role_to_source.values()))
        partitioner = fds.partitioners[split]
        identity_map = (
            partitioner.partition_id_to_natural_id
            if p.scheme == "natural_id"
            else partitioner.partition_id_to_natural_ids
        )

    raw_clients = [
        _client_raw_partitions(
            fds,
            partition_id=partition_id,
            client_id=client_id,
            merged_source=merged_source,
            initial_merged=initial_merged,
            initial_partitions=initial_partitions,
            role_to_source=role_to_source,
            settings=settings,
            target_column=resolved.target_column,
        )
        for client_id, partition_id in enumerate(selected_partition_ids)
    ]
    clients = []
    client_targets = []
    observed_targets = []
    input_shape = None
    target_shape = None
    input_kind = None
    for cid, (partition_id, raw_partitions) in enumerate(
        zip(selected_partition_ids, raw_clients)
    ):
        client_directory = output_directory / "clients" / f"client_{cid}"
        client_directory.mkdir(parents=True)

        converted = {}
        for role, partition in raw_partitions.items():
            x, y, kind = _convert_partition(partition, resolved)
            converted[role] = (x, y)
            current_input_shape = list(x.shape[1:])
            current_target_shape = list(y.shape[1:])
            if input_shape is None:
                input_shape, target_shape, input_kind = (
                    current_input_shape,
                    current_target_shape,
                    kind,
                )
            elif current_input_shape != input_shape or current_target_shape != target_shape:
                raise ValueError("source splits do not share one input and target shape")

        x_train, y_train = converted["train"]
        x_validation, y_validation = converted["validation"]
        train_indices = range(len(y_train))
        validation_indices = range(len(y_validation))
        x_test, y_test = converted["test"]

        _save(client_directory / "train.pt", x_train, y_train, train_indices)
        _save(
            client_directory / "validation.pt",
            x_validation,
            y_validation,
            validation_indices,
        )
        _save(client_directory / "test.pt", x_test, y_test, range(len(y_test)))

        train_targets = y_train[torch.as_tensor(list(train_indices), dtype=torch.long)]
        validation_targets = y_validation[
            torch.as_tensor(list(validation_indices), dtype=torch.long)
        ]
        observed_targets.extend([train_targets, validation_targets, y_test])
        client_targets.append(
            {"train": train_targets, "validation": validation_targets, "test": y_test}
        )
        client = {
            "client_id": cid,
            "train": len(train_targets),
            "validation": len(validation_targets),
            "test": len(y_test),
        }
        if identity_map is not None:
            key = "source_client_id" if p.scheme == "natural_id" else "source_client_ids"
            source_ids = identity_map[partition_id]
            client[key] = (
                list(source_ids) if p.scheme == "grouped_natural_id" else source_ids
            )
        clients.append(client)

    target_spec = {
        "dtype": str(observed_targets[0].dtype).removeprefix("torch."),
        "shape": target_shape,
    }
    if resolved.task == "classification":
        if resolved.class_names is not None:
            num_classes = len(resolved.class_names)
        else:
            labels = torch.cat([target.reshape(-1) for target in observed_targets])
            if labels.numel() == 0 or int(labels.min()) < 0:
                raise ValueError("classification targets must be nonnegative integer class ids")
            num_classes = int(labels.max()) + 1
        target_spec.update(
            {"num_classes": num_classes, "class_names": resolved.class_names}
        )
        for client, targets_by_role in zip(clients, client_targets):
            for role, key in (("train", "train_label_hist"),
                              ("validation", "validation_label_hist"),
                              ("test", "test_label_hist")):
                client[key] = torch.bincount(
                    targets_by_role[role].long(), minlength=num_classes
                ).tolist()

    return {
        "backend": "flower",
        "source": {
            "dataset": settings.source_dataset,
            "subset": resolved.subset,
            "revision": settings.source_revision,
            "splits": resolved.splits.model_dump(mode="json"),
            "client_split": (
                settings.client_split.model_dump(mode="json")
                if settings.client_split is not None
                else None
            ),
            "input_column": resolved.input_column,
            "target_column": resolved.target_column,
            "data_transform": {
                "name": settings.data_transform,
                **resolved.data_transform.identity(),
                "fitted_parameters": fitted_parameters or None,
            },
        },
        "task": resolved.task,
        "num_clients": num_clients,
        "input_spec": {"kind": input_kind, "shape": input_shape},
        "target_spec": target_spec,
        "clients": clients,
    }
