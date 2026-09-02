"""Generic Hugging Face dataset partitioning through Flower Datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from datasets import (
    ClassLabel,
    Image,
    get_dataset_config_names,
    get_dataset_split_names,
    load_dataset_builder,
)

from rigfl.data.builder import _train_val_indices
from rigfl.data.config import FlowerDatasetSettings, SourceSplits


@dataclass(frozen=True)
class ResolvedFlowerSource:
    """Source choices resolved from Hugging Face metadata before partitioning."""

    subset: str | None
    splits: SourceSplits
    input_column: str
    target_column: str
    task: str
    features: object
    class_names: list[str] | None


def _available(values) -> str:
    return "\n".join(f"- {value}" for value in values) or "- (none)"


def inspect_flower_source(settings: FlowerDatasetSettings) -> ResolvedFlowerSource:
    """Resolve split and feature roles without creating a client partition."""
    subsets = list(get_dataset_config_names(settings.source_dataset))
    requested_subset = settings.source_subset
    if requested_subset is not None and requested_subset not in subsets:
        raise ValueError(
            f"source_subset={requested_subset!r} does not exist.\n"
            f"Available source subsets:\n{_available(subsets)}"
        )
    if requested_subset is None and len(subsets) == 1:
        requested_subset = subsets[0]
    try:
        builder = load_dataset_builder(settings.source_dataset, name=requested_subset)
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
        get_dataset_split_names(settings.source_dataset, config_name=subset)
    )
    features = builder.info.features
    if features is None:
        raise ValueError(
            f"{settings.source_dataset!r} does not publish a feature schema; "
            "RigFL cannot infer its model input and target columns"
        )
    columns = list(features)

    if settings.source_splits is None:
        missing = [name for name in ("train", "test") if name not in splits]
        if missing:
            raise ValueError(
                "RigFL could not infer the required train/test source splits.\n"
                f"Available source splits:\n{_available(splits)}\n"
                "Set source_splits in the dataset configuration."
            )
        source_splits = SourceSplits(train="train", test="test")
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

    if settings.target_column is not None:
        if settings.target_column not in features:
            raise ValueError(
                f"target_column={settings.target_column!r} does not exist.\n"
                f"Available columns:\n{_available(columns)}"
            )
        target_column = settings.target_column
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

    if settings.input_column is not None:
        if settings.input_column not in features:
            raise ValueError(
                f"input_column={settings.input_column!r} does not exist.\n"
                f"Available columns:\n{_available(columns)}"
            )
        input_column = settings.input_column
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
    task = inferred_task if settings.task == "auto" else settings.task
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
    class_names = list(target_feature.names) if isinstance(target_feature, ClassLabel) else None
    return ResolvedFlowerSource(
        subset=subset,
        splits=source_splits,
        input_column=input_column,
        target_column=target_column,
        task=task,
        features=features,
        class_names=class_names,
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


def _image_tensor(values, mean, std) -> torch.Tensor:
    tensors = []
    for value in values:
        array = np.asarray(value)
        if array.ndim == 2:
            array = array[:, :, None]
        if array.ndim != 3:
            raise ValueError(f"image input must have 2 or 3 dimensions, got {array.shape}")
        tensor = torch.from_numpy(np.array(array, copy=True)).permute(2, 0, 1).float()
        if np.issubdtype(array.dtype, np.integer):
            tensor = tensor / float(np.iinfo(array.dtype).max)
        tensors.append(tensor)
    try:
        output = torch.stack(tensors)
    except RuntimeError as exc:
        shapes = sorted({tuple(tensor.shape) for tensor in tensors})
        raise ValueError(
            f"images have inconsistent shapes {shapes}; configure a resizing transform "
            "before using this dataset"
        ) from exc
    if mean is not None:
        if output.shape[1] != len(mean):
            raise ValueError(
                f"normalization defines {len(mean)} channels but inputs have {output.shape[1]}"
            )
        mean_tensor = torch.tensor(mean, dtype=output.dtype).view(1, -1, 1, 1)
        std_tensor = torch.tensor(std, dtype=output.dtype).view(1, -1, 1, 1)
        output = (output - mean_tensor) / std_tensor
    return output


def _convert_partition(partition, resolved: ResolvedFlowerSource, settings):
    feature = partition.features[resolved.input_column]
    preprocessing = settings.preprocessing
    conversion = preprocessing.type
    if conversion == "auto":
        conversion = "image" if isinstance(feature, Image) else "numeric"
    values = partition[resolved.input_column]
    if conversion == "image":
        inputs = _image_tensor(values, preprocessing.mean, preprocessing.std)
        input_kind = "image"
    else:
        if preprocessing.mean is not None:
            raise ValueError("mean/std normalization is supported only for image inputs")
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


def generate_flower_partition(
    settings: FlowerDatasetSettings,
    output_directory: Path,
) -> dict:
    """Generate client tensor files and return metadata for the artifact manifest."""
    from flwr_datasets import FederatedDataset

    resolved = inspect_flower_source(settings)
    partition_factory = FLOWER_PARTITIONERS[settings.partition.scheme]
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
    fds = FederatedDataset(
        dataset=settings.source_dataset,
        subset=resolved.subset,
        partitioners=partitioners,
        shuffle=settings.partition.shuffle,
        seed=settings.partition.partition_seed,
    )

    p = settings.partition
    initial_partitions, num_clients = _initialize_partitions(
        fds, role_to_source, p.scheme
    )
    limits = {
        "train": p.train_per_client,
        "validation": p.validation_per_client,
        "test": p.test_per_client,
    }
    clients = []
    client_targets = []
    observed_targets = []
    input_shape = None
    target_shape = None
    input_kind = None
    for cid in range(num_clients):
        client_directory = output_directory / "clients" / f"client_{cid}"
        client_directory.mkdir(parents=True)

        converted = {}
        for offset, (role, source_split) in enumerate(role_to_source.items()):
            partition = (
                initial_partitions[role]
                if cid == 0
                else fds.load_partition(cid, source_split)
            )
            partition = _cap(
                partition,
                limits[role],
                p.partition_seed + cid * 17 + offset,
            )
            x, y, kind = _convert_partition(partition, resolved, settings)
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
        if "validation" in converted:
            x_validation, y_validation = converted["validation"]
            train_indices = range(len(y_train))
            validation_indices = range(len(y_validation))
        else:
            generator = torch.Generator().manual_seed(p.partition_seed + cid)
            train_indices, validation_indices = _train_val_indices(
                len(y_train), None, p.val_frac, generator=generator
            )
            x_validation, y_validation = x_train, y_train
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
        clients.append(
            {
                "client_id": cid,
                "train": len(train_targets),
                "validation": len(validation_targets),
                "test": len(y_test),
            }
        )

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
            "splits": resolved.splits.model_dump(mode="json"),
            "input_column": resolved.input_column,
            "target_column": resolved.target_column,
        },
        "task": resolved.task,
        "num_clients": num_clients,
        "input_spec": {"kind": input_kind, "shape": input_shape},
        "target_spec": target_spec,
        "clients": clients,
    }
