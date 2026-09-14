"""Generate a configured client-data partition before running experiments.

    python -m rigfl.data.generate --dataset cifar10
"""

from __future__ import annotations

import argparse

from rigfl.data.config import (
    BioSiloDatasetSettings,
    FlowerDatasetSettings,
    dataset_settings,
)
from rigfl.data.partitions import (
    DEFAULT_DATA_DIR,
    DEFAULT_DATASET_CONFIG,
    generate_partition,
)


def _nonnegative_seed(value: str) -> int:
    seed = int(value)
    if seed < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the configured client-data partition for a dataset."
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dataset-config", default=DEFAULT_DATASET_CONFIG)
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--partition-seed", type=_nonnegative_seed)
    parser.add_argument("--split-seed", type=_nonnegative_seed)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = dataset_settings(args.dataset, args.dataset_config)
    if isinstance(settings, BioSiloDatasetSettings):
        from rigfl.data.biosilo import generate_biosilo_partition

        updates = {}
        if args.partition_seed is not None:
            updates["parameters"] = {
                **settings.parameters,
                "seed": args.partition_seed,
            }
        if args.split_seed is not None:
            updates["split_seed"] = args.split_seed
        if updates:
            settings = settings.model_copy(update=updates)
        artifact, created = generate_biosilo_partition(settings, data_dir=args.data_dir)
    else:
        if not isinstance(settings, FlowerDatasetSettings):
            raise TypeError(f"unsupported dataset backend: {settings.backend!r}")
        overrides = {
            name: value
            for name, value in {
                "partition_seed": args.partition_seed,
                "split_seed": args.split_seed,
            }.items()
            if value is not None
        }
        if overrides:
            settings = settings.model_copy(
                update={
                    "partition": settings.partition.model_copy(update=overrides)
                }
            )
        artifact, created = generate_partition(
            args.dataset,
            config_path=args.dataset_config,
            data_dir=args.data_dir,
            settings=settings,
        )
    action = "generated" if created else "already exists"
    print(f"{action}: {artifact.path}")
    print(f"partition fingerprint: {artifact.partition_id}")


if __name__ == "__main__":
    main()
