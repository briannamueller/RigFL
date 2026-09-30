"""Generate a configured client-data partition before running experiments.

    rigfl data generate --dataset mnist
"""

from __future__ import annotations

import argparse

from rigfl.data.config import (
    BioSiloDatasetSettings,
    dataset_settings,
    with_seed_overrides,
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


def parse_args(
    argv: list[str] | None = None, *, prog: str | None = None
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Generate the configured client-data partition for a dataset."
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument(
        "--config", default=DEFAULT_DATASET_CONFIG,
        help="dataset registry YAML (default: configs/datasets.yaml)",
    )
    parser.add_argument(
        "--data-dir", default=DEFAULT_DATA_DIR,
        help="directory where partitions are stored (default: data)",
    )
    parser.add_argument(
        "--partition-seed", type=_nonnegative_seed,
        help="override the dataset entry's partition seed",
    )
    parser.add_argument(
        "--split-seed", type=_nonnegative_seed,
        help="override the dataset entry's split seed",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None, *, prog: str | None = None) -> None:
    args = parse_args(argv, prog=prog)
    settings = with_seed_overrides(
        dataset_settings(args.dataset, args.config),
        args.partition_seed,
        args.split_seed,
    )
    if isinstance(settings, BioSiloDatasetSettings):
        from rigfl.data.biosilo import generate_biosilo_partition

        artifact, created = generate_biosilo_partition(settings, data_dir=args.data_dir)
    else:
        artifact, created = generate_partition(
            args.dataset,
            config_path=args.config,
            data_dir=args.data_dir,
            settings=settings,
        )
    action = "generated" if created else "already exists"
    print(f"{action}: {artifact.path}")
    print(f"partition fingerprint: {artifact.partition_id}")


if __name__ == "__main__":
    main()
