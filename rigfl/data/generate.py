"""Generate a configured client-data partition before running experiments.

    python -m rigfl.data.generate --dataset cifar10
"""

from __future__ import annotations

import argparse

from rigfl.data.partitions import (
    DEFAULT_DATASET_CONFIG,
    DEFAULT_DATA_DIR,
    generate_partition,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the configured client-data partition for a dataset."
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dataset-config", default=DEFAULT_DATASET_CONFIG)
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifact, created = generate_partition(
        args.dataset, config_path=args.dataset_config, data_dir=args.data_dir
    )
    action = "generated" if created else "already exists"
    print(f"{action}: {artifact.path}")
    print(f"partition fingerprint: {artifact.partition_id}")


if __name__ == "__main__":
    main()
