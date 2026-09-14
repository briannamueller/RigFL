"""Run one RigFL grid task inside an AWS Batch array job.

AWS Batch numbers array children from zero while RigFL grid tasks are numbered
from one. This module owns that conversion, downloads the immutable sweep grid
from S3, runs exactly one task, and uploads only completed result artifacts.

Credentials are never passed in the job configuration. Boto3 obtains temporary
credentials from the AWS Batch job role.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
from urllib.parse import urlparse

from rigfl.experiment.launch import run_task


class BatchConfigurationError(ValueError):
    """An invalid or incomplete AWS Batch worker configuration."""


@dataclass(frozen=True)
class S3Location:
    bucket: str
    key: str

    @classmethod
    def parse(cls, uri: str, *, name: str) -> S3Location:
        parsed = urlparse(uri)
        key = parsed.path.lstrip("/")
        if (
            parsed.scheme != "s3"
            or not parsed.netloc
            or not key
            or parsed.params
            or parsed.query
            or parsed.fragment
        ):
            raise BatchConfigurationError(
                f"{name} must be an S3 URI with a bucket and key, got {uri!r}"
            )
        return cls(parsed.netloc, key)

    def child(self, relative: str) -> S3Location:
        return S3Location(self.bucket, f"{self.key.rstrip('/')}/{relative.lstrip('/')}")


class S3Client(Protocol):
    def download_file(self, bucket: str, key: str, filename: str) -> object: ...

    def upload_file(self, filename: str, bucket: str, key: str) -> object: ...

    def put_object(self, **kwargs) -> object: ...


def batch_task_id(
    array_index: int | None = None,
    *,
    environ: Mapping[str, str] | None = None,
) -> tuple[int, int]:
    """Return ``(zero_based_index, one_based_rigfl_task_id)``."""
    if array_index is None:
        raw = (environ or os.environ).get("AWS_BATCH_JOB_ARRAY_INDEX")
        if raw is None:
            raise BatchConfigurationError(
                "AWS_BATCH_JOB_ARRAY_INDEX is missing; run as an array job or pass "
                "--array-index"
            )
        try:
            array_index = int(raw)
        except ValueError as error:
            raise BatchConfigurationError(
                f"AWS_BATCH_JOB_ARRAY_INDEX must be an integer, got {raw!r}"
            ) from error
    if array_index < 0:
        raise BatchConfigurationError("the AWS Batch array index cannot be negative")
    return array_index, array_index + 1


def _work_name(job_id: str, task_id: int) -> str:
    safe_job_id = re.sub(r"[^A-Za-z0-9_.-]", "_", job_id)
    return f"{safe_job_id}-task-{task_id:05d}"


def run_batch_task(
    *,
    grid_uri: str,
    results_uri: str,
    array_index: int,
    job_id: str,
    work_root: Path,
    s3: S3Client,
) -> list[str]:
    """Stage, execute, and publish one array child; return uploaded result keys."""
    index, task_id = batch_task_id(array_index)
    grid = S3Location.parse(grid_uri, name="grid URI")
    results = S3Location.parse(results_uri, name="results URI")

    work_dir = work_root / _work_name(job_id, task_id)
    local_grid = work_dir / "grid.jsonl"
    local_results = work_dir / "results"
    local_results.mkdir(parents=True, exist_ok=True)

    print(
        f"AWS Batch job {job_id}: array index {index} -> RigFL task {task_id}; "
        f"grid={grid_uri}"
    )
    s3.download_file(grid.bucket, grid.key, str(local_grid))
    run_task(str(local_grid), task_id, local_results / "runs")

    files = sorted(path for path in local_results.rglob("*") if path.is_file())
    if not files:
        raise RuntimeError(f"RigFL task {task_id} completed without a result artifact")

    uploaded: list[str] = []
    for path in files:
        destination = results.child(path.relative_to(local_results).as_posix())
        s3.upload_file(str(path), destination.bucket, destination.key)
        uploaded.append(destination.key)

    marker = results.child(f"_batch/task-{task_id:05d}.json")
    body = json.dumps(
        {
            "array_index": index,
            "job_id": job_id,
            "result_keys": uploaded,
            "task_id": task_id,
        },
        sort_keys=True,
    ).encode()
    s3.put_object(
        Bucket=marker.bucket,
        Key=marker.key,
        Body=body,
        ContentType="application/json",
    )
    print(f"uploaded {len(uploaded)} result artifact(s) to {results_uri}")
    return uploaded


def _s3_client() -> S3Client:
    try:
        import boto3
    except ImportError as error:
        raise SystemExit(
            'AWS Batch support requires the AWS extra: pip install ".[aws]"'
        ) from error
    return boto3.client("s3")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run one RigFL AWS Batch array task")
    parser.add_argument(
        "--grid-uri",
        default=os.environ.get("RIGFL_GRID_URI"),
        help="S3 URI for grid.jsonl (or RIGFL_GRID_URI)",
    )
    parser.add_argument(
        "--results-uri",
        default=os.environ.get("RIGFL_RESULTS_URI"),
        help="S3 prefix for completed results (or RIGFL_RESULTS_URI)",
    )
    parser.add_argument("--array-index", type=int)
    parser.add_argument("--work-root", type=Path, default=Path("/tmp/rigfl-batch"))
    args = parser.parse_args(argv)

    if not args.grid_uri:
        parser.error("--grid-uri or RIGFL_GRID_URI is required")
    if not args.results_uri:
        parser.error("--results-uri or RIGFL_RESULTS_URI is required")
    try:
        index, _ = batch_task_id(args.array_index)
        run_batch_task(
            grid_uri=args.grid_uri,
            results_uri=args.results_uri,
            array_index=index,
            job_id=os.environ.get("AWS_BATCH_JOB_ID", "local"),
            work_root=args.work_root,
            s3=_s3_client(),
        )
    except BatchConfigurationError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
