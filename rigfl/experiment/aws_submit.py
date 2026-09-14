"""Preview or submit a RigFL grid as an AWS Batch array job.

The command is intentionally safe by default: without ``--submit`` it performs
no AWS calls and prints the exact S3 locations and Batch request it would use.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


class SubmissionError(ValueError):
    """A grid or AWS submission setting is invalid."""


class S3Client(Protocol):
    def put_object(self, **kwargs) -> object: ...


class BatchClient(Protocol):
    def submit_job(self, **kwargs) -> dict: ...


@dataclass(frozen=True)
class SweepSubmission:
    name: str
    task_count: int
    grid_key: str
    grid_uri: str
    results_uri: str
    digest: str
    grid_body: bytes


def _slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_-]+", "-", value).strip("-")
    if not slug:
        raise SubmissionError("sweep name must contain a letter or number")
    return slug[:80]


def prepare_submission(grid_path: Path, *, bucket: str, name: str) -> SweepSubmission:
    """Validate a JSONL grid and derive immutable, content-addressed S3 paths."""
    if not bucket or "/" in bucket or bucket.startswith("s3://"):
        raise SubmissionError("bucket must be an S3 bucket name, not an S3 URI")
    try:
        body = grid_path.read_bytes()
    except OSError as error:
        raise SubmissionError(f"cannot read grid {grid_path}: {error}") from error

    lines = body.splitlines()
    if not 2 <= len(lines) <= 10_000:
        raise SubmissionError(
            "AWS Batch array jobs require between 2 and 10,000 grid tasks; "
            f"found {len(lines)}"
        )
    for task_id, line in enumerate(lines, 1):
        try:
            task = json.loads(line)
        except json.JSONDecodeError as error:
            raise SubmissionError(
                f"grid task {task_id} is not valid JSON: {error}"
            ) from error
        if not isinstance(task, dict) or not isinstance(task.get("algorithm"), str):
            raise SubmissionError(f"grid task {task_id} is missing an algorithm")
        if not isinstance(task.get("experiment"), dict):
            raise SubmissionError(f"grid task {task_id} is missing experiment settings")
        if not isinstance(task.get("algorithm_config"), dict):
            raise SubmissionError(f"grid task {task_id} is missing algorithm settings")

    digest = hashlib.sha256(body).hexdigest()
    prefix = f"sweeps/{_slug(name)}/{digest[:16]}"
    grid_key = f"{prefix}/grid.jsonl"
    return SweepSubmission(
        name=_slug(name),
        task_count=len(lines),
        grid_key=grid_key,
        grid_uri=f"s3://{bucket}/{grid_key}",
        results_uri=f"s3://{bucket}/{prefix}/results/",
        digest=digest,
        grid_body=body,
    )


def job_request(
    submission: SweepSubmission,
    *,
    queue: str,
    job_definition: str,
) -> dict:
    """Build the boto3 SubmitJob request for a validated sweep."""
    if not queue:
        raise SubmissionError("job queue is required")
    if not job_definition:
        raise SubmissionError("job definition is required")
    return {
        "jobName": f"rigfl-{submission.name}-{submission.digest[:8]}",
        "jobQueue": queue,
        "arrayProperties": {"size": submission.task_count},
        "jobDefinition": job_definition,
        "containerOverrides": {
            "environment": [
                {"name": "RIGFL_GRID_URI", "value": submission.grid_uri},
                {"name": "RIGFL_RESULTS_URI", "value": submission.results_uri},
            ]
        },
    }


def submit_sweep(
    submission: SweepSubmission,
    *,
    bucket: str,
    request: dict,
    s3: S3Client,
    batch: BatchClient,
) -> dict:
    """Upload the immutable grid and submit its array job."""
    s3.put_object(
        Bucket=bucket,
        Key=submission.grid_key,
        Body=submission.grid_body,
        ContentType="application/x-ndjson",
        Metadata={"sha256": submission.digest},
    )
    return batch.submit_job(**request)


def _clients(profile: str, region: str) -> tuple[S3Client, BatchClient]:
    try:
        import boto3
    except ImportError as error:
        raise SystemExit('AWS submission requires: pip install ".[aws]"') from error
    session = boto3.Session(profile_name=profile, region_name=region)
    return session.client("s3"), session.client("batch")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Preview or submit a RigFL grid to AWS Batch"
    )
    parser.add_argument("--grid", type=Path, required=True, help="local grid.jsonl")
    parser.add_argument("--bucket", required=True, help="artifacts bucket name")
    parser.add_argument("--job-queue", required=True, help="Batch queue name or ARN")
    parser.add_argument(
        "--job-definition", required=True, help="Batch job definition name or ARN"
    )
    parser.add_argument("--name", help="sweep name; defaults to the grid's directory")
    parser.add_argument("--profile", default="rigfl")
    parser.add_argument("--region", default="us-east-2")
    parser.add_argument(
        "--submit",
        action="store_true",
        help="perform the S3 upload and Batch submission (default: preview only)",
    )
    args = parser.parse_args(argv)

    try:
        submission = prepare_submission(
            args.grid,
            bucket=args.bucket,
            name=args.name or args.grid.parent.name,
        )
        request = job_request(
            submission,
            queue=args.job_queue,
            job_definition=args.job_definition,
        )
    except SubmissionError as error:
        parser.error(str(error))

    print(f"Sweep: {submission.name}")
    print(
        f"Tasks: {submission.task_count} (AWS indices 0..{submission.task_count - 1})"
    )
    print(f"Grid: {submission.grid_uri}")
    print(f"Results: {submission.results_uri}")
    print(f"Queue: {args.job_queue}")
    print(f"Job definition: {args.job_definition}")

    if not args.submit:
        print(
            "\nPreview only: no AWS calls were made. Add --submit to launch the sweep."
        )
        return

    s3, batch = _clients(args.profile, args.region)
    response = submit_sweep(
        submission,
        bucket=args.bucket,
        request=request,
        s3=s3,
        batch=batch,
    )
    print(f"\nSubmitted AWS Batch job: {response['jobId']}")


if __name__ == "__main__":
    main()
