import json
from pathlib import Path

import pytest

from rigfl.experiment import aws_batch


class FakeS3:
    def __init__(self):
        self.downloads = []
        self.uploads = []
        self.objects = []

    def download_file(self, bucket, key, filename):
        self.downloads.append((bucket, key, filename))
        Path(filename).write_text('{"algorithm": "Local"}\n')

    def upload_file(self, filename, bucket, key):
        self.uploads.append((filename, bucket, key))

    def put_object(self, **kwargs):
        self.objects.append(kwargs)


def test_batch_task_id_converts_zero_based_array_index():
    assert aws_batch.batch_task_id(environ={"AWS_BATCH_JOB_ARRAY_INDEX": "7"}) == (
        7,
        8,
    )


@pytest.mark.parametrize("value", ["not-an-int", "-1"])
def test_batch_task_id_rejects_invalid_environment(value):
    with pytest.raises(aws_batch.BatchConfigurationError):
        aws_batch.batch_task_id(environ={"AWS_BATCH_JOB_ARRAY_INDEX": value})


def test_s3_location_requires_bucket_and_key():
    assert aws_batch.S3Location.parse(
        "s3://rigfl-bucket/sweeps/demo/grid.jsonl", name="grid URI"
    ) == aws_batch.S3Location("rigfl-bucket", "sweeps/demo/grid.jsonl")
    with pytest.raises(aws_batch.BatchConfigurationError):
        aws_batch.S3Location.parse("s3://rigfl-bucket", name="grid URI")


def test_run_batch_task_downloads_runs_and_uploads(monkeypatch, tmp_path):
    s3 = FakeS3()
    called = {}

    def fake_run_task(grid_path, task_id, out_dir):
        called.update(grid_path=grid_path, task_id=task_id, out_dir=out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "result.json").write_text('{"ok": true}\n')

    monkeypatch.setattr(aws_batch, "run_task", fake_run_task)
    uploaded = aws_batch.run_batch_task(
        grid_uri="s3://rigfl-bucket/sweeps/demo/grid.jsonl",
        results_uri="s3://rigfl-bucket/sweeps/demo/results/",
        array_index=2,
        job_id="job:child",
        work_root=tmp_path,
        s3=s3,
    )

    assert called["task_id"] == 3
    assert Path(called["grid_path"]).read_text() == '{"algorithm": "Local"}\n'
    assert uploaded == ["sweeps/demo/results/runs/result.json"]
    assert s3.uploads[0][1:] == (
        "rigfl-bucket",
        "sweeps/demo/results/runs/result.json",
    )
    marker = s3.objects[0]
    assert marker["Key"] == "sweeps/demo/results/_batch/task-00003.json"
    assert json.loads(marker["Body"])["task_id"] == 3
