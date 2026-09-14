import json

import pytest

from rigfl.experiment import aws_submit


def _grid(tmp_path, count=3):
    path = tmp_path / "demo" / "grid.jsonl"
    path.parent.mkdir()
    task = {
        "algorithm": "local",
        "experiment": {"seed": 0},
        "algorithm_config": {},
    }
    path.write_text(
        "".join(
            json.dumps(task | {"experiment": {"seed": i}}) + "\n" for i in range(count)
        )
    )
    return path


class FakeS3:
    def __init__(self):
        self.calls = []

    def put_object(self, **kwargs):
        self.calls.append(kwargs)


class FakeBatch:
    def __init__(self):
        self.calls = []

    def submit_job(self, **kwargs):
        self.calls.append(kwargs)
        return {"jobId": "batch-job-123"}


def test_prepare_submission_uses_content_addressed_paths(tmp_path):
    grid = _grid(tmp_path)
    prepared = aws_submit.prepare_submission(
        grid, bucket="rigfl-artifacts", name="Demo sweep"
    )

    assert prepared.task_count == 3
    assert prepared.name == "Demo-sweep"
    assert prepared.grid_uri.startswith("s3://rigfl-artifacts/sweeps/Demo-sweep/")
    assert prepared.grid_uri.endswith("/grid.jsonl")
    assert prepared.results_uri.endswith("/results/")


@pytest.mark.parametrize("count", [0, 1, 10_001])
def test_prepare_submission_enforces_batch_array_size(tmp_path, count):
    grid = _grid(tmp_path, count)
    with pytest.raises(aws_submit.SubmissionError, match="between 2 and 10,000"):
        aws_submit.prepare_submission(grid, bucket="rigfl-artifacts", name="demo")


def test_submit_uploads_grid_then_submits_array(tmp_path):
    submission = aws_submit.prepare_submission(
        _grid(tmp_path), bucket="rigfl-artifacts", name="demo"
    )
    request = aws_submit.job_request(
        submission, queue="rigfl-gpu", job_definition="rigfl-gpu:1"
    )
    s3, batch = FakeS3(), FakeBatch()

    response = aws_submit.submit_sweep(
        submission,
        bucket="rigfl-artifacts",
        request=request,
        s3=s3,
        batch=batch,
    )

    assert response["jobId"] == "batch-job-123"
    assert s3.calls[0]["Key"] == submission.grid_key
    assert s3.calls[0]["Metadata"] == {"sha256": submission.digest}
    assert batch.calls[0]["arrayProperties"] == {"size": 3}
    environment = batch.calls[0]["containerOverrides"]["environment"]
    assert {item["name"]: item["value"] for item in environment} == {
        "RIGFL_GRID_URI": submission.grid_uri,
        "RIGFL_RESULTS_URI": submission.results_uri,
    }


def test_preview_makes_no_aws_clients(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(
        aws_submit,
        "_clients",
        lambda *args: pytest.fail("preview must not create AWS clients"),
    )
    grid = _grid(tmp_path)
    aws_submit.main(
        [
            "--grid",
            str(grid),
            "--bucket",
            "rigfl-artifacts",
            "--job-queue",
            "rigfl-gpu",
            "--job-definition",
            "rigfl-gpu:1",
        ]
    )
    assert "Preview only: no AWS calls were made" in capsys.readouterr().out
