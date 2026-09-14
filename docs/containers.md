# RigFL containers

RigFL has two container targets built from the same repository snapshot:

- `cpu` runs locally on either Apple-silicon or Intel machines. It is the fast
  validation target and does not require a GPU.
- `gpu` is an x86_64 CUDA image for NVIDIA-backed AWS Batch jobs. It is not
  intended to run natively on an Apple-silicon Mac.

Both targets install the same pinned RigFL runtime dependencies. The base image,
Python version, PyTorch version, and GPU CUDA runtime are pinned in `Dockerfile`
and `constraints/container.txt`.

## Local smoke test

From the repository root:

```bash
docker build \
  --target cpu \
  --build-arg RIGFL_REVISION="$(git rev-parse HEAD)" \
  --tag rigfl:cpu .

docker run --rm rigfl:cpu
```

The image's default command runs `examples/smoke.py`. A successful run prints a
result row for each supported algorithm and exits with status zero.

To run another Python command in the same image, place its arguments after the
image name:

```bash
docker run --rm rigfl:cpu -c \
  "import platform, torch; print(platform.python_version(), torch.__version__)"
```

The production GPU image is built for AWS with:

```bash
docker build \
  --platform linux/amd64 \
  --target gpu \
  --build-arg RIGFL_REVISION="$(git rev-parse HEAD)" \
  --tag rigfl:gpu .
```

Building the GPU target does not start an AWS resource or create a charge. A
later deployment step will push this image to Amazon ECR and reference its
immutable digest from an AWS Batch job definition.

## AWS Batch worker contract

The GPU image starts `python -m rigfl.experiment.aws_batch`. Each AWS Batch array
child receives three values:

- `AWS_BATCH_JOB_ARRAY_INDEX`, supplied automatically by AWS Batch and numbered
  from zero;
- `RIGFL_GRID_URI`, an `s3://` URI for the sweep's `grid.jsonl`;
- `RIGFL_RESULTS_URI`, an `s3://` prefix for completed run records.

The worker converts the array index to RigFL's one-based task number, downloads
the grid, executes exactly one task, and uploads completed result JSON. It writes
an `_batch/task-NNNNN.json` completion marker only after every result upload
succeeds. AWS credentials are not embedded in the image; Boto3 uses temporary
credentials supplied by the Batch job role.
