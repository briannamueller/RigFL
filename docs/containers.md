# RigFL containers

RigFL provides a CPU container for local validation on Apple-silicon and Intel
machines. It installs the pinned runtime dependencies defined in `Dockerfile`
and `constraints/container.txt` and does not require a GPU.

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
