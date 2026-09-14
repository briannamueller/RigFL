# syntax=docker/dockerfile:1.7

ARG GPU_PLATFORM=linux/amd64

# The CPU target is multi-platform, including Apple-silicon Macs. The digest
# pins the multi-platform Python 3.12.13 image rather than a moving tag.
FROM python:3.12.13-slim-bookworm@sha256:4766d8b510c428e595d74b9cc5bbb2fae8e26316fffb4adc89908d79aacd58a2 AS cpu-runtime

ARG TORCH_VERSION=2.13.0
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install --yes --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && python -m pip install "torch==${TORCH_VERSION}" \
        --index-url https://download.pytorch.org/whl/cpu


# The GPU target is linux/amd64 because AWS Batch NVIDIA instances are x86_64.
# CUDA is supplied by the image; the host only needs a compatible NVIDIA driver.
FROM --platform=${GPU_PLATFORM} pytorch/pytorch:2.13.0-cuda12.6-cudnn9-runtime@sha256:6acf597eeb8e376a96580dde4952f37cc017fef732bb40bfc73f28f25e3f64b4 AS gpu-runtime

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1


FROM cpu-runtime AS cpu

ARG RIGFL_REVISION=unknown
LABEL org.opencontainers.image.source="https://github.com/briannamueller/RigFL" \
      org.opencontainers.image.revision="${RIGFL_REVISION}" \
      org.opencontainers.image.title="RigFL CPU"

WORKDIR /opt/rigfl
COPY . .
RUN python -m pip install --constraint constraints/container.txt . \
    && useradd --create-home --uid 10001 rigfl \
    && chown -R rigfl:rigfl /opt/rigfl

USER rigfl
ENV HF_HOME=/home/rigfl/.cache/huggingface \
    XDG_CACHE_HOME=/home/rigfl/.cache

ENTRYPOINT ["python"]
CMD ["examples/smoke.py"]


FROM gpu-runtime AS gpu

ARG RIGFL_REVISION=unknown
LABEL org.opencontainers.image.source="https://github.com/briannamueller/RigFL" \
      org.opencontainers.image.revision="${RIGFL_REVISION}" \
      org.opencontainers.image.title="RigFL GPU"

WORKDIR /opt/rigfl
COPY . .
RUN python -m pip install --constraint constraints/container.txt ".[aws]" \
    && useradd --create-home --uid 10001 rigfl \
    && chown -R rigfl:rigfl /opt/rigfl

USER rigfl
ENV HF_HOME=/home/rigfl/.cache/huggingface \
    XDG_CACHE_HOME=/home/rigfl/.cache

ENTRYPOINT ["python"]
CMD ["-m", "rigfl.experiment.aws_batch"]
