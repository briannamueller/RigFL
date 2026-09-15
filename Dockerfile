# syntax=docker/dockerfile:1.7

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
