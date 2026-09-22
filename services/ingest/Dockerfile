FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_SYSTEM_PYTHON=1

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      build-essential libglib2.0-0 libgl1 \
 && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /srv

COPY pyproject.toml ./
# cu128, not cu124. The RTX 5090 is sm_120 (Blackwell) and cu124 ships no
# kernels for it — torch imports fine and then fails on the first .cuda()
# call with "no kernel image is available for execution on the device".
# Ingestion is the one place that genuinely wants the card, so it gets a
# CUDA build; the rag service uses CPU torch instead.
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu128
# --index-strategy unsafe-best-match is required, not optional. uv defaults
# to "first index that has the package wins", so the torch index shadows
# PyPI for anything it happens to mirror (urllib3, filelock, ...) and
# resolution fails outright. Both indexes here are first-party and pinned,
# so considering all of them is the correct behaviour.
RUN uv pip install --system \
      --extra-index-url ${TORCH_INDEX} \
      --index-strategy unsafe-best-match \
      -r pyproject.toml

COPY ingest ./ingest

# No CMD — always invoked explicitly:
#   docker compose run --rm ingest python -m ingest.cli run
