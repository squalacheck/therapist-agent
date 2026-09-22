FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_SYSTEM_PYTHON=1

RUN apt-get update \
 && apt-get install -y --no-install-recommends curl build-essential \
 && rm -rf /var/lib/apt/lists/*

# uv, to match the toolchain used across your other projects.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /srv

COPY pyproject.toml ./
# CPU torch, deliberately, and it makes the image ~3 GB smaller.
#
# Two reasons. First, cu124 has no sm_120 kernels at all, so it could never
# have driven a 5090 — any CUDA build here must be cu128 or later. Second,
# and decisive: the 27B weights are 22.9 GiB once the vision tower is
# dropped, which leaves under a gigabyte spare after the KV cache. The
# embedder and reranker want ~2.4 GB. They do not fit on the card next to
# the model, so query-time embedding runs on the 9800X3D and is fast enough.
#
# Batch ingestion is the case that really wants the GPU, and it gets it:
# services/ingest builds against cu128 and runs with vLLM stopped.
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
# --index-strategy unsafe-best-match is required, not optional. uv defaults
# to "first index that has the package wins", so the torch index shadows
# PyPI for anything it happens to mirror (urllib3, filelock, ...) and
# resolution fails outright. Both indexes here are first-party and pinned,
# so considering all of them is the correct behaviour.
# --extra dev, because `make test` runs pytest inside this container and
# the base dependency set does not contain it. Without this the documented
# test target fails with "No module named pytest", which is a bad way to
# find out that nothing has been running the suite.
RUN uv pip install --system \
      --extra-index-url ${TORCH_INDEX} \
      --index-strategy unsafe-best-match \
      --extra dev \
      -r pyproject.toml

COPY app ./app
COPY tests ./tests

EXPOSE 8080

HEALTHCHECK --interval=15s --timeout=5s --start-period=120s --retries=5 \
  CMD curl -sf http://localhost:8080/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080", "--no-access-log"]
