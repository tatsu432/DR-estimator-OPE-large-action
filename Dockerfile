# syntax=docker/dockerfile:1
# Reproducible CPU image: README.md is required for Hatchling (pyproject readme field).
#
# PyPI torch 2.11+ on Linux pulls cuda-bindings, cuda-toolkit, and many nvidia-* wheels into
# uv.lock. `--no-install-package torch` alone still installs those siblings. We run
# docker/cpu_only_sync.sh first (skip torch + CUDA packages), then install CPU torch from
# PyTorch's CPU index — pre-installing torch before sync causes uv to uninstall it.
FROM python:3.12-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg \
    UV_NO_CACHE=1

RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

RUN useradd --create-home --uid 10001 --shell /bin/bash appuser

WORKDIR /app

COPY pyproject.toml uv.lock README.md LICENSE ./
COPY src ./src
COPY docker/cpu_only_sync.sh ./docker/cpu_only_sync.sh

RUN chown -R appuser:appuser /app \
    && chmod +x /app/docker/cpu_only_sync.sh

USER appuser

ENV VIRTUAL_ENV=/app/.venv
ENV PATH="/app/.venv/bin:${PATH}"

# UV_NO_CACHE=1: avoid ~/.cache/uv + site-packages duplicating wheels (lowers peak disk during build).
# Omit UV_LINK_MODE=copy so uv can hardlink/clones from its temp store instead of copying twice.
# RAM-backed /tmp for this step: avoids writing huge wheel unpacks onto the Docker disk image
# (helps when the overlay is tight) and avoids tiny default /tmp. Requires BuildKit (default in Compose).
RUN --mount=type=tmpfs,target=/tmp,size=3221225472 \
    uv venv /app/.venv \
    && ./docker/cpu_only_sync.sh \
    && uv pip install "torch==2.11.0" --index-url https://download.pytorch.org/whl/cpu \
    && rm -rf /home/appuser/.cache/uv 2>/dev/null || true

VOLUME ["/app/outputs"]

# --no-sync: CPU torch is installed with uv pip after sync; a default `uv run` sync would drop it.
ENTRYPOINT ["uv", "run", "--no-sync", "python", "-m", "synthetic.run_experiment"]
CMD ["experiment=beta", "scale=fastest"]
