#!/usr/bin/env bash
# Skip torch + Linux CUDA stack from uv.lock; install CPU torch after this script (see Dockerfile).
set -euo pipefail
exec uv sync --frozen --no-dev \
  --no-install-package torch \
  --no-install-package triton \
  --no-install-package cuda-bindings \
  --no-install-package cuda-pathfinder \
  --no-install-package cuda-toolkit \
  --no-install-package nvidia-cublas \
  --no-install-package nvidia-cuda-cupti \
  --no-install-package nvidia-cuda-nvrtc \
  --no-install-package nvidia-cuda-runtime \
  --no-install-package nvidia-cudnn-cu13 \
  --no-install-package nvidia-cufft \
  --no-install-package nvidia-cufile \
  --no-install-package nvidia-curand \
  --no-install-package nvidia-cusolver \
  --no-install-package nvidia-cusparse \
  --no-install-package nvidia-cusparselt-cu13 \
  --no-install-package nvidia-nccl-cu13 \
  --no-install-package nvidia-nvjitlink \
  --no-install-package nvidia-nvshmem-cu13 \
  --no-install-package nvidia-nvtx \
  "$@"
