# Single image that works on both GPU and CPU hosts.
#
# Strategy: use ``python:3.11-slim`` as the base and let PyPI pick the
# right ``torch`` wheel for the architecture pip resolves it on:
#
#   linux/amd64 (RunPod, lab servers): CUDA-built wheel.  Runs on GPU
#                                      when the host exposes one via
#                                      nvidia-container-toolkit, falls
#                                      back to CPU otherwise.
#   linux/arm64 (Mac Docker Desktop):  CPU-only wheel.  No CUDA wheel
#                                      exists for ARM64, so this is the
#                                      only option there anyway.
#
# The runtime device choice is made by ``src.utils.device.resolve_device``
# from the ``device`` field in ``configs/default.yaml`` (default
# ``"auto"``), so the researcher does not pick CPU vs GPU manually.

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System libraries required by Pillow, torchvision image ops, and git
# (timm / transformers occasionally fetch via git+https).
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies in a dedicated layer so source-only edits
# don't invalidate the (slow) install.  pyproject.toml's setuptools
# config requires src/ and plugins/ to exist on disk for the build
# backend to enumerate packages, so we copy those two before installing.
#
# The layer is built FROM THE LOCK.  ``uv export --frozen`` turns
# uv.lock into a hashed requirements file and pip installs exactly those
# wheels: ``--require-hashes`` refuses any file the lock does not list
# and ``--no-deps`` refuses to resolve anything on its own.  Two builds
# of the same commit therefore install the same package set, where the
# previous ``pip install -e .`` re-resolved every version range at build
# time (the 2026-08-21 image drifted from the lock on 20 packages; see
# docs/environment.md).  uv stays in the image so the lock can be
# regenerated from the ``shell`` service on a host without Python.
#
# CUDA vs CPU is unchanged: the lock pins torch from PyPI, whose wheel is
# the CUDA 12.8 build on linux/amd64 (``2.8.0+cu128`` at runtime) and the
# CPU-only build on linux/arm64.  The platform, not a flag, still decides,
# and the run manifest records which one was used.
COPY pyproject.toml uv.lock ./
COPY src ./src
COPY plugins ./plugins
RUN python -m pip install --upgrade pip "uv==0.9.5" \
    && uv export --frozen --no-dev --no-emit-project -o /tmp/requirements.lock.txt \
    && python -m pip install --no-deps --require-hashes -r /tmp/requirements.lock.txt \
    && python -m pip install --no-deps --no-build-isolation -e . \
    && python -m pip check \
    && rm /tmp/requirements.lock.txt
# The ``telemetry`` extra adds NVML + psutil so per-step GPU power and
# multi-process CPU accounting are read in-process; without them
# telemetry degrades to an ``nvidia-smi`` subprocess and getrusage (see
# src/utils/telemetry.py).  uv.lock predates the extra, and regenerating
# the lock is a reviewed dependency operation (docs/reliability-sdd/V01.md),
# so these two small binding wheels are the only range-resolved packages
# in the image until that lands.  Same ranges as pyproject.toml.
RUN python -m pip install --no-deps \
        "psutil>=5.9,<7.0" \
        "nvidia-ml-py>=12.535,<13.0" \
    && python -m pip check

# Copy the rest of the source tree.
COPY . .

# First-run directories so steps that assume they exist don't error out.
RUN mkdir -p \
        data/raw data/processed data/embeddings \
        results/finetuning results/statistical \
        logs \
        checkpoints/extraction checkpoints/training checkpoints/finetuning

ENTRYPOINT ["python"]
CMD ["main.py"]
