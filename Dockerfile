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
# don't invalidate the (slow) pip install.  pyproject.toml's setuptools
# config requires src/ and plugins/ to exist on disk for the build
# backend to enumerate packages, so we copy those two before installing.
COPY pyproject.toml ./
COPY src ./src
COPY plugins ./plugins
RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install -e .

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
