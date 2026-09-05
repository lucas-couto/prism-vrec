# Runtime environment

Inventory of the environment the pipeline and its tests actually run in,
and how it is now made reproducible (SDD task V01, finding F14,
requirement Q18). Regenerate the inventory section after every image
rebuild; the commands are given inline.

## 1. Where the environment is declared

| Declaration | Role | Enforced by |
|---|---|---|
| `pyproject.toml` | Dependency **ranges** (`torch>=2.8,<2.9`, `numpy<2.0`, ...) and `requires-python >=3.11,<3.13` | `tests/test_environment_lock.py` |
| `uv.lock` | Exact resolution of those ranges (115 packages, PyPI wheels with SHA-256) | `Dockerfile` (`uv export --frozen` + `pip --require-hashes`) |
| `Dockerfile` | `python:3.11-slim` + the locked set + the `telemetry` extra | `docker compose build` |
| `.github/workflows/ci.yml` | CPU wheels for `torch==2.8.0`/`torchvision==0.23.0` (lock versions) + `pip install -e ".[dev]"` | GitHub Actions |

The image is the reference environment: every experiment and every
test run goes through it (there is no Python on the workstation).

## 2. Installed inventory — image `prism-vrec-shell:latest`

Captured 2026-09-05 from the image that ran the whole 2026-08/09 audit and
battery work. **This image was built on 2026-08-21 with the pre-V01
Dockerfile (`pip install -e ".[telemetry]"` from ranges), so its content
is what the ranges resolved to on that day, not what `uv.lock` pins** —
see §3.

```
docker image inspect prism-vrec-shell:latest --format '{{.Id}} {{.Created}} {{.Size}} {{.Architecture}}'
sha256:b0054c25d312fb7e7e1d01f2155bdd7ce47be83e420923eb06a7669d9fe40a11 2026-08-21T11:57:03 12387631992 bytes amd64
```

| Component | Installed | Note |
|---|---|---|
| Python | 3.11.16 (GCC 14.2.0), `python:3.11-slim` | inside `requires-python` |
| torch | 2.8.0+cu128 — CUDA build 12.8, cuDNN 91002 | PyPI linux/amd64 wheel |
| torchvision | 0.23.0+cu128 | |
| NumPy | 1.26.4 | |
| scikit-learn / SciPy | 1.5.2 / 1.14.1 | |
| timm | **1.0.28** | lock pins 1.0.27 |
| transformers | 4.49.0 | |
| optuna | 4.0.0 | |
| pandas / Pillow | 2.2.3 / 10.4.0 | |
| psutil, nvidia-ml-py (`telemetry` extra) | **absent** | image predates the extra; telemetry runs on the `nvidia-smi` + `getrusage` fallbacks |
| pytest | 9.1.1, **not in the image** | lives in the host-mounted `.cache/.local` user site (`HOME=/app/.cache`); lock pins 8.4.2 |
| ruff | 0.15.17 in `.cache/.local/bin` | equals the lock and the CI pin |
| pip / setuptools | 26.2.1 / 84.0.0 | installer, not a runtime dependency |

Full `pip freeze` of the image (`docker run --rm --entrypoint pip prism-vrec-shell:latest freeze`):

```
alembic==1.19.1
annotated-types==0.8.0
certifi==2026.7.22
charset-normalizer==3.5.1
colorlog==6.12.0
contourpy==1.3.3
cycler==0.12.1
filelock==3.32.3
fonttools==4.63.0
fsspec==2026.7.0
ftfy==6.3.1
greenlet==3.5.5
huggingface-hub==0.26.5
idna==3.19
Jinja2==3.1.6
joblib==1.5.3
kiwisolver==1.5.0
Mako==1.4.1
MarkupSafe==3.0.3
matplotlib==3.9.4
mpmath==1.3.0
networkx==3.6.1
numpy==1.26.4
nvidia-cublas-cu12==12.8.4.1
nvidia-cuda-cupti-cu12==12.8.90
nvidia-cuda-nvrtc-cu12==12.8.93
nvidia-cuda-runtime-cu12==12.8.90
nvidia-cudnn-cu12==9.10.2.21
nvidia-cufft-cu12==11.3.3.83
nvidia-cufile-cu12==1.13.1.3
nvidia-curand-cu12==10.3.9.90
nvidia-cusolver-cu12==11.7.3.90
nvidia-cusparse-cu12==12.5.8.93
nvidia-cusparselt-cu12==0.7.1
nvidia-nccl-cu12==2.27.3
nvidia-nvjitlink-cu12==12.8.93
nvidia-nvtx-cu12==12.8.90
open_clip_torch==2.32.0
optuna==4.0.0
packaging==26.3
pandas==2.2.3
pillow==10.4.0
# Editable install with no version control (prism-vrec==2.5.0)
-e /app
pydantic==2.13.4
pydantic_core==2.46.4
pyparsing==3.3.2
python-dateutil==2.9.0.post0
pytz==2026.3.post1
PyYAML==6.0.3
regex==2026.7.19
requests==2.34.2
safetensors==0.8.0
scikit-learn==1.5.2
scipy==1.14.1
seaborn==0.13.2
six==1.17.0
SQLAlchemy==2.0.52
sympy==1.14.0
threadpoolctl==3.6.0
timm==1.0.28
tokenizers==0.21.4
torch==2.8.0
torchvision==0.23.0
tqdm==4.70.0
transformers==4.49.0
triton==3.4.0
typing-inspection==0.4.4
typing_extensions==4.16.0
tzdata==2026.3
urllib3==2.7.0
wcwidth==0.8.2
```

## 3. Mismatches found (2026-09-05)

| # | Where | Mismatch | Status |
|---|---|---|---|
| 1 | Dockerfile | Installed `pip install -e ".[telemetry]"` from ranges; `uv.lock` was never read at build time. | **Fixed**: locked, hash-verified install (§4). |
| 2 | image vs `uv.lock` | 20 packages differ (image / lock): timm 1.0.28 / 1.0.27, tqdm 4.70.0 / 4.68.3, filelock 3.32.3 / 3.29.4, fsspec 2026.7.0 / 2026.6.0, regex 2026.7.19 / 2026.5.9, sqlalchemy 2.0.52 / 2.0.51, alembic 1.19.1 / 1.18.4, mako 1.4.1 / 1.3.12, greenlet 3.5.5 / 3.5.2, packaging 26.3 / 26.2, typing-extensions 4.16.0 / 4.15.0, typing-inspection 0.4.4 / 0.4.2, annotated-types 0.8.0 / 0.7.0, certifi 2026.7.22 / 2026.6.17, charset-normalizer 3.5.1 / 3.4.7, idna 3.19 / 3.18, colorlog 6.12.0 / 6.10.1, pytz 2026.3.post1 / 2026.2, tzdata 2026.3 / 2026.2, wcwidth 0.8.2 / 0.8.1. All inside pyproject's ranges; **timm is the only numerical library affected**. | Reported, not regenerated. The next image build installs the lock's versions (timm 1.0.27). |
| 3 | `uv.lock` vs `pyproject.toml` | `uv lock --check` fails: the lock's `prism-vrec` metadata is 2.5.0 (commit 4050469, #22) and lacks the `telemetry` extra added in 2.6.0 (#23). `uv lock --dry-run` would change only: `prism-vrec 2.5.0 -> (dynamic)`, `psutil 7.2.2 -> 6.1.1`, `nvidia-ml-py 13.610.43 -> 12.575.51` (the extra's `<7.0` / `<13.0` ceilings are tighter than codecarbon's). | Reported. Regenerating the lock is a reviewed dependency operation; `tests/test_environment_lock.py::test_lock_records_the_same_requires_dist_as_pyproject` is a strict `xfail` until it lands. |
| 4 | `ci.yml` | Installed `torch>=2.1.0,<2.6` / `torchvision>=0.16.0,<0.21` (three jobs), contradicting `torch>=2.8,<2.9`; `pip install -e .` then pulled the CUDA torch 2.8 from PyPI on top. | **Fixed**: pins `torch==2.8.0` / `torchvision==0.23.0` (lock versions) from the CPU index. |
| 5 | image vs Dockerfile | The running image lacks the `telemetry` extra the Dockerfile has installed since 2.6.0 — the image was never rebuilt. | Rebuild needed; the manifest's `hardware.ram_total_gb` is `null` and telemetry probes are the fallbacks until then. |
| 6 | `.dockerignore` | `.cache/` (2.0 GB of weights, host user site) and `.env` (HF token) were part of the build context and of `COPY . .`. | **Fixed**: both ignored. |

## 4. How the image is built now

```
COPY pyproject.toml uv.lock ./
pip install --upgrade pip "uv==0.9.5"
uv export --frozen --no-dev --no-emit-project -o /tmp/requirements.lock.txt
pip install --no-deps --require-hashes -r /tmp/requirements.lock.txt
pip install --no-deps --no-build-isolation -e .
pip check
pip install --no-deps "psutil>=5.9,<7.0" "nvidia-ml-py>=12.535,<13.0"   # telemetry extra, see mismatch 3
```

- `--frozen` uses the lock as committed (no re-resolution); `--require-hashes` rejects any wheel whose SHA-256 the lock does not list; `--no-deps` forbids pip from resolving on its own; `--no-build-isolation` builds the editable package with the locked setuptools instead of a fresh, range-resolved one.
- The only range-resolved packages left are pip itself, uv (pinned) and the two `telemetry` wheels (mismatch 3).
- **CUDA vs CPU** is unchanged from before: the lock pins `torch==2.8.0` from PyPI, whose linux/amd64 wheel is the CUDA 12.8 build (`torch.__version__ == "2.8.0+cu128"`, ~3 GB with the `nvidia-*-cu12` libraries, all in the lock with hashes) and whose linux/arm64 wheel is CPU-only. No flag selects it; the platform does. CI is the third variant: same versions from `download.pytorch.org/whl/cpu` (`2.8.0+cpu`).
- Validated on 2026-09-05 without building the image: in a throwaway `python:3.11-slim` container the export succeeded (72 requirement lines, 510 hashes), `uv pip install --dry-run` resolved all 71 packages for Python 3.11 and 3.12 on `x86_64-manylinux_2_28`, pip accepted the hashed file (`--dry-run --require-hashes` on the 56 non-torch lines), the telemetry line resolves to `psutil 6.1.1` + `nvidia-ml-py 12.575.51`, and the editable install produced `prism-vrec 2.12.1` with setuptools 82.0.1. The full image build (a ~3 GB torch download) has **not** been run; the first `docker compose build` after this change is the remaining evidence.

## 5. CPU-test versus GPU-test environments

The same image serves both; the distinction is recorded, not configured:

| Signal | CPU test / CI | GPU run |
|---|---|---|
| Invocation | `docker compose run ... -e CUDA_VISIBLE_DEVICES=` (tests), or GitHub Actions with `+cpu` wheels | `docker compose up` with the nvidia device reservation |
| `torch.__version__` | `2.8.0+cu128` with no visible device (container) or `2.8.0+cpu` (CI) | `2.8.0+cu128` |
| `torch.cuda.is_available()` | `False` | `True` |
| Run manifest (`src/utils/manifest.py`) | `device.requested`/`device.resolved` = `cpu`; `hardware.torch_cuda_available: false`; no `cuda_version` | `device.resolved: cuda`; `hardware.gpu_name`, `gpu_total_memory_mb`, `gpu_count`, `cuda_version: "12.8"` |
| Telemetry probe | `none` / `getrusage` (or `psutil` once the extra is present) | `nvml` (extra) or `nvidia-smi` fallback |

A passing CPU suite therefore certifies the CPU path of the locked
environment only; CUDA memory behaviour needs a run whose manifest shows
`torch_cuda_available: true` (task V02).

## 6. Regenerating this inventory

```
docker image inspect prism-vrec-shell:latest --format '{{.Id}} {{.Created}} {{.Size}} {{.Architecture}}'
docker run --rm --entrypoint python prism-vrec-shell:latest -c "import sys, torch, torchvision, numpy, sklearn, scipy, timm, transformers, optuna; print(sys.version, torch.__version__, torch.version.cuda, torchvision.__version__, numpy.__version__, sklearn.__version__, scipy.__version__, timm.__version__, transformers.__version__, optuna.__version__)"
docker run --rm --entrypoint pip prism-vrec-shell:latest freeze
```

To regenerate the lock (a reviewed dependency operation, not part of a
reliability fix): `docker compose --profile tools run --rm shell -c "uv lock"`
once the image contains uv, then re-run `tests/test_environment_lock.py`
and remove its `xfail` marker.
