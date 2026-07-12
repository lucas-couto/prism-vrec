"""Reproducibility manifest written once per pipeline invocation.

Each call to :func:`start_run` creates a directory
``results/runs/<run_id>/`` and writes a ``manifest.json`` capturing
everything needed to reproduce the run later: the git SHA, whether the
working tree was dirty, the merged config snapshot, the random seed,
the hardware, and the curated package versions.

:func:`finish_run` is called at the end (in a ``finally`` block in
``main.py``) and adds the finish timestamp plus the wall-clock
duration.  A manifest without a ``finished_at`` field is, by
construction, the trace of an interrupted run.

Manifests are deliberately *not* committed: they are execution
artefacts, not source.  The ``results/`` directory is gitignored.
For citation purposes, the manifest of any experiments published or
otherwise referenced should be uploaded alongside the corresponding
data release (e.g. on Zenodo, with a DOI), so that the exact code
state, configuration, and environment can be recovered later.
"""

from __future__ import annotations

import json
import os
import platform
import socket
import subprocess
import time
from datetime import UTC, datetime
from importlib import metadata as _metadata
from pathlib import Path
from typing import Any

from src.utils.atomic_io import atomic_write
from src.utils.dataloader import describe as describe_dataloader_tune
from src.utils.device import resolve_device
from src.utils.logging import get_logger
from src.utils.timing import step_timings as collect_step_timings

logger = get_logger(__name__)


_TRACKED_PACKAGES = (
    "torch",
    "torchvision",
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn",
    "transformers",
    "timm",
    "open_clip_torch",
    "huggingface_hub",
    "pyyaml",
    "tqdm",
    "Pillow",
)


def start_run(
    config_snapshot: dict,
    *,
    results_root: str | Path = "results/runs",
    seed: int | None = None,
) -> Path:
    """Create a new run directory and write the initial manifest.

    Returns the absolute path to the run directory.  The caller is
    expected to pass the same path to :func:`finish_run` (or to read it
    from the returned ``Path``).
    """
    run_id = _make_run_id()
    run_dir = Path(results_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "run_id": run_id,
        "started_at": _now_iso(),
        "started_at_epoch": time.time(),
        "git": _git_info(),
        "seed": seed if seed is not None else config_snapshot.get("seed"),
        "hostname": socket.gethostname(),
        "platform": _platform_info(),
        "hardware": _hardware_info(),
        "device": _device_info(config_snapshot.get("device", "auto")),
        "dataloader_autotune": describe_dataloader_tune(config_snapshot),
        "package_versions": _package_versions(_TRACKED_PACKAGES),
        "config_snapshot": config_snapshot,
        "finished_at": None,
        "duration_seconds": None,
        "exit_status": None,
    }

    _write_manifest(run_dir, manifest)
    logger.info("Run manifest started at %s", run_dir / "manifest.json")
    return run_dir


def finish_run(
    run_dir: str | Path,
    *,
    exit_status: str = "ok",
) -> None:
    """Update the manifest at ``run_dir`` with finish metadata.

    ``exit_status`` is a free-form string (``"ok"`` / ``"error"`` /
    ``"interrupted"``).  Manifests with ``exit_status != "ok"`` should
    not be cited as authoritative results, they document an aborted
    run, not a finished one.
    """
    run_dir = Path(run_dir)
    manifest_path = run_dir / "manifest.json"

    if not manifest_path.exists():
        logger.warning(
            "finish_run called but %s does not exist, skipping update.",
            manifest_path,
        )
        return

    try:
        with open(manifest_path, encoding="utf-8") as fh:
            manifest = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Could not read existing manifest %s: %s", manifest_path, exc)
        return

    finished_epoch = time.time()
    started_epoch = manifest.get("started_at_epoch", finished_epoch)

    manifest["finished_at"] = _now_iso()
    manifest["finished_at_epoch"] = finished_epoch
    # Per-cell durations live in the sidecar step_timings.json so the
    # manifest stays small.
    manifest["steps"] = collect_step_timings()
    manifest["duration_seconds"] = round(finished_epoch - started_epoch, 3)
    manifest["exit_status"] = exit_status

    _write_manifest(run_dir, manifest)
    logger.info(
        "Run manifest closed: %s (status=%s, duration=%.1fs)",
        manifest_path,
        exit_status,
        manifest["duration_seconds"],
    )


def _make_run_id() -> str:
    """``YYYY-MM-DDTHH-MM-SS_<short_sha>`` (UTC)."""
    ts = datetime.now(UTC).strftime("%Y-%m-%dT%H-%M-%SZ")
    short_sha = _git_sha(short=True) or "nogit"
    return f"{ts}_{short_sha}"


def _now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _git_info() -> dict[str, Any]:
    return {
        "sha": _git_sha(),
        "short_sha": _git_sha(short=True),
        "branch": _git_branch(),
        "dirty": _git_dirty(),
        "remote_url": _git_remote_url(),
    }


def _git(args: list[str]) -> str | None:
    try:
        out = subprocess.check_output(
            ["git", *args],
            stderr=subprocess.DEVNULL,
            cwd=Path(__file__).resolve().parents[2],
        )
        return out.decode("utf-8").strip() or None
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None


def _git_sha(short: bool = False) -> str | None:
    return _git(["rev-parse", "--short", "HEAD"]) if short else _git(["rev-parse", "HEAD"])


def _git_branch() -> str | None:
    return _git(["rev-parse", "--abbrev-ref", "HEAD"])


def _git_dirty() -> bool | None:
    status = _git(["status", "--porcelain"])
    if status is None:
        return None
    return bool(status)


def _git_remote_url() -> str | None:
    return _git(["config", "--get", "remote.origin.url"])


def _platform_info() -> dict[str, str]:
    return {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "python": platform.python_version(),
    }


def _device_info(requested: str) -> dict[str, Any]:
    """Record both the requested device value and the one actually used.

    Researchers reading the manifest can tell at a glance whether the
    run executed on GPU or CPU without inspecting hardware fields.
    """
    return {
        "requested": requested,
        "resolved": resolve_device(requested),
    }


def _hardware_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "cpu_count": os.cpu_count(),
    }

    try:
        import torch

        info["torch_cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            info["gpu_name"] = props.name
            info["gpu_total_memory_mb"] = round(props.total_memory / (1024 * 1024))
            info["gpu_count"] = torch.cuda.device_count()
            info["cuda_version"] = torch.version.cuda
    except Exception as exc:  # noqa: BLE001
        info["gpu_query_error"] = repr(exc)

    try:
        import psutil

        info["ram_total_gb"] = round(psutil.virtual_memory().total / (1024**3), 1)
    except ImportError:
        info["ram_total_gb"] = None

    return info


def _package_versions(names: tuple[str, ...]) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = _metadata.version(name)
        except _metadata.PackageNotFoundError:  # noqa: PERF203
            versions[name] = None
    return versions


def _write_manifest(run_dir: Path, manifest: dict) -> None:
    """Durable atomic write via :func:`atomic_write` (fsync + retried replace)."""

    def _write(tmp: str) -> None:
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2, sort_keys=False, default=str)

    atomic_write(_write, run_dir / "manifest.json")
