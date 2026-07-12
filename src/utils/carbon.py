"""Optional carbon-emissions tracking via ``codecarbon``.

ML venues (NeurIPS 2022+, EMNLP 2023+, SIGIR) ask authors to declare
the energy footprint of trained models.  ``codecarbon`` queries
NVIDIA NVML for GPU power draw, RAPL / battery / system files for the
CPU, multiplies the integral by the country-specific carbon intensity
of the electricity grid, and returns kilograms of CO₂-equivalent.

This module is a thin opt-in wrapper.  Two reasons to keep it
optional:

1. ``codecarbon`` pulls in a non-trivial dependency tree (its own
   sqlite store, geocoding, etc.) which we do not want to force on
   every user who just wants to extract some embeddings.
2. The tracker spawns a background thread that polls every
   ``measure_power_secs`` (default 15 s).  Opting in is a deliberate
   choice, most quick smoke runs do not benefit from the noise.

Activation requires both:

* the ``carbon`` extra installed (``pip install -e .[carbon]``); and
* the env var ``PRISM_TRACK_CARBON=1`` at run time.

When either is missing the helpers silently return a no-op so
``main.py`` can call them unconditionally.

The result is embedded in the run manifest under ``carbon`` so a
researcher publishing later can quote the figures without re-running.
"""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from src.utils.atomic_io import atomic_write
from src.utils.logging import get_logger

logger = get_logger(__name__)


def is_enabled() -> bool:
    """Return True iff the user opted in and codecarbon is importable."""
    if os.environ.get("PRISM_TRACK_CARBON", "0") not in {"1", "true", "yes"}:
        return False
    try:
        import codecarbon  # noqa: F401
    except ImportError:
        return False
    return True


@contextmanager
def tracker(run_dir: Path):
    """Yield a started ``EmissionsTracker`` (or ``None`` when disabled).

    On exit:

    * stops the tracker (always),
    * appends the result to ``manifest.json['carbon']`` so it lives
      alongside the rest of the run snapshot,
    * never re-raises a codecarbon error, broken tracking should
      not fail the pipeline.
    """
    if not is_enabled():
        yield None
        return

    from codecarbon import EmissionsTracker

    tracker = EmissionsTracker(
        project_name=Path(run_dir).name,
        output_dir=str(run_dir),
        save_to_file=False,
        log_level="error",
    )
    try:
        tracker.start()
    except Exception as exc:  # noqa: BLE001, never block the pipeline
        logger.warning("codecarbon failed to start: %r, proceeding without it.", exc)
        yield None
        return

    try:
        yield tracker
    finally:
        try:
            emissions_kg = tracker.stop()
        except Exception as exc:  # noqa: BLE001
            logger.warning("codecarbon stop failed: %r", exc)
            emissions_kg = None

        if emissions_kg is not None:
            _record_in_manifest(run_dir, tracker, emissions_kg)


def _record_in_manifest(run_dir: Path, tracker: Any, emissions_kg: float) -> None:
    """Append a ``carbon`` block to ``manifest.json`` next to the run dir."""
    manifest_path = Path(run_dir) / "manifest.json"
    if not manifest_path.exists():
        return

    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning(
            "could not read %s (%r); emissions measured this run are discarded.",
            manifest_path,
            exc,
        )
        return

    data = tracker.final_emissions_data
    manifest["carbon"] = {
        "emissions_kg_co2": round(float(emissions_kg), 6),
        "energy_kwh": round(float(getattr(data, "energy_consumed", 0.0)), 6),
        "duration_seconds": round(float(getattr(data, "duration", 0.0)), 3),
        "country_name": getattr(data, "country_name", None),
        "region": getattr(data, "region", None),
        "cpu_model": getattr(data, "cpu_model", None),
        "gpu_model": getattr(data, "gpu_model", None),
        "codecarbon_version": getattr(data, "codecarbon_version", None),
    }

    payload = json.dumps(manifest, indent=2, default=str)
    try:
        atomic_write(lambda tmp: Path(tmp).write_text(payload), manifest_path)
    except OSError as exc:
        logger.warning("failed to persist carbon block to %s: %r", manifest_path, exc)
