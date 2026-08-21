"""Per-step runtime telemetry: throughput, utilisation and energy cost.

``timing.py`` answers *how long* each pipeline step took.  This module
answers *how fast it went* and *what it cost* while it ran, which is
what a reader of the manifest needs to judge whether a 4-hour extract
was bandwidth-bound, GPU-bound or simply idle.

Two independent sources feed the numbers:

* **Counters** — monotonic totals the pipeline itself increments as
  work happens: bytes pulled off the network (:func:`add_bytes`),
  floating-point operations dispatched (:func:`add_flops`) and items
  processed (:func:`add_items`).  Cheap enough to call in a hot loop:
  one lock acquisition and an addition.
* **Gauges** — instantaneous hardware readings sampled by a background
  thread: GPU utilisation, GPU power draw, GPU memory, process CPU
  time and resident set size.

A single sampler runs for the whole pipeline invocation rather than
one per step.  Steps and cells only record a *marker* (a
``perf_counter`` stamp) on entry and ask for a summary of the window
on exit.  Nesting therefore works for free — a ``time_cell`` inside a
step slices the same sample series the step will slice — and no
thread is started or joined per step.

Entering and leaving a window each force a sample, so a step's
counter deltas are exact even when it finishes between two ticks; a
step that short simply reports a mean with no min/max spread.

Rates are derived from consecutive counter snapshots, so ``min`` and
``max`` describe the slowest and fastest *sampling window* of the
step, not the slowest and fastest single chunk.  With the default
1 s interval a 20-minute download is summarised from ~1200 windows,
which is enough to expose a stalled connection without storing a
sample per socket read.

Energy is the trapezoidal integral of sampled GPU power over the
window.  It covers the GPU only: consumer NVIDIA cards expose board
power through NVML, whereas CPU package power (RAPL) is not readable
from inside an unprivileged container.  For a full-system figure
including CPU and DRAM, use the optional ``codecarbon`` integration in
:mod:`src.utils.carbon`, which this module deliberately does not
duplicate.

The two halves of a summary have different coverage under
multiprocessing.  Gauges are sampled from the device and from the
whole process tree, so they include forked workers.  Counters live in
this process's singleton, so a subprocess spawned by
:mod:`multiprocessing` or joblib increments its own (discarded) copy —
the same constraint that makes per-cell timings unavailable for
parallel hyperparameter search.  A parallel ``train`` step therefore
reports accurate energy and utilisation but no ``items_per_s``.

Everything degrades quietly.  On a host with no GPU, the ``cost``
block simply carries no GPU series; if no probe can be constructed at
all the summary still reports duration, counters and CPU.  Telemetry
never fails a run.
"""

from __future__ import annotations

import contextlib
import json
import os
import platform
import resource
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

from src.utils.atomic_io import atomic_write
from src.utils.logging import get_logger

logger = get_logger(__name__)

# Sampling cadence when the config says nothing.  1 s keeps the series
# small enough to hold in memory for a multi-hour run (~3.6 k samples
# per hour) while still resolving a stall that lasts a few seconds.
DEFAULT_INTERVAL_SECONDS = 1.0

# Below this many samples a window cannot produce a meaningful min/max
# spread, so the summary reports the mean only (a step that finished in
# under one sampling interval has no distribution to describe).
_MIN_SAMPLES_FOR_SPREAD = 3


class _Counters:
    """Monotonic totals incremented by the pipeline as work happens."""

    def __init__(self) -> None:
        self.bytes = 0
        self.flops = 0.0
        self.items = 0

    def snapshot(self) -> tuple[int, float, int]:
        return (self.bytes, self.flops, self.items)


class _Sample:
    """One instant: counter totals plus hardware gauges.

    ``__slots__`` matters here — a 10-hour run at 1 s holds ~36 k of
    these, and the dict-free layout keeps that under a few MB.
    """

    __slots__ = (
        "t",
        "bytes",
        "flops",
        "items",
        "cpu_seconds",
        "rss_mb",
        "gpu_util",
        "gpu_power",
        "gpu_mem",
    )

    def __init__(
        self,
        t: float,
        counters: tuple[int, float, int],
        cpu_seconds: float,
        rss_mb: float | None,
        gpu: dict[str, float] | None,
    ) -> None:
        self.t = t
        self.bytes, self.flops, self.items = counters
        self.cpu_seconds = cpu_seconds
        self.rss_mb = rss_mb
        self.gpu_util = gpu.get("util") if gpu else None
        self.gpu_power = gpu.get("power") if gpu else None
        self.gpu_mem = gpu.get("memory") if gpu else None


# ---------------------------------------------------------------------------
# GPU probes
# ---------------------------------------------------------------------------


class _NvmlProbe:
    """Read utilisation / power / memory straight from NVML.

    Preferred probe: an in-process C call per sample, no subprocess and
    no parsing.  Requires ``nvidia-ml-py`` (the ``telemetry`` extra).
    """

    name = "nvml"

    def __init__(self) -> None:
        import pynvml

        self._nvml = pynvml
        pynvml.nvmlInit()
        self._handle = pynvml.nvmlDeviceGetHandleByIndex(0)

    def read(self) -> dict[str, float] | None:
        try:
            util = self._nvml.nvmlDeviceGetUtilizationRates(self._handle)
            mem = self._nvml.nvmlDeviceGetMemoryInfo(self._handle)
            # NVML reports milliwatts.
            power = self._nvml.nvmlDeviceGetPowerUsage(self._handle) / 1000.0
        except Exception:  # noqa: BLE001 - a transient NVML error must not kill sampling
            return None
        return {
            "util": float(util.gpu),
            "power": float(power),
            "memory": mem.used / (1024 * 1024),
        }

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self._nvml.nvmlShutdown()


class _NvidiaSmiProbe:
    """Fallback probe driving one long-lived ``nvidia-smi`` process.

    Spawning ``nvidia-smi`` per sample would cost 30-80 ms of CPU every
    tick.  Instead we start it once in loop mode (``-lms``) and read the
    CSV lines it emits, so the per-sample cost is a buffered read of the
    most recent line.  This is the probe that works in the stock image,
    where ``nvidia-smi`` is injected by nvidia-container-toolkit but no
    Python NVML binding is installed.
    """

    name = "nvidia-smi"

    _QUERY = "utilization.gpu,power.draw,memory.used"

    def __init__(self, interval_seconds: float) -> None:
        binary = shutil.which("nvidia-smi")
        if binary is None:
            raise RuntimeError("nvidia-smi not on PATH")
        interval_ms = max(100, int(interval_seconds * 1000))
        self._proc = subprocess.Popen(  # noqa: S603 - fixed argv, no shell
            [
                binary,
                f"--query-gpu={self._QUERY}",
                "--format=csv,noheader,nounits",
                f"-lms={interval_ms}",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        self._latest: dict[str, float] | None = None
        self._lock = threading.Lock()
        self._reader = threading.Thread(target=self._pump, name="nvsmi-reader", daemon=True)
        self._reader.start()

    def _pump(self) -> None:
        """Consume stdout continuously so the pipe never fills and blocks."""
        stream = self._proc.stdout
        if stream is None:
            return
        for line in stream:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) != 3:
                continue
            try:
                reading = {
                    "util": float(parts[0]),
                    "power": float(parts[1]),
                    "memory": float(parts[2]),
                }
            except ValueError:
                # "[N/A]" — cards that do not report power draw.
                continue
            with self._lock:
                self._latest = reading

    def read(self) -> dict[str, float] | None:
        with self._lock:
            return dict(self._latest) if self._latest else None

    def close(self) -> None:
        try:
            self._proc.terminate()
            self._proc.wait(timeout=5)
        except Exception:  # noqa: BLE001
            self._proc.kill()


class _TorchProbe:
    """Last-resort probe: allocated GPU memory only, straight from torch.

    No utilisation and no power, so a step summarised through this probe
    reports memory but no ``energy_joules``.  Better than nothing on a
    host where NVML is unreachable.
    """

    name = "torch"

    def __init__(self) -> None:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("cuda unavailable")
        self._torch = torch

    def read(self) -> dict[str, float] | None:
        try:
            return {"memory": self._torch.cuda.memory_allocated() / (1024 * 1024)}
        except Exception:  # noqa: BLE001
            return None

    def close(self) -> None:
        pass


def _build_gpu_probe(interval_seconds: float):
    """Return the best available GPU probe, or ``None`` on a CPU host."""
    for factory in (
        _NvmlProbe,
        lambda: _NvidiaSmiProbe(interval_seconds),
        _TorchProbe,
    ):
        try:
            probe = factory()
        except Exception as exc:  # noqa: BLE001 - each probe is genuinely optional
            logger.debug("GPU telemetry probe unavailable: %r", exc)
            continue
        logger.debug("GPU telemetry probe: %s", probe.name)
        return probe
    return None


# ---------------------------------------------------------------------------
# CPU probe
# ---------------------------------------------------------------------------


class _CpuProbe:
    """Cumulative CPU seconds and RSS for this process *and its children*.

    DataLoader workers and joblib jobs are separate processes, so a
    self-only reading would understate extraction badly.  ``psutil``
    walks live children and is used when present.  The fallback,
    :func:`resource.getrusage` with ``RUSAGE_CHILDREN``, only accounts
    for children that have already been reaped — it undercounts while
    workers are alive, which is why psutil is preferred and shipped in
    the ``telemetry`` extra.
    """

    def __init__(self) -> None:
        self._psutil = None
        self._proc = None
        try:
            import psutil

            self._psutil = psutil
            self._proc = psutil.Process()
        except Exception:  # noqa: BLE001
            logger.debug("psutil unavailable; CPU telemetry falls back to getrusage")

    @property
    def name(self) -> str:
        return "psutil" if self._psutil else "getrusage"

    def read(self) -> tuple[float, float | None]:
        """Return ``(cumulative_cpu_seconds, rss_mb)``."""
        if self._psutil and self._proc:
            try:
                cpu = 0.0
                rss = 0
                for proc in [self._proc, *self._proc.children(recursive=True)]:
                    try:
                        times = proc.cpu_times()
                        cpu += times.user + times.system
                        rss += proc.memory_info().rss
                    except Exception:  # noqa: BLE001, PERF203 - process died mid-walk
                        continue
                return cpu, rss / (1024 * 1024)
            except Exception:  # noqa: BLE001
                pass

        own = resource.getrusage(resource.RUSAGE_SELF)
        kids = resource.getrusage(resource.RUSAGE_CHILDREN)
        cpu = own.ru_utime + own.ru_stime + kids.ru_utime + kids.ru_stime
        # ru_maxrss is kilobytes on Linux, bytes on macOS.
        divisor = 1024 * 1024 if platform.system() == "Darwin" else 1024
        return cpu, own.ru_maxrss / divisor


# ---------------------------------------------------------------------------
# Sampler
# ---------------------------------------------------------------------------


class _Sampler:
    """Background thread accumulating :class:`_Sample` instances."""

    def __init__(self, interval_seconds: float) -> None:
        self.interval = max(0.05, float(interval_seconds))
        self.counters = _Counters()
        self._samples: list[_Sample] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._cpu = _CpuProbe()
        self._gpu = None
        self._cpu_count = os.cpu_count() or 1

    def start(self) -> None:
        self._gpu = _build_gpu_probe(self.interval)
        self.take_sample()  # baseline for the first step
        self._thread = threading.Thread(target=self._loop, name="telemetry", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.wait(self.interval):
            try:
                self.take_sample()
            except Exception as exc:  # noqa: BLE001 - sampling must never kill the run
                logger.debug("telemetry sample failed: %r", exc)

    def take_sample(self) -> float:
        """Record one sample now and return its timestamp.

        The timestamp is what callers use to bound a window, so that the
        edge sample is inside the window it opens rather than one tick
        before it.
        """
        cpu_seconds, rss_mb = self._cpu.read()
        gpu = self._gpu.read() if self._gpu else None
        sample = _Sample(time.perf_counter(), self.counters.snapshot(), cpu_seconds, rss_mb, gpu)
        with self._lock:
            self._samples.append(sample)
        return sample.t

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=self.interval * 2 + 1)
        self.take_sample()  # final edge, so the last window is closed
        if self._gpu:
            self._gpu.close()

    def add(self, *, nbytes: int = 0, flops: float = 0.0, items: int = 0) -> None:
        """Increment the monotonic counters.  Called from hot loops."""
        with self._lock:
            self.counters.bytes += nbytes
            self.counters.flops += flops
            self.counters.items += items

    def window(self, t0: float, t1: float) -> list[_Sample]:
        with self._lock:
            return [s for s in self._samples if t0 <= s.t <= t1]

    def all_samples(self) -> list[_Sample]:
        with self._lock:
            return list(self._samples)

    @property
    def probes(self) -> dict[str, str]:
        return {
            "cpu": self._cpu.name,
            "gpu": self._gpu.name if self._gpu else "none",
        }


_SAMPLER: _Sampler | None = None
_RUN_DIR: Path | None = None
_SAVE_SAMPLES = False
# Survives stop() so the manifest, written afterwards, can still record
# which backends produced the numbers.
_LAST_PROBES: dict[str, str] = {"cpu": "none", "gpu": "none"}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def start(config: dict[str, Any] | None = None, run_dir: Path | str | None = None) -> None:
    """Start the process-wide sampler.

    Called once by ``main.py`` right after the manifest opens.  Reads
    ``telemetry.enabled``, ``telemetry.sample_interval_seconds`` and
    ``telemetry.save_samples`` from the merged config.  A second call
    without an intervening :func:`stop` is a no-op, so a nested entry
    point cannot start two samplers.
    """
    global _SAMPLER, _RUN_DIR, _SAVE_SAMPLES

    if _SAMPLER is not None:
        return

    settings = (config or {}).get("telemetry", {}) or {}
    if not settings.get("enabled", True):
        logger.debug("telemetry disabled by config")
        return

    interval = float(settings.get("sample_interval_seconds", DEFAULT_INTERVAL_SECONDS))
    _SAVE_SAMPLES = bool(settings.get("save_samples", False))
    _RUN_DIR = Path(run_dir) if run_dir else None

    sampler = _Sampler(interval)
    try:
        sampler.start()
    except Exception as exc:  # noqa: BLE001
        logger.warning("could not start telemetry sampler: %r", exc)
        return

    global _LAST_PROBES
    _SAMPLER = sampler
    _LAST_PROBES = sampler.probes
    logger.debug("telemetry started (interval=%.2fs, probes=%s)", interval, sampler.probes)


def stop() -> None:
    """Stop the sampler and, when configured, flush the raw series."""
    global _SAMPLER

    sampler = _SAMPLER
    if sampler is None:
        return
    _SAMPLER = None

    try:
        sampler.stop()
    except Exception as exc:  # noqa: BLE001
        logger.debug("telemetry stop failed: %r", exc)

    if _SAVE_SAMPLES and _RUN_DIR is not None:
        _flush_samples(sampler, _RUN_DIR)


def is_active() -> bool:
    """True when a sampler is running and counters are being collected."""
    return _SAMPLER is not None


def mark() -> float | None:
    """Stamp the start of a window; pass the result to :func:`summarise_since`.

    Takes a sample immediately so the window has an exact left edge.
    Without it, work done between the mark and the sampler's next tick
    would be silently attributed to the *previous* step, and a step
    shorter than one interval would see no counter movement at all.
    """
    sampler = _SAMPLER
    if sampler is None:
        return None
    return sampler.take_sample()


def summarise_since(marker: float | None) -> dict[str, Any] | None:
    """Aggregate every sample taken between ``marker`` and now.

    Returns ``None`` when telemetry is off or the marker is missing, so
    callers can attach the result unconditionally and simply omit the
    key when there is nothing to report.
    """
    sampler = _SAMPLER
    if sampler is None or marker is None:
        return None
    # Closing edge, mirroring the one mark() opened.
    return _summarise(sampler, marker, sampler.take_sample())


def add_bytes(n: int) -> None:
    """Record ``n`` bytes transferred (network download throughput)."""
    sampler = _SAMPLER
    if sampler is not None:
        sampler.add(nbytes=n)


def add_flops(n: float) -> None:
    """Record ``n`` floating-point operations dispatched."""
    sampler = _SAMPLER
    if sampler is not None:
        sampler.add(flops=n)


def add_items(n: int) -> None:
    """Record ``n`` units of work (images, interactions, users) processed."""
    sampler = _SAMPLER
    if sampler is not None:
        sampler.add(items=n)


def probes() -> dict[str, str]:
    """Report which probe backends produced the numbers.

    Reads through to the last started sampler rather than the live one:
    the manifest is written after :func:`stop`, and a report of
    ``"none"`` there would misattribute perfectly good measurements.
    """
    sampler = _SAMPLER
    if sampler is not None:
        return sampler.probes
    return dict(_LAST_PROBES)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _stats(values: list[float]) -> dict[str, float] | None:
    """min / max / mean, or mean alone when the window is too short."""
    clean = [v for v in values if v is not None]
    if not clean:
        return None
    mean = sum(clean) / len(clean)
    if len(clean) < _MIN_SAMPLES_FOR_SPREAD:
        return {"mean": round(mean, 4), "samples": len(clean)}
    return {
        "min": round(min(clean), 4),
        "max": round(max(clean), 4),
        "mean": round(mean, 4),
        "samples": len(clean),
    }


def _summarise(sampler: _Sampler, t0: float, t1: float) -> dict[str, Any]:
    samples = sampler.window(t0, t1)
    summary: dict[str, Any] = {
        "sample_interval_seconds": round(sampler.interval, 3),
        "samples": len(samples),
    }

    if len(samples) < 2:
        # Only reachable if sampling failed outright; nothing to derive.
        summary["note"] = "no samples captured for this window"
        return summary

    # --- rates between consecutive samples ---------------------------------
    net_rates: list[float] = []
    flop_rates: list[float] = []
    item_rates: list[float] = []
    cpu_rates: list[float] = []

    for prev, cur in zip(samples, samples[1:], strict=False):
        dt = cur.t - prev.t
        if dt <= 0:
            continue
        net_rates.append((cur.bytes - prev.bytes) / dt / (1024 * 1024))
        flop_rates.append((cur.flops - prev.flops) / dt)
        item_rates.append((cur.items - prev.items) / dt)
        # CPU seconds burned per wall second, expressed as % of one core.
        cpu_rates.append((cur.cpu_seconds - prev.cpu_seconds) / dt * 100.0)

    first, last = samples[0], samples[-1]
    total_bytes = last.bytes - first.bytes
    total_flops = last.flops - first.flops
    total_items = last.items - first.items

    throughput: dict[str, Any] = {}
    if total_bytes > 0:
        throughput["network_mb_per_s"] = _stats(net_rates)
        throughput["total_bytes"] = total_bytes
        throughput["total_mb"] = round(total_bytes / (1024 * 1024), 3)
    if total_flops > 0:
        throughput["flops_per_s"] = _stats(flop_rates)
        throughput["total_flops"] = total_flops
        throughput["total_tflops"] = round(total_flops / 1e12, 4)
    if total_items > 0:
        throughput["items_per_s"] = _stats(item_rates)
        throughput["total_items"] = total_items
    if throughput:
        summary["throughput"] = throughput

    # --- cost gauges -------------------------------------------------------
    cost: dict[str, Any] = {}
    cpu_stats = _stats(cpu_rates)
    if cpu_stats:
        cost["cpu_util_percent"] = cpu_stats
        cost["cpu_cores_used_mean"] = round(cpu_stats["mean"] / 100.0, 3)
    rss = _stats([s.rss_mb for s in samples if s.rss_mb is not None])
    if rss:
        cost["rss_mb"] = rss

    gpu_util = _stats([s.gpu_util for s in samples if s.gpu_util is not None])
    if gpu_util:
        cost["gpu_util_percent"] = gpu_util
    gpu_mem = _stats([s.gpu_mem for s in samples if s.gpu_mem is not None])
    if gpu_mem:
        cost["gpu_mem_mb"] = gpu_mem

    power_samples = [(s.t, s.gpu_power) for s in samples if s.gpu_power is not None]
    power_stats = _stats([p for _, p in power_samples])
    if power_stats:
        cost["gpu_power_watts"] = power_stats
        joules = _integrate(power_samples)
        cost["energy_joules"] = round(joules, 2)
        cost["energy_wh"] = round(joules / 3600.0, 5)

    if cost:
        summary["cost"] = cost

    return summary


def _integrate(series: list[tuple[float, float]]) -> float:
    """Trapezoidal integral of a ``(timestamp, value)`` series.

    Trapezoid rather than rectangle: GPU power ramps between samples,
    and a left-rectangle sum systematically over-reports the tail of a
    step that finishes mid-window.
    """
    total = 0.0
    for (t0, v0), (t1, v1) in zip(series, series[1:], strict=False):
        total += (v0 + v1) / 2.0 * (t1 - t0)
    return total


def _flush_samples(sampler: _Sampler, run_dir: Path) -> None:
    """Write the raw series to ``telemetry_samples.jsonl`` for plotting."""
    path = run_dir / "telemetry_samples.jsonl"
    samples = sampler.all_samples()
    if not samples:
        return
    origin = samples[0].t

    def _write(tmp: str) -> None:
        with open(tmp, "w", encoding="utf-8") as fh:
            for s in samples:
                fh.write(
                    json.dumps(
                        {
                            "t": round(s.t - origin, 3),
                            "bytes": s.bytes,
                            "flops": s.flops,
                            "items": s.items,
                            "cpu_seconds": round(s.cpu_seconds, 3),
                            "rss_mb": round(s.rss_mb, 2) if s.rss_mb is not None else None,
                            "gpu_util": s.gpu_util,
                            "gpu_power": s.gpu_power,
                            "gpu_mem_mb": s.gpu_mem,
                        }
                    )
                    + "\n"
                )

    try:
        atomic_write(_write, path)
        logger.info("Raw telemetry series written to %s (%d samples)", path, len(samples))
    except OSError as exc:
        logger.warning("failed to write %s: %r", path, exc)


def reset_for_tests() -> None:
    """Tear down any active sampler.  Test-only escape hatch."""
    global _SAMPLER, _RUN_DIR, _SAVE_SAMPLES, _LAST_PROBES
    if _SAMPLER is not None:
        with contextlib.suppress(Exception):
            _SAMPLER.stop()
    _SAMPLER = None
    _RUN_DIR = None
    _SAVE_SAMPLES = False
    _LAST_PROBES = {"cpu": "none", "gpu": "none"}
