"""Budget helpers for sizing process pools: memory AND cpu.

Several steps fan work out across worker *processes* (fusion via
``ProcessPoolExecutor``, hyperparameter search via
``torch.multiprocessing``).  Sizing those pools from ``os.cpu_count()``
alone is unsafe: a fusion worker holding two native embedding matrices
for a 350K-item catalogue peaks at several GB of RSS, so 12 concurrent
workers on a 16-core host ask for ~100 GB of RAM.  When the container
runs without a memory limit the kernel cannot contain the damage to the
container, it declares a *global* OOM and starts killing the host's own
processes, which is how a runaway pool takes a desktop down.

This module answers one question: *how many workers of a known
footprint fit in the memory this process is allowed to use?*  The budget
is the cgroup limit when one is set (container) and the host's total RAM
otherwise, minus a reserve that keeps the parent process, the page cache
and, on an unconstrained host, the desktop session alive.

The functions are pure reads, no allocation, safe to call at import
time.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from src.utils.logging import get_logger

logger = get_logger(__name__)

# cgroup v1 with no limit set returns a sentinel close to ``2 ** 63``;
# any value above this threshold is treated as "no limit".
_CGROUP_NO_LIMIT_THRESHOLD = 1 << 60

_FALLBACK_MEMORY_GB = 4.0  # used when neither cgroup nor sysconf works

#: Memory never handed to a worker pool: the parent process (which holds
#: the config, the task list and, in fusion, the PCA-aligned sources),
#: the page cache backing the ``.npy`` reads, and the host session when
#: no cgroup limit confines this process.
RESERVED_BYTES = 4 * 1024**3


def memory_budget_bytes() -> int:
    """Return the strictest memory budget that applies to this process.

    Resolution order: cgroup v2 -> cgroup v1 -> host total memory ->
    a 4 GB fallback so the function never returns 0.
    """
    cgroup_v2 = _read_int_file(Path("/sys/fs/cgroup/memory.max"))
    if cgroup_v2 is not None and cgroup_v2 < _CGROUP_NO_LIMIT_THRESHOLD:
        return cgroup_v2

    cgroup_v1 = _read_int_file(Path("/sys/fs/cgroup/memory/memory.limit_in_bytes"))
    if cgroup_v1 is not None and cgroup_v1 < _CGROUP_NO_LIMIT_THRESHOLD:
        return cgroup_v1

    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError):
        pass

    return int(_FALLBACK_MEMORY_GB * 1024**3)


def available_host_bytes() -> int | None:
    """Host memory this process can still allocate, or ``None`` when unknown.

    The conservative minimum of the cgroup headroom (limit minus current
    usage, v2 then v1) and the kernel's ``MemAvailable``; a memory-mapped
    feature file's page cache counts against the cgroup, which is why
    the cgroup figure is consulted first.  Pure read, no allocation.
    Callers admitting a single unavoidable allocation (the evaluator's
    full score vector) treat ``None`` as "unverifiable": they log and
    proceed, because there is no smaller alternative to fall back to.
    """
    candidates: list[int] = []
    limit = memory_budget_bytes()
    usage = _read_int_file(Path("/sys/fs/cgroup/memory.current"))
    if usage is None:
        usage = _read_int_file(Path("/sys/fs/cgroup/memory/memory.usage_in_bytes"))
    if usage is not None and limit < _CGROUP_NO_LIMIT_THRESHOLD:
        candidates.append(max(0, limit - usage))
    mem_available = _read_meminfo_available()
    if mem_available is not None:
        candidates.append(mem_available)
    return min(candidates) if candidates else None


def _read_meminfo_available() -> int | None:
    """``MemAvailable`` from ``/proc/meminfo`` in bytes, or ``None``."""
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        return None
    return None


def available_cpus() -> int:
    """CPU cores THIS process may actually use, honouring the cgroup quota.

    ``os.cpu_count()`` reports the machine, not the allowance: a
    container limited with ``cpus: 10.4`` still sees all 16 cores and
    sizes its pools for 16, so the pool oversubscribes its own quota and
    spends the difference on context switching.  This is the CPU twin of
    :func:`memory_budget_bytes`.

    Resolution order: cgroup v2 ``cpu.max`` -> cgroup v1
    ``cpu.cfs_quota_us`` / ``cpu.cfs_period_us`` -> ``os.cpu_count()``.
    A fractional quota rounds DOWN (10.4 cores -> 10), so a pool never
    asks for a core the scheduler will not give it, and the result is
    never below 1.

    :returns: Usable core count, at least 1.
    """
    quota = _read_cgroup_cpu_quota()
    host = os.cpu_count() or 1
    if quota is None:
        return max(1, host)
    return max(1, min(host, quota))


def _read_cgroup_cpu_quota() -> int | None:
    """Whole cores the cgroup allows, or ``None`` when unconstrained."""
    v2 = Path("/sys/fs/cgroup/cpu.max")
    try:
        raw = v2.read_text(encoding="utf-8").split()
        if raw and raw[0] != "max":
            return int(int(raw[0]) // int(raw[1]))
    except (OSError, ValueError, IndexError):
        pass

    period = _read_int_file(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us"))
    quota = _read_int_file(Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us"))
    if period and quota and quota > 0:
        return int(quota // period)
    return None


def plan_pool_workers(
    *,
    per_worker_bytes: int,
    hard_cap: int,
    reserve_bytes: int = RESERVED_BYTES,
    label: str = "pool",
) -> int:
    """Return how many workers of *per_worker_bytes* fit in the budget.

    The result is clamped to ``[1, hard_cap]``: a single worker always
    runs even when the estimate says nothing fits, because refusing to
    make progress is worse than one process the kernel may swap.  The
    caller is responsible for the footprint estimate; overestimating is
    the safe direction.

    :param per_worker_bytes:
        Peak resident bytes one worker is expected to hold.  Values
        ``<= 0`` mean "unknown / negligible" and yield *hard_cap*.
    :param hard_cap:
        Upper bound from the caller's own constraints (CPU count, number
        of pending tasks).
    :param reserve_bytes:
        Memory withheld from the pool for the parent process and the
        host.  See :data:`RESERVED_BYTES`.
    :param label:
        Name used in the log line, so a reader of the run log can tell
        which pool was resized.
    :returns:
        Worker count in ``[1, hard_cap]``.
    """
    if hard_cap <= 1:
        return max(1, hard_cap)
    if per_worker_bytes <= 0:
        return hard_cap

    budget = memory_budget_bytes() - reserve_bytes
    fits = int(budget // per_worker_bytes)
    n_workers = max(1, min(hard_cap, fits))

    if n_workers < hard_cap:
        logger.info(
            "%s: memory-capped to %d workers (budget=%.1f GB after %.1f GB "
            "reserve, ~%.1f GB/worker, cpu/task cap was %d)",
            label,
            n_workers,
            budget / 1024**3,
            reserve_bytes / 1024**3,
            per_worker_bytes / 1024**3,
            hard_cap,
        )
    return n_workers


def _read_int_file(path: Path) -> int | None:
    """Read *path* and parse it as an integer (cgroup interface convention).

    Returns ``None`` when the file does not exist, is unreadable, or
    contains a non-numeric value (e.g. cgroup v2 ``"max"``).
    """
    try:
        text = path.read_text().strip()
    except (FileNotFoundError, PermissionError, OSError):
        return None
    try:
        return int(text)
    except ValueError:
        return None


# ---------------------------------------------------------------------
# Host admission (M05/M06, C06)
# ---------------------------------------------------------------------

#: Bytes per parameter of a trainable table under Adam: the weight, its
#: gradient and the two moment estimates, all float32.
_ADAM_BYTES_PER_PARAM = 4 * 4

#: Default latent width when a job declares none (``common.total_dim``).
_DEFAULT_TOTAL_DIM = 128

FEATURE_RESIDENCY_POLICIES = ("dense", "lazy", "auto")


class AdmissionError(RuntimeError):
    """A job's declared minimum memory exceeds the resolved budget.

    Raised (or recorded as a failed outcome) *before* the job is
    launched, so a job that cannot fit is never started repeatedly.
    """


@dataclass(frozen=True)
class ResourcesConfig:
    """Resolved ``resources:`` block (C06 proposal; every key optional).

    ``host_budget_bytes`` — explicit host budget; ``None`` resolves the
    container/host limit.  ``headroom_bytes`` — withheld from every pool
    for the parent, page cache and transfer buffers (default
    :data:`RESERVED_BYTES`).  ``max_workers`` — hard cap on concurrent
    workers; ``None`` leaves the CPU/VRAM caps in charge.
    ``feature_residency`` — ``dense`` (today's resident matrices),
    ``lazy`` (bounded row reads, M01/M02) or ``auto`` (lazy only when the
    dense payload does not fit the budget).  Zero or absent values never
    mean "unlimited": an absent budget is resolved conservatively and an
    explicit ``0`` admits nothing.
    """

    host_budget_bytes: int | None = None
    headroom_bytes: int = RESERVED_BYTES
    max_workers: int | None = None
    feature_residency: str = "dense"


def _non_negative_int(block: dict, key: str, default: int | None) -> int | None:
    value = block.get(key, default)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float) or value != value:
        raise ValueError(f"resources.{key} must be a finite non-negative integer, got {value!r}")
    if value < 0 or value in (float("inf"), float("-inf")):
        raise ValueError(f"resources.{key} must be a finite non-negative integer, got {value!r}")
    return int(value)


def resolve_resources(config: dict | None) -> ResourcesConfig:
    """Validate and resolve the optional ``resources:`` block of *config*."""
    block = (config or {}).get("resources") or {}
    if not isinstance(block, dict):
        raise ValueError(f"resources must be a mapping, got {type(block).__name__}")
    unknown = set(block) - {
        "host_budget_bytes",
        "headroom_bytes",
        "max_workers",
        "feature_residency",
    }
    if unknown:
        raise ValueError(f"resources has unknown keys: {sorted(unknown)}")
    residency = str(block.get("feature_residency", "dense"))
    if residency not in FEATURE_RESIDENCY_POLICIES:
        raise ValueError(
            f"resources.feature_residency must be one of {FEATURE_RESIDENCY_POLICIES}, "
            f"got {residency!r}"
        )
    headroom = _non_negative_int(block, "headroom_bytes", RESERVED_BYTES)
    return ResourcesConfig(
        host_budget_bytes=_non_negative_int(block, "host_budget_bytes", None),
        headroom_bytes=int(headroom if headroom is not None else RESERVED_BYTES),
        max_workers=_non_negative_int(block, "max_workers", None),
        feature_residency=residency,
    )


@dataclass(frozen=True)
class HostBudget:
    """The host memory limit this process must plan against, and where it came from."""

    limit_bytes: int
    source: str  # "config" | "cgroup_v2" | "cgroup_v1" | "host" | "fallback"
    available_bytes: int | None = None


def resolve_host_budget(config: dict | None = None) -> HostBudget:
    """Resolve the host budget conservatively; never unlimited.

    ``resources.host_budget_bytes`` wins when set; otherwise the cgroup v2
    limit, the cgroup v1 limit, the host's physical memory, and finally
    the 4 GiB fallback (source ``fallback``) when nothing is readable.
    The result also carries the current headroom (:func:`available_host_bytes`).
    """
    explicit = resolve_resources(config).host_budget_bytes if config is not None else None
    if explicit is not None:
        return HostBudget(explicit, "config", available_host_bytes())
    cgroup_v2 = _read_int_file(Path("/sys/fs/cgroup/memory.max"))
    if cgroup_v2 is not None and cgroup_v2 < _CGROUP_NO_LIMIT_THRESHOLD:
        return HostBudget(cgroup_v2, "cgroup_v2", available_host_bytes())
    cgroup_v1 = _read_int_file(Path("/sys/fs/cgroup/memory/memory.limit_in_bytes"))
    if cgroup_v1 is not None and cgroup_v1 < _CGROUP_NO_LIMIT_THRESHOLD:
        return HostBudget(cgroup_v1, "cgroup_v1", available_host_bytes())
    try:
        host = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError):
        return HostBudget(int(_FALLBACK_MEMORY_GB * 1024**3), "fallback", None)
    return HostBudget(int(host), "host", available_host_bytes())


@dataclass(frozen=True)
class AdmissionPlan:
    """Outcome of admitting workers of a known footprint against a budget."""

    n_workers: int
    per_worker_bytes: int
    budget_bytes: int
    headroom_bytes: int
    budget_source: str
    admitted: bool
    reason: str

    @property
    def usable_bytes(self) -> int:
        return max(0, self.budget_bytes - self.headroom_bytes)


def admit_workers(
    per_worker_bytes: int,
    *,
    hard_cap: int,
    budget: HostBudget,
    headroom_bytes: int = RESERVED_BYTES,
    max_workers: int | None = None,
    label: str = "pool",
) -> AdmissionPlan:
    """Admit the largest worker count whose aggregate commitment fits the budget.

    Unlike :func:`plan_pool_workers` this never rounds up to one: a
    footprint larger than ``budget - headroom`` is *not admitted*
    (``admitted=False``) and the caller must refuse the job instead of
    launching it.  ``per_worker_bytes <= 0`` is treated as *unknown*, not
    as free: one worker is admitted with the reason recorded.
    """
    usable = max(0, budget.limit_bytes - headroom_bytes)
    cap = max(1, hard_cap)
    if max_workers is not None:
        cap = max(1, min(cap, max_workers)) if max_workers > 0 else 0
    if cap == 0:
        return AdmissionPlan(
            0,
            per_worker_bytes,
            budget.limit_bytes,
            headroom_bytes,
            budget.source,
            False,
            "resources.max_workers is 0: nothing admitted",
        )
    if per_worker_bytes <= 0:
        plan = AdmissionPlan(
            1,
            per_worker_bytes,
            budget.limit_bytes,
            headroom_bytes,
            budget.source,
            True,
            "worker footprint unknown; one worker admitted conservatively",
        )
        logger.info("%s: %s", label, plan.reason)
        return plan
    fits = usable // per_worker_bytes
    if fits < 1:
        return AdmissionPlan(
            0,
            per_worker_bytes,
            budget.limit_bytes,
            headroom_bytes,
            budget.source,
            False,
            f"declared minimum {per_worker_bytes / 1024**3:.2f} GB exceeds the usable budget "
            f"{usable / 1024**3:.2f} GB ({budget.source} limit {budget.limit_bytes / 1024**3:.2f} GB "
            f"minus {headroom_bytes / 1024**3:.2f} GB headroom)",
        )
    n_workers = int(min(cap, fits))
    plan = AdmissionPlan(
        n_workers,
        per_worker_bytes,
        budget.limit_bytes,
        headroom_bytes,
        budget.source,
        True,
        f"{n_workers} worker(s) x {per_worker_bytes / 1024**3:.2f} GB within "
        f"{usable / 1024**3:.2f} GB usable ({budget.source})",
    )
    if n_workers < cap:
        logger.info("%s: memory-capped to %s", label, plan.reason)
    return plan


def choose_lazy_features(policy: str, payload_bytes: int, plan_usable_bytes: int) -> bool:
    """Whether a job should read its features lazily under *policy*.

    ``dense`` never, ``lazy`` always, ``auto`` when the resident payload
    alone would not leave room in the usable budget (half of it, so the
    model state and the ranking workspace still fit).
    """
    if policy == "lazy":
        return True
    if policy == "dense":
        return False
    return payload_bytes > plan_usable_bytes // 2


def estimate_model_state_bytes(
    n_users: int, n_items: int, total_dim: int | None, *, visual_dim: int = 0
) -> int:
    """Bytes of parameters + gradients + Adam moments of the trainable tables.

    ``(n_users + n_items) x d`` embeddings plus a ``visual_dim x d``
    projection, each charged :data:`_ADAM_BYTES_PER_PARAM`.  Bias vectors
    and small heads are inside the rounding.
    """
    dim = int(total_dim or _DEFAULT_TOTAL_DIM)
    params = (int(n_users) + int(n_items)) * dim + int(visual_dim) * dim
    return int(params * _ADAM_BYTES_PER_PARAM)


def npy_payload_bytes(path: str | Path) -> tuple[int, int]:
    """``(payload_bytes, trailing_width)`` of a ``.npy`` from its header only."""
    import numpy as np

    with Path(path).open("rb") as handle:
        version = np.lib.format.read_magic(handle)
        reader = (
            np.lib.format.read_array_header_1_0
            if version == (1, 0)
            else np.lib.format.read_array_header_2_0
        )
        shape, _fortran, dtype = reader(handle)
    shape = tuple(int(s) for s in shape)
    width = int(np.prod(shape[1:], dtype=np.int64)) if len(shape) > 1 else 1
    return int(np.prod(shape, dtype=np.int64)) * int(np.dtype(dtype).itemsize), width
