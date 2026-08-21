"""Host-memory budget helpers for sizing process pools.

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
