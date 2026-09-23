"""Recognise a CUDA fault that has poisoned this process's context.

A GPU hang (NVRM Xid 8, "the launch timed out and was terminated") is
not an error of the job that was running: every later CUDA call in the
same process fails the same way, and only a new process gets a working
context back.  The pipeline answers such a fault by exiting with
:data:`src.supervisor.EXIT_CUDA_FAULT`, so the supervisor can resume the
run in a fresh child (see :mod:`src.supervisor`).

Two ways to tell, used together:

* :func:`is_fatal_cuda_error` reads the exception chain.  Only the
  messages of context-destroying faults count; an OOM, a device-side
  assert or a shape error is a property of the code or the job and is
  never retried.
* :func:`cuda_context_lost` probes the device.  A step that catches
  per-cell exceptions and re-raises a summary loses the original chain;
  the probe still sees the context is unusable.
"""

from __future__ import annotations

import sys

#: Substrings of ``CUDA error: ...`` messages that mean the context is gone.
FATAL_CUDA_MARKERS = (
    "the launch timed out and was terminated",
    "unspecified launch failure",
    "cuda error: unknown error",
    "busy or unavailable",
)


class CudaContextLostError(RuntimeError):
    """Raised when a CUDA fault leaves the process unable to run more work."""


def is_fatal_cuda_error(exc: BaseException) -> bool:
    """True when *exc* or anything in its cause/context chain is a context fault.

    :param exc: The exception that ended a job, a step or the run.
    :returns: Whether the fault destroyed the CUDA context.
    """
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, CudaContextLostError):
            return True
        message = str(current).lower()
        if any(marker in message for marker in FATAL_CUDA_MARKERS):
            return True
        current = current.__cause__ or current.__context__
    return False


def cuda_context_lost() -> bool:
    """Probe the device; True only when a CUDA call fails with a context fault.

    Never initialises CUDA: a process that has not touched the GPU has
    no context to lose, and creating one here would cost VRAM.

    :returns: Whether this process's CUDA context is unusable.
    """
    torch = sys.modules.get("torch")
    if torch is None:
        return False
    try:
        if not torch.cuda.is_initialized():
            return False
        torch.zeros(1, device="cuda")
        torch.cuda.synchronize()
    except Exception as exc:  # noqa: BLE001 — the probe classifies, it does not handle
        return is_fatal_cuda_error(exc)
    return False
