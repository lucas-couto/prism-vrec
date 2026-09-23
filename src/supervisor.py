"""Keep one run alive across CUDA context faults.

A Xid 8 launch timeout (or any other GPU hang) poisons the CUDA context
of the process that hit it, and nothing inside that process can recover
it.  Before this module a fault ended the invocation, ``restart:
on-failure`` in the compose file started a fresh one, and every fault
therefore left a second run directory, a second session log and a
second manifest behind.

``main.py`` now runs as two processes:

* the **supervisor** (this module, the container's PID 1) never imports
  torch and never touches the GPU.  It launches the pipeline as a child
  and waits for it;
* the **child** runs the pipeline exactly as before.  When it dies from
  a CUDA context fault (exit code :data:`EXIT_CUDA_FAULT`) or is killed
  by a signal, the supervisor launches a new child -- with a clean CUDA
  context -- that REOPENS the same run directory, appends to the same
  session log and manifest, and resumes from what is already on disk.

A new run begins only when a new supervisor begins, i.e. on ``docker
compose up -d``.  An ordinary exception ends the run as it always did:
retrying a deterministic defect would only repeat it.  A fault that
recurs without any new work completing between attempts ends the run
after :data:`MAX_RESTARTS_WITHOUT_PROGRESS` restarts, because by then
the GPU is gone rather than hung.

Supervisor and child talk through environment variables and a private
state directory (never through ``results/``): the child records the run
directory it opened and bumps a progress counter whenever a unit of new
work completes.  This module is stdlib-only on purpose -- importing
``src.utils`` would pull torch into PID 1.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

#: Exit code of a child whose CUDA context was lost (``EX_TEMPFAIL``).
EXIT_CUDA_FAULT = 75

#: Restarts allowed in a row without any new work completing.
MAX_RESTARTS_WITHOUT_PROGRESS = 3

#: Seconds between a fault and the next attempt, so the driver finishes
#: resetting the channel before a new context is created on it.
RESTART_DELAY_S = 30.0

ENV_SUPERVISED = "PRISM_SUPERVISED"
ENV_STATE_DIR = "PRISM_SUPERVISOR_STATE"
ENV_ATTEMPT = "PRISM_RUN_ATTEMPT"
ENV_RESTART_REASON = "PRISM_RESTART_REASON"
ENV_RUN_ID = "PRISM_RUN_ID"

#: Signals that kill a child without meaning "stop the run".  SIGTERM and
#: SIGINT are excluded: they are how a person stops the pipeline.
_RETRYABLE_SIGNALS = frozenset({signal.SIGKILL, signal.SIGSEGV, signal.SIGBUS, signal.SIGABRT})

#: Steps whose cells are recorded on every attempt even when nothing new
#: happens (``download`` never skips a cell), so they are not progress.
NON_PROGRESS_STEPS = frozenset({"download"})

_RUN_DIRS_FILE = "run_dirs.json"
_FAULT_FILE = "fault.txt"
_PROGRESS_PREFIX = "progress."
_LOG_FORMAT = "[%s] [src.supervisor] [%s] %s"

_INTERRUPT_HINT = (
    "Ctrl+C ignored: the run keeps going. Detach with Ctrl+P Ctrl+Q; "
    "stop the run with `docker compose stop`."
)

_progress_count = 0


# -- child side -------------------------------------------------------------


def is_supervised() -> bool:
    """True inside a child launched by :func:`supervise`."""
    return os.environ.get(ENV_SUPERVISED) == "1"


def attempt() -> int:
    """1-based attempt number of this child (1 when unsupervised)."""
    try:
        return max(1, int(os.environ.get(ENV_ATTEMPT, "1")))
    except ValueError:
        return 1


def restart_reason() -> str | None:
    """Why the supervisor launched this child again, or ``None`` on attempt 1."""
    return os.environ.get(ENV_RESTART_REASON) or None


def recorded_run_dir(key: str) -> Path | None:
    """The run directory an earlier attempt opened under *key*, if any."""
    state = _state_dir()
    if state is None:
        return None
    path = _read_json(state / _RUN_DIRS_FILE).get(key)
    return Path(path) if path else None


def remember_run_dir(key: str, run_dir: Path) -> None:
    """Record *run_dir* under *key* so a later attempt reopens it."""
    state = _state_dir()
    if state is None:
        return
    entries = _read_json(state / _RUN_DIRS_FILE)
    entries[key] = str(Path(run_dir).resolve())
    _write_text(state / _RUN_DIRS_FILE, json.dumps(entries))


def note_progress() -> None:
    """Count one completed unit of new work (a job, a cell) for this process."""
    global _progress_count
    state = _state_dir()
    if state is None:
        return
    _progress_count += 1
    _write_text(state / f"{_PROGRESS_PREFIX}{os.getpid()}", str(_progress_count))


def note_fault(message: str) -> None:
    """Leave the fault's first line for the supervisor to log and pass on."""
    state = _state_dir()
    if state is None:
        return
    first_line = (message.strip().splitlines() or [""])[0]
    _write_text(state / _FAULT_FILE, first_line[:300])


# -- supervisor side --------------------------------------------------------


def supervise(
    argv: list[str],
    *,
    delay_s: float = RESTART_DELAY_S,
    sleep: Callable[[float], None] = time.sleep,
    ignore_interrupt: bool | None = None,
) -> int:
    """Run *argv* as a supervised child until it ends for a non-retryable reason.

    :param argv: Command line of the child (``[sys.executable, "main.py"]``).
    :param delay_s: Seconds to wait between a fault and the next attempt.
    :param sleep: Sleep function (tests pass a no-op).
    :param ignore_interrupt: Ignore SIGINT; ``None`` ignores it only as PID 1,
        i.e. inside the container, where Ctrl+C comes from a person following
        the output (``docker attach``) and must not end a multi-day run.
    :returns: The exit code the container should end with.
    """
    state = Path(tempfile.mkdtemp(prefix="prism-supervisor-"))
    env = {**os.environ, ENV_SUPERVISED: "1", ENV_STATE_DIR: str(state)}
    # The child's logger names the session log after this id, so every
    # attempt appends to the same ``logs/run_<id>.log``.
    env.setdefault(ENV_RUN_ID, datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"))
    if ignore_interrupt is None:
        ignore_interrupt = os.getpid() == 1
    stopper = _Stopper(ignore_interrupt=ignore_interrupt)
    try:
        return _attempt_loop(argv, env, state, stopper, delay_s=delay_s, sleep=sleep)
    finally:
        stopper.restore()
        shutil.rmtree(state, ignore_errors=True)


def _attempt_loop(argv, env, state: Path, stopper: _Stopper, *, delay_s, sleep) -> int:
    log = _SessionLog(env[ENV_RUN_ID])
    streak = 0
    progress = 0
    number = 1
    while True:
        env[ENV_ATTEMPT] = str(number)
        (state / _FAULT_FILE).unlink(missing_ok=True)
        code = _run_child(argv, env, stopper)
        if stopper.requested or not is_retryable(code):
            return _exit_status(code)
        now = read_progress(state)
        streak = 0 if now > progress else streak + 1
        progress = now
        reason = f"{describe_exit(code)}: {_read_text(state / _FAULT_FILE) or 'no detail'}"
        if streak > MAX_RESTARTS_WITHOUT_PROGRESS:
            log.write(
                "ERROR",
                f"attempt {number} ended by {reason}; {streak - 1} restart(s) "
                "in a row completed no new work, giving up",
            )
            return _exit_status(code)
        number += 1
        log.write(
            "WARNING",
            f"attempt {number - 1} ended by {reason}; resuming the same run "
            f"as attempt {number} in {delay_s:.0f}s",
        )
        env[ENV_RESTART_REASON] = reason
        sleep(delay_s)
        if stopper.requested:
            return _exit_status(code)


def _run_child(argv: list[str], env: dict[str, str], stopper: _Stopper) -> int:
    # Its own session, so the whole group (spawned workers included) can
    # be signalled and swept when the child is gone.
    proc = subprocess.Popen(argv, env=env, start_new_session=True)  # noqa: S603 - fixed argv
    stopper.child = proc
    try:
        return proc.wait()
    finally:
        stopper.child = None
        _kill_group(proc.pid)


def is_retryable(code: int) -> bool:
    """True for a CUDA fault exit or a kill by a non-stop signal."""
    if code == EXIT_CUDA_FAULT:
        return True
    return code < 0 and -code in _RETRYABLE_SIGNALS


def describe_exit(code: int) -> str:
    """Human-readable cause of a child exit (``exit 75``, ``signal SIGKILL``)."""
    if code < 0:
        try:
            return f"signal {signal.Signals(-code).name}"
        except ValueError:
            return f"signal {-code}"
    if code == EXIT_CUDA_FAULT:
        return "CUDA context fault"
    return f"exit {code}"


def read_progress(state: Path) -> int:
    """Units of new work every child process has reported so far."""
    total = 0
    for path in state.glob(f"{_PROGRESS_PREFIX}*"):
        try:
            total += int(path.read_text() or 0)
        except (OSError, ValueError):
            continue
    return total


def _exit_status(code: int) -> int:
    return 128 - code if code < 0 else code


def _kill_group(pgid: int) -> None:
    try:
        os.killpg(pgid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        return
    # PID 1 inherits orphaned workers; reap them so none stays a zombie.
    for _ in range(50):
        try:
            pid, _status = os.waitpid(-1, os.WNOHANG)
        except ChildProcessError:
            return
        if pid == 0:
            time.sleep(0.1)


class _Stopper:
    """Forward stop signals to the child and remember that a stop was asked.

    SIGTERM always stops the run (``docker compose stop``).  SIGINT stops
    it only outside the container: inside, Ctrl+C in ``docker attach`` is
    a person who wanted to stop watching, so it is ignored with a hint.
    The child runs in its own session, so the terminal never delivers the
    interrupt to it directly.
    """

    def __init__(self, *, ignore_interrupt: bool = False) -> None:
        self.requested = False
        self.child: subprocess.Popen | None = None
        self._ignore_interrupt = ignore_interrupt
        self._previous = {
            sig: signal.signal(sig, self._handle) for sig in (signal.SIGTERM, signal.SIGINT)
        }

    def _handle(self, signum: int, _frame: object) -> None:
        if signum == signal.SIGINT and self._ignore_interrupt:
            print(_INTERRUPT_HINT, file=sys.stderr, flush=True)
            return
        self.requested = True
        child = self.child
        if child is not None and child.poll() is None:
            try:
                os.killpg(child.pid, signum)
            except (ProcessLookupError, PermissionError):
                return

    def restore(self) -> None:
        for sig, handler in self._previous.items():
            signal.signal(sig, handler)


class _SessionLog:
    """Append supervisor lines to stderr and to the run's session log."""

    def __init__(self, run_id: str, log_dir: str = "logs") -> None:
        self._path = Path(log_dir) / f"run_{run_id}.log"

    def write(self, level: str, message: str) -> None:
        stamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S,000")
        line = _LOG_FORMAT % (stamp, level, message)
        print(line, file=sys.stderr, flush=True)
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._path, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        except OSError:
            return


def _state_dir() -> Path | None:
    raw = os.environ.get(ENV_STATE_DIR)
    return Path(raw) if raw and is_supervised() else None


def _read_json(path: Path) -> dict[str, str]:
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _read_text(path: Path) -> str:
    try:
        return path.read_text().strip()
    except OSError:
        return ""


def _write_text(path: Path, text: str) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(text)
        os.replace(tmp, path)
    except OSError:
        return
