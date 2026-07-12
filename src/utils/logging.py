"""Standardised logger factory.

Every module in the project obtains its logger through
:func:`get_logger`; output is consistently formatted, written to the
console, persisted to a per-module file *and* streamed into a single
chronological session log so a long-running pod produces one file you
can ``tail -f`` to follow the entire pipeline at once.

Layout under ``logs/``::

    logs/
    ├── run_<timestamp>.log     # unified, chronological — one per run
    ├── src.steps.train.log     # per-module drill-down
    ├── src.steps.fuse.log
    └── ...

The unified file is created lazily on the first :func:`get_logger`
call.  Override the timestamp with the ``PRISM_RUN_ID`` env var (the
legacy ``HVR_RUN_ID`` still works) when launching the pipeline if you
want a stable name (handy when mounting the log directory from outside
the container/pod).
"""

from __future__ import annotations

import logging
import os
from datetime import UTC, datetime
from pathlib import Path

_LOG_FORMAT = "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s"

# Track configured loggers so repeated calls with the same *name* do
# not duplicate handlers.
_CONFIGURED_LOGGERS: set[str] = set()

# Resolved once on first use; subsequent ``get_logger`` calls reuse the
# same path so every module writes into the *same* session file.
_SESSION_LOG_PATH: Path | None = None


def _resolve_session_log_path(log_dir: Path) -> Path:
    """Return the unified-session-log path, creating its directory.

    The filename is ``run_<PRISM_RUN_ID or UTC timestamp>.log``; supplying
    ``PRISM_RUN_ID`` from outside the process keeps the name stable
    across forks (useful when the orchestrator launches sub-processes).
    The legacy ``HVR_RUN_ID`` name is still honoured.
    """
    global _SESSION_LOG_PATH
    if _SESSION_LOG_PATH is not None:
        return _SESSION_LOG_PATH

    log_dir.mkdir(parents=True, exist_ok=True)

    # Prefer the PRISM_ prefix; fall back to the legacy HVR_ name so
    # existing pod/orchestrator scripts keep working.
    run_id = os.environ.get("PRISM_RUN_ID") or os.environ.get("HVR_RUN_ID")
    if not run_id:
        run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    # Persist under both names so child processes share the same log file
    # regardless of which variable they read.
    os.environ["PRISM_RUN_ID"] = run_id
    os.environ["HVR_RUN_ID"] = run_id

    _SESSION_LOG_PATH = log_dir / f"run_{run_id}.log"
    return _SESSION_LOG_PATH


def get_logger(name: str, log_dir: str = "logs") -> logging.Logger:
    """Return a logger with console + per-module file + session-log handlers.

    On the first call for a given *name* the logger is set up with:

    * A :class:`~logging.StreamHandler` writing to ``stderr``.
    * A :class:`~logging.FileHandler` writing to ``<log_dir>/<name>.log``
      (per-module drill-down at DEBUG level).
    * A shared :class:`~logging.FileHandler` writing to
      ``<log_dir>/run_<id>.log`` (chronological view at INFO level —
      every module of a single run interleaves into this one file).

    Subsequent calls with the same *name* return the existing logger
    without adding duplicate handlers.
    """
    logger = logging.getLogger(name)

    if name in _CONFIGURED_LOGGERS:
        return logger

    logger.setLevel(logging.DEBUG)
    logger.propagate = False  # avoid double-emission via the root logger

    formatter = logging.Formatter(_LOG_FORMAT)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_path / f"{name}.log", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    session_path = _resolve_session_log_path(log_path)
    session_handler = logging.FileHandler(session_path, encoding="utf-8")
    session_handler.setLevel(logging.INFO)
    session_handler.setFormatter(formatter)
    logger.addHandler(session_handler)

    _CONFIGURED_LOGGERS.add(name)
    return logger


def session_log_path() -> Path | None:
    """Return the unified-session-log path, or ``None`` before init.

    Useful for callers (manifests, status banners) that want to point
    the user at the file they should ``tail -f``.
    """
    return _SESSION_LOG_PATH
