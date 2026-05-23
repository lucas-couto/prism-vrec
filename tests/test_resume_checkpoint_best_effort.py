"""The per-trial resume checkpoint must be best-effort.

It only lets a killed trial resume instead of restarting; it never
affects the trial's result (metrics come from evaluation, best weights
are saved separately). On some networked filesystems a freshly written
temp file can stay invisible to the subsequent rename even after
fsync+retry, raising ``OSError``. That must NOT propagate and kill the
whole run. Real bugs (non-OSError) must still surface.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.utils.training import _best_effort_resume_checkpoint


class _Mgr:
    def __init__(self, exc: Exception | None) -> None:
        self.exc = exc
        self.calls: list[dict] = []

    def save_training_checkpoint(self, **kwargs) -> None:
        self.calls.append(kwargs)
        if self.exc is not None:
            raise self.exc


def _logger() -> SimpleNamespace:
    msgs: list[str] = []
    return SimpleNamespace(warning=lambda *a, **k: msgs.append(a), _msgs=msgs)


def test_swallows_oserror_and_continues() -> None:
    mgr = _Mgr(FileNotFoundError(2, "No such file or directory"))
    log = _logger()

    _best_effort_resume_checkpoint(mgr, log, run_id="r", epoch=3)

    assert len(mgr.calls) == 1
    assert log._msgs


def test_non_oserror_still_propagates() -> None:
    mgr = _Mgr(ValueError("real bug"))

    with pytest.raises(ValueError, match="real bug"):
        _best_effort_resume_checkpoint(mgr, _logger(), run_id="r")


def test_passes_kwargs_through_on_success() -> None:
    mgr = _Mgr(None)

    _best_effort_resume_checkpoint(mgr, _logger(), run_id="r", epoch=7)

    assert mgr.calls == [{"run_id": "r", "epoch": 7}]
