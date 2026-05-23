"""Regression test for hyperparameter-search budget accounting.

``len(study.trials)`` also counts FAIL trials (infra crashes) and stale
RUNNING trials (process killed mid-trial). Counting those toward
``n_trials`` truncates or entirely skips the search for affected cells.
Only COMPLETE and PRUNED are legitimate search outcomes and may count
toward the budget; this test pins that contract.
"""

from __future__ import annotations

from types import SimpleNamespace

from src.steps.train import _legit_trial_count


def _study(*state_names: str) -> SimpleNamespace:
    trials = [SimpleNamespace(state=SimpleNamespace(name=s)) for s in state_names]
    return SimpleNamespace(trials=trials)


def test_counts_only_complete_and_pruned() -> None:
    study = _study("COMPLETE", "PRUNED", "FAIL", "RUNNING", "COMPLETE", "WAITING")

    assert _legit_trial_count(study) == 3


def test_all_fail_counts_zero() -> None:
    study = _study(*(["FAIL"] * 30))

    assert _legit_trial_count(study) == 0


def test_empty_study_is_zero() -> None:
    assert _legit_trial_count(_study()) == 0
