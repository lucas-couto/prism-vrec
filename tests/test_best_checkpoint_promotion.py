"""Audit D2/D3 guards on the ``_best.pt`` checkpoint.

D2 — promotion timing: intermediate validation wins go to a TRIAL-LOCAL
file; ``_best.pt`` is written once, at normal trial completion, so a
pruned trial can never leave a winner that Optuna's ``best_params``
(replay seeds, export_best) cannot see.

D3 — protocol fingerprint: every checkpoint is stamped with a hash of
the selection protocol; a disk best carrying a different (or missing)
fingerprint is not comparable and is overwritten, never silently kept.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.utils.training import (
    _promote_trial_best,
    _save_best_model,
    _save_trial_best,
    _trial_best_path,
    selection_protocol_fingerprint,
)


def _fingerprint(**over) -> str:
    base = {
        "dataset_name": "ds",
        "es_metric": "ndcg@10",
        "eval_sample_size": 1000,
        "eval_sample_seed": 42,
        "tiebreak_seed": 42,
        "k_values": [10],
    }
    base.update(over)
    return selection_protocol_fingerprint(**base)


def _save(root: Path, metric: float, fingerprint: str, weight: float = 1.0) -> None:
    _save_best_model(
        {"w": torch.tensor([weight])},
        {"latent_dim": 8},
        metric,
        3,
        4,
        "ds",
        "vbpr",
        "emb",
        fingerprint,
        results_root=root,
    )


def _best_path(root: Path) -> Path:
    return root / "models" / "ds" / "vbpr_emb_best.pt"


def _load_best(root: Path) -> dict:
    return torch.load(_best_path(root), map_location="cpu", weights_only=False)


class TestSelectionProtocolFingerprint:
    def test_is_deterministic(self) -> None:
        assert _fingerprint() == _fingerprint()

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("dataset_name", "other_ds"),
            ("es_metric", "recall@10"),
            ("eval_sample_size", None),
            ("eval_sample_seed", 7),
            ("tiebreak_seed", 7),
            ("k_values", [5]),
        ],
    )
    def test_changes_when_any_protocol_field_changes(self, field, value) -> None:
        assert _fingerprint(**{field: value}) != _fingerprint()


class TestSaveBestModelFingerprint:
    def test_keeps_existing_winner_under_same_protocol(self, tmp_path: Path) -> None:
        _save(tmp_path, 0.5, _fingerprint(), weight=1.0)

        _save(tmp_path, 0.4, _fingerprint(), weight=2.0)

        payload = _load_best(tmp_path)
        assert payload["best_metric"] == 0.5
        assert float(payload["model_state"]["w"]) == 1.0

    def test_overwrites_higher_stale_metric_on_fingerprint_mismatch(self, tmp_path: Path) -> None:
        _save(tmp_path, 0.9, _fingerprint(eval_sample_size=None), weight=1.0)

        _save(tmp_path, 0.4, _fingerprint(), weight=2.0)

        payload = _load_best(tmp_path)
        assert payload["best_metric"] == 0.4
        assert float(payload["model_state"]["w"]) == 2.0

    def test_legacy_checkpoint_without_fingerprint_is_overwritten(self, tmp_path: Path) -> None:
        path = _best_path(tmp_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state": {}, "hyperparams": {}, "best_metric": 0.9}, path)

        _save(tmp_path, 0.1, _fingerprint())

        assert _load_best(tmp_path)["best_metric"] == 0.1

    def test_stores_fingerprint_in_payload(self, tmp_path: Path) -> None:
        _save(tmp_path, 0.5, _fingerprint())

        assert _load_best(tmp_path)["selection_fingerprint"] == _fingerprint()


class TestTrialLocalPromotion:
    def _trial_path(self, root: Path) -> Path:
        return _trial_best_path(root, "ds", "vbpr", "emb", "abc123")

    def _write_trial(self, root: Path, metric: float) -> Path:
        trial_path = self._trial_path(root)
        _save_trial_best(
            trial_path,
            torch.nn.Linear(1, 1),
            {"latent_dim": 8},
            metric,
            3,
            4,
            _fingerprint(),
        )
        return trial_path

    def test_intermediate_save_never_touches_best_pt(self, tmp_path: Path) -> None:
        # A trial pruned after this point must leave no winner behind.
        self._write_trial(tmp_path, 0.7)

        assert not _best_path(tmp_path).exists()

    def test_promotion_writes_best_from_trial_payload(self, tmp_path: Path) -> None:
        trial_path = self._write_trial(tmp_path, 0.7)

        _promote_trial_best(
            trial_path,
            best_metric=0.7,
            dataset_name="ds",
            model_name="vbpr",
            embedding_name="emb",
            results_root=tmp_path,
            log=_SilentLog(),
        )

        payload = _load_best(tmp_path)
        assert payload["best_metric"] == 0.7
        assert payload["hyperparams"] == {"latent_dim": 8}
        assert payload["selection_fingerprint"] == _fingerprint()

    def test_promotion_respects_disk_best_comparison(self, tmp_path: Path) -> None:
        _save(tmp_path, 0.9, _fingerprint(), weight=1.0)
        trial_path = self._write_trial(tmp_path, 0.7)

        _promote_trial_best(
            trial_path,
            best_metric=0.7,
            dataset_name="ds",
            model_name="vbpr",
            embedding_name="emb",
            results_root=tmp_path,
            log=_SilentLog(),
        )

        assert _load_best(tmp_path)["best_metric"] == 0.9

    def test_promotion_without_trial_file_is_a_warned_noop(self, tmp_path: Path) -> None:
        log = _SilentLog()

        _promote_trial_best(
            self._trial_path(tmp_path),
            best_metric=0.5,
            dataset_name="ds",
            model_name="vbpr",
            embedding_name="emb",
            results_root=tmp_path,
            log=log,
        )

        assert not _best_path(tmp_path).exists()
        assert log.warnings

    def test_trial_files_are_invisible_to_best_pt_globs(self, tmp_path: Path) -> None:
        # export_best / evaluate discover checkpoints via ``*_best.pt``.
        trial_path = self._write_trial(tmp_path, 0.7)

        assert trial_path.exists()
        assert list(trial_path.parent.glob("*_best.pt")) == []


class _SilentLog:
    def __init__(self) -> None:
        self.warnings: list[tuple] = []

    def warning(self, *args, **kwargs) -> None:
        self.warnings.append(args)
