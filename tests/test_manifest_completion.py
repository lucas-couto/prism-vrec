"""Battery / K-fold completion depends on validated artifacts and provenance (E07, Q14).

The manifest is published atomically; a ``done`` entry stands only while
its recorded artifact binding validates on disk; a fold artifact counts
as complete only for the requested k, partition seed, profile rule and
split content.  Incompleteness reaches the caller (main's
``_require_complete``) through the returned manifest.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

import src.battery.manifest as manifest_mod
from src.battery.cells import BatteryCell
from src.battery.runner import run_battery
from src.evaluation.persistence import CellMetadata, artifact_paths, write_cell_artifact
from src.folds.runner import _cell_done, run_folds
from tests.test_folds_runner import _config as _folds_config
from tests.test_folds_runner import _write_dataset as _write_folds_dataset

BatteryManifest = manifest_mod.BatteryManifest


def _records(ranks: list[int]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_idx": list(range(len(ranks))),
            "rank": ranks,
            "n_candidates": [10] * len(ranks),
            "tie_block_size": [1] * len(ranks),
            "top_items": [[1]] * len(ranks),
        }
    )


def _meta(cell: BatteryCell, **extra) -> CellMetadata:
    return CellMetadata(
        dataset=cell.dataset,
        visual_config=cell.visual_config,
        recommender=cell.recommender,
        seed=cell.seed,
        d=4,
        split="test",
        **extra,
    )


def _publish(cell: BatteryCell, results: Path, ranks: list[int] = (1, 2, 3)) -> Path:
    return write_cell_artifact(_records(list(ranks)), _meta(cell), results)


class TestManifestFile:
    def test_save_is_atomic_and_a_failed_rewrite_keeps_the_previous_file(
        self, tmp_path, monkeypatch
    ) -> None:
        manifest = BatteryManifest.load(tmp_path / "battery" / "manifest.json")
        manifest.cells = {"a": {"state": "done"}}
        manifest.save()
        before = manifest.path.read_bytes()

        def _boom(tmp):
            Path(tmp).write_text("{ partial")
            raise OSError("disk full")

        monkeypatch.setattr(manifest_mod, "atomic_write", lambda fn, path: _boom(str(path) + ".x"))
        manifest.cells["b"] = {"state": "pending"}
        with pytest.raises(OSError):
            manifest.save()

        assert manifest.path.read_bytes() == before
        assert BatteryManifest.load(manifest.path).cells == {"a": {"state": "done"}}

    def test_unreadable_manifest_is_refused_not_guessed(self, tmp_path) -> None:
        path = tmp_path / "manifest.json"
        path.write_text("{ torn")
        with pytest.raises(manifest_mod.ManifestError, match="refusing to guess"):
            BatteryManifest.load(path)


def _cells(tmp_path: Path) -> tuple[list[BatteryCell], str, str]:
    from tests.test_battery import _CONFIG, _fixture

    proc, emb = _fixture(tmp_path)
    from src.battery.cells import enumerate_cells

    return enumerate_cells(_CONFIG, processed_dir=proc, embeddings_dir=emb), proc, emb


def _executor(results: Path, *, writes: bool = True):
    called: list[str] = []

    def _execute(cell: BatteryCell, config: dict) -> dict:
        called.append(cell.key())
        if writes:
            _publish(cell, results)
        return {"seed": cell.seed}

    return _execute, called


class TestBatteryDoneEntries:
    def _first_run(self, tmp_path):
        from tests.test_battery import _CONFIG

        cells, proc, emb = _cells(tmp_path)
        execute, called = _executor(tmp_path)
        manifest = run_battery(_CONFIG, tmp_path, execute, processed_dir=proc, embeddings_dir=emb)
        assert manifest.summary()["done"] == len(cells) == len(called)
        assert all("artifact" in e for e in manifest.cells.values())
        return cells, proc, emb, manifest

    def _rerun(self, tmp_path, proc, emb, *, writes=True):
        from tests.test_battery import _CONFIG

        execute, called = _executor(tmp_path, writes=writes)
        manifest = run_battery(_CONFIG, tmp_path, execute, processed_dir=proc, embeddings_dir=emb)
        return manifest, called

    def test_valid_done_entries_are_skipped(self, tmp_path) -> None:
        _, proc, emb, _ = self._first_run(tmp_path)
        manifest, called = self._rerun(tmp_path, proc, emb)
        assert called == [] and manifest.summary()["failed"] == 0

    def test_replaced_artifact_invalidates_the_done_entry(self, tmp_path) -> None:
        cells, proc, emb, _ = self._first_run(tmp_path)
        _publish(cells[0], tmp_path, ranks=[9, 9, 9])

        manifest, called = self._rerun(tmp_path, proc, emb)

        assert called == [cells[0].key()]
        assert "done entry invalid" in manifest.cells[cells[0].key()].get("note", "")

    def test_missing_artifact_under_a_done_entry_reruns_the_cell(self, tmp_path) -> None:
        cells, proc, emb, _ = self._first_run(tmp_path)
        records, _ = artifact_paths(tmp_path, _meta(cells[1]))
        records.unlink()

        _, called = self._rerun(tmp_path, proc, emb)

        assert called == [cells[1].key()]

    def test_legacy_done_entry_without_binding_reruns_the_cell(self, tmp_path) -> None:
        cells, proc, emb, manifest = self._first_run(tmp_path)
        entry = manifest.cells[cells[2].key()]
        del entry["artifact"]
        manifest.save()

        _, called = self._rerun(tmp_path, proc, emb)

        assert called == [cells[2].key()]

    def test_artifact_of_another_cell_is_not_accepted(self, tmp_path) -> None:
        cells, proc, emb, manifest = self._first_run(tmp_path)
        _, meta_path = artifact_paths(tmp_path, _meta(cells[0]))
        meta = json.loads(meta_path.read_text())
        meta["seed"] = 999
        meta_path.write_text(json.dumps(meta))

        valid, reason = manifest_mod.done_entry_valid(
            manifest.cells[cells[0].key()], cells[0], tmp_path
        )

        assert not valid and "provenance" in reason

    def test_executor_without_a_validated_artifact_fails_the_cell(self, tmp_path) -> None:
        from tests.test_battery import _CONFIG

        cells, proc, emb = _cells(tmp_path)
        execute, _ = _executor(tmp_path, writes=False)

        manifest = run_battery(_CONFIG, tmp_path, execute, processed_dir=proc, embeddings_dir=emb)

        assert manifest.summary()["failed"] == len(cells)
        assert all("validated per-user artifact" in e["error"] for e in manifest.cells.values())

    def test_present_artifact_without_entry_is_adopted_with_its_binding(self, tmp_path) -> None:
        from tests.test_battery import _CONFIG

        cells, proc, emb = _cells(tmp_path)
        _publish(cells[0], tmp_path)
        execute, called = _executor(tmp_path)

        manifest = run_battery(_CONFIG, tmp_path, execute, processed_dir=proc, embeddings_dir=emb)

        assert cells[0].key() not in called
        entry = manifest.cells[cells[0].key()]
        assert entry["note"] == "artifact already present" and entry["artifact"]["row_count"] == 3


class TestFoldCompletion:
    def _cell(self) -> BatteryCell:
        return BatteryCell(
            dataset="ds", visual_config="resnet50", recommender="vbpr", seed=7, role="search"
        )

    def _write(self, results: Path, *, k: int = 2, seed: int = 7, plan: str = "plan-A") -> None:
        cell = self._cell()
        meta = _meta(
            cell,
            fold={"k": k, "seeds": [seed + i for i in range(k)], "n_users_per_fold": [2, 1]},
            config_hash=plan,
        )
        write_cell_artifact(_records([1, 2, 3]), meta, results)

    def test_matching_plan_is_done(self, tmp_path) -> None:
        self._write(tmp_path)
        assert _cell_done(self._cell(), 7, tmp_path, k=2, plan_digest="plan-A")

    @pytest.mark.parametrize(
        ("k", "seed", "plan"),
        [(3, 7, "plan-A"), (2, 8, "plan-A"), (2, 7, "plan-B")],
        ids=["changed_k", "changed_partition_seed", "changed_plan"],
    )
    def test_changed_k_seed_or_plan_is_not_done(self, tmp_path, k, seed, plan) -> None:
        self._write(tmp_path)
        assert not _cell_done(self._cell(), seed, tmp_path, k=k, plan_digest=plan)

    def test_partial_or_leave_one_out_artifact_is_not_done(self, tmp_path) -> None:
        cell = self._cell()
        write_cell_artifact(_records([1]), _meta(cell), tmp_path)
        assert not _cell_done(cell, 7, tmp_path, k=2, plan_digest="plan-A")

    def test_torn_artifact_is_not_done(self, tmp_path) -> None:
        self._write(tmp_path)
        records, _ = artifact_paths(tmp_path, _meta(self._cell()))
        records.write_bytes(b"not the published payload")
        assert not _cell_done(self._cell(), 7, tmp_path, k=2, plan_digest="plan-A")

    def test_run_folds_reruns_cells_when_k_changes(self, tmp_path) -> None:
        processed, embeddings = _write_folds_dataset(tmp_path)
        cfg = _folds_config(tmp_path, processed, embeddings)
        results = Path(cfg["paths"]["results"])
        first = run_folds(cfg, results)
        assert first.summary()["done"] == 2
        calls: list[str] = []

        def _stub(cell, config, plan, frames, *, results_dir, device):
            calls.append(cell.key())
            return {}

        same = run_folds(cfg, results, execute=_stub)
        assert calls == [] and same.summary()["done"] == 2

        cfg["folds"]["k"] = 3
        changed = run_folds(cfg, results, execute=_stub)

        assert sorted(calls) == sorted(changed.cells)
