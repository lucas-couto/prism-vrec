"""Battery runner: enumeration, manifest, idempotency, resume, smoke (Task I)."""

from __future__ import annotations

import numpy as np
import torch

from src.battery.cells import BatteryCell, enumerate_cells
from src.battery.manifest import BatteryManifest, is_cell_complete, project_cost
from src.battery.runner import run_battery
from src.evaluation.derive_metrics import metrics_frame
from src.evaluation.paired_loader import load_paired
from src.evaluation.persistence import CellMetadata, read_cell_artifact, write_cell_artifact
from src.evaluation.protocol import Evaluator

_CONFIG = {
    "datasets": ["synthetic"],
    "recommenders_enabled": ["bpr", "vbpr", "deepstyle", "avbpr"],
    "extractors_enabled": ["resnet50"],
    "fusion_strategies_enabled": ["mean"],
    "seeds": [1, 2],
    "seed": 1,
}


class TestEnumeration:
    def test_rules_applied(self) -> None:
        cells = enumerate_cells(_CONFIG)
        by_rec: dict[str, list[BatteryCell]] = {}
        for c in cells:
            by_rec.setdefault(c.recommender, []).append(c)

        # AVBPR excluded.
        assert "avbpr" not in by_rec
        # BPR once per (dataset, seed), feature-blind.
        assert len(by_rec["bpr"]) == 2
        assert all(c.visual_config == "none" for c in by_rec["bpr"])
        # Visual model: one cell per visual config (resnet50 + mean) per seed.
        assert len(by_rec["vbpr"]) == 4
        assert {c.visual_config for c in by_rec["vbpr"]} == {"resnet50", "mean"}
        # DeepStyle present (not skipped on any dataset).
        assert "deepstyle" in by_rec

    def test_primary_seed_is_search_others_replay(self) -> None:
        cells = enumerate_cells(_CONFIG)
        roles = {c.seed: c.role for c in cells}
        assert roles[1] == "search"  # primary (first seed)
        assert roles[2] == "replay"


class TestManifest:
    def test_state_transitions_and_summary(self, tmp_path) -> None:
        m = BatteryManifest.load(tmp_path / "m.json")
        m.sync_cells(enumerate_cells(_CONFIG))
        total = len(m.cells)
        m.set_state(next(iter(m.cells)), "done", duration_seconds=1.0)
        assert m.summary()["done"] == 1
        assert m.summary()["pending"] == total - 1

    def test_cost_projection_uses_done_durations(self, tmp_path) -> None:
        m = BatteryManifest.load(tmp_path / "m.json")
        m.cells = {
            "a": {"state": "done", "role": "search", "duration_seconds": 3600.0},
            "b": {"state": "pending", "role": "search"},
            "c": {"state": "pending", "role": "search"},
        }
        proj = project_cost(m)
        assert proj["estimated_remaining_hours"] == 2.0  # 2 pending * 1h mean


def _synthetic_eval() -> tuple[Evaluator, object, int]:
    n_users, n_items = 12, 30
    train = {u: {u % 4} for u in range(n_users)}
    test = {u: {10 + u} for u in range(n_users)}
    ev = Evaluator(train, test, n_items, k_values=[5, 10], tiebreak_seed=1)
    scores = np.random.default_rng(0).standard_normal((n_users, n_items)).astype(np.float32)

    class _M:
        def eval(self):
            pass

        def predict_batch(self, uids, items):
            return torch.tensor(scores[np.ix_(uids.cpu().numpy(), items.cpu().numpy())])

    return ev, _M(), n_items


def _make_light_execute(results_dir):
    ev, model, n_items = _synthetic_eval()
    called: list[str] = []

    def _execute(cell: BatteryCell, config: dict) -> dict:
        called.append(cell.key())
        records = ev.per_user_records(model, device="cpu")
        meta = CellMetadata(
            dataset=cell.dataset,
            visual_config=cell.visual_config,
            recommender=cell.recommender,
            seed=cell.seed,
            d=8,
            split="test",
            n_users=len(records),
            n_items=n_items,
        )
        write_cell_artifact(records, meta, results_dir)
        return {}

    return _execute, called


class TestSmokeEndToEnd:
    def test_runner_produces_valid_artifacts_and_paired_matrix(self, tmp_path) -> None:
        execute, called = _make_light_execute(tmp_path)
        manifest = run_battery(_CONFIG, tmp_path, execute)

        # Every cell ran and is done.
        assert manifest.summary()["done"] == len(manifest.cells)
        assert len(called) == len(manifest.cells)

        # Artifacts read back, metrics derive, paired matrix assembles.
        cells = enumerate_cells(_CONFIG)
        seed1 = [c for c in cells if c.seed == 1]
        one = seed1[0]
        from src.evaluation.persistence import artifact_paths

        recs_path, _ = artifact_paths(
            tmp_path,
            CellMetadata(one.dataset, one.visual_config, one.recommender, one.seed, 8, "test"),
        )
        meta, df = read_cell_artifact(recs_path)
        assert not df.empty and "rank" in df.columns
        assert not metrics_frame(df, [5, 10]).empty

        matrix = load_paired(tmp_path, "synthetic", seed=1, metric="ndcg", k=10)
        # bpr__none + vbpr x{resnet50,mean} + deepstyle x{resnet50,mean} = 5 systems.
        assert matrix.shape[1] == 5

    def test_idempotent_rerun_skips_completed(self, tmp_path) -> None:
        execute, called = _make_light_execute(tmp_path)
        run_battery(_CONFIG, tmp_path, execute)
        first = len(called)

        execute2, called2 = _make_light_execute(tmp_path)
        run_battery(_CONFIG, tmp_path, execute2)
        # Second run: everything already done → nothing re-executed.
        assert first == len(enumerate_cells(_CONFIG))
        assert called2 == []


class TestResumeAndFailure:
    def test_failed_cell_isolated_then_retried(self, tmp_path) -> None:
        boom = {"cell": None}
        good, _ = _make_light_execute(tmp_path)

        def _flaky(cell, config):
            if cell.key() == boom["cell"]:
                raise RuntimeError("simulated interruption")
            return good(cell, config)

        cells = enumerate_cells(_CONFIG)
        boom["cell"] = cells[0].key()

        m = run_battery(_CONFIG, tmp_path, _flaky)
        assert m.summary()["failed"] == 1
        assert m.summary()["done"] == len(cells) - 1
        assert m.state_of(cells[0].key()) == "failed"

        # Resume with retry: the failed cell runs, the done ones are skipped.
        good2, called2 = _make_light_execute(tmp_path)
        m2 = run_battery(_CONFIG, tmp_path, good2, retry_failed=True)
        assert m2.summary()["failed"] == 0
        assert called2 == [cells[0].key()]

    def test_is_cell_complete_detects_written_artifact(self, tmp_path) -> None:
        cell = enumerate_cells(_CONFIG)[0]
        assert is_cell_complete(cell, tmp_path) is False
        execute, _ = _make_light_execute(tmp_path)
        execute(cell, _CONFIG)
        assert is_cell_complete(cell, tmp_path) is True
