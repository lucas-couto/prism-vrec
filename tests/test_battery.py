"""Battery runner: enumeration, manifest, idempotency, resume, smoke (Task I).

Enumeration reads the ACTUAL feature artifacts on disk (the visual_config
is the real embedding stem), so the tests lay down a tiny processed +
embeddings fixture.
"""

from __future__ import annotations

import json
from pathlib import Path

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
    "seeds": [1, 2],
    "seed": 1,
}
_NU, _NI = 12, 30


def _fixture(tmp_path: Path) -> tuple[str, str]:
    """Minimal on-disk processed + embeddings for enumeration."""
    proc = tmp_path / "processed" / "synthetic"
    emb = tmp_path / "embeddings" / "synthetic"
    proc.mkdir(parents=True)
    emb.mkdir(parents=True)
    (proc / "user2idx.json").write_text(json.dumps({str(i): i for i in range(_NU)}))
    (proc / "item2idx.json").write_text(json.dumps({str(i): i for i in range(_NI)}))
    np.save(emb / "resnet50.npy", np.zeros((_NI, 4), dtype=np.float32))
    return str(tmp_path / "processed"), str(tmp_path / "embeddings")


def _cells(tmp_path: Path) -> list[BatteryCell]:
    proc, emb = _fixture(tmp_path)
    return enumerate_cells(_CONFIG, processed_dir=proc, embeddings_dir=emb)


class TestEnumeration:
    def test_rules_applied(self, tmp_path) -> None:
        cells = _cells(tmp_path)
        by_rec: dict[str, list[BatteryCell]] = {}
        for c in cells:
            by_rec.setdefault(c.recommender, []).append(c)

        assert "avbpr" not in by_rec  # excluded
        assert len(by_rec["bpr"]) == 2  # once per (dataset, seed)
        assert all(c.visual_config == "none" for c in by_rec["bpr"])  # feature-blind
        assert {c.visual_config for c in by_rec["vbpr"]} == {"resnet50"}  # real stem
        assert "deepstyle" in by_rec  # not skipped

    def test_primary_seed_is_search_others_replay(self, tmp_path) -> None:
        roles = {c.seed: c.role for c in _cells(tmp_path)}
        assert roles[1] == "search"
        assert roles[2] == "replay"

    def test_finetuned_condition_adds_distinct_cells(self, tmp_path) -> None:
        proc, emb = _fixture(tmp_path)
        # A finetuned feature appears as a distinct '_finetuned' stem.
        np.save(Path(emb) / "synthetic" / "resnet50_finetuned.npy", np.zeros((_NI, 4), np.float32))

        stems = {
            c.visual_config
            for c in enumerate_cells(_CONFIG, processed_dir=proc, embeddings_dir=emb)
        }

        assert "resnet50" in stems  # frozen kept
        assert "resnet50_finetuned" in stems  # finetuned added (condition axis)

    def test_bpr_not_duplicated_for_finetuned(self, tmp_path) -> None:
        proc, emb = _fixture(tmp_path)
        np.save(Path(emb) / "synthetic" / "resnet50_finetuned.npy", np.zeros((_NI, 4), np.float32))

        cells = enumerate_cells(_CONFIG, processed_dir=proc, embeddings_dir=emb)
        bpr = [c for c in cells if c.recommender == "bpr"]

        # BPR is feature-blind: 'none' in frozen only, never a finetuned twin.
        assert len(bpr) == 2  # one per seed
        assert all(c.visual_config == "none" for c in bpr)


class TestManifest:
    def test_state_transitions_and_summary(self, tmp_path) -> None:
        m = BatteryManifest.load(tmp_path / "m.json")
        m.sync_cells(_cells(tmp_path))
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
        assert proj["estimated_remaining_hours"] == 2.0


def _synthetic_eval() -> tuple[Evaluator, object]:
    train = {u: {u % 4} for u in range(_NU)}
    test = {u: {10 + u} for u in range(_NU)}
    ev = Evaluator(train, test, _NI, k_values=[5, 10], tiebreak_seed=1)
    scores = np.random.default_rng(0).standard_normal((_NU, _NI)).astype(np.float32)

    class _M:
        def eval(self):
            pass

        def predict_batch(self, uids, items):
            return torch.tensor(scores[np.ix_(uids.cpu().numpy(), items.cpu().numpy())])

    return ev, _M()


def _make_light_execute(results_dir):
    ev, model = _synthetic_eval()
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
            n_items=_NI,
        )
        write_cell_artifact(records, meta, results_dir)
        return {}

    return _execute, called


class TestSmokeEndToEnd:
    def test_runner_produces_valid_artifacts_and_paired_matrix(self, tmp_path) -> None:
        proc, emb = _fixture(tmp_path)
        execute, called = _make_light_execute(tmp_path)
        manifest = run_battery(_CONFIG, tmp_path, execute, processed_dir=proc, embeddings_dir=emb)

        assert manifest.summary()["done"] == len(manifest.cells)
        assert len(called) == len(manifest.cells)

        # 3 systems for seed 1: bpr__none, vbpr__resnet50, deepstyle__resnet50.
        matrix = load_paired(tmp_path, "synthetic", seed=1, metric="ndcg", k=10)
        assert matrix.shape == (_NU, 3)

        from src.evaluation.persistence import artifact_paths

        one = next(c for c in manifest.cells)
        meta = manifest.cells[one]
        recs_path, _ = artifact_paths(
            tmp_path,
            CellMetadata(
                meta["dataset"], meta["visual_config"], meta["recommender"], meta["seed"], 8, "test"
            ),
        )
        _, df = read_cell_artifact(recs_path)
        assert not df.empty and not metrics_frame(df, [5, 10]).empty

    def test_idempotent_rerun_skips_completed(self, tmp_path) -> None:
        proc, emb = _fixture(tmp_path)
        execute, _ = _make_light_execute(tmp_path)
        run_battery(_CONFIG, tmp_path, execute, processed_dir=proc, embeddings_dir=emb)

        execute2, called2 = _make_light_execute(tmp_path)
        run_battery(_CONFIG, tmp_path, execute2, processed_dir=proc, embeddings_dir=emb)
        assert called2 == []


class TestResumeAndFailure:
    def test_failed_cell_isolated_then_retried(self, tmp_path) -> None:
        proc, emb = _fixture(tmp_path)
        cells = enumerate_cells(_CONFIG, processed_dir=proc, embeddings_dir=emb)
        good, _ = _make_light_execute(tmp_path)
        boom_key = cells[0].key()

        def _flaky(cell, config):
            if cell.key() == boom_key:
                raise RuntimeError("simulated interruption")
            return good(cell, config)

        m = run_battery(_CONFIG, tmp_path, _flaky, processed_dir=proc, embeddings_dir=emb)
        assert m.summary()["failed"] == 1
        assert m.summary()["done"] == len(cells) - 1

        good2, called2 = _make_light_execute(tmp_path)
        m2 = run_battery(
            _CONFIG, tmp_path, good2, retry_failed=True, processed_dir=proc, embeddings_dir=emb
        )
        assert m2.summary()["failed"] == 0
        assert called2 == [boom_key]

    def test_is_cell_complete_detects_written_artifact(self, tmp_path) -> None:
        cell = _cells(tmp_path)[0]
        assert is_cell_complete(cell, tmp_path) is False
        execute, _ = _make_light_execute(tmp_path)
        execute(cell, _CONFIG)
        assert is_cell_complete(cell, tmp_path) is True


class TestExecuteCellSeedIsolation:
    def test_checkpoint_seed_isolated_but_f_artifact_shared(self, tmp_path, monkeypatch) -> None:
        # Fix #1: training/checkpoint use a seed-suffixed results dir so a
        # replay never inherits the search seed's _best.pt; the F artifact
        # still lands in the shared base results dir.
        import src.battery.execute as ex
        import src.steps.train as train_mod

        proc, emb = _fixture(tmp_path)
        config = {
            "seed": 1,
            "device": "cpu",
            "paths": {
                "data_processed": proc,
                "embeddings": emb,
                "results": str(tmp_path / "res"),
            },
        }
        seen: dict = {}
        monkeypatch.setattr(
            train_mod,
            "_optimize_one_cell",
            lambda *a, **k: seen.__setitem__("train_results", k["config"]["paths"]["results"]),
        )
        monkeypatch.setattr(
            ex,
            "_evaluate_one_cell",
            lambda cell, cfg, *a, **k: seen.update(
                train_eval_results=cfg["paths"]["results"], f_out=k["f_out_dir"]
            ),
        )

        ex.execute_cell(BatteryCell("synthetic", "resnet50", "vbpr", 7, "search"), config)

        assert seen["train_results"] == str(tmp_path / "res") + "_seed7"
        assert seen["train_eval_results"] == str(tmp_path / "res") + "_seed7"
        assert seen["f_out"] == str(tmp_path / "res")
