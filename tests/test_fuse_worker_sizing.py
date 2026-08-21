"""Tests for the memory-aware sizing of the fusion process pool.

Sizing the pool at ``os.cpu_count()`` regardless of what each worker
holds is what let the fusion step exhaust host memory.  The estimate is
now regime-aware: sidecar tasks cost nothing, streamed tasks are bounded
by the chunk size (plus the PCA fit matrix), and anything still running
in memory is charged for the whole catalogue.
"""

from __future__ import annotations

import numpy as np

from src.steps import fuse as fuse_mod
from src.utils import memory as memory_mod

GB = 1024**3


def _npy(tmp_path, name: str, rows: int, dim: int) -> str:
    path = tmp_path / name
    np.save(path, np.zeros((rows, dim), dtype=np.float32))
    return str(path)


def _task(paths, *, strategy="concat", train_items=None, sidecar=None) -> dict:
    return {
        "strategy_name": strategy,
        "emb_list_paths": paths,
        "train_items": train_items,
        "sidecar_payload": sidecar,
    }


class TestTaskPeakBytes:
    def test_sidecar_task_costs_nothing(self, tmp_path):
        paths = [_npy(tmp_path, "a.npy", 1000, 256)]

        assert fuse_mod._task_peak_bytes(_task(paths, sidecar={"components": []})) == 0

    def test_streamed_concat_is_bounded_by_the_chunk_not_the_catalogue(self, tmp_path):
        """The whole point: peak must not grow with the number of rows."""
        small = [_npy(tmp_path, "small.npy", fuse_mod.CHUNK_ROWS * 2, 64)]
        large = [_npy(tmp_path, "large.npy", fuse_mod.CHUNK_ROWS * 200, 64)]

        peak_small = fuse_mod._task_peak_bytes(_task(small))
        peak_large = fuse_mod._task_peak_bytes(_task(large))

        assert peak_small == peak_large

    def test_in_memory_strategy_is_charged_for_the_whole_catalogue(self, tmp_path):
        paths = [_npy(tmp_path, "a.npy", 100_000, 64)]

        peak = fuse_mod._task_peak_bytes(_task(paths, strategy="some_plugin"))

        raw = 100_000 * 64 * 4
        assert peak >= raw * fuse_mod._FUSION_PEAK_FACTOR

    def test_pca_task_is_charged_for_its_fit_matrix(self, tmp_path):
        paths = [_npy(tmp_path, "a.npy", 100_000, 64)]
        train_items = list(range(80_000))

        with_fit = fuse_mod._task_peak_bytes(_task(paths, strategy="pca", train_items=train_items))
        without_fit = fuse_mod._task_peak_bytes(_task(paths, strategy="concat"))

        assert with_fit - without_fit >= 80_000 * 64 * 4

    def test_missing_source_does_not_zero_the_estimate(self, tmp_path):
        present = _npy(tmp_path, "a.npy", 500, 128)
        paths = [present, str(tmp_path / "gone.npy")]

        assert fuse_mod._task_peak_bytes(_task(paths, strategy="some_plugin")) > 0

    def test_unreadable_sources_only_yield_zero(self, tmp_path):
        paths = [str(tmp_path / "gone.npy")]

        assert fuse_mod._task_peak_bytes(_task(paths)) == 0


class TestPlanFusionWorkers:
    def test_streamed_tasks_keep_every_core(self, tmp_path, monkeypatch):
        """After streaming, a catalogue-sized concat no longer caps the pool."""
        monkeypatch.setattr(fuse_mod.os, "cpu_count", lambda: 16)
        monkeypatch.setattr(memory_mod, "memory_budget_bytes", lambda: 24 * GB)
        # ~4 GB of sources — the shape that used to force 1 worker.
        paths = [_npy(tmp_path, "a.npy", 2 * 1024 * 1024, 512)]
        pending = [_task(paths) for _ in range(12)]

        assert fuse_mod._plan_fusion_workers(pending) == 12

    def test_in_memory_tasks_are_still_capped(self, tmp_path, monkeypatch):
        monkeypatch.setattr(fuse_mod.os, "cpu_count", lambda: 16)
        monkeypatch.setattr(memory_mod, "memory_budget_bytes", lambda: 24 * GB)
        paths = [_npy(tmp_path, "a.npy", 2 * 1024 * 1024, 512)]
        pending = [_task(paths, strategy="some_plugin") for _ in range(12)]

        assert fuse_mod._plan_fusion_workers(pending) == 1

    def test_the_heaviest_task_sets_the_pool_size(self, tmp_path, monkeypatch):
        """Homogeneous slots: one fat task must not be sized by the light ones."""
        monkeypatch.setattr(fuse_mod.os, "cpu_count", lambda: 16)
        monkeypatch.setattr(memory_mod, "memory_budget_bytes", lambda: 24 * GB)
        light = [_npy(tmp_path, "light.npy", 100, 32)]
        heavy = [_npy(tmp_path, "heavy.npy", 2 * 1024 * 1024, 512)]
        pending = [_task(light) for _ in range(11)]
        pending.append(_task(heavy, strategy="some_plugin"))

        assert fuse_mod._plan_fusion_workers(pending) == 1

    def test_never_more_workers_than_pending_tasks(self, tmp_path, monkeypatch):
        monkeypatch.setattr(fuse_mod.os, "cpu_count", lambda: 16)
        monkeypatch.setattr(memory_mod, "memory_budget_bytes", lambda: 64 * GB)
        paths = [_npy(tmp_path, "a.npy", 100, 32)]
        pending = [_task(paths) for _ in range(3)]

        assert fuse_mod._plan_fusion_workers(pending) == 3

    def test_sidecar_only_batch_is_not_memory_capped(self, monkeypatch):
        monkeypatch.setattr(fuse_mod.os, "cpu_count", lambda: 4)
        monkeypatch.setattr(memory_mod, "memory_budget_bytes", lambda: 5 * GB)
        pending = [_task([], sidecar={"components": []}) for _ in range(4)]

        assert fuse_mod._plan_fusion_workers(pending) == 4

    def test_always_at_least_one_worker(self, tmp_path, monkeypatch):
        monkeypatch.setattr(fuse_mod.os, "cpu_count", lambda: 16)
        monkeypatch.setattr(memory_mod, "memory_budget_bytes", lambda: 5 * GB)
        paths = [_npy(tmp_path, "a.npy", 4 * 1024 * 1024, 512)]
        pending = [_task(paths, strategy="some_plugin") for _ in range(12)]

        assert fuse_mod._plan_fusion_workers(pending) == 1
