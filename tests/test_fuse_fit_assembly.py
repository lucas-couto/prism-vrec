"""The PCA fit matrix is assembled in row blocks and a lost worker is named.

Before 3.0.0rc1, ``_assemble_fit_matrix`` fancy-indexed a whole source
with the training index and normalised the result, holding two extra
source-sized arrays on top of the fit matrix (about 8 GB for tradesy);
two such workers were OOM-killed inside a 16 GB container while the
planner had charged 5.6 GB each.  ``BrokenProcessPool`` then surfaced
with no cause.
"""

from __future__ import annotations

import os
import tracemalloc

import numpy as np
import pytest

from src.fusions import streaming
from src.fusions.strategies import l2_normalize
from src.steps import fuse as fuse_mod

N_ROWS, DIM, CHUNK = 20_000, 256, 1024


def _memmap(tmp_path, name: str, seed: int) -> np.ndarray:
    path = tmp_path / name
    np.save(path, np.random.default_rng(seed).standard_normal((N_ROWS, DIM)).astype(np.float32))
    return np.load(path, mmap_mode="r")


@pytest.fixture
def sources(tmp_path):
    return [_memmap(tmp_path, "a.npy", 1), _memmap(tmp_path, "b.npy", 2)]


@pytest.fixture
def fit_idx():
    return np.sort(np.random.default_rng(3).choice(N_ROWS, size=16_000, replace=False))


class TestFitMatrixAssembly:
    @pytest.mark.parametrize("normalize", [True, False])
    def test_matches_the_in_memory_concatenation(self, sources, fit_idx, normalize):
        expected = np.concatenate(
            [
                l2_normalize(np.asarray(s)[fit_idx]) if normalize else np.asarray(s)[fit_idx]
                for s in sources
            ],
            axis=1,
        )

        got = streaming._assemble_fit_matrix(
            sources, fit_idx, normalize, 2 * DIM, np.float32, CHUNK
        )

        np.testing.assert_array_equal(got, expected)

    def test_peak_is_bounded_by_the_chunk_not_by_the_source(self, sources, fit_idx):
        fit_bytes = fit_idx.shape[0] * 2 * DIM * 4
        chunk_bytes = CHUNK * DIM * 4

        tracemalloc.start()
        try:
            streaming._assemble_fit_matrix(sources, fit_idx, True, 2 * DIM, np.float32, CHUNK)
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()

        # The old path held a whole gathered source plus its normalised
        # copy (2 * 16_000 * DIM * 4 = 32.8 MB) on top of the fit matrix.
        assert peak - fit_bytes < 4 * chunk_bytes + 1024**2

    def test_per_model_block_gather_matches_and_is_bounded(self, sources, fit_idx):
        src = sources[0]
        out = np.empty((fit_idx.shape[0], DIM), dtype=np.float32)

        tracemalloc.start()
        try:
            streaming._gather_rows(src, fit_idx, True, CHUNK, out=out)
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()

        np.testing.assert_array_equal(out, l2_normalize(np.asarray(src)[fit_idx]))
        assert peak < 4 * CHUNK * DIM * 4 + 1024**2


# Module-level so a spawned/forked pool worker can unpickle it by import path.
def _vanish(**_kwargs) -> str:
    os._exit(3)


def _ok(**kwargs) -> str:
    return f"done {kwargs['strategy_name']}"


class TestLostWorker:
    def test_lost_worker_is_reported_with_its_cause(self):
        pending = [{"strategy_name": "concat"}]

        with pytest.raises(fuse_mod.FusionWorkerLostError) as info:
            fuse_mod._run_fusion_pool(pending, 1, worker=_vanish)

        message = str(info.value)
        assert "0/1 fusions completed on 1 worker(s)" in message
        assert "oom_kill" in message

    def test_healthy_pool_reports_completed_count(self):
        pending = [{"strategy_name": "concat"}, {"strategy_name": "pca"}]

        assert fuse_mod._run_fusion_pool(pending, 1, worker=_ok) == 2

    def test_cgroup_counter_is_optional(self, monkeypatch):
        monkeypatch.setattr(
            fuse_mod.Path, "read_text", lambda *a, **k: (_ for _ in ()).throw(OSError())
        )

        assert fuse_mod._cgroup_oom_kills() is None
