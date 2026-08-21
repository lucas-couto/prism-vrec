"""Equivalence tests for the chunked fusion executor.

The streaming path exists purely to bound memory: it must produce the
*same* arrays as the in-memory strategies it replaces, or every result
in the paper changes silently. These tests pin that equivalence.

``concat`` is row-wise end to end, so equality is exact. The PCA
strategies fit on an identically-assembled matrix (equal components)
and only chunk the ``transform``, whose output rows each depend solely
on their own input row — equal to floating-point tolerance, since BLAS
is free to block a large matmul differently from a small one.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.fusions import streaming
from src.fusions.strategies import fuse_concat, fuse_pca, fuse_pca_per_model
from src.utils.atomic_io import atomic_np_memmap_save

RNG = np.random.default_rng(20260821)


@pytest.fixture
def sources(tmp_path):
    """Two native-dim sources with mismatched dims, as in the real pipeline."""
    a = RNG.normal(size=(97, 16)).astype(np.float32)
    b = RNG.normal(size=(97, 7)).astype(np.float32)
    paths = []
    for name, arr in (("a.npy", a), ("b.npy", b)):
        path = tmp_path / name
        np.save(path, arr)
        paths.append(str(path))
    return [a, b], paths


@pytest.fixture
def train_items():
    """A realistic fit set: most rows, unsorted gaps, never all of them."""
    return np.array(sorted(RNG.choice(97, size=70, replace=False)))


class TestConcatEquivalence:
    @pytest.mark.parametrize("normalize", [True, False])
    @pytest.mark.parametrize("chunk_rows", [1, 7, 96, 97, 500])
    def test_streamed_concat_is_bit_identical(self, sources, tmp_path, normalize, chunk_rows):
        arrays, paths = sources
        out = tmp_path / "fused.npy"
        expected = fuse_concat(arrays, normalize=normalize)

        shape = streaming.run_streamed(
            "concat", paths, out, normalize=normalize, chunk_rows=chunk_rows
        )

        actual = np.load(out)
        assert shape == expected.shape
        assert actual.dtype == expected.dtype
        np.testing.assert_array_equal(actual, expected)

    def test_chunk_size_never_changes_the_result(self, sources, tmp_path):
        _, paths = sources
        outs = []
        for chunk in (3, 40, 1000):
            out = tmp_path / f"fused_{chunk}.npy"
            streaming.run_streamed("concat", paths, out, normalize=True, chunk_rows=chunk)
            outs.append(np.load(out))

        np.testing.assert_array_equal(outs[0], outs[1])
        np.testing.assert_array_equal(outs[1], outs[2])


class TestPcaEquivalence:
    @pytest.mark.parametrize("normalize", [True, False])
    def test_streamed_pca_matches_in_memory(self, sources, tmp_path, train_items, normalize):
        arrays, paths = sources
        out = tmp_path / "fused.npy"
        expected = fuse_pca(
            arrays,
            normalize=normalize,
            n_components=5,
            random_state=42,
            train_items=train_items,
        )

        streaming.run_streamed(
            "pca",
            paths,
            out,
            normalize=normalize,
            train_items=train_items,
            n_components=5,
            random_state=42,
            chunk_rows=11,
        )

        actual = np.load(out)
        assert actual.shape == expected.shape
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)

    def test_streamed_pca_per_model_matches_in_memory(self, sources, tmp_path, train_items):
        arrays, paths = sources
        out = tmp_path / "fused.npy"
        expected = fuse_pca_per_model(
            arrays,
            normalize=True,
            n_components=4,
            random_state=42,
            train_items=train_items,
        )

        streaming.run_streamed(
            "pca_per_model",
            paths,
            out,
            normalize=True,
            train_items=train_items,
            n_components=4,
            random_state=42,
            chunk_rows=13,
        )

        actual = np.load(out)
        assert actual.shape == expected.shape
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)

    def test_n_components_is_clamped_like_the_in_memory_path(self, sources, tmp_path, train_items):
        """Asking for more components than the fit matrix can supply."""
        arrays, paths = sources
        out = tmp_path / "fused.npy"
        expected = fuse_pca(
            arrays, normalize=True, n_components=999, random_state=42, train_items=train_items
        )

        shape = streaming.run_streamed(
            "pca",
            paths,
            out,
            normalize=True,
            train_items=train_items,
            n_components=999,
            random_state=42,
        )

        assert shape == expected.shape

    def test_omitted_k_falls_back_to_each_strategy_own_default(
        self, sources, tmp_path, train_items
    ):
        """The two PCA strategies default to different k (128 vs 64).

        A single shared fallback would silently change the fused
        dimensionality of whichever strategy it did not match.
        """
        arrays, paths = sources
        expected = fuse_pca_per_model(arrays, normalize=True, train_items=train_items)
        out = tmp_path / "fused.npy"

        shape = streaming.run_streamed(
            "pca_per_model", paths, out, normalize=True, train_items=train_items
        )

        assert shape == expected.shape
        np.testing.assert_allclose(np.load(out), expected, rtol=1e-5, atol=1e-5)

    def test_unknown_kwargs_are_discarded_not_crashed_on(self, sources, tmp_path, train_items):
        """Mirrors the in-memory plugin contract: warn, do not raise."""
        _, paths = sources

        shape = streaming.run_streamed(
            "pca",
            paths,
            tmp_path / "fused.npy",
            normalize=True,
            train_items=train_items,
            n_component=5,  # typo'd: consumed by nothing
        )

        # k falls back to pca's default 128, clamped to the 23 available
        # dims — the typo contributed nothing, as intended.
        assert shape[1] == 23

    def test_pca_without_train_items_is_refused(self, sources, tmp_path):
        """The transductive fit is a leak; streaming offers no opt-in."""
        _, paths = sources

        with pytest.raises(ValueError, match="train_items is required"):
            streaming.run_streamed(
                "pca", paths, tmp_path / "x.npy", normalize=True, train_items=None
            )


class TestPcaAlign:
    def test_stream_pca_align_matches_the_in_memory_helper(self, sources, tmp_path, train_items):
        from src.fusions.strategies import pca_align

        arrays, paths = sources
        [expected] = pca_align([arrays[0]], 5, train_items=train_items, random_state=42)
        out = tmp_path / "aligned.npy"

        shape = streaming.stream_pca_align(paths[0], out, 5, train_items, random_state=42)

        actual = np.load(out)
        assert shape == expected.shape
        assert actual.dtype == np.float32
        np.testing.assert_allclose(actual, expected.astype(np.float32), rtol=1e-5, atol=1e-5)


class TestRouting:
    def test_is_streamable_covers_exactly_the_row_wise_strategies(self):
        assert streaming.is_streamable("concat")
        assert streaming.is_streamable("pca")
        assert streaming.is_streamable("pca_per_model")
        # Element-wise strategies run online (or in memory), never here.
        assert not streaming.is_streamable("mean")
        assert not streaming.is_streamable("adaptive_gated")
        assert not streaming.is_streamable("some_plugin_strategy")

    def test_unstreamable_name_is_refused(self, sources, tmp_path):
        _, paths = sources

        with pytest.raises(ValueError, match="no streaming implementation"):
            streaming.run_streamed("mean", paths, tmp_path / "x.npy", normalize=True)

    def test_mismatched_row_counts_are_rejected(self, tmp_path):
        paths = []
        for name, rows in (("a.npy", 10), ("b.npy", 11)):
            path = tmp_path / name
            np.save(path, np.zeros((rows, 4), dtype=np.float32))
            paths.append(str(path))

        with pytest.raises(ValueError, match="same number of rows"):
            streaming.run_streamed("concat", paths, tmp_path / "x.npy", normalize=True)


class TestAtomicMemmapSave:
    def test_produces_a_normal_npy(self, tmp_path):
        arr = RNG.normal(size=(20, 6)).astype(np.float32)
        streamed = tmp_path / "streamed.npy"

        atomic_np_memmap_save(
            streamed,
            dtype=arr.dtype,
            shape=arr.shape,
            fill=lambda out: out.__setitem__(slice(None), arr),
        )

        assert streamed.read_bytes()[:6] == b"\x93NUMPY"
        loaded = np.load(streamed)
        assert (loaded.dtype, loaded.shape) == (arr.dtype, arr.shape)
        np.testing.assert_array_equal(loaded, arr)

    def test_failure_leaves_no_partial_file(self, tmp_path):
        dest = tmp_path / "out.npy"

        def _boom(out):
            out[0] = 1.0
            raise RuntimeError("interrupted mid-fill")

        with pytest.raises(RuntimeError, match="interrupted mid-fill"):
            atomic_np_memmap_save(dest, dtype=np.float32, shape=(4, 3), fill=_boom)

        assert not dest.exists()
        assert list(tmp_path.glob("*.tmp")) == []

    def test_previous_file_survives_a_failed_rewrite(self, tmp_path):
        dest = tmp_path / "out.npy"
        original = np.ones((4, 3), dtype=np.float32)
        np.save(dest, original)

        def _boom(out):
            raise RuntimeError("nope")

        with pytest.raises(RuntimeError):
            atomic_np_memmap_save(dest, dtype=np.float32, shape=(9, 9), fill=_boom)

        np.testing.assert_array_equal(np.load(dest), original)
