"""SDD M01: bounded feature-source adapters (C01) and the lazy loader form.

Pins the contract of :mod:`src.data.feature_source`: rows come back in
caller order (repeated and unordered ids included), an empty request
keeps the trailing shape and dtype, invalid ids fail, fp16 component
artifacts stay fp16, sidecar sources are combined per gathered row
only, and memory-map handles are owned by the process that reads.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import pickle
from pathlib import Path

import numpy as np
import pytest

from src.data.feature_source import (
    ArrayFeatureSource,
    ConcatFeatureSource,
    FeatureSource,
    NpyFeatureSource,
    StackedFeatureSource,
    is_feature_source,
    source_bytes,
)
from src.fusions.online import RaggedSources, load_embedding

N_ITEMS = 11


def _pooled(n: int = N_ITEMS, d: int = 5, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal((n, d)).astype(np.float32)


def _components(n: int = N_ITEMS, m: int = 3, d: int = 4) -> np.ndarray:
    return np.random.default_rng(1).standard_normal((n, m, d)).astype(np.float16)


@pytest.fixture()
def pooled_npy(tmp_path: Path) -> tuple[Path, np.ndarray]:
    arr = _pooled()
    path = tmp_path / "resnet50.npy"
    np.save(path, arr)
    return path, arr


@pytest.fixture()
def comp_npy(tmp_path: Path) -> tuple[Path, np.ndarray]:
    arr = _components()
    path = tmp_path / "resnet50_comp.npy"
    np.save(path, arr)
    return path, arr


class TestNpySource:
    def test_arbitrary_repeated_and_unordered_ids_keep_caller_order(self, pooled_npy) -> None:
        path, arr = pooled_npy
        ids = np.array([7, 2, 2, 10, 0, 7], dtype=np.int64)

        rows = NpyFeatureSource(path).read_rows(ids)

        np.testing.assert_array_equal(rows, arr[ids])
        assert rows.dtype == np.float32

    def test_empty_request_keeps_trailing_shape_and_dtype(self, comp_npy) -> None:
        path, _ = comp_npy

        rows = NpyFeatureSource(path).read_rows(np.array([], dtype=np.int64))

        assert rows.shape == (0, 3, 4)
        assert rows.dtype == np.float16

    def test_component_fp16_is_preserved(self, comp_npy) -> None:
        path, arr = comp_npy
        source = NpyFeatureSource(path)

        rows = source.read_rows(np.array([3, 1]))

        assert source.dtype == np.float16 and rows.dtype == np.float16
        np.testing.assert_array_equal(rows, arr[[3, 1]])

    @pytest.mark.parametrize("bad", [[N_ITEMS], [-1], [0, 99]])
    def test_out_of_bounds_ids_fail(self, pooled_npy, bad: list[int]) -> None:
        source = NpyFeatureSource(pooled_npy[0])

        with pytest.raises(IndexError, match="out of range"):
            source.read_rows(np.array(bad))

    def test_non_integer_and_non_1d_ids_fail(self, pooled_npy) -> None:
        source = NpyFeatureSource(pooled_npy[0])

        with pytest.raises(TypeError, match="integers"):
            source.read_rows(np.array([0.5, 1.0]))
        with pytest.raises(ValueError, match="1-D"):
            source.read_rows(np.array([[0, 1]]))

    def test_rows_are_host_owned_copies_not_memmap_views(self, pooled_npy) -> None:
        path, arr = pooled_npy
        source = NpyFeatureSource(path)

        rows = source.read_rows(np.array([1, 2]))
        rows[0, 0] = 123.0  # must not touch the file

        assert not isinstance(rows, np.memmap)
        assert rows.flags["C_CONTIGUOUS"] and rows.flags["WRITEABLE"]
        assert float(np.load(path)[1, 0]) == float(arr[1, 0])

    def test_header_only_at_construction_handle_opened_on_first_read(self, pooled_npy) -> None:
        source = NpyFeatureSource(pooled_npy[0])

        assert source.shape == (N_ITEMS, 5) and source.dtype == np.float32
        assert not source.is_open
        source.read_rows(np.array([0]))
        assert source.is_open
        source.close()
        assert not source.is_open
        # Usable after close: the next read reopens.
        np.testing.assert_array_equal(source.read_rows(np.array([4])), pooled_npy[1][[4]])

    def test_missing_file_fails_at_construction(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            NpyFeatureSource(tmp_path / "absent.npy")

    def test_one_dimensional_file_is_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "vector.npy"
        np.save(path, np.zeros(4, dtype=np.float32))

        with pytest.raises(ValueError, match="at least 2-D"):
            NpyFeatureSource(path)

    def test_pickle_drops_the_handle_and_reads_after_unpickling(self, pooled_npy) -> None:
        path, arr = pooled_npy
        source = NpyFeatureSource(path)
        source.read_rows(np.array([0]))

        clone = pickle.loads(pickle.dumps(source))

        assert source.is_open and not clone.is_open
        np.testing.assert_array_equal(clone.read_rows(np.array([9, 9])), arr[[9, 9]])

    def test_spawned_process_opens_its_own_handle(self, pooled_npy) -> None:
        path, arr = pooled_npy
        source = NpyFeatureSource(path)
        source.read_rows(np.array([0]))  # parent owns a handle

        ctx = mp.get_context("spawn")
        with ctx.Pool(1) as pool:
            rows, child_pid, open_on_arrival = pool.apply(_read_in_child, (source, [5, 1]))

        assert child_pid != os.getpid()
        assert open_on_arrival is False
        np.testing.assert_array_equal(rows, arr[[5, 1]])
        assert source.is_open  # the parent's handle is untouched


def _read_in_child(source: NpyFeatureSource, ids: list[int]):
    """Spawn target: report whether the handle arrived open, then read."""
    open_on_arrival = source.is_open
    rows = source.read_rows(np.array(ids))
    return rows, os.getpid(), open_on_arrival


class TestMultiSource:
    def test_concat_gathers_ragged_sources_per_row(self, tmp_path: Path) -> None:
        a, b = _pooled(d=6, seed=2), _pooled(d=4, seed=3)
        np.save(tmp_path / "a.npy", a)
        np.save(tmp_path / "b.npy", b)
        source = ConcatFeatureSource(
            [NpyFeatureSource(tmp_path / "a.npy"), NpyFeatureSource(tmp_path / "b.npy")],
            strategy="mean",
            aligned_dim=5,
            normalize=False,
            fusion_kwargs={"weights": [0.3, 0.7]},
        )
        ids = np.array([4, 4, 0, 10])

        rows = source.read_rows(ids)

        assert source.shape == (N_ITEMS, 10)
        assert source.source_dims == [6, 4]
        assert (source.strategy, source.aligned_dim, source.normalize) == ("mean", 5, False)
        assert source.fusion_kwargs == {"weights": [0.3, 0.7]}
        np.testing.assert_array_equal(rows, np.concatenate([a, b], axis=1)[ids])
        assert source.read_rows(np.array([], dtype=np.int64)).shape == (0, 10)

    def test_stack_gathers_equal_dim_sources_per_row(self) -> None:
        a, b = _pooled(seed=4), _pooled(seed=5)
        source = StackedFeatureSource([ArrayFeatureSource(a), ArrayFeatureSource(b)])
        ids = np.array([2, 9, 2])

        rows = source.read_rows(ids)

        assert source.shape == (N_ITEMS, 2, 5)
        np.testing.assert_array_equal(rows, np.stack([a, b], axis=1)[ids])
        assert source.read_rows(np.array([], dtype=np.int64)).shape == (0, 2, 5)

    def test_row_count_mismatch_fails(self) -> None:
        with pytest.raises(ValueError, match="n_items"):
            ConcatFeatureSource(
                [ArrayFeatureSource(_pooled(n=4)), ArrayFeatureSource(_pooled(n=5))],
                strategy="mean",
                aligned_dim=3,
            )

    def test_stack_shape_mismatch_fails(self) -> None:
        with pytest.raises(ValueError, match="share a shape"):
            StackedFeatureSource(
                [ArrayFeatureSource(_pooled(d=5)), ArrayFeatureSource(_pooled(d=6))]
            )

    def test_close_propagates_to_every_source(self, tmp_path: Path) -> None:
        np.save(tmp_path / "a.npy", _pooled())
        np.save(tmp_path / "b.npy", _pooled())
        parts = [NpyFeatureSource(tmp_path / "a.npy"), NpyFeatureSource(tmp_path / "b.npy")]
        source = StackedFeatureSource(parts)
        source.read_rows(np.array([0]))
        assert all(p.is_open for p in parts)

        source.close()

        assert not any(p.is_open for p in parts)


class TestProtocolAndBytes:
    def test_adapters_satisfy_the_protocol_and_arrays_do_not(self, pooled_npy) -> None:
        assert isinstance(NpyFeatureSource(pooled_npy[0]), FeatureSource)
        assert is_feature_source(ArrayFeatureSource(_pooled()))
        assert not is_feature_source(_pooled())
        assert not is_feature_source(
            RaggedSources(_pooled(), source_dims=[5], strategy="mean", aligned_dim=3)
        )

    def test_source_bytes_is_payload_not_file_size(self, comp_npy) -> None:
        assert source_bytes(NpyFeatureSource(comp_npy[0])) == N_ITEMS * 3 * 4 * 2
        assert source_bytes(_pooled()) == N_ITEMS * 5 * 4


class TestLoadEmbeddingLazy:
    def _write_sidecar(self, tmp_path: Path, payload: dict, name: str) -> Path:
        path = tmp_path / name
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_default_form_is_unchanged(self, pooled_npy) -> None:
        out = load_embedding(pooled_npy[0])

        assert isinstance(out, np.ndarray) and not is_feature_source(out)

    def test_pooled_npy_becomes_a_memmap_source(self, pooled_npy) -> None:
        source = load_embedding(pooled_npy[0], lazy=True)

        assert isinstance(source, NpyFeatureSource)
        assert source.shape == (N_ITEMS, 5)

    def test_component_npy_keeps_fp16(self, comp_npy) -> None:
        source = load_embedding(comp_npy[0], lazy=True)

        assert isinstance(source, NpyFeatureSource)
        assert source.dtype == np.float16 and source.shape == (N_ITEMS, 3, 4)

    def test_meta_mismatch_still_fails_lazily(self, pooled_npy) -> None:
        path, _ = pooled_npy
        (path.parent / "resnet50.meta.json").write_text(
            json.dumps({"name": "resnet50", "native_dim": 2048})
        )

        with pytest.raises(ValueError, match="native_dim"):
            load_embedding(path, lazy=True)

    def test_learned_sidecar_becomes_an_ordered_concat_source(self, tmp_path: Path) -> None:
        a, b = _pooled(d=6, seed=6), _pooled(d=4, seed=7)
        np.save(tmp_path / "resnet50.npy", a)
        np.save(tmp_path / "vit_b16.npy", b)
        payload = {
            "strategy": "sigmoid_gated",
            "online": True,
            "alignment": "learned",
            "dim": 3,
            "components": ["resnet50.npy", "vit_b16.npy"],
            "normalize": True,
            "fusion_kwargs": {"logits": [0.2, -0.2]},
        }
        path = self._write_sidecar(tmp_path, payload, "hybrid_sigmoid_gated_learned_D3.json")

        source = load_embedding(path, lazy=True)
        eager = load_embedding(path)

        assert isinstance(source, ConcatFeatureSource)
        assert source.shape == eager.shape == (N_ITEMS, 10)
        assert source.source_dims == eager.source_dims == [6, 4]
        assert (source.strategy, source.aligned_dim, source.normalize) == ("sigmoid_gated", 3, True)
        assert source.fusion_kwargs == {"logits": [0.2, -0.2]}
        np.testing.assert_array_equal(source.read_rows(np.arange(N_ITEMS)), np.asarray(eager))

    def test_equal_dim_sidecar_becomes_a_stacked_source(self, tmp_path: Path) -> None:
        a, b = _pooled(seed=8), _pooled(seed=9)
        np.save(tmp_path / "a.npy", a)
        np.save(tmp_path / "b.npy", b)
        payload = {
            "strategy": "adaptive_gated",
            "online": True,
            "components": ["a.npy", "b.npy"],
            "normalize": False,
        }
        path = self._write_sidecar(tmp_path, payload, "hybrid_adaptive_gated_pca_D5.json")

        source = load_embedding(path, lazy=True)

        assert isinstance(source, StackedFeatureSource)
        assert source.shape == (N_ITEMS, 2, 5)
        np.testing.assert_array_equal(
            source.read_rows(np.array([3, 0])), np.stack([a, b], axis=1)[[3, 0]]
        )

    def test_mismatched_components_fail_lazily_like_eagerly(self, tmp_path: Path) -> None:
        np.save(tmp_path / "a.npy", np.zeros((4, 8), dtype=np.float32))
        np.save(tmp_path / "b.npy", np.zeros((4, 16), dtype=np.float32))
        path = self._write_sidecar(
            tmp_path, {"strategy": "adaptive_gated", "components": ["a.npy", "b.npy"]}, "h.json"
        )

        with pytest.raises(ValueError, match="expected"):
            load_embedding(path, lazy=True)

    def test_missing_component_fails_lazily(self, tmp_path: Path) -> None:
        path = self._write_sidecar(
            tmp_path, {"strategy": "mean", "components": ["ghost.npy"]}, "h.json"
        )

        with pytest.raises(FileNotFoundError, match="missing component"):
            load_embedding(path, lazy=True)
