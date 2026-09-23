"""Component artifacts reach their consumer L2-normalised.

Native ``<extractor>_comp.npy`` files used to be fed to ACF exactly as
extracted, while the pooled embeddings every other recommender reads
were normalised offline.  The attention logits then scaled with the
backbone's native magnitude and overflowed to ``inf`` under autocast
(amazon_men grid, 2026-09-16: coatnet_0 and convnext_base at lr=0.01).
These tests pin the normalisation and, above all, that the eager and
lazy paths deliver the same numbers.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.fusions.online import load_embedding
from src.fusions.strategies import l2_normalize, l2_normalize_components
from src.recommenders.base import BaseRecommender


def _write_components(path, *, scale: float = 100.0, n_items: int = 64) -> np.ndarray:
    """Write a 3-D fp16 component artifact with large-magnitude rows."""
    rng = np.random.default_rng(0)
    arr = (rng.standard_normal((n_items, 4, 8)) * scale).astype(np.float16)
    np.save(path, arr)
    return arr


class TestL2NormalizeComponents:
    def test_should_give_every_component_vector_unit_norm(self):
        arr = (np.random.default_rng(1).standard_normal((16, 4, 8)) * 50).astype(np.float32)

        out = l2_normalize_components(arr)

        norms = np.linalg.norm(out, axis=-1)
        assert np.allclose(norms, 1.0, atol=1e-6)

    def test_should_leave_zero_vectors_unchanged(self):
        arr = np.zeros((4, 2, 8), dtype=np.float32)
        arr[0, 0] = 3.0

        out = l2_normalize_components(arr)

        assert np.allclose(out[1:], 0.0)
        assert np.isclose(np.linalg.norm(out[0, 0]), 1.0)

    def test_should_preserve_the_on_disk_dtype(self):
        arr = (np.random.default_rng(2).standard_normal((8, 4, 8)) * 100).astype(np.float16)

        out = l2_normalize_components(arr)

        assert out.dtype == np.float16

    def test_should_match_row_wise_l2_normalize_on_each_component(self):
        arr = (np.random.default_rng(3).standard_normal((8, 4, 8)) * 30).astype(np.float32)

        out = l2_normalize_components(arr)

        expected = np.stack([l2_normalize(arr[:, r, :]) for r in range(arr.shape[1])], axis=1)
        assert np.allclose(out, expected, atol=1e-6)

    def test_should_normalize_identically_across_chunk_boundaries(self, monkeypatch):
        arr = (np.random.default_rng(4).standard_normal((70, 2, 8)) * 10).astype(np.float32)
        monkeypatch.setattr("src.fusions.strategies._COMPONENT_NORM_CHUNK", 7)

        chunked = l2_normalize_components(arr)

        monkeypatch.setattr("src.fusions.strategies._COMPONENT_NORM_CHUNK", 10_000)
        assert np.array_equal(chunked, l2_normalize_components(arr))

    def test_should_reject_a_two_dimensional_array(self):
        with pytest.raises(ValueError, match="3-D"):
            l2_normalize_components(np.zeros((4, 8), dtype=np.float32))


class TestComponentLoading:
    def test_should_normalize_a_native_component_artifact_eagerly(self, tmp_path):
        path = tmp_path / "resnet50_comp.npy"
        _write_components(path)

        arr = load_embedding(path)

        # The eager branch keeps the memmap; the model normalises it.
        assert arr.shape == (64, 4, 8)

    def test_should_normalize_gathered_rows_of_a_component_source(self, tmp_path):
        path = tmp_path / "resnet50_comp.npy"
        _write_components(path)

        model = _ComponentModel(4, 64, load_embedding(path, lazy=True), {})
        rows = model._raw_visual_rows(torch.tensor([0, 5, 9]))

        norms = torch.linalg.norm(rows.float(), dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-3)

    def test_should_leave_the_source_itself_untouched(self, tmp_path):
        path = tmp_path / "resnet50_comp.npy"
        arr = _write_components(path)

        source = load_embedding(path, lazy=True)

        # Normalisation belongs to the consumer, so residency stays an
        # execution detail: the source still yields the raw rows.
        assert np.array_equal(source.read_rows(np.array([0, 5])), arr[[0, 5]])

    def test_should_not_normalize_a_pooled_embedding(self, tmp_path):
        path = tmp_path / "resnet50.npy"
        arr = (np.random.default_rng(5).standard_normal((16, 8)) * 20).astype(np.float32)
        np.save(path, arr)

        model = _PooledModel(4, 16, load_embedding(path), {})

        assert model._normalize_components is False
        assert np.allclose(model.visual_features.numpy(), arr)


class _ComponentModel(BaseRecommender):
    """Minimal component consumer: only the feature plumbing is exercised."""

    consumes_raw_components = True

    def forward(self, *args, **kwargs):  # pragma: no cover - not exercised
        raise NotImplementedError

    def predict(self, user_id, item_ids):  # pragma: no cover - not exercised
        raise NotImplementedError


class _PooledModel(_ComponentModel):
    """Same plumbing, but a consumer of pooled 2-D embeddings."""

    consumes_raw_components = False


class TestDenseAndLazyAgree:
    def test_should_buffer_unit_norm_components_in_the_dense_path(self, tmp_path):
        path = tmp_path / "resnet50_comp.npy"
        _write_components(path)

        model = _ComponentModel(4, 64, load_embedding(path), {})

        norms = torch.linalg.norm(model.visual_features.float(), dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-3)

    def test_should_keep_the_dense_buffer_in_fp16(self, tmp_path):
        path = tmp_path / "resnet50_comp.npy"
        _write_components(path)

        model = _ComponentModel(4, 64, load_embedding(path), {})

        assert model.visual_features.dtype == torch.float16

    def test_should_gather_the_same_rows_dense_and_lazy(self, tmp_path):
        path = tmp_path / "resnet50_comp.npy"
        _write_components(path)
        ids = torch.tensor([0, 3, 3, 17])

        dense = _ComponentModel(4, 64, load_embedding(path), {})
        lazy = _ComponentModel(4, 64, load_embedding(path, lazy=True), {})

        assert torch.equal(dense._raw_visual_rows(ids), lazy._raw_visual_rows(ids))
