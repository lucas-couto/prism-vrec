"""SDD M02/M04 isolated loader test: staging memory follows the block, not N.

Three ``.npy`` feature files of N, 2N and 4N rows feed the same small
VBPR (fixed trainable state per N is not the point here — the item
table grows with N and is attributed separately).  A fixed-size gather
through the lazy path must allocate host staging proportional to the
block, and the raw feature bytes resident in the model must be zero,
for every N.  ``tracemalloc`` tracks numpy allocations (the memmap
gather returns a numpy copy); torch tensors on CPU share that memory,
so nothing native escapes the count here.  Native CUDA allocations are
outside this test's reach and are recorded as a limitation.
"""

from __future__ import annotations

import tracemalloc
from pathlib import Path

import numpy as np
import pytest
import torch

from src.data.feature_source import NpyFeatureSource, source_bytes
from src.recommenders.vbpr import VBPR

BASE_N, DV, BLOCK, N_USERS = 2_000, 64, 128, 5


def _source(tmp_path: Path, n_items: int) -> NpyFeatureSource:
    path = tmp_path / f"resnet50_{n_items}.npy"
    np.save(path, np.random.default_rng(n_items).standard_normal((n_items, DV)).astype(np.float32))
    return NpyFeatureSource(path)


def _staging_peak(model: VBPR, ids: torch.Tensor) -> int:
    """Peak host bytes allocated while gathering *ids* through the lazy path."""
    model._raw_visual_rows(ids[:1])  # open the handle outside the measurement
    tracemalloc.start()
    tracemalloc.reset_peak()
    rows = model._raw_visual_rows(ids)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert rows.shape == (BLOCK, DV)
    return peak


@pytest.mark.parametrize("scale", [1, 2, 4])
def test_lazy_model_holds_no_raw_feature_bytes(tmp_path: Path, scale: int) -> None:
    n_items = BASE_N * scale
    model = VBPR(N_USERS, n_items, _source(tmp_path, n_items), {"latent_dim": 4, "visual_dim": 3})

    raw_bytes = sum(b.numel() * b.element_size() for b in model.buffers())

    assert raw_bytes == 0
    assert source_bytes(model._feature_source) == n_items * DV * 4


def test_staging_memory_scales_with_the_block_not_the_catalogue(tmp_path: Path) -> None:
    peaks = {}
    for scale in (1, 2, 4):
        n_items = BASE_N * scale
        model = VBPR(
            N_USERS, n_items, _source(tmp_path, n_items), {"latent_dim": 4, "visual_dim": 3}
        )
        ids = torch.from_numpy(np.random.default_rng(scale).integers(0, n_items, size=BLOCK))
        peaks[scale] = _staging_peak(model, ids)

    block_bytes = BLOCK * DV * 4
    smallest, largest = min(peaks.values()), max(peaks.values())
    # Each peak is a small multiple of one block (gather copy + inverse
    # index bookkeeping) and does not track the 4x growth of the source.
    assert largest <= 4 * block_bytes, peaks
    assert largest <= 1.5 * smallest, peaks
    assert largest < 0.05 * source_bytes(model._feature_source), peaks
