"""Example fusion plugin: element-wise max pooling across extractors.

How to use this file
--------------------
1. Copy ``_example.py`` to a new file with a descriptive name, e.g.
   ``plugins/fusions/my_max_pool.py``.  The leading underscore on *this*
   file is what keeps the auto-discovery from importing it — files
   (and dataset directories) starting with ``_`` are skipped on purpose
   so the example never registers itself.
2. Rename the function and the ``register_fusion_strategy("my_max_pool"
   ...)`` key.
3. Add the same key to ``configs/fusion.yaml ->
   fusion_strategies_enabled`` and run the pipeline.

The contract is just a callable:

    fn(embeddings, normalize=True, **kwargs) -> np.ndarray

* ``embeddings`` — list of ``(N, dm)`` matrices, one per extractor in
  ``fusion_extractors``.
* Returns a single ``(N, Dh)`` matrix.

Set ``equal_dim_required=True`` when every input must share its second
dimension (the orchestrator skips invalid combinations otherwise).

Full guide: ``docs/extending.md``.
"""

from __future__ import annotations

import numpy as np

from src.fusions.registry import register_fusion_strategy


def my_max_pool(
    embeddings: list[np.ndarray],
    normalize: bool = True,
    **kwargs,  # noqa: ARG001 — accept and ignore unused expand_grid kwargs
) -> np.ndarray:
    """Element-wise maximum across the M input embeddings.

    Parameters
    ----------
    embeddings:
        List of ``(N, D)`` matrices — one per extractor.  All matrices
        must share the same ``(N, D)``.
    normalize:
        L2-normalise the fused output along the embedding axis.
    """
    if not embeddings:
        raise ValueError("my_max_pool received an empty embeddings list")

    stacked = np.stack(embeddings, axis=0)  # (M, N, D)
    fused = stacked.max(axis=0)  # (N, D)

    if normalize:
        norms = np.linalg.norm(fused, axis=1, keepdims=True) + 1e-12
        fused = fused / norms

    return fused.astype(np.float32, copy=False)


register_fusion_strategy(
    "my_max_pool",
    my_max_pool,
    equal_dim_required=True,
)
