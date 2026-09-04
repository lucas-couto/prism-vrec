"""Spatial pooling of per-item component features before they are saved.

ACF (Chen et al. 2017) attends over the ``M`` spatial regions of an
image feature map.  At native resolution ``M`` is 49 (7×7, ResNet-50 /
ConvNeXt / CoAtNet / CLIP ViT-B/32), 196 (14×14, ViT-B/16) or 256
(16×16, DINOv2), which makes the ``*_comp.npy`` artifacts 50–500× the
pooled ones (≈1.1 TB for the four datasets and eight backbones) and
puts a single one past the container's RAM cap.  ``component_grid``
(``configs/extractors.yaml``) pools the ``√M × √M`` map to a ``g × g``
grid with adaptive average pooling — ``g = 2`` keeps four quadrant
descriptors per item, ``g = 3`` nine regions — so the mechanism (one
attention weight per region) is preserved at a fraction of the size.
This is a declared divergence from the paper's 49 regions
(``docs/protocol.md`` §7).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def pool_components(components: torch.Tensor, grid: int | None) -> torch.Tensor:
    """Pool ``(B, M, D)`` region features to ``(B, grid², D)``.

    :param components: Per-item region features, ``M`` a perfect square
        laid out row-major over the ``√M × √M`` spatial map (the order
        every built-in backbone emits).
    :param grid: Target side ``g``; ``None`` returns the input unchanged.
    :returns: ``(B, g*g, D)`` adaptive-average-pooled features, in the
        input dtype.
    :raises ValueError: If ``M`` is not a perfect square, or ``grid``
        exceeds the native side (pooling cannot upsample).
    """
    if grid is None:
        return components
    if components.dim() != 3:
        raise ValueError(f"components must be (B, M, D), got shape {tuple(components.shape)}")
    batch, n_regions, dim = components.shape
    side = math.isqrt(n_regions)
    if side * side != n_regions:
        raise ValueError(f"component_grid needs a square spatial map; got M={n_regions} regions")
    if grid < 1 or grid > side:
        raise ValueError(f"component_grid must be in [1, {side}] for M={n_regions}, got {grid}")
    if grid == side:
        return components
    spatial = components.transpose(1, 2).reshape(batch, dim, side, side)
    pooled = F.adaptive_avg_pool2d(spatial, grid)  # (B, D, g, g)
    return pooled.reshape(batch, dim, grid * grid).transpose(1, 2).contiguous()


__all__ = ["pool_components"]
