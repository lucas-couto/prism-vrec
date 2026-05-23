"""Tests for the per-item adaptive gated fusion module.

This is the implementation of the ``adaptive_gated`` strategy
specified in the qualification document: a per-item per-dimension
gate produced by a small MLP, applied as a convex combination of
two equal-dimensional embeddings, trained jointly with the
recommender via the BPR loss.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from src.fusions import (
    AdaptiveGatedFusion,
    is_online_strategy,
    load_embedding,
    online_module_for,
    registered_fusion_strategies,
)


def test_module_output_shape_matches_inputs() -> None:
    fusion = AdaptiveGatedFusion(dim=8)
    e1 = torch.randn(4, 8)
    e2 = torch.randn(4, 8)

    h = fusion(e1, e2)

    assert h.shape == (4, 8)


def test_module_initialises_to_uniform_fusion() -> None:
    """Final layer is zero-initialised, so initial gate is sigmoid(0)=0.5
    everywhere — fused output equals 0.5 * (e1 + e2) at trial 0."""
    fusion = AdaptiveGatedFusion(dim=4)
    e1 = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    e2 = torch.tensor([[5.0, 6.0, 7.0, 8.0]])

    h = fusion(e1, e2)

    expected = 0.5 * (e1 + e2)
    torch.testing.assert_close(h, expected, atol=1e-5, rtol=1e-5)


def test_module_rejects_mismatched_shapes() -> None:
    fusion = AdaptiveGatedFusion(dim=8)

    with pytest.raises(ValueError, match="share shape"):
        fusion(torch.randn(4, 8), torch.randn(4, 16))


def test_module_rejects_wrong_dim() -> None:
    fusion = AdaptiveGatedFusion(dim=8)

    with pytest.raises(ValueError, match="trailing dim"):
        fusion(torch.randn(4, 16), torch.randn(4, 16))


def test_module_parameters_receive_gradient() -> None:
    """Backprop reaches every gate parameter once training is under way.

    The gate is zero-initialised at ``gate[0]`` so the network starts at
    the uniform fusion (sigmoid(0) = 0.5). On the very first backward,
    ``∂L/∂gate[2].weight = 0`` because ``a₁ = Tanh(0) = 0``. A single
    SGD step on the freely-updating parameters breaks the zero at
    ``gate[0]``; from the second backward onward every parameter
    receives a non-zero gradient. This test verifies that full path.
    """
    torch.manual_seed(0)

    fusion = AdaptiveGatedFusion(dim=4)
    e1 = torch.randn(2, 4, requires_grad=True)
    e2 = torch.randn(2, 4, requires_grad=True)

    optimiser = torch.optim.SGD(fusion.parameters(), lr=0.1)

    # Warm-up step: moves gate[0] off zero so a₁ stops being identically 0.
    optimiser.zero_grad()
    fusion(e1, e2).sum().backward()
    optimiser.step()

    # Real assertion: after one step, every parameter sees gradient.
    optimiser.zero_grad()
    fusion(e1, e2).sum().backward()

    for name, param in fusion.named_parameters():
        assert param.grad is not None, f"{name}: no grad"
        assert param.grad.abs().sum().item() > 0, (
            f"{name}: zero grad — gate is decoupled from output"
        )


def test_gate_values_in_unit_interval() -> None:
    fusion = AdaptiveGatedFusion(dim=8)
    # Push the linear layer away from zero so sigmoid actually varies.
    with torch.no_grad():
        torch.nn.init.xavier_uniform_(fusion.gate[-1].weight)

    e1 = torch.randn(16, 8)
    e2 = torch.randn(16, 8)
    g = fusion.gate_values(e1, e2)

    assert g.shape == (16, 8)
    assert (g >= 0.0).all()
    assert (g <= 1.0).all()


def test_adaptive_gated_registered_as_online() -> None:
    assert "adaptive_gated" in registered_fusion_strategies()
    assert is_online_strategy("adaptive_gated")


def test_factory_returns_module_for_known_strategy() -> None:
    module = online_module_for("adaptive_gated", dim=16)

    assert isinstance(module, AdaptiveGatedFusion)
    assert module.dim == 16


def test_factory_rejects_unknown_strategy() -> None:
    with pytest.raises(ValueError, match="Unknown online fusion strategy"):
        online_module_for("does_not_exist", dim=16)


def test_load_embedding_npy_passes_through(tmp_path: Path) -> None:
    arr = np.random.rand(4, 8).astype(np.float32)
    path = tmp_path / "plain.npy"
    np.save(path, arr)

    out = load_embedding(path)

    np.testing.assert_array_equal(out, arr)


def test_load_embedding_json_sidecar_stacks_components(tmp_path: Path) -> None:
    e1 = np.random.rand(4, 8).astype(np.float32)
    e2 = np.random.rand(4, 8).astype(np.float32)
    np.save(tmp_path / "resnet50_D8.npy", e1)
    np.save(tmp_path / "vit_b16_D8.npy", e2)

    sidecar = tmp_path / "hybrid_adaptive_gated_D8.json"
    sidecar.write_text(
        json.dumps(
            {
                "strategy": "adaptive_gated",
                "online": True,
                "components": ["resnet50_D8.npy", "vit_b16_D8.npy"],
                "normalize": True,
            }
        )
    )

    out = load_embedding(sidecar)

    assert out.shape == (4, 2, 8)
    np.testing.assert_array_equal(out[:, 0], e1)
    np.testing.assert_array_equal(out[:, 1], e2)


def test_load_embedding_rejects_mismatched_components(tmp_path: Path) -> None:
    np.save(tmp_path / "a.npy", np.zeros((4, 8), dtype=np.float32))
    np.save(tmp_path / "b.npy", np.zeros((4, 16), dtype=np.float32))

    sidecar = tmp_path / "hybrid_adaptive_gated_D8.json"
    sidecar.write_text(
        json.dumps(
            {
                "strategy": "adaptive_gated",
                "components": ["a.npy", "b.npy"],
            }
        )
    )

    with pytest.raises(ValueError, match="expected"):
        load_embedding(sidecar)


def test_base_recommender_detects_3d_buffer_and_creates_gate() -> None:
    """When visual_embeddings is 3-D, BaseRecommender must instantiate
    the matching online fusion module so backprop can update the gate."""
    from src.recommenders.base import BaseRecommender

    class _Toy(BaseRecommender):
        def forward(self, u, p, n):
            return torch.zeros_like(u, dtype=torch.float32), torch.zeros_like(
                u,
                dtype=torch.float32,
            )

        def predict(self, u, items):
            return torch.zeros(items.shape[0], dtype=torch.float32)

    stacked = np.random.rand(10, 2, 8).astype(np.float32)
    rec = _Toy(
        n_users=5,
        n_items=10,
        visual_embeddings=stacked,
        config={"l2_reg": 0.0},
    )

    assert rec.visual_dim_raw == 8
    assert rec._online_fusion is not None
    assert isinstance(rec._online_fusion, AdaptiveGatedFusion)

    # _resolve_visual must return (B, D), not (B, M, D).
    item_ids = torch.tensor([0, 3, 7])
    out = rec._resolve_visual(item_ids)
    assert out.shape == (3, 8)


def test_base_recommender_2d_buffer_does_not_create_gate() -> None:
    from src.recommenders.base import BaseRecommender

    class _Toy(BaseRecommender):
        def forward(self, u, p, n):
            return torch.zeros_like(u, dtype=torch.float32), torch.zeros_like(
                u,
                dtype=torch.float32,
            )

        def predict(self, u, items):
            return torch.zeros(items.shape[0], dtype=torch.float32)

    plain = np.random.rand(10, 8).astype(np.float32)
    rec = _Toy(n_users=5, n_items=10, visual_embeddings=plain, config={})

    assert rec._online_fusion is None
    assert rec.visual_dim_raw == 8
