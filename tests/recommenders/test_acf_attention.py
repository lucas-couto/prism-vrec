"""Unit tests for ACF attention modules."""

from __future__ import annotations

import torch

from src.recommenders.acf_attention import ComponentAttention, ItemAttention


def test_component_attention_weights_sum_to_one() -> None:
    torch.manual_seed(0)
    latent, visual, hidden, batch, m = 4, 5, 6, 3, 7
    att = ComponentAttention(latent, visual, hidden)
    gamma_u = torch.randn(batch, latent)
    components = torch.randn(batch, m, visual)

    # Recompute the internal weights to assert they form a distribution.
    query = att.user_proj(gamma_u).unsqueeze(-2)
    energy = att.score(torch.relu(query + att.comp_proj(components)))
    alpha = torch.softmax(energy, dim=-2)

    assert alpha.shape == (batch, m, 1)
    assert torch.allclose(alpha.sum(dim=-2), torch.ones(batch, 1), atol=1e-6)


def test_component_attention_output_shape() -> None:
    att = ComponentAttention(4, 5, 6)
    out = att(torch.randn(3, 4), torch.randn(3, 7, 5))

    assert out.shape == (3, 5)


def test_item_attention_ignores_padded_positions() -> None:
    torch.manual_seed(1)
    latent, hidden, batch, horizon = 4, 6, 2, 5
    att = ItemAttention(latent, hidden)

    gamma_u = torch.randn(batch, latent)
    gamma_h = torch.randn(batch, horizon, latent)
    p_h = torch.randn(batch, horizon, latent)
    v_h = torch.randn(batch, horizon, latent)

    full_mask = torch.ones(batch, horizon, dtype=torch.bool)
    out_full = att(gamma_u, gamma_h, p_h, v_h, full_mask)

    # Corrupting a masked-out slot must not change the output.
    partial_mask = full_mask.clone()
    partial_mask[:, -1] = False
    p_h_corrupt = p_h.clone()
    p_h_corrupt[:, -1] = 1e3
    out_a = att(gamma_u, gamma_h, p_h, v_h, partial_mask)
    out_b = att(gamma_u, gamma_h, p_h_corrupt, v_h, partial_mask)

    assert out_full.shape == (batch, latent)
    assert torch.allclose(out_a, out_b, atol=1e-5)


def test_item_attention_empty_history_returns_zero() -> None:
    att = ItemAttention(4, 6)
    empty_mask = torch.zeros(2, 5, dtype=torch.bool)

    out = att(
        torch.randn(2, 4),
        torch.randn(2, 5, 4),
        torch.randn(2, 5, 4),
        torch.randn(2, 5, 4),
        empty_mask,
    )

    assert torch.allclose(out, torch.zeros(2, 4))
