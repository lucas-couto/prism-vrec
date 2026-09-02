"""Attention modules for ACF (Chen et al., SIGIR 2017).

Two small networks implement ACF's two attention levels:

* :class:`ComponentAttention` weights an item's ``M`` visual components
  (spatial feature-map cells / patch tokens) conditioned on the user
  (Eqs. 10-12).
* :class:`ItemAttention` weights the items in a user's history to build
  the augmented user profile (Eqs. 8-9).

Both broadcast over arbitrary leading batch dimensions so the same
module serves single-item, candidate-list, and history tensors.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _init_linear(layer: nn.Linear) -> None:
    """Xavier-uniform weights, zero bias."""
    nn.init.xavier_uniform_(layer.weight)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)


class ComponentAttention(nn.Module):
    """Component-level attention over an item's ``M`` projected components.

    Eqs. 10-12: ``b(i,l,m) = w_2^T φ(W_2u u_i + W_2x x_{lm} + b_2) + c_2``,
    ``β(i,l,m) = softmax_m b(i,l,m)`` and ``x̄_l = Σ_m β(i,l,m) x_{lm}``.
    """

    def __init__(self, latent_dim: int, visual_dim: int, hidden: int) -> None:
        super().__init__()
        self.user_proj = nn.Linear(latent_dim, hidden)
        self.comp_proj = nn.Linear(visual_dim, hidden)
        self.score = nn.Linear(hidden, 1)
        for layer in (self.user_proj, self.comp_proj, self.score):
            _init_linear(layer)

    def forward(self, gamma_u: torch.Tensor, components: torch.Tensor) -> torch.Tensor:
        """Return the attended visual vector ``x̄`` of shape ``(..., visual_dim)``.

        ``gamma_u`` has shape ``(..., latent_dim)`` and ``components`` has
        shape ``(..., M, visual_dim)`` with matching leading dims.
        """
        query = self.user_proj(gamma_u).unsqueeze(-2)  # (..., 1, hidden)
        energy = self.score(torch.relu(query + self.comp_proj(components)))  # (..., M, 1)
        alpha = torch.softmax(energy, dim=-2)
        return (alpha * components).sum(dim=-2)


class ItemAttention(nn.Module):
    """Item-level attention building the augmented user profile.

    Eq. 8: ``a(i,l) = w_1^T φ(W_1u u_i + W_1v v_l + W_1p p_l + W_1x x̄_l + b_1) + c_1``
    with ``α(i,l) = softmax_l a(i,l)`` over the user's history ``R(i)``
    (Eq. 9).  The module returns ``Σ_{l∈R(i)} α(i,l) p_l`` — the history
    term of Eq. 6.  The item latent ``v_l`` and the component-attended
    visual ``x̄_l`` (already mapped to the latent space) enter the
    attention *energy* only; neither is aggregated into the profile.
    """

    def __init__(self, latent_dim: int, hidden: int) -> None:
        super().__init__()
        self.user = nn.Linear(latent_dim, hidden)
        self.item = nn.Linear(latent_dim, hidden)
        self.aux = nn.Linear(latent_dim, hidden)
        self.vis = nn.Linear(latent_dim, hidden)
        self.score = nn.Linear(hidden, 1)
        for layer in (self.user, self.item, self.aux, self.vis, self.score):
            _init_linear(layer)

    def forward(
        self,
        gamma_u: torch.Tensor,
        gamma_h: torch.Tensor,
        p_h: torch.Tensor,
        v_h: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return ``Σ_l α(u,l) p_l`` of shape ``(B, latent_dim)``.

        ``gamma_u`` is ``(B, k)``; ``gamma_h`` (item latents ``v_l``),
        ``p_h`` (auxiliary ``p_l``) and ``v_h`` (visual ``x̄_l`` in latent
        space) are ``(B, H, k)``; ``mask`` is ``(B, H)`` with ``True`` for
        valid history slots.  Users with empty history contribute zero.
        """
        query = self.user(gamma_u).unsqueeze(1)  # (B, 1, hidden)
        energy = self.score(
            torch.relu(query + self.item(gamma_h) + self.aux(p_h) + self.vis(v_h))
        )  # (B, H, 1)
        energy = energy.masked_fill(~mask.unsqueeze(-1), float("-inf"))
        alpha = torch.nan_to_num(torch.softmax(energy, dim=1))  # empty rows -> 0
        return (alpha * p_h).sum(dim=1)
