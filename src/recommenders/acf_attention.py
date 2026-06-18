"""Attention modules for ACF (Chen et al., SIGIR 2017).

Two small networks implement ACF's two attention levels:

* :class:`ComponentAttention` weights an item's ``M`` visual components
  (spatial feature-map cells / patch tokens) conditioned on the user.
* :class:`ItemAttention` weights the items in a user's history to build
  the augmented user profile.

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
    """Component-level attention over an item's ``M`` projected components."""

    def __init__(self, latent_dim: int, visual_dim: int, hidden: int) -> None:
        super().__init__()
        self.user_proj = nn.Linear(latent_dim, hidden)
        self.comp_proj = nn.Linear(visual_dim, hidden)
        self.score = nn.Linear(hidden, 1)
        for layer in (self.user_proj, self.comp_proj, self.score):
            _init_linear(layer)

    def forward(self, gamma_u: torch.Tensor, components: torch.Tensor) -> torch.Tensor:
        """Return the attended visual vector ``(..., visual_dim)``.

        ``gamma_u`` has shape ``(..., latent_dim)`` and ``components`` has
        shape ``(..., M, visual_dim)`` with matching leading dims.
        """
        query = self.user_proj(gamma_u).unsqueeze(-2)  # (..., 1, hidden)
        energy = self.score(torch.relu(query + self.comp_proj(components)))  # (..., M, 1)
        alpha = torch.softmax(energy, dim=-2)
        return (alpha * components).sum(dim=-2)


class ItemAttention(nn.Module):
    """Item-level attention building the augmented user profile.

    Aggregates, over the user's history, ``alpha_i * (p_i + v_i)`` where
    ``p_i`` is the auxiliary item embedding and ``v_i`` the item's
    component-attended visual vector mapped to the latent space.
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
        """Return the profile contribution ``(B, latent_dim)``.

        ``gamma_u`` is ``(B, k)``; ``gamma_h``/``p_h``/``v_h`` are
        ``(B, H, k)``; ``mask`` is ``(B, H)`` with ``True`` for valid
        history slots.  Users with empty history contribute zero.
        """
        query = self.user(gamma_u).unsqueeze(1)  # (B, 1, hidden)
        energy = self.score(
            torch.relu(query + self.item(gamma_h) + self.aux(p_h) + self.vis(v_h))
        )  # (B, H, 1)
        energy = energy.masked_fill(~mask.unsqueeze(-1), float("-inf"))
        alpha = torch.nan_to_num(torch.softmax(energy, dim=1))  # empty rows -> 0
        return (alpha * (p_h + v_h)).sum(dim=1)
