"""BPR -- Bayesian Personalised Ranking (baseline, no visual features).

Prediction rule:
    y_hat_ui = gamma_u^T gamma_i + beta_i

References
----------
Rendle, S. et al. (2009). BPR: Bayesian Personalized Ranking from
Implicit Feedback.  UAI.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from src.recommenders.base import BaseRecommender


class BPR(BaseRecommender):
    """Matrix-factorisation BPR without visual features.

    Parameters
    ----------
    n_users, n_items:
        Vocabulary sizes.
    visual_embeddings:
        Ignored (kept for interface compatibility).  Should be ``None``.
    config:
        Must contain ``latent_dim`` (int).  ``l2_reg`` is optional
        (default 0).
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        visual_embeddings: np.ndarray | None = None,
        config: dict | None = None,
    ) -> None:
        config = config or {}
        super().__init__(n_users, n_items, visual_embeddings=None, config=config)

        k: int = config["latent_dim"]

        self.user_embedding = nn.Embedding(n_users, k)
        self.item_embedding = nn.Embedding(n_items, k)
        self.item_bias = nn.Embedding(n_items, 1)

        self._init_embedding(self.user_embedding)
        self._init_embedding(self.item_embedding)
        nn.init.zeros_(self.item_bias.weight)

    def forward(
        self,
        user_ids: torch.Tensor,
        pos_item_ids: torch.Tensor,
        neg_item_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        gamma_u = self.user_embedding(user_ids)
        gamma_pos = self.item_embedding(pos_item_ids)
        gamma_neg = self.item_embedding(neg_item_ids)
        beta_pos = self.item_bias(pos_item_ids).squeeze(-1)
        beta_neg = self.item_bias(neg_item_ids).squeeze(-1)

        score_pos = (gamma_u * gamma_pos).sum(dim=-1) + beta_pos
        score_neg = (gamma_u * gamma_neg).sum(dim=-1) + beta_neg
        return score_pos, score_neg

    def predict(self, user_id: int, item_ids: torch.Tensor) -> torch.Tensor:
        gamma_u = self.user_embedding.weight[user_id]
        gamma_i = self.item_embedding(item_ids)
        beta_i = self.item_bias(item_ids).squeeze(-1)
        return (gamma_u * gamma_i).sum(dim=-1) + beta_i

    def predict_batch(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        gamma_u = self.user_embedding(user_ids)
        gamma_i = self.item_embedding(item_ids)
        beta_i = self.item_bias(item_ids).squeeze(-1)
        return gamma_u @ gamma_i.T + beta_i.unsqueeze(0)
