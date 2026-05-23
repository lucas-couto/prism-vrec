"""VBPR -- Visual Bayesian Personalised Ranking.

Prediction rule:
    y_hat_ui = gamma_u^T gamma_i + alpha_u^T (W_vis @ f_i) + beta_i

References
----------
He, R. & McAuley, J. (2016). VBPR: Visual Bayesian Personalized Ranking
from Implicit Feedback.  AAAI.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from src.recommenders.base import BaseRecommender


class VBPR(BaseRecommender):
    """VBPR with a linear visual projection.

    Parameters
    ----------
    n_users, n_items:
        Vocabulary sizes.
    visual_embeddings:
        Pre-extracted visual features of shape ``(n_items, D_v)``.
    config:
        Must contain ``latent_dim`` (k) and ``visual_dim`` (k_v).
        ``l2_reg`` is optional (default 0).
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        visual_embeddings: np.ndarray | None = None,
        config: dict | None = None,
    ) -> None:
        config = config or {}
        super().__init__(n_users, n_items, visual_embeddings, config)

        k: int = config["latent_dim"]
        kv: int = config["visual_dim"]

        assert self.visual_features is not None, "VBPR requires visual embeddings"
        dv: int = self.visual_dim_raw

        self.user_embedding = nn.Embedding(n_users, k)
        self.item_embedding = nn.Embedding(n_items, k)
        self.item_bias = nn.Embedding(n_items, 1)

        self.visual_user_embedding = nn.Embedding(n_users, kv)  # alpha_u
        self.visual_projection = nn.Linear(dv, kv, bias=False)  # W_vis

        self._init_embedding(self.user_embedding)
        self._init_embedding(self.item_embedding)
        self._init_embedding(self.visual_user_embedding)
        nn.init.zeros_(self.item_bias.weight)
        nn.init.xavier_uniform_(self.visual_projection.weight)

        self._item_proj_cache: torch.Tensor | None = None

    def train(self, mode: bool = True) -> VBPR:
        self._item_proj_cache = None
        return super().train(mode)

    def _visual_item(self, item_ids: torch.Tensor) -> torch.Tensor:
        """Project raw visual features for the given items: W_vis @ f_i.

        With an online fusion (3-D buffer) the cache is bypassed since
        the gate's output depends on trainable parameters and changes
        every optimisation step.
        """
        cache_eligible = self._online_fusion is None
        if (
            cache_eligible
            and self._item_proj_cache is not None
            and item_ids.shape[0] == self.n_items
        ):
            return self._item_proj_cache
        f_i = self._resolve_visual(item_ids)
        proj = self.visual_projection(f_i)
        if cache_eligible and item_ids.shape[0] == self.n_items:
            self._item_proj_cache = proj
        return proj

    def forward(
        self,
        user_ids: torch.Tensor,
        pos_item_ids: torch.Tensor,
        neg_item_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        gamma_u = self.user_embedding(user_ids)
        alpha_u = self.visual_user_embedding(user_ids)

        # Combine pos and neg item lookups into a single (2B,)-batched
        # forward through item_embedding, item_bias and visual_projection.
        # Each row is independent, so splitting after the matmul produces
        # the same result as two separate B-sized passes while letting the
        # GPU amortise launch overhead over a larger matmul.
        B = pos_item_ids.shape[0]
        all_items = torch.cat([pos_item_ids, neg_item_ids], dim=0)
        gamma_all = self.item_embedding(all_items)
        beta_all = self.item_bias(all_items).squeeze(-1)
        theta_all = self._visual_item(all_items)

        gamma_pos, gamma_neg = gamma_all[:B], gamma_all[B:]
        beta_pos, beta_neg = beta_all[:B], beta_all[B:]
        theta_pos, theta_neg = theta_all[:B], theta_all[B:]

        score_pos = (gamma_u * gamma_pos).sum(-1) + (alpha_u * theta_pos).sum(-1) + beta_pos
        score_neg = (gamma_u * gamma_neg).sum(-1) + (alpha_u * theta_neg).sum(-1) + beta_neg
        return score_pos, score_neg

    def predict(self, user_id: int, item_ids: torch.Tensor) -> torch.Tensor:
        gamma_u = self.user_embedding.weight[user_id]
        gamma_i = self.item_embedding(item_ids)
        beta_i = self.item_bias(item_ids).squeeze(-1)
        alpha_u = self.visual_user_embedding.weight[user_id]
        theta_i = self._visual_item(item_ids)

        return (gamma_u * gamma_i).sum(-1) + (alpha_u * theta_i).sum(-1) + beta_i

    def predict_batch(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        gamma_u = self.user_embedding(user_ids)
        gamma_i = self.item_embedding(item_ids)
        beta_i = self.item_bias(item_ids).squeeze(-1)
        alpha_u = self.visual_user_embedding(user_ids)
        theta_i = self._visual_item(item_ids)

        return gamma_u @ gamma_i.T + alpha_u @ theta_i.T + beta_i.unsqueeze(0)
