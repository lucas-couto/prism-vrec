"""DeepStyle -- Style-aware recommendation with visual features.

Prediction rule:
    y_hat_ui = gamma_u^T gamma_i + s_u^T S(f_i) + beta_i

S(.) is a small MLP that projects the raw visual feature f_i into a
low-dimensional *style* space of dimension k_s.

References
----------
Liu, Q. et al. (2017). DeepStyle: Learning User Preferences for Visual
Recommendation.  SIGIR.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from src.recommenders.base import BaseRecommender


class DeepStyle(BaseRecommender):
    """DeepStyle with an MLP-based style projector.

    Parameters
    ----------
    n_users, n_items:
        Vocabulary sizes.
    visual_embeddings:
        Pre-extracted visual features of shape ``(n_items, D_v)``.
    config:
        Must contain ``latent_dim`` (k), ``style_dim`` (k_s).
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
        ks: int = config["style_dim"]

        assert self.visual_features is not None, "DeepStyle requires visual embeddings"
        dv: int = self.visual_dim_raw

        self.user_embedding = nn.Embedding(n_users, k)
        self.item_embedding = nn.Embedding(n_items, k)
        self.item_bias = nn.Embedding(n_items, 1)

        self.style_user_embedding = nn.Embedding(n_users, ks)  # s_u
        self.style_projector = nn.Sequential(
            nn.Linear(dv, (dv + ks) // 2),
            nn.ReLU(inplace=True),
            nn.Linear((dv + ks) // 2, ks),
        )

        self._init_embedding(self.user_embedding)
        self._init_embedding(self.item_embedding)
        self._init_embedding(self.style_user_embedding)
        nn.init.zeros_(self.item_bias.weight)
        for module in self.style_projector:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

        self._item_proj_cache: torch.Tensor | None = None

    def train(self, mode: bool = True) -> DeepStyle:
        self._item_proj_cache = None
        return super().train(mode)

    def _style_item(self, item_ids: torch.Tensor) -> torch.Tensor:
        """Project visual features into style space: S(f_i).

        Cache is bypassed when an online fusion is active because the
        gate's output depends on trainable parameters and changes
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
        proj = self.style_projector(f_i)
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
        s_u = self.style_user_embedding(user_ids)

        # Combine pos and neg item lookups into a single (2B,)-batched
        # forward through item_embedding, item_bias and the style MLP.
        # Linear and ReLU are point-wise, so splitting after the matmul
        # is mathematically equivalent to two B-sized passes while letting
        # the GPU amortise the heavy first Linear (dv -> (dv+ks)/2) over
        # one large batch.
        B = pos_item_ids.shape[0]
        all_items = torch.cat([pos_item_ids, neg_item_ids], dim=0)
        gamma_all = self.item_embedding(all_items)
        beta_all = self.item_bias(all_items).squeeze(-1)
        style_all = self._style_item(all_items)

        gamma_pos, gamma_neg = gamma_all[:B], gamma_all[B:]
        beta_pos, beta_neg = beta_all[:B], beta_all[B:]
        style_pos, style_neg = style_all[:B], style_all[B:]

        score_pos = (gamma_u * gamma_pos).sum(-1) + (s_u * style_pos).sum(-1) + beta_pos
        score_neg = (gamma_u * gamma_neg).sum(-1) + (s_u * style_neg).sum(-1) + beta_neg
        return score_pos, score_neg

    def predict(self, user_id: int, item_ids: torch.Tensor) -> torch.Tensor:
        gamma_u = self.user_embedding.weight[user_id]
        gamma_i = self.item_embedding(item_ids)
        beta_i = self.item_bias(item_ids).squeeze(-1)
        s_u = self.style_user_embedding.weight[user_id]
        style_i = self._style_item(item_ids)

        return (gamma_u * gamma_i).sum(-1) + (s_u * style_i).sum(-1) + beta_i

    def predict_batch(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        gamma_u = self.user_embedding(user_ids)
        gamma_i = self.item_embedding(item_ids)
        beta_i = self.item_bias(item_ids).squeeze(-1)
        s_u = self.style_user_embedding(user_ids)
        style_i = self._style_item(item_ids)

        return gamma_u @ gamma_i.T + s_u @ style_i.T + beta_i.unsqueeze(0)
