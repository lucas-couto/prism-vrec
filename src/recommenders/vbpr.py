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

from src.recommenders._scoring import LinearVisualScoreMixin
from src.recommenders.base import BaseRecommender


class VBPR(LinearVisualScoreMixin, BaseRecommender):
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

        if self.visual_features is None:
            raise RuntimeError("VBPR requires visual embeddings")
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

    def _visual_user_table(self) -> nn.Embedding:
        return self.visual_user_embedding

    def _item_visual_term(self, item_ids: torch.Tensor) -> torch.Tensor:
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
