"""AVBPR -- Attentional Visual BPR.

Prediction rule:
    theta_i   = W_vis @ f_i
    a_i       = softmax(MLP_att(theta_i))
    theta_hat = theta_i * a_i                (element-wise)
    y_hat_ui  = gamma_u^T gamma_i + alpha_u^T theta_hat + beta_i

MLP_att is a small attention network that produces per-dimension
importance weights over the projected visual features.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from src.recommenders._scoring import LinearVisualScoreMixin
from src.recommenders.base import BaseRecommender


class AVBPR(LinearVisualScoreMixin, BaseRecommender):
    """VBPR extended with a learned attention mechanism over the visual
    projection.

    Parameters
    ----------
    n_users, n_items:
        Vocabulary sizes.
    visual_embeddings:
        Pre-extracted visual features of shape ``(n_items, D_v)``.
    config:
        Must contain ``latent_dim`` (k) and ``att_hidden`` (int, hidden
        size of the attention MLP).
        ``visual_dim`` defaults to ``latent_dim`` if not supplied.
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
        kv: int = config.get("visual_dim", k)
        att_hidden: int = config["att_hidden"]

        if self.visual_features is None:
            raise RuntimeError("AVBPR requires visual embeddings")
        dv: int = self.visual_dim_raw

        self.user_embedding = nn.Embedding(n_users, k)
        self.item_embedding = nn.Embedding(n_items, k)
        self.item_bias = nn.Embedding(n_items, 1)

        self.visual_user_embedding = nn.Embedding(n_users, kv)  # alpha_u
        self.visual_projection = nn.Linear(dv, kv, bias=False)  # W_vis

        # softmax is applied in _attended_visual below
        self.attention_net = nn.Sequential(
            nn.Linear(kv, att_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(att_hidden, kv),
        )

        self._init_embedding(self.user_embedding)
        self._init_embedding(self.item_embedding)
        self._init_embedding(self.visual_user_embedding)
        nn.init.zeros_(self.item_bias.weight)
        nn.init.xavier_uniform_(self.visual_projection.weight)
        for module in self.attention_net:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

        self._item_proj_cache: torch.Tensor | None = None

    def _visual_user_table(self) -> nn.Embedding:
        return self.visual_user_embedding

    def _item_visual_term(self, item_ids: torch.Tensor) -> torch.Tensor:
        """Compute attention-weighted visual embedding for items.

        Returns theta_hat = theta_i * softmax(MLP_att(theta_i)).

        With an online fusion (3-D buffer) the cache is bypassed since
        the gate's output depends on trainable parameters.
        """
        cache_eligible = self._online_fusion is None
        if (
            cache_eligible
            and self._item_proj_cache is not None
            and item_ids.shape[0] == self.n_items
        ):
            return self._item_proj_cache
        f_i = self._resolve_visual(item_ids)
        theta_i = self.visual_projection(f_i)
        a_i = torch.softmax(self.attention_net(theta_i), dim=-1)
        proj = theta_i * a_i
        if cache_eligible and item_ids.shape[0] == self.n_items:
            self._item_proj_cache = proj
        return proj
