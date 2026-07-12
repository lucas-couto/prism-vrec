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

from src.recommenders.base import BaseRecommender


class AVBPR(BaseRecommender):
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

    def train(self, mode: bool = True) -> AVBPR:
        self._item_proj_cache = None
        return super().train(mode)

    def _attended_visual(self, item_ids: torch.Tensor) -> torch.Tensor:
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

    def forward(
        self,
        user_ids: torch.Tensor,
        pos_item_ids: torch.Tensor,
        neg_item_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        gamma_u = self.user_embedding(user_ids)
        alpha_u = self.visual_user_embedding(user_ids)

        # Combine pos and neg item lookups into a single (2B,)-batched
        # forward.  attention_net softmax applies per-row (dim=-1), so
        # concatenating along dim 0 is mathematically equivalent to two
        # independent B-sized calls while reducing kernel launches.
        B = pos_item_ids.shape[0]
        all_items = torch.cat([pos_item_ids, neg_item_ids], dim=0)
        gamma_all = self.item_embedding(all_items)
        beta_all = self.item_bias(all_items).squeeze(-1)
        theta_all = self._attended_visual(all_items)

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
        theta_i = self._attended_visual(item_ids)

        return (gamma_u * gamma_i).sum(-1) + (alpha_u * theta_i).sum(-1) + beta_i

    def predict_batch(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        gamma_u = self.user_embedding(user_ids)
        gamma_i = self.item_embedding(item_ids)
        beta_i = self.item_bias(item_ids).squeeze(-1)
        alpha_u = self.visual_user_embedding(user_ids)
        theta_i = self._attended_visual(item_ids)

        return gamma_u @ gamma_i.T + alpha_u @ theta_i.T + beta_i.unsqueeze(0)
