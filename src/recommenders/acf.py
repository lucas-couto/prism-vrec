"""ACF -- Attentive Collaborative Filtering (Chen et al., SIGIR 2017).

Full two-level attention, adapted to the framework's BPR-pairwise
protocol (like VBPR/AVBPR):

    c_{l,m}  = W_c f_{l,m}                      (component projection)
    x_l      = component_attention(gamma_u, c)  (component-level attention)
    v_l      = W_v x_l                          (visual -> latent space)
    p_hat_u  = gamma_u + Σ_{i∈R(u)} a_{u,i} (p_i + v_i)   (item-level attention)
    y_hat_ul = p_hat_u · (gamma_l + v_l) + beta_l

The user history ``R(u)`` is built from training interactions only, so
validation/test items never enter the profile (no leakage). Faithful to
the paper, the sampled BPR positive remains in ``R(u)`` during training.

Reference
---------
Chen, J. et al. (2017). Attentive Collaborative Filtering: Multimedia
Recommendation with Item- and Component-Level Attention. SIGIR.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from src.recommenders.acf_attention import ComponentAttention, ItemAttention
from src.recommenders.base import BaseRecommender


class ACF(BaseRecommender):
    """Attentive Collaborative Filtering with component- and item-level attention.

    Consumes per-item *component* embeddings of shape ``(n_items, M, D)``
    (the ``*_comp`` artifacts) and the user's training history.

    Config keys: ``latent_dim`` (k), ``att_hidden`` (attention hidden
    size), ``visual_dim`` (kv, defaults to k), ``max_history`` (H, default
    50), and optional ``l2_reg``.
    """

    consumes_raw_components = True
    wants_history = True

    def __init__(
        self,
        n_users: int,
        n_items: int,
        visual_embeddings: np.ndarray | None = None,
        config: dict | None = None,
        *,
        train_interactions: dict[int, set[int]] | None = None,
    ) -> None:
        config = config or {}
        super().__init__(
            n_users, n_items, visual_embeddings, config, train_interactions=train_interactions
        )

        if self.visual_features is None or self.visual_features.dim() != 3:
            raise RuntimeError("ACF requires 3-D component embeddings (n_items, M, D).")
        if train_interactions is None:
            raise RuntimeError("ACF requires train_interactions to build the user history.")

        k: int = config["latent_dim"]
        kv: int = config.get("visual_dim", k)
        att_hidden: int = config["att_hidden"]
        self.max_history = int(config.get("max_history", 50))
        self.n_components = int(self.visual_features.shape[1])
        dv: int = self.visual_dim_raw

        self.user_embedding = nn.Embedding(n_users, k)
        self.item_embedding = nn.Embedding(n_items, k)  # q (gamma)
        self.aux_embedding = nn.Embedding(n_items, k)  # p
        self.item_bias = nn.Embedding(n_items, 1)
        self.comp_projection = nn.Linear(dv, kv, bias=False)  # W_c
        self.visual_to_latent = nn.Linear(kv, k, bias=False)  # W_v
        self.component_attention = ComponentAttention(k, kv, att_hidden)
        self.item_attention = ItemAttention(k, att_hidden)

        self._init_embedding(self.user_embedding)
        self._init_embedding(self.item_embedding)
        self._init_embedding(self.aux_embedding)
        nn.init.zeros_(self.item_bias.weight)
        nn.init.xavier_uniform_(self.comp_projection.weight)
        nn.init.xavier_uniform_(self.visual_to_latent.weight)

        self._build_history(train_interactions)
        self._comp_cache: torch.Tensor | None = None
        self._comp_hidden_cache: torch.Tensor | None = None

    def train(self, mode: bool = True) -> ACF:
        self._comp_cache = None
        self._comp_hidden_cache = None
        return super().train(mode)

    def _build_history(self, interactions: dict[int, set[int]]) -> None:
        """Materialise padded ``(n_users, H)`` history buffers (train-only)."""
        items = torch.zeros(self.n_users, self.max_history, dtype=torch.long)
        mask = torch.zeros(self.n_users, self.max_history, dtype=torch.bool)
        for user, item_set in interactions.items():
            if user < 0 or user >= self.n_users or not item_set:
                continue
            chosen = sorted(item_set)[: self.max_history]  # deterministic truncation
            length = len(chosen)
            items[user, :length] = torch.tensor(chosen, dtype=torch.long)
            mask[user, :length] = True
        self.register_buffer("history_items", items)
        self.register_buffer("history_mask", mask)

    def _projected_components(self, item_ids: torch.Tensor) -> torch.Tensor:
        """Return ``W_c f`` for items: ``(B, M, kv)``. Cached for all-items lookups."""
        all_items = item_ids.shape[0] == self.n_items
        if all_items and self._comp_cache is not None:
            return self._comp_cache
        projected = self.comp_projection(self.visual_features[item_ids])
        if all_items:
            self._comp_cache = projected
        return projected

    def _visual_latent(self, components: torch.Tensor, gamma_u: torch.Tensor) -> torch.Tensor:
        """Component-attend then map to latent space: ``(..., k)``."""
        attended = self.component_attention(gamma_u, components)
        return self.visual_to_latent(attended)

    def _augmented_user(self, user_ids: torch.Tensor, gamma_u: torch.Tensor) -> torch.Tensor:
        """Build ``p_hat_u`` from the user's history: ``(B, k)``."""
        hist = self.history_items[user_ids]  # (B, H)
        mask = self.history_mask[user_ids]  # (B, H)
        batch, horizon = hist.shape
        comps = self._projected_components(hist.reshape(-1)).reshape(
            batch, horizon, self.n_components, -1
        )
        gamma_h = self.item_embedding(hist)  # (B, H, k)
        p_h = self.aux_embedding(hist)  # (B, H, k)
        gu_expanded = gamma_u.unsqueeze(1).expand(-1, horizon, -1)  # (B, H, k)
        v_h = self._visual_latent(comps, gu_expanded)  # (B, H, k)
        return gamma_u + self.item_attention(gamma_u, gamma_h, p_h, v_h, mask)

    def _item_rep(self, item_ids: torch.Tensor, gamma_u: torch.Tensor) -> torch.Tensor:
        """Item representation ``gamma_l + v_l`` conditioned on the user: ``(B, k)``."""
        comps = self._projected_components(item_ids)
        v_l = self._visual_latent(comps, gamma_u)
        return self.item_embedding(item_ids) + v_l

    def forward(
        self,
        user_ids: torch.Tensor,
        pos_item_ids: torch.Tensor,
        neg_item_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        gamma_u = self.user_embedding(user_ids)
        p_hat = self._augmented_user(user_ids, gamma_u)

        r_pos = self._item_rep(pos_item_ids, gamma_u)
        r_neg = self._item_rep(neg_item_ids, gamma_u)
        beta_pos = self.item_bias(pos_item_ids).squeeze(-1)
        beta_neg = self.item_bias(neg_item_ids).squeeze(-1)

        score_pos = (p_hat * r_pos).sum(-1) + beta_pos
        score_neg = (p_hat * r_neg).sum(-1) + beta_neg
        return score_pos, score_neg

    def predict(self, user_id: int, item_ids: torch.Tensor) -> torch.Tensor:
        uid = torch.tensor([user_id], device=item_ids.device)
        gamma_u = self.user_embedding(uid)  # (1, k)
        p_hat = self._augmented_user(uid, gamma_u)  # (1, k)

        gu_expanded = gamma_u.expand(item_ids.shape[0], -1)  # (N, k)
        r_l = self._item_rep(item_ids, gu_expanded)  # (N, k)
        beta_l = self.item_bias(item_ids).squeeze(-1)
        return (p_hat * r_l).sum(-1) + beta_l

    #: Element budget per evaluation tile (users × items × M × hidden).
    #: A deterministic constant (no device query) — chunking only bounds
    #: peak memory, it never changes the scores.  2**27 fp32 ≈ 0.5 GB for
    #: the dominant ``relu(query + comp_hidden)`` intermediate.
    _EVAL_TILE_ELEMENTS = 2**27

    def predict_batch(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        """Score every (user, item) pair, tiled over users × items.

        Replaces the former per-user Python loop (one GPU→CPU sync per
        user, ~20k iterations on Tradesy).  The math per user is
        identical: the user profile ``p_hat`` is computed once per user,
        and the user-independent attention term ``comp_proj(components)``
        is computed once per evaluation (cached, invalidated by
        ``train()``) instead of once per user.  Tiling bounds the
        ``(U_c, N_c, M, hidden)`` attention intermediate.
        """
        n_users_b = user_ids.shape[0]
        n_items_b = item_ids.shape[0]

        gamma_u = self.user_embedding(user_ids)  # (B, k)
        p_hat = self._augmented_user(user_ids, gamma_u)  # (B, k)

        comps = self._projected_components(item_ids)  # (N, M, kv)
        all_items = item_ids.shape[0] == self.n_items
        if all_items and self._comp_hidden_cache is not None:
            comp_hidden = self._comp_hidden_cache
        else:
            comp_hidden = self.component_attention.precompute_components(comps)
            if all_items:
                self._comp_hidden_cache = comp_hidden

        gamma_items = self.item_embedding(item_ids)  # (N, k)
        beta = self.item_bias(item_ids).squeeze(-1)  # (N,)
        queries = self.component_attention.user_proj(gamma_u)  # (B, hidden)

        hidden = comp_hidden.shape[-1]
        users_per_tile = min(n_users_b, 16)
        items_per_tile = max(
            1, self._EVAL_TILE_ELEMENTS // (users_per_tile * self.n_components * hidden)
        )

        out = torch.empty(n_users_b, n_items_b, device=gamma_u.device, dtype=p_hat.dtype)
        for u_start in range(0, n_users_b, users_per_tile):
            u_end = min(u_start + users_per_tile, n_users_b)
            query_tile = queries[u_start:u_end].unsqueeze(1).unsqueeze(2)  # (Uc,1,1,h)
            p_hat_tile = p_hat[u_start:u_end].unsqueeze(1)  # (Uc,1,k)
            for i_start in range(0, n_items_b, items_per_tile):
                i_end = min(i_start + items_per_tile, n_items_b)
                ch = comp_hidden[i_start:i_end].unsqueeze(0)  # (1,Nc,M,h)
                energy = self.component_attention.score(torch.relu(query_tile + ch))  # (Uc,Nc,M,1)
                alpha = torch.softmax(energy, dim=-2)
                attended = (alpha * comps[i_start:i_end].unsqueeze(0)).sum(dim=-2)  # (Uc,Nc,kv)
                v_l = self.visual_to_latent(attended)  # (Uc,Nc,k)
                r_l = gamma_items[i_start:i_end].unsqueeze(0) + v_l
                out[u_start:u_end, i_start:i_end] = (p_hat_tile * r_l).sum(-1) + beta[
                    i_start:i_end
                ].unsqueeze(0)
        return out
