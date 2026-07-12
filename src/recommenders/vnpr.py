"""VNPR -- Visual Neural Personalised Ranking.

Prediction rule:
    v_i   = ReLU(W_v @ f_i + b_v)
    x_ui  = concat(u_u, q_i, v_i)
    y_hat = MLP(x_ui)              (scalar output)

The MLP has configurable hidden layers and uses ReLU activations.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from src.recommenders.base import BaseRecommender


def _autotune_chunk_pairs() -> int:
    """Pick the (user, item) chunk size for full-ranking eval from VRAM.

    Bounded peaks for an 8 GB / 16 GB / 24+ GB GPU when k=128 and
    hidden_layers up to [512, 256, 128].  CPU and small-VRAM hosts fall
    into the conservative 500_000 tier.
    """
    try:
        if not torch.cuda.is_available():
            return 500_000
        total_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    except (RuntimeError, AssertionError):
        return 500_000

    if total_gb < 12:
        return 500_000
    if total_gb < 24:
        return 2_000_000
    return 5_000_000


_PREDICT_BATCH_CHUNK_PAIRS = _autotune_chunk_pairs()


class VNPR(BaseRecommender):
    """Neural pairwise ranking with visual features.

    Parameters
    ----------
    n_users, n_items:
        Vocabulary sizes.
    visual_embeddings:
        Pre-extracted visual features of shape ``(n_items, D_v)``.
    config:
        Must contain ``latent_dim`` (embedding size for users and items),
        ``hidden_layers`` (list[int], e.g. ``[256, 128, 64]``).
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
        hidden_layers: list[int] = config["hidden_layers"]

        if self.visual_features is None:
            raise RuntimeError("VNPR requires visual embeddings")
        dv: int = self.visual_dim_raw

        self.user_embedding = nn.Embedding(n_users, k)  # u_u
        self.item_embedding = nn.Embedding(n_items, k)  # q_i

        # Visual transform: v_i = ReLU(W_v f_i + b_v)
        self.visual_transform = nn.Linear(dv, k)

        # Scoring MLP: input is concat(u_u, q_i, v_i) -> dim = 3*k
        layers: list[nn.Module] = []
        in_dim = 3 * k
        for h_dim in hidden_layers:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU(inplace=True))
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, 1))
        self.mlp = nn.Sequential(*layers)

        self._init_embedding(self.user_embedding)
        self._init_embedding(self.item_embedding)
        nn.init.xavier_uniform_(self.visual_transform.weight)
        nn.init.zeros_(self.visual_transform.bias)
        for module in self.mlp:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

        # Cache of [q_i, v_i] for the full item catalogue, populated lazily
        # during evaluation and invalidated on every train() call.
        self._item_feats_cache: torch.Tensor | None = None

    def train(self, mode: bool = True) -> VNPR:
        self._item_feats_cache = None
        return super().train(mode)

    def _item_feats(self, item_ids: torch.Tensor) -> torch.Tensor:
        """Return concat([q_i, v_i]) for the given items, with caching.

        When called with the full catalogue the result is cached and
        reused across calls until the next ``train()`` toggle.  Cache
        is bypassed when an online fusion is active because the gate
        depends on trainable parameters.
        """
        cache_eligible = self._online_fusion is None
        if (
            cache_eligible
            and self._item_feats_cache is not None
            and item_ids.shape[0] == self.n_items
        ):
            return self._item_feats_cache
        q_i = self.item_embedding(item_ids)
        f_i = self._resolve_visual(item_ids)
        v_i = torch.relu(self.visual_transform(f_i))
        feats = torch.cat([q_i, v_i], dim=-1)
        if cache_eligible and item_ids.shape[0] == self.n_items:
            self._item_feats_cache = feats
        return feats

    def _score(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        """Compute scalar scores for (user, item) pairs."""
        u_u = self.user_embedding(user_ids)
        q_i = self.item_embedding(item_ids)
        f_i = self._resolve_visual(item_ids)
        v_i = torch.relu(self.visual_transform(f_i))

        x_ui = torch.cat([u_u, q_i, v_i], dim=-1)
        return self.mlp(x_ui).squeeze(-1)

    def forward(
        self,
        user_ids: torch.Tensor,
        pos_item_ids: torch.Tensor,
        neg_item_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        score_pos = self._score(user_ids, pos_item_ids)
        score_neg = self._score(user_ids, neg_item_ids)
        return score_pos, score_neg

    def predict(self, user_id: int, item_ids: torch.Tensor) -> torch.Tensor:
        n = item_ids.shape[0]
        user_ids = torch.full((n,), user_id, dtype=torch.long, device=item_ids.device)
        return self._score(user_ids, item_ids)

    def predict_batch(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        """Score every (user, item) pair in the cartesian product.

        Vectorised across users in chunks bounded by
        :data:`_PREDICT_BATCH_CHUNK_PAIRS`.  This keeps peak VRAM bounded
        regardless of ``B * N`` while still feeding the MLP with large
        contiguous batches (orders of magnitude faster than scoring one
        user at a time).  Item-side features are computed once per call
        via :meth:`_item_feats` and reused across chunks.
        """
        B = user_ids.shape[0]
        N = item_ids.shape[0]

        item_feats = self._item_feats(item_ids)
        u_u = self.user_embedding(user_ids)

        users_per_chunk = max(1, _PREDICT_BATCH_CHUNK_PAIRS // max(N, 1))

        out = torch.empty(B, N, device=u_u.device, dtype=u_u.dtype)
        for start in range(0, B, users_per_chunk):
            end = min(start + users_per_chunk, B)
            u_chunk = u_u[start:end]
            b = u_chunk.shape[0]

            u_expanded = u_chunk.unsqueeze(1).expand(b, N, -1)
            item_expanded = item_feats.unsqueeze(0).expand(b, N, -1)
            x_ui = torch.cat([u_expanded, item_expanded], dim=-1)
            x_flat = x_ui.reshape(b * N, -1)

            scores = self.mlp(x_flat).squeeze(-1)
            out[start:end] = scores.view(b, N)
        return out
