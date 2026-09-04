"""ACF -- Attentive Collaborative Filtering (Chen et al., SIGIR 2017).

Two-level attention, adapted to the framework's BPR-pairwise protocol.
Notation follows the paper (``u_i`` user, ``v_j`` item, ``p_l``
auxiliary item, ``x̄_l`` component-attended visual of item ``l``):

    Eq. 6   R̂_ij = (u_i + Σ_{l∈R(i)} α(i,l) p_l)^T v_j
    Eq. 8   a(i,l) = w_1^T φ(W_1u u_i + W_1v v_l + W_1p p_l + W_1x x̄_l + b_1) + c_1
    Eq. 10-12  x̄_l = Σ_m β(i,l,m) x_{lm}    (component-level attention)
    Eq. 5   L = Σ_(i,j,k) -ln σ(R̂_ij - R̂_ik) + λ(‖U‖² + ‖V‖² + ‖P‖²)

The score carries NO visual term and NO item bias: the visual ``x̄_l``
of the *history* items enters only as an input of the item-level
attention energy (Eq. 8), and the candidate item is represented by its
latent ``v_j`` alone.  Only ``U``, ``V`` and ``P`` are penalised; the
attention networks and the visual projections ``Θ`` are not (Eq. 5).

With uniform attention ``α(i,l) = 1/|R(i)|`` the model degenerates to
SVD++ (Section 4.1 of the paper) — a property the tests exercise.

The user history ``R(i)`` is built from training interactions only, so
validation/test items never enter the profile (no leakage).  Faithful to
the paper, the sampled BPR positive remains in ``R(i)`` during training.
The paper uses the *complete* ``R(i)``; bounding it by ``max_history``
is an implementation decision (memory), and when a history exceeds the
bound the kept items are a uniform random sample without replacement
seeded from ``history_seed`` — never an ordered prefix, because item
ids correlate with popularity in the DVBPR splits.

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
    size), ``visual_dim`` (kv, defaults to k), ``max_history`` (H: an int
    bound, or ``None`` for the complete history as in the paper; default
    50), ``history_seed`` (seed of the uniform sub-sampling applied when
    ``|R(u)| > H``; default 42) and ``l2_reg`` (``λ`` of Eq. 5).
    """

    consumes_raw_components = True
    wants_history = True

    #: BPR-Opt L2 (Eq. 5): ``U``/``V`` rows come from the base tables and
    #: ``P`` rows (plus the ``V`` rows of the history) are gathered in
    #: :meth:`_l2_gathered_terms`; listing ``aux_embedding`` here keeps
    #: its full matrix out of the shared term.  ``Θ`` — both attention
    #: nets and the visual projections — is left unpenalised.
    _L2_EXTRA_GATHERED_TABLES = ("aux_embedding",)
    _L2_UNREGULARIZED = (
        "component_attention",
        "item_attention",
        "comp_projection",
        "visual_to_latent",
    )

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
        raw_max = config.get("max_history", 50)
        self.max_history: int | None = None if raw_max is None else int(raw_max)
        self.history_seed = int(config.get("history_seed", 42))
        self.n_components = int(self.visual_features.shape[1])
        dv: int = self.visual_dim_raw

        self.user_embedding = nn.Embedding(n_users, k)  # U
        self.item_embedding = nn.Embedding(n_items, k)  # V
        self.aux_embedding = nn.Embedding(n_items, k)  # P
        self.comp_projection = nn.Linear(dv, kv, bias=False)  # W_c
        self.visual_to_latent = nn.Linear(kv, k, bias=False)  # W_v
        self.component_attention = ComponentAttention(k, kv, att_hidden)
        self.item_attention = ItemAttention(k, att_hidden)

        self._init_embedding(self.user_embedding)
        self._init_embedding(self.item_embedding)
        self._init_embedding(self.aux_embedding)
        nn.init.xavier_uniform_(self.comp_projection.weight)
        nn.init.xavier_uniform_(self.visual_to_latent.weight)

        self._build_history(train_interactions)
        self._comp_cache: torch.Tensor | None = None

    def train(self, mode: bool = True) -> ACF:
        self._comp_cache = None
        return super().train(mode)

    # ------------------------------------------------------------------ history
    def _build_history(self, interactions: dict[int, set[int]]) -> None:
        """Materialise padded ``(n_users, H)`` history buffers (train-only).

        ``H`` is ``max_history`` or, when ``None``, the longest training
        history in the dataset.  Users exceeding ``H`` keep a uniform
        sample of ``H`` items (see :meth:`_select_history`).
        """
        valid = {u: s for u, s in interactions.items() if 0 <= u < self.n_users and s}
        longest = max((len(s) for s in valid.values()), default=0)
        horizon = longest if self.max_history is None else self.max_history
        horizon = max(int(horizon), 1)
        items = torch.zeros(self.n_users, horizon, dtype=torch.long)
        mask = torch.zeros(self.n_users, horizon, dtype=torch.bool)
        for user, item_set in valid.items():
            chosen = self._select_history(user, sorted(item_set), horizon)
            items[user, : len(chosen)] = torch.tensor(chosen, dtype=torch.long)
            mask[user, : len(chosen)] = True
        self.register_buffer("history_items", items, persistent=False)
        self.register_buffer("history_mask", mask, persistent=False)

    def rebuild_user_state(self, interactions: dict[int, set[int]]) -> None:
        """Rewrite the history rows of the users in ``interactions`` only.

        Fold-in hook: ``R(u)`` is non-parametric state, so a folded-in
        user's profile must be written into the existing
        ``history_items`` / ``history_mask`` buffers in place (the
        horizon ``H`` is fixed at construction; a profile longer than
        ``H`` is sub-sampled by the same seeded rule as
        :meth:`_select_history`).  Rows of every other user are left
        untouched.  Users outside ``[0, n_users)`` are ignored, as in
        :meth:`_build_history`; a user with an empty set gets an empty
        history.  Invalidates the eval-mode component cache.
        """
        horizon = int(self.history_items.shape[1])
        device = self.history_items.device
        for user, item_set in interactions.items():
            if not 0 <= user < self.n_users:
                continue
            chosen = self._select_history(user, sorted(item_set), horizon)
            self.history_items[user] = 0
            self.history_mask[user] = False
            if chosen:
                self.history_items[user, : len(chosen)] = torch.tensor(
                    chosen, dtype=torch.long, device=device
                )
                self.history_mask[user, : len(chosen)] = True
        self._comp_cache = None

    def _select_history(self, user: int, pool: list[int], horizon: int) -> list[int]:
        """Uniform sample without replacement of ``horizon`` items when needed.

        The generator is seeded per user from ``(history_seed, user)`` so
        the selection is reproducible and independent of dict ordering.
        Slot order is irrelevant to the attention.
        """
        if len(pool) <= horizon:
            return pool
        rng = np.random.default_rng([self.history_seed, user])
        return rng.choice(np.asarray(pool, dtype=np.int64), size=horizon, replace=False).tolist()

    # ------------------------------------------------------------- regularisation
    def _l2_gathered_terms(
        self,
        user_ids: torch.Tensor,
        pos_item_ids: torch.Tensor,
        neg_item_ids: torch.Tensor,
    ) -> list[tuple[str, torch.Tensor]]:
        """Eq. 5 rows touched by the batch: ``U`` (user), ``V`` (pos/neg) and,
        through each user's ``R(u)``, the ``V``/``P`` rows of the history.

        History rows are penalised per occurrence across the batch's
        users, like every gathered row.  All groups share ``λ = l2_reg``.
        """
        terms = super()._l2_gathered_terms(user_ids, pos_item_ids, neg_item_ids)
        touched = self.history_items[user_ids][self.history_mask[user_ids]]
        terms.append(("l2_reg", self.item_embedding(touched)))
        terms.append(("l2_reg", self.aux_embedding(touched)))
        return terms

    # -------------------------------------------------------------------- scoring
    def _projected_components(self, item_ids: torch.Tensor) -> torch.Tensor:
        """``W_c f`` for the given items: ``(..., M, kv)``.

        In eval mode the catalogue projection is computed once and
        cached (invalidated by :meth:`train`) so successive history
        lookups only index it.
        """
        if self.training:
            return self.comp_projection(self.visual_features[item_ids])
        if self._comp_cache is None:
            self._comp_cache = self.comp_projection(self.visual_features)
        return self._comp_cache[item_ids]

    def _augmented_user(self, user_ids: torch.Tensor, gamma_u: torch.Tensor) -> torch.Tensor:
        """Eq. 6 profile ``p̂_u = u + Σ_{l∈R(u)} α(u,l) p_l``: ``(B, k)``."""
        hist = self.history_items[user_ids]  # (B, H)
        mask = self.history_mask[user_ids]  # (B, H)
        horizon = hist.shape[1]
        comps = self._projected_components(hist)  # (B, H, M, kv)
        gamma_h = self.item_embedding(hist)  # (B, H, k)   v_l
        p_h = self.aux_embedding(hist)  # (B, H, k)   p_l
        gu_expanded = gamma_u.unsqueeze(1).expand(-1, horizon, -1)  # (B, H, k)
        v_h = self.visual_to_latent(self.component_attention(gu_expanded, comps))  # x̄_l
        return gamma_u + self.item_attention(gamma_u, gamma_h, p_h, v_h, mask)

    def forward(
        self,
        user_ids: torch.Tensor,
        pos_item_ids: torch.Tensor,
        neg_item_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        gamma_u = self.user_embedding(user_ids)
        p_hat = self._augmented_user(user_ids, gamma_u)
        score_pos = (p_hat * self.item_embedding(pos_item_ids)).sum(-1)
        score_neg = (p_hat * self.item_embedding(neg_item_ids)).sum(-1)
        return score_pos, score_neg

    def predict(self, user_id: int, item_ids: torch.Tensor) -> torch.Tensor:
        uid = torch.tensor([user_id], device=item_ids.device)
        p_hat = self._augmented_user(uid, self.user_embedding(uid))  # (1, k)
        return self.item_embedding(item_ids) @ p_hat.squeeze(0)  # (N,)

    def predict_batch(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        """Score every (user, item) pair as ``p̂_u @ V^T``: ``(B, N)``.

        Both attention levels run over the users' histories only (Eq. 6
        has no per-candidate term), so the candidate side is a single
        GEMM against the item table.
        """
        gamma_u = self.user_embedding(user_ids)  # (B, k)
        p_hat = self._augmented_user(user_ids, gamma_u)  # (B, k)
        return p_hat @ self.item_embedding(item_ids).T
