"""VNPR -- Visual Neural Personalized Ranking (Niu, Caverlee & Lu, 2018).

Reference
---------
Wei Niu, James Caverlee and Haokai Lu.  *Neural Personalized Ranking
for Image Recommendation*.  Proceedings of the Eleventh ACM
International Conference on Web Search and Data Mining (WSDM '18),
Marina Del Rey, CA, USA, 2018, pp. 423-431.
https://doi.org/10.1145/3159652.3159728

This module implements the visual variant VNPR (Section 5.2) of the
NPR architecture (Sections 4.1-4.3).

Architecture (Sections 4.1 and 5.2)
-----------------------------------
NPR is a *pairwise* neural network with two mirrored branches, one for
the positive item ``i`` and one for the negative item ``j``.  The user
embedding is shared by both branches; each branch owns its own item
embedding table.  VNPR adds a second user embedding that lives in the
image-feature space so the user can be matched against the raw image
feature of each item::

    p_h  = W_u[h]      ∈ R^k     user latent factors        (shared)
    q_i  = W_i[i]      ∈ R^k     positive-branch item factors
    q'_j = W_i'[j]     ∈ R^k     negative-branch item factors
    v_h  = W_v[h]      ∈ R^dv    user *visual* factors      (shared)
    f_i               ∈ R^dv    pre-extracted image feature (frozen)

The paper transforms the USER into the dimension of the image feature
and consumes ``f_i`` as is -- there is no transformation over the
image.  Each branch merges user and item by element-wise product and
concatenates the two products::

    m'_hi = [ p_h ∘ q_i , v_h ∘ f_i ]                 ∈ R^{k+dv}      (Eq. 9)
    r_hi  = ReLU( w^T m'_hi + b )                     one-neuron dense (Eq. 3)

    r_hj  = ReLU( w^T [ p_h ∘ q'_j , v_h ∘ f_j ] + b )  mirrored branch

The dense layer ``(w, b)`` is shared by the two branches.  Dropout is
applied to the embedding vectors ``p_h``, ``q_i``, ``q'_j`` and ``v_h``
during training only.

Objective (Section 4.2)
-----------------------
The network is trained on the pairwise probability that user ``h``
prefers ``i`` over ``j``, ``σ(r_hi - r_hj)``, with the negative
log-likelihood plus an L2 penalty over the embedding matrices::

    L = -Σ_(h,i,j) ln σ(r_hi - r_hj)
        + λ ( ‖W_u‖² + ‖W_i‖² + ‖W_i'‖² + ‖W_v‖² )

:meth:`BaseRecommender.bpr_loss` supplies ``-ln σ(r_pos - r_neg)``;
the regulariser is delegated to the group machinery of the base class
configured so that the WHOLE embedding matrices are penalised (see
below).

Inference (Section 4.3)
-----------------------
Both branches are scoring functions of the same form, so the paper
feeds ``(h, i, i)`` to the network and averages the two outputs::

    r̂_hi = ½ ( r_hi + r'_hi )

:meth:`VNPR.predict` and :meth:`VNPR.predict_batch` implement this
average.  ``predict_batch`` never materialises ``(B, N, k+dv)``: since
``w^T [p∘q, v∘f] = (p ⊙ w_q)·q + (v ⊙ w_f)·f``, each branch is two GEMMs
``(B,k)@(k,N)`` and ``(B,dv)@(dv,N)`` plus ``b`` followed by the ReLU.

Fidelity to the paper
---------------------
Faithful:

* shared user embedding + two item tables (mirrored branches);
* user visual embedding in the raw image-feature dimension, image
  feature used unchanged;
* element-wise product merge, concatenation of the products, single
  ReLU neuron shared by the branches;
* pairwise ``-ln σ(r_hi - r_hj)`` objective;
* L2 over the whole embedding matrices, dense layer unpenalised;
* dropout on the embeddings, training only;
* inference as the average of the two branches.

Declared divergences:

* **Regularisation IS BPR-Opt, not the paper's whole matrices.**  The
  paper adds ``λ(‖W_u‖² + ‖W_i‖² + ‖W_i'‖² + ‖W_v‖²)`` to every step.
  Reproduced literally under Adam (validation profile, 2026-09-04) that
  term trained the three collaborative tables to row norms of EXACTLY
  0.0 in every cell: Adam normalises the gradient magnitude, so a row
  that only receives the L2 gradient moves ≈ lr towards zero each step
  regardless of ``λ`` and is dead within the first epoch; only ``W_v``,
  fed a dense gradient by the image feature, survived, leaving a
  visual-only ``ReLU(v_h·f_i + b)``.  VNPR therefore penalises the rows
  gathered by the batch like every other recommender: ``W_u`` / ``W_v``
  by the users, ``W_i`` by the positives, ``W_i'`` by the negatives
  (see :meth:`VNPR._l2_gathered_terms`), all under the single
  ``l2_reg`` weight.  ``dense`` stays unpenalised via
  ``_L2_UNREGULARIZED``, as in the paper.  Full account in
  ``docs/protocol.md`` ("VNPR regularisation"); guarded by
  ``tests/recommenders/test_vnpr_paper.py``.  A learned online-fusion
  module, when present, is a framework-level component and keeps the
  framework's default (penalised in the shared term).
* **No learning-rate decay and no per-model patience.**  The paper
  decays the learning rate and stops after 3 epochs without
  improvement.  The early-stopping budget of this framework is shared
  by every model of a dataset (``src/recommenders/hp_budget.py``
  forbids per-model budget keys), so neither is reproduced here.
* **Dropout rate is fixed in the config**, not searched: set
  ``dropout`` (default ``0.0``) in ``configs/recommenders.yaml``.
* Image features come from the framework's extractors (possibly fused)
  instead of the paper's AlexNet features; ``dv`` follows the feature.
* Negative sampling, optimiser and initialisation (Xavier-uniform for
  the tables, zero bias) are the framework's, not the paper's.

Config
------
``latent_dim`` (required) -- ``k``; ``dropout`` (optional, default 0)
-- embedding dropout rate; ``l2_reg`` (optional, default 0) -- ``λ``.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from src.recommenders.base import BaseRecommender


class VNPR(BaseRecommender):
    """Visual Neural Personalized Ranking (see the module docstring).

    Parameters
    ----------
    n_users, n_items:
        Vocabulary sizes.
    visual_embeddings:
        Pre-extracted image features ``(n_items, dv)`` (or the 3-D /
        ragged layouts the base class resolves through an online
        fusion).  Required.
    config:
        ``latent_dim`` (required), ``dropout`` and ``l2_reg`` (optional).
    """

    # BPR-Opt regularisation (declared divergence, see the module
    # docstring): user rows of both user tables, positive rows of
    # ``W_i``, negative rows of ``W_i'`` — the latter split is done in
    # :meth:`_l2_gathered_terms`; listing both item tables here keeps
    # them out of the shared (whole-matrix) term.  Dense layer free.
    _L2_USER_TABLES: tuple[str, ...] = ("user_embedding", "visual_user_embedding")
    _L2_ITEM_TABLES: tuple[str, ...] = ("item_embedding", "item_embedding_neg")
    _L2_UNREGULARIZED: tuple[str, ...] = ("dense",)
    #: Fold-in: the user-indexed tables are ``W_u`` (``p_h``) and ``W_v``
    #: (``v_h``).
    _USER_TABLES: tuple[str, ...] = ("user_embedding", "visual_user_embedding")

    #: Beyond the returned score matrix, :meth:`predict_batch` holds the
    #: shared visual term and one branch's GEMM live -- two fp32
    #: ``(B, N)`` buffers, 8 bytes per pair.
    PREDICT_BATCH_BYTES_PER_ELEMENT: int = 8

    def __init__(
        self,
        n_users: int,
        n_items: int,
        visual_embeddings: np.ndarray | None = None,
        config: dict | None = None,
    ) -> None:
        config = config or {}
        super().__init__(n_users, n_items, visual_embeddings, config)

        if not self.has_visual_features:
            raise RuntimeError("VNPR requires visual embeddings")

        k: int = config["latent_dim"]
        dv: int = self.visual_dim_raw
        self.latent_dim = k

        self.user_embedding = nn.Embedding(n_users, k)  # W_u  -> p_h
        self.item_embedding = nn.Embedding(n_items, k)  # W_i  -> q_i  (positive branch)
        self.item_embedding_neg = nn.Embedding(n_items, k)  # W_i' -> q'_j (negative branch)
        self.visual_user_embedding = nn.Embedding(n_users, dv)  # W_v  -> v_h
        self.dense = nn.Linear(k + dv, 1)  # r = ReLU(w^T m' + b)
        self.dropout = nn.Dropout(float(config.get("dropout", 0.0)))

        for table in (
            self.user_embedding,
            self.item_embedding,
            self.item_embedding_neg,
            self.visual_user_embedding,
        ):
            self._init_embedding(table)
        nn.init.xavier_uniform_(self.dense.weight)
        nn.init.zeros_(self.dense.bias)

        # Full-catalogue image features: in eval only (no online fusion)
        # a zero-copy alias of the buffer, invalidated by every train()
        # call (see :meth:`_catalogue_visual`).
        self._catalogue_visual_cache: torch.Tensor | None = None

    def train(self, mode: bool = True) -> VNPR:
        self._catalogue_visual_cache = None
        return super().train(mode)

    def _l2_gathered_terms(
        self,
        user_ids: torch.Tensor,
        pos_item_ids: torch.Tensor,
        neg_item_ids: torch.Tensor,
    ) -> list[tuple[str, torch.Tensor]]:
        """BPR-Opt rows of the mirrored branches.

        ``W_i`` only ever scores the positive branch and ``W_i'`` the
        negative one, so each table is gathered by its own side of the
        triple instead of the base class's pos+neg rows for every table.
        """
        return [
            (self._l2_lambda_key("user_embedding", "user"), self.user_embedding(user_ids)),
            (
                self._l2_lambda_key("visual_user_embedding", "user"),
                self.visual_user_embedding(user_ids),
            ),
            (self._l2_lambda_key("item_embedding", "pos"), self.item_embedding(pos_item_ids)),
            (
                self._l2_lambda_key("item_embedding_neg", "neg"),
                self.item_embedding_neg(neg_item_ids),
            ),
        ]

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------
    def _branch(
        self,
        p: torch.Tensor,
        q: torch.Tensor,
        v: torch.Tensor,
        f: torch.Tensor,
    ) -> torch.Tensor:
        """``ReLU(w^T [p ∘ q, v ∘ f] + b)`` for aligned ``(B, ·)`` tensors."""
        return torch.relu(self._preactivation(p, q, v, f))

    def _preactivation(
        self,
        p: torch.Tensor,
        q: torch.Tensor,
        v: torch.Tensor,
        f: torch.Tensor,
    ) -> torch.Tensor:
        """``w^T [p ∘ q, v ∘ f] + b`` — the branch before its ReLU (Eq. 3)."""
        merged = torch.cat([p * q, v * f], dim=-1)
        return self.dense(merged).squeeze(-1)

    def diagnostic_branches(
        self,
        user_ids: torch.Tensor,
        pos_item_ids: torch.Tensor,
        neg_item_ids: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Pre-ReLU branch values of :meth:`forward` (SDD S04 diagnostics).

        Same computation as :meth:`forward` up to the ReLU — dropout
        follows the module's current mode, so the training view is
        obtained in ``train()`` and the inference view ``(u, i, i)`` in
        ``eval()``.  Read-only: no parameter, cache or buffer changes.
        """
        p = self.dropout(self.user_embedding(user_ids))
        v = self.dropout(self.visual_user_embedding(user_ids))
        q_pos = self.dropout(self.item_embedding(pos_item_ids))
        q_neg = self.dropout(self.item_embedding_neg(neg_item_ids))
        f_pos = self._resolve_visual(pos_item_ids)
        f_neg = self._resolve_visual(neg_item_ids)
        return {
            "pos": self._preactivation(p, q_pos, v, f_pos),
            "neg": self._preactivation(p, q_neg, v, f_neg),
        }

    def forward(
        self,
        user_ids: torch.Tensor,
        pos_item_ids: torch.Tensor,
        neg_item_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``(r_pos, r_neg)``: positive branch on ``W_i``, negative on ``W_i'``."""
        p = self.dropout(self.user_embedding(user_ids))
        v = self.dropout(self.visual_user_embedding(user_ids))
        q_pos = self.dropout(self.item_embedding(pos_item_ids))
        q_neg = self.dropout(self.item_embedding_neg(neg_item_ids))
        f_pos = self._resolve_visual(pos_item_ids)
        f_neg = self._resolve_visual(neg_item_ids)
        return self._branch(p, q_pos, v, f_pos), self._branch(p, q_neg, v, f_neg)

    def predict(self, user_id: int, item_ids: torch.Tensor) -> torch.Tensor:
        """``½ (r_ui + r'_ui)`` -- both branches fed ``(u, i, i)`` (Section 4.3)."""
        n = item_ids.shape[0]
        user_ids = torch.full((n,), user_id, dtype=torch.long, device=item_ids.device)
        p = self.user_embedding(user_ids)
        v = self.visual_user_embedding(user_ids)
        f = self._resolve_visual(item_ids)
        r_pos = self._branch(p, self.item_embedding(item_ids), v, f)
        r_neg = self._branch(p, self.item_embedding_neg(item_ids), v, f)
        return 0.5 * (r_pos + r_neg)

    def predict_batch(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        """Score the cartesian product ``(B, N)`` without a ``(B, N, k+dv)`` tensor.

        ``w^T [p∘q, v∘f] = (p ⊙ w_q)·q + (v ⊙ w_f)·f``: the visual term
        and the bias are shared by the two branches, each branch adds
        its own ``(B,k)@(k,N)`` item GEMM before the ReLU, and the two
        are averaged.  Mathematically identical to :meth:`predict`
        (float reductions are reordered, so not bit-identical).

        Exactly three ``(B, N)`` buffers are live at the peak -- the
        shared visual term, the accumulating result and one branch's
        GEMM -- which is what
        :attr:`PREDICT_BATCH_BYTES_PER_ELEMENT` declares to the
        evaluator's batch planner.  The fused ``addmm`` / in-place ReLU
        below is what keeps it at three: the naive expression form holds
        five, and the planner sizing a VNPR batch as if it held one is
        how a full-ranking pass over a 326K-item catalogue OOM-ed.
        """
        k = self.latent_dim
        w = self.dense.weight.squeeze(0)
        w_q, w_f = w[:k], w[k:]

        p = self.user_embedding(user_ids) * w_q  # (B, k)
        v = self.visual_user_embedding(user_ids) * w_f  # (B, dv)
        full = self._is_full_catalogue(item_ids)
        if self.is_lazy_visual:
            return self._predict_batch_blocked(p, v, item_ids, full)

        f = self._catalogue_visual(item_ids)  # (N, dv)
        q_pos = self._branch_items(self.item_embedding, item_ids, full)
        q_neg = self._branch_items(self.item_embedding_neg, item_ids, full)
        return self._score_block(p, v, f, q_pos, q_neg)

    def _score_block(
        self,
        p: torch.Tensor,
        v: torch.Tensor,
        f: torch.Tensor,
        q_pos: torch.Tensor,
        q_neg: torch.Tensor,
    ) -> torch.Tensor:
        """``½(ReLU(shared + p·q) + ReLU(shared + p·q'))`` for one item block."""
        # beta*input + mat1 @ mat2 in one kernel: the bias broadcasts
        # over (B, N) instead of allocating a second matrix for the sum.
        shared = torch.addmm(self.dense.bias, v, f.T)  # (B, N)

        scores = torch.addmm(shared, p, q_pos.T)
        torch.relu_(scores)
        other = torch.addmm(shared, p, q_neg.T)
        torch.relu_(other)
        scores.add_(other).mul_(0.5)
        del other
        return scores

    def _predict_batch_blocked(
        self,
        p: torch.Tensor,
        v: torch.Tensor,
        item_ids: torch.Tensor,
        full_catalogue: bool,
    ) -> torch.Tensor:
        """Lazy-source ``predict_batch``: item blocks, never ``(N, dv)`` at once.

        Every column of the ``(B, N)`` result depends on its own item
        only, so filling it block by block is the same computation with
        the raw feature staging bounded by :attr:`_LAZY_ITEM_BLOCK`
        rows instead of the catalogue.
        """
        n = item_ids.shape[0]
        out = torch.empty((p.shape[0], n), device=p.device, dtype=p.dtype)
        block = self._LAZY_ITEM_BLOCK
        for start in range(0, n, block):
            stop = min(start + block, n)
            ids = item_ids[start:stop]
            f = self._resolve_visual(ids)  # (n_blk, dv)
            if full_catalogue:
                q_pos = self.item_embedding.weight[start:stop]
                q_neg = self.item_embedding_neg.weight[start:stop]
            else:
                q_pos = self.item_embedding(ids)
                q_neg = self.item_embedding_neg(ids)
            out[:, start:stop] = self._score_block(p, v, f, q_pos, q_neg)
        return out

    def _is_full_catalogue(self, item_ids: torch.Tensor) -> bool:
        """Whether ``item_ids`` is exactly ``arange(n_items)``."""
        if item_ids.shape[0] != self.n_items:
            return False
        expected = torch.arange(self.n_items, device=item_ids.device)
        return bool(torch.equal(item_ids, expected))

    @staticmethod
    def _branch_items(
        table: nn.Embedding, item_ids: torch.Tensor, full_catalogue: bool
    ) -> torch.Tensor:
        """``q`` rows for *item_ids*, without copying the whole table.

        Evaluation scores the full catalogue in order, and an
        ``nn.Embedding`` lookup for ``arange(n_items)`` materialises a
        second copy of the table -- 167 MB per branch at ``N`` = 326K,
        ``k`` = 128, twice over.  ``weight`` carries the same values in
        the same order, so the gather is pure waste there.
        """
        if full_catalogue:
            return table.weight
        return table(item_ids)

    def _catalogue_visual(self, item_ids: torch.Tensor) -> torch.Tensor:
        """Image features of ``item_ids``; the dense full catalogue is an alias.

        In eval mode, without an online fusion (whose output depends on
        trainable parameters) and when ``item_ids`` is exactly
        ``arange(n_items)``, the answer IS the resident 2-D buffer in its
        own order, so the "cache" is a zero-copy alias of it (SDD M03:
        no derived full-catalogue copy) — previously an ``(N, dv)`` fp32
        duplicate of the buffer.  Invalidated by :meth:`train`.  Lazy
        sources never reach here (see :meth:`_predict_batch_blocked`).
        """
        cacheable = (
            not self.training and self._online_fusion is None and self._is_full_catalogue(item_ids)
        )
        if not cacheable:
            return self._resolve_visual(item_ids)
        if self._catalogue_visual_cache is None:
            self._catalogue_visual_cache = self.visual_features
        return self._catalogue_visual_cache
