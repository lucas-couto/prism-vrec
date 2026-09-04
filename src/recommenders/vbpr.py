"""VBPR -- Visual Bayesian Personalised Ranking.

Prediction rule (He & McAuley 2016, Eq. 4)::

    x_hat_ui = alpha + beta_u + beta_i + gamma_u^T gamma_i
               + theta_u^T (E f_i) + beta'^T f_i

where ``f_i`` is the item's (native) visual feature, ``E`` the learned
projection ``W_vis``, ``theta_u`` the visual user factors ``alpha_u`` and
``beta'`` the *visual bias* vector.  The global offset ``alpha`` and the
user bias ``beta_u`` are omitted: both cancel in the pairwise difference
``x_hat_ui - x_hat_uj`` that BPR optimises, so they are unidentifiable
under the training objective.

Regularisation follows the paper's three constants:

* ``lambda_Theta`` (config ``l2_reg``) — latent factors and ``theta_u``,
  plus ``beta_i``;
* ``lambda_beta`` (config ``l2_reg_visual_bias``) — the visual bias
  ``beta'``; falls back to ``l2_reg`` when absent;
* ``lambda_E`` (config ``l2_reg_projection``) — the projection ``E``;
  defaults to ``0`` as in the paper's experiments.

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
    """VBPR with a linear visual projection and a visual bias.

    Parameters
    ----------
    n_users, n_items:
        Vocabulary sizes.
    visual_embeddings:
        Pre-extracted visual features of shape ``(n_items, D_v)``.
    config:
        Must contain ``latent_dim`` (k) and ``visual_dim`` (k_v).
        ``l2_reg`` (``lambda_Theta``), ``l2_reg_visual_bias``
        (``lambda_beta``, default ``l2_reg``) and ``l2_reg_projection``
        (``lambda_E``, default ``0``) are optional.

    Notes
    -----
    ``beta'`` lives in the *native* feature space: its size is
    ``visual_dim_raw``, the dimension ``E`` consumes.  With an online
    fusion (3-D buffer or ragged learned alignment) ``f_i`` is the fused
    output and ``beta'`` therefore has the aligned dimension.
    """

    #: BPR-Opt L2: gather ``alpha_u`` rows alongside ``gamma_u``; the
    #: dense projection ``W_vis`` and ``beta'`` stay in the shared term
    #: (every triple touches them).
    _L2_USER_TABLES = ("user_embedding", "visual_user_embedding")
    #: Fold-in: both user-indexed tables (``gamma_u`` and ``alpha_u``)
    #: are re-initialised and optimised for a folded-in user.
    _USER_TABLES = ("user_embedding", "visual_user_embedding")
    _L2_LAMBDA_KEYS = {
        ("visual_projection", "shared"): "l2_reg_projection",
        ("visual_bias", "shared"): "l2_reg_visual_bias",
    }
    _L2_LAMBDA_DEFAULTS = {"l2_reg_projection": 0.0}

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
        # beta' (Eq. 4): zero-initialised on purpose so a fresh model
        # scores exactly as it did before the term was added.
        self.visual_bias = nn.Parameter(torch.zeros(dv))

        self._init_embedding(self.user_embedding)
        self._init_embedding(self.item_embedding)
        self._init_embedding(self.visual_user_embedding)
        nn.init.zeros_(self.item_bias.weight)
        nn.init.xavier_uniform_(self.visual_projection.weight)

        self._item_proj_cache: torch.Tensor | None = None
        self._item_visual_bias_cache: torch.Tensor | None = None

    def train(self, mode: bool = True):
        self._item_visual_bias_cache = None
        return super().train(mode)

    def _visual_user_table(self) -> nn.Embedding:
        return self.visual_user_embedding

    def _item_visual_term(self, item_ids: torch.Tensor) -> torch.Tensor:
        """Project raw visual features for the given items: W_vis @ f_i.

        With an online fusion (3-D buffer) the cache is bypassed since
        the gate's output depends on trainable parameters and changes
        every optimisation step.
        """
        return self._full_catalog_cache(
            item_ids, lambda ids: self.visual_projection(self._resolve_visual(ids))
        )

    def _item_visual_bias(self, item_ids: torch.Tensor) -> torch.Tensor:
        """Visual bias ``beta'^T f_i`` of shape ``(B,)``.

        Cached under the rules of :meth:`_full_catalog_lookup` in its own
        slot,
        invalidated by :meth:`train`.
        """
        eligible = self._full_catalog_lookup(item_ids)
        if eligible and self._item_visual_bias_cache is not None:
            return self._item_visual_bias_cache
        result = self._resolve_visual(item_ids) @ self.visual_bias
        if eligible:
            self._item_visual_bias_cache = result
        return result
