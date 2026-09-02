"""AVBPR -- Attentional Visual BPR.

Prediction rule (VBPR Eq. 4 with an attended visual term)::

    theta_i   = E f_i                        (E = W_vis)
    a_i       = softmax(MLP_att(theta_i))
    theta_hat = theta_i * a_i                (element-wise)
    x_hat_ui  = alpha + beta_u + beta_i + gamma_u^T gamma_i
                + theta_u^T theta_hat + beta'^T f_i

MLP_att is a small attention network that produces per-dimension
importance weights over the projected visual features.  Everything
else — ``gamma``, ``theta_u`` (``alpha_u``), ``beta_i`` and the visual
bias ``beta'`` — is VBPR's, so the attention is the only difference
between the two models.  The global offset ``alpha`` and the user bias
``beta_u`` are omitted because they cancel in the pairwise difference
BPR optimises.

Regularisation uses VBPR's three constants (``l2_reg`` for latent
factors, ``theta_u`` and ``beta_i``; ``l2_reg_visual_bias`` for
``beta'``, default ``l2_reg``; ``l2_reg_projection`` for ``E``, default
``0``).  The attention network stays under ``l2_reg``.
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
        ``l2_reg``, ``l2_reg_visual_bias`` (default ``l2_reg``) and
        ``l2_reg_projection`` (default ``0``) are optional.

    Notes
    -----
    ``beta'`` has size ``visual_dim_raw`` — the native feature dimension
    ``E`` consumes, or the aligned dimension when an online fusion
    produces ``f_i``.
    """

    #: BPR-Opt L2: gather ``alpha_u`` rows alongside ``gamma_u``; the
    #: dense ``W_vis``, ``beta'`` and attention MLP stay in the shared
    #: term (every triple touches them).
    _L2_USER_TABLES = ("user_embedding", "visual_user_embedding")
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
        # beta' (VBPR Eq. 4): zero-initialised on purpose so a fresh
        # model scores exactly as it did before the term was added.
        self.visual_bias = nn.Parameter(torch.zeros(dv))

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
        self._item_visual_bias_cache: torch.Tensor | None = None

    def train(self, mode: bool = True):
        self._item_visual_bias_cache = None
        return super().train(mode)

    def _visual_user_table(self) -> nn.Embedding:
        return self.visual_user_embedding

    def _item_visual_term(self, item_ids: torch.Tensor) -> torch.Tensor:
        """Compute attention-weighted visual embedding for items.

        Returns theta_hat = theta_i * softmax(MLP_att(theta_i)).

        With an online fusion (3-D buffer) the cache is bypassed since
        the gate's output depends on trainable parameters.
        """

        def _attend(ids: torch.Tensor) -> torch.Tensor:
            theta_i = self.visual_projection(self._resolve_visual(ids))
            a_i = torch.softmax(self.attention_net(theta_i), dim=-1)
            return theta_i * a_i

        return self._full_catalog_cache(item_ids, _attend)

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
