"""BPR-MF -- Bayesian Personalised Ranking with matrix factorisation.

Prediction rule (Rendle et al., 2009, Section 4.3.1):

    x_hat_ui = <w_u, h_i> = gamma_u^T gamma_i

The model is a plain matrix factorisation ``X_hat = W H^T`` with a
user matrix ``W`` (``user_embedding``) and an item matrix ``H``
(``item_embedding``).  There is NO item bias: the ``beta_i`` term found
in older versions of this file belongs to the VBPR formulation of He &
McAuley (2016), not to BPR-MF.  Since the BPR criterion only ever sees
score differences ``x_hat_uij = x_hat_ui - x_hat_uj`` for the same user,
a per-item bias is the only extra parameter that would survive the
difference — and the paper deliberately leaves it out.

Regularisation (paper, Section 4.3.1) uses three constants applied to
the parameters touched by each sampled triple ``(u, i, j)``:

    lambda_W    on the user row  w_u   -> config key ``l2_reg``
    lambda_H+   on the positive  h_i   -> config key ``l2_reg_item_pos``
    lambda_H-   on the negative  h_j   -> config key ``l2_reg_item_neg``

When ``l2_reg_item_pos`` / ``l2_reg_item_neg`` are absent from the
config they fall back to ``l2_reg`` (see
:meth:`BaseRecommender._l2_lambda`), which recovers the single-``λ``
behaviour of the original implementation.

References
----------
Rendle, S., Freudenthaler, C., Gantner, Z. & Schmidt-Thieme, L. (2009).
BPR: Bayesian Personalized Ranking from Implicit Feedback.  UAI.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from src.recommenders.base import BaseRecommender


class BPR(BaseRecommender):
    """Matrix-factorisation BPR (BPR-MF) without visual features.

    Parameters
    ----------
    n_users, n_items:
        Vocabulary sizes.
    visual_embeddings:
        Ignored (kept for interface compatibility).  Should be ``None``.
    config:
        Must contain ``latent_dim`` (int).  ``l2_reg`` (``λ_W``) is
        optional (default 0); ``l2_reg_item_pos`` (``λ_H+``) and
        ``l2_reg_item_neg`` (``λ_H-``) are optional and default to
        ``l2_reg``.
    """

    #: BPR-MF has no item bias — only the two factor matrices are
    #: gathered per batch.
    _L2_ITEM_TABLES = ("item_embedding",)

    #: Paper's three regularisation constants: ``λ_W`` for the user row,
    #: ``λ_H+`` / ``λ_H-`` for the positive / negative item rows.
    _L2_LAMBDA_KEYS: dict[str | tuple[str, str], str] = {
        ("user_embedding", "user"): "l2_reg",
        ("item_embedding", "pos"): "l2_reg_item_pos",
        ("item_embedding", "neg"): "l2_reg_item_neg",
    }

    def __init__(
        self,
        n_users: int,
        n_items: int,
        visual_embeddings: np.ndarray | None = None,
        config: dict | None = None,
    ) -> None:
        config = config or {}
        super().__init__(n_users, n_items, visual_embeddings=None, config=config)

        k: int = config["latent_dim"]

        self.user_embedding = nn.Embedding(n_users, k)  # W
        self.item_embedding = nn.Embedding(n_items, k)  # H

        self._init_embedding(self.user_embedding)
        self._init_embedding(self.item_embedding)

    def forward(
        self,
        user_ids: torch.Tensor,
        pos_item_ids: torch.Tensor,
        neg_item_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        gamma_u = self.user_embedding(user_ids)
        gamma_pos = self.item_embedding(pos_item_ids)
        gamma_neg = self.item_embedding(neg_item_ids)

        score_pos = (gamma_u * gamma_pos).sum(dim=-1)
        score_neg = (gamma_u * gamma_neg).sum(dim=-1)
        return score_pos, score_neg

    def predict(self, user_id: int, item_ids: torch.Tensor) -> torch.Tensor:
        gamma_u = self.user_embedding.weight[user_id]
        gamma_i = self.item_embedding(item_ids)
        return (gamma_u * gamma_i).sum(dim=-1)

    def predict_batch(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        gamma_u = self.user_embedding(user_ids)
        gamma_i = self.item_embedding(item_ids)
        return gamma_u @ gamma_i.T
