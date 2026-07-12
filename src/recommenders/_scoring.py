"""Shared scoring logic for the linear visual-BPR recommender family.

VBPR, AVBPR and DeepStyle share the exact same score decomposition::

    y_hat_ui = gamma_u . gamma_i  +  alpha_u . theta_i  +  beta_i

and therefore the same ``forward`` / ``predict`` / ``predict_batch`` /
``train`` (cache-invalidation) bodies.  They differ only in

* how the item's visual term ``theta_i`` is produced
  (linear projection, attention-weighted, or an MLP style projector), and
* the name of the per-user visual embedding table (``alpha_u`` / ``s_u``).

:class:`LinearVisualScoreMixin` captures the shared bodies; each model
supplies the two hooks below.  The mixin creates no parameters, so a
model's ``state_dict`` and its seeded weights are unchanged by adopting
it.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LinearVisualScoreMixin:
    """Mixin providing the shared linear visual-BPR scoring methods.

    Concrete models must define, in addition to the standard
    ``user_embedding`` / ``item_embedding`` / ``item_bias`` parameters:

    * :meth:`_item_visual_term` — return ``theta_i`` of shape
      ``(len(item_ids), kv)`` for the given items (the cache-guarded
      visual/style projection).
    * :meth:`_visual_user_table` — return the :class:`nn.Embedding`
      holding the per-user visual weights ``alpha_u`` / ``s_u``.

    The mixin must precede :class:`BaseRecommender` in the MRO so its
    methods take precedence.
    """

    def _item_visual_term(self, item_ids: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError

    def _visual_user_table(self) -> nn.Embedding:  # pragma: no cover
        raise NotImplementedError

    def train(self, mode: bool = True):
        self._item_proj_cache = None
        return super().train(mode)

    def forward(
        self,
        user_ids: torch.Tensor,
        pos_item_ids: torch.Tensor,
        neg_item_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        gamma_u = self.user_embedding(user_ids)
        alpha_u = self._visual_user_table()(user_ids)

        # Combine pos and neg item lookups into a single (2B,)-batched
        # forward.  Every op is row-independent (or point-wise), so
        # splitting after the matmul is mathematically equivalent to two
        # B-sized passes while amortising kernel-launch / matmul cost.
        b = pos_item_ids.shape[0]
        all_items = torch.cat([pos_item_ids, neg_item_ids], dim=0)
        gamma_all = self.item_embedding(all_items)
        beta_all = self.item_bias(all_items).squeeze(-1)
        theta_all = self._item_visual_term(all_items)

        gamma_pos, gamma_neg = gamma_all[:b], gamma_all[b:]
        beta_pos, beta_neg = beta_all[:b], beta_all[b:]
        theta_pos, theta_neg = theta_all[:b], theta_all[b:]

        score_pos = (gamma_u * gamma_pos).sum(-1) + (alpha_u * theta_pos).sum(-1) + beta_pos
        score_neg = (gamma_u * gamma_neg).sum(-1) + (alpha_u * theta_neg).sum(-1) + beta_neg
        return score_pos, score_neg

    def predict(self, user_id: int, item_ids: torch.Tensor) -> torch.Tensor:
        gamma_u = self.user_embedding.weight[user_id]
        gamma_i = self.item_embedding(item_ids)
        beta_i = self.item_bias(item_ids).squeeze(-1)
        alpha_u = self._visual_user_table().weight[user_id]
        theta_i = self._item_visual_term(item_ids)

        return (gamma_u * gamma_i).sum(-1) + (alpha_u * theta_i).sum(-1) + beta_i

    def predict_batch(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        gamma_u = self.user_embedding(user_ids)
        gamma_i = self.item_embedding(item_ids)
        beta_i = self.item_bias(item_ids).squeeze(-1)
        alpha_u = self._visual_user_table()(user_ids)
        theta_i = self._item_visual_term(item_ids)

        return gamma_u @ gamma_i.T + alpha_u @ theta_i.T + beta_i.unsqueeze(0)
