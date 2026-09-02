"""Shared scoring logic for the linear visual-BPR recommender family.

VBPR, AVBPR and DeepStyle share the same score decomposition::

    y_hat_ui = gamma_u . gamma_i  +  alpha_u . theta_i  +  beta_i  +  beta'_i

and therefore the same ``forward`` / ``predict`` / ``predict_batch`` /
``train`` (cache-invalidation) bodies.  They differ only in

* how the item's visual term ``theta_i`` is produced
  (linear projection, attention-weighted, or projection minus a learned
  category vector),
* which table holds the per-user visual weights ``alpha_u`` / ``s_u``
  (DeepStyle shares ``gamma_u`` — one user vector, as in its paper),
* whether the model carries an item bias ``beta_i`` (VBPR Eq. 4 does;
  DeepStyle Eq. 3 does not), and
* whether the model carries VBPR's visual bias ``beta'_i = beta'^T f_i``.

:class:`LinearVisualScoreMixin` captures the shared bodies; each model
supplies the hooks below.  The mixin creates no parameters, so a
model's ``state_dict`` and its seeded weights are unchanged by adopting
it.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn


class LinearVisualScoreMixin:
    """Mixin providing the shared linear visual-BPR scoring methods.

    Concrete models must define ``user_embedding`` / ``item_embedding``
    plus:

    * :meth:`_item_visual_term` — return ``theta_i`` of shape
      ``(len(item_ids), kv)`` for the given items (the cache-guarded
      visual/style projection).
    * :meth:`_visual_user_table` — return the :class:`nn.Embedding`
      holding the per-user visual weights ``alpha_u`` / ``s_u``.

    Optional hooks with neutral defaults:

    * :meth:`_item_bias_term` — ``beta_i`` per item, or ``None`` for a
      model without an item bias.  Default: the ``item_bias`` table when
      the model has one.
    * :meth:`_item_visual_bias` — ``beta'^T f_i`` per item, or ``None``.
      Default ``None``: models that do not define the term score exactly
      as before.

    The mixin must precede :class:`BaseRecommender` in the MRO so its
    methods take precedence.
    """

    def _item_visual_term(self, item_ids: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError

    def _visual_user_table(self) -> nn.Embedding:  # pragma: no cover
        raise NotImplementedError

    def _item_bias_term(self, item_ids: torch.Tensor) -> torch.Tensor | None:
        table = getattr(self, "item_bias", None)
        return None if table is None else table(item_ids).squeeze(-1)

    def _item_visual_bias(self, item_ids: torch.Tensor) -> torch.Tensor | None:
        return None

    def train(self, mode: bool = True):
        self._item_proj_cache = None
        return super().train(mode)

    def _full_catalog_lookup(self, item_ids: torch.Tensor) -> bool:
        """Whether a per-item result for ``item_ids`` may be cached.

        Only evaluation lookups over the whole catalogue in order
        (``item_ids == arange(n_items)``) qualify, and never with an
        online fusion active (the fused features depend on trainable
        parameters).  A training batch that happens to be ``n_items``
        long, or an eval ``forward`` whose pos+neg concat has that
        length, therefore never poisons later lookups.
        """
        if self.training or self._online_fusion is not None:
            return False
        if item_ids.shape[0] != self.n_items:
            return False
        expected = torch.arange(self.n_items, device=item_ids.device)
        return bool(torch.equal(item_ids, expected))

    def _full_catalog_cache(
        self,
        item_ids: torch.Tensor,
        compute: Callable[[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        """Evaluate ``compute(item_ids)``, caching full-catalogue results.

        See :meth:`_full_catalog_lookup` for the eligibility rule.
        """
        eligible = self._full_catalog_lookup(item_ids)
        if eligible and self._item_proj_cache is not None:
            return self._item_proj_cache
        result = compute(item_ids)
        if eligible:
            self._item_proj_cache = result
        return result

    def _item_terms(
        self, item_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``(gamma_i, theta_i, bias_i)`` with the two biases folded into one."""
        gamma_i = self.item_embedding(item_ids)
        theta_i = self._item_visual_term(item_ids)
        bias = torch.zeros(item_ids.shape[0], device=gamma_i.device, dtype=gamma_i.dtype)
        beta_i = self._item_bias_term(item_ids)
        if beta_i is not None:
            bias = bias + beta_i
        visual_bias = self._item_visual_bias(item_ids)
        if visual_bias is not None:
            bias = bias + visual_bias
        return gamma_i, theta_i, bias

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
        gamma_all, theta_all, bias_all = self._item_terms(all_items)

        gamma_pos, gamma_neg = gamma_all[:b], gamma_all[b:]
        theta_pos, theta_neg = theta_all[:b], theta_all[b:]
        bias_pos, bias_neg = bias_all[:b], bias_all[b:]

        score_pos = (gamma_u * gamma_pos).sum(-1) + (alpha_u * theta_pos).sum(-1) + bias_pos
        score_neg = (gamma_u * gamma_neg).sum(-1) + (alpha_u * theta_neg).sum(-1) + bias_neg
        return score_pos, score_neg

    def predict(self, user_id: int, item_ids: torch.Tensor) -> torch.Tensor:
        gamma_u = self.user_embedding.weight[user_id]
        alpha_u = self._visual_user_table().weight[user_id]
        gamma_i, theta_i, bias_i = self._item_terms(item_ids)

        return (gamma_u * gamma_i).sum(-1) + (alpha_u * theta_i).sum(-1) + bias_i

    def predict_batch(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        gamma_u = self.user_embedding(user_ids)
        alpha_u = self._visual_user_table()(user_ids)
        gamma_i, theta_i, bias_i = self._item_terms(item_ids)

        return gamma_u @ gamma_i.T + alpha_u @ theta_i.T + bias_i.unsqueeze(0)
