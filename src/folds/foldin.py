"""Fold-in of held-out users into a frozen, trained BPR recommender.

Under the user-level K-fold protocol the users of fold ``k`` never take
part in training.  To evaluate the trained model on them, each user is
*folded in*: the user-indexed rows of the model are re-initialised and
fitted on the user's **profile** while every other parameter stays
frozen, and the user is then scored on a disjoint **target** item.

Rendle et al. (BPR: Bayesian Personalized Ranking from Implicit
Feedback, UAI 2009, Section 2) note that the fold-in strategy used for
matrix factorisation carries over to BPR: with the item side fixed, a
new user's factors are obtained by optimising the same pairwise
criterion over that user's feedback only.  This module is exactly that
strategy, generalised to every recommender of the framework through
two model-declared hooks: :attr:`BaseRecommender._USER_TABLES` (the
user-indexed ``nn.Embedding`` tables) and
:meth:`BaseRecommender.rebuild_user_state` (non-parametric per-user
state, e.g. ACF's history buffers).  No branch on the model's name.

Guarantees, all covered by ``tests/test_fold_in.py``:

* no parameter outside the user tables changes;
* within the user tables, only the rows of the profile users change —
  enforced by a gradient row-mask on each table, so it holds even for
  a regulariser that touches the whole matrix (none of the built-ins
  since VNPR moved to BPR-Opt on 2026-09-04; the guarantee is kept and
  tested against a re-created whole-matrix VNPR);
* the target set never enters the loss: the sampler is built over the
  profile alone.  A target item is therefore a legitimate NEGATIVE draw
  for its user, exactly as the training protocol may sample a user's
  held-out item — expected, not a leak;
* the model's ``requires_grad`` flags and train/eval mode are restored.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn

from src.utils.amp_compat import get_grad_scaler
from src.utils.training import BPRBatchSampler, bpr_step

Interactions = dict[int, set[int]]


@dataclass(frozen=True)
class FoldInConfig:
    """Optimisation budget of one fold-in call.

    :param epochs: Passes over the profile interactions (``>= 1``).
    :param learning_rate: Adam learning rate for the user rows.
    :param batch_size: Triples per step.
    :param seed: Seeds the row re-initialisation and the negative
        sampler; a stochastic forward pass (dropout) still draws from
        the global RNG, which this routine deliberately leaves alone.
    """

    epochs: int
    learning_rate: float
    batch_size: int
    seed: int

    def __post_init__(self) -> None:
        if self.epochs < 1:
            raise ValueError(f"FoldInConfig.epochs must be >= 1, got {self.epochs}.")
        if self.batch_size < 1:
            raise ValueError(f"FoldInConfig.batch_size must be >= 1, got {self.batch_size}.")
        if self.learning_rate <= 0:
            raise ValueError(f"FoldInConfig.learning_rate must be > 0, got {self.learning_rate}.")


@dataclass(frozen=True)
class FoldInReport:
    """What one fold-in call did.

    :param n_users: Users folded in (``len(profile)``).
    :param n_interactions: Profile triples seen per epoch.
    :param epochs: Epochs run.
    :param final_loss: Mean batch loss of the last epoch.
    :param user_tables: The ``_USER_TABLES`` actually present on the model.
    """

    n_users: int
    n_interactions: int
    epochs: int
    final_loss: float
    user_tables: tuple[str, ...]


def _validate_profile(profile: Interactions, n_users: int, n_items: int) -> None:
    if not profile or all(not items for items in profile.values()):
        raise ValueError("fold_in_users: profile is empty.")
    bad_users = [u for u in profile if not 0 <= u < n_users]
    if bad_users:
        raise ValueError(f"fold_in_users: users outside [0, {n_users}): {sorted(bad_users)[:5]}")
    bad_items = {i for items in profile.values() for i in items if not 0 <= i < n_items}
    if bad_items:
        raise ValueError(f"fold_in_users: items outside [0, {n_items}): {sorted(bad_items)[:5]}")


def _user_tables(model: nn.Module) -> list[tuple[str, nn.Embedding]]:
    """``(name, table)`` for each declared user table the model owns."""
    tables: list[tuple[str, nn.Embedding]] = []
    for name in getattr(model, "_USER_TABLES", ()):
        table = getattr(model, name, None)
        if isinstance(table, nn.Embedding):
            tables.append((name, table))
    return tables


def _reinit_rows(table: nn.Embedding, rows: torch.Tensor, generator: torch.Generator) -> None:
    """Xavier-uniform draw for ``rows`` only, with the FULL table's bound.

    ``nn.init.xavier_uniform_`` derives its bound from the whole
    ``(n_users, dim)`` shape, so the folded-in rows follow the same
    distribution as the rows initialised at construction.
    """
    fan_out, fan_in = table.weight.shape
    bound = math.sqrt(6.0 / float(fan_in + fan_out))
    fresh = torch.empty(rows.shape[0], fan_in).uniform_(-bound, bound, generator=generator)
    with torch.no_grad():
        table.weight[rows] = fresh.to(device=table.weight.device, dtype=table.weight.dtype)


def _row_mask_hook(table: nn.Embedding, rows: torch.Tensor):
    """Zero the gradient of every row not in ``rows`` before it is applied.

    Sparse index gradients already leave unused rows at zero; the mask
    additionally covers models whose L2 term is dense over the table
    (VNPR), so no row outside the profile can ever move.
    """
    keep = torch.zeros(
        table.weight.shape[0], 1, dtype=table.weight.dtype, device=table.weight.device
    )
    keep[rows] = 1.0
    return table.weight.register_hook(lambda grad, keep=keep: grad * keep)


def _optimise(
    model: nn.Module,
    weights: list[torch.Tensor],
    profile: Interactions,
    config: FoldInConfig,
    *,
    n_items: int,
    device: str,
) -> float:
    """Adam over ``weights`` with the model's own ``bpr_loss``; returns the last epoch's mean loss."""
    use_cuda = device != "cpu" and torch.cuda.is_available()
    optimizer = torch.optim.Adam(weights, lr=config.learning_rate)
    scaler = get_grad_scaler(enabled=use_cuda)
    sampler = BPRBatchSampler(profile, n_items, config.batch_size, seed=config.seed)
    loss_device = torch.device(device) if use_cuda else torch.device("cpu")
    model.train()
    avg_loss = float("nan")
    for epoch in range(config.epochs):
        total = torch.zeros((), device=loss_device)
        n_batches = 0
        for users, pos, neg in sampler.epoch(epoch):
            total += bpr_step(
                model, optimizer, scaler, users, pos, neg, device=device, use_cuda=use_cuda
            )
            n_batches += 1
        avg_loss = (total / max(n_batches, 1)).item()
    return avg_loss


def fold_in_users(
    model: nn.Module,
    profile: Interactions,
    config: FoldInConfig,
    *,
    n_items: int,
    device: str,
) -> FoldInReport:
    """Fit the user rows of ``profile``'s users on a frozen ``model``.

    Steps: (a) re-initialise ONLY those rows in every
    :attr:`_USER_TABLES` table (Xavier-uniform, seeded); (b)
    ``model.rebuild_user_state(profile)``; (c) freeze every parameter
    and unfreeze the user tables' weights; (d) Adam + the model's
    ``bpr_loss`` over a :class:`BPRBatchSampler` built on ``profile``
    alone (negatives are drawn outside each user's profile), for
    ``config.epochs``, via :func:`bpr_step`; (e) restore the original
    ``requires_grad`` flags and train/eval mode.

    :param model: Trained recommender, already on ``device``.
    :param profile: ``{user_idx: set(item_idx)}`` to fit; must not
        contain the target items.
    :param config: Optimisation budget and seed.
    :param n_items: Catalogue size for negative sampling.
    :param device: ``"cpu"`` or a CUDA device string.
    :returns: A :class:`FoldInReport`.
    :raises ValueError: On an empty profile or out-of-range ids.
    """
    _validate_profile(profile, int(getattr(model, "n_users", 0)), n_items)
    tables = _user_tables(model)
    rows = torch.tensor(sorted(profile), dtype=torch.long)
    generator = torch.Generator().manual_seed(config.seed)
    for _, table in tables:
        _reinit_rows(table, rows, generator)
    model.rebuild_user_state(profile)

    original_grad = [(p, p.requires_grad) for p in model.parameters()]
    was_training = model.training
    hooks = []
    try:
        for param, _ in original_grad:
            param.requires_grad_(False)
        weights = [table.weight.requires_grad_(True) for _, table in tables]
        hooks = [_row_mask_hook(table, rows) for _, table in tables]
        final_loss = _optimise(model, weights, profile, config, n_items=n_items, device=device)
    finally:
        for hook in hooks:
            hook.remove()
        for param, flag in original_grad:
            param.requires_grad_(flag)
        model.train(was_training)

    return FoldInReport(
        n_users=len(profile),
        n_interactions=sum(len(items) for items in profile.values()),
        epochs=config.epochs,
        final_loss=final_loss,
        user_tables=tuple(name for name, _ in tables),
    )
