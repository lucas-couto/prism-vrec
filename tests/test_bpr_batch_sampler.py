"""Vectorized BPR negative sampler contract (v2 training protocol)."""

from __future__ import annotations

import torch

from src.utils.training import BPRBatchSampler

N_ITEMS = 50


def _interactions() -> dict[int, set[int]]:
    return {0: {1, 2, 3}, 1: {10, 11}, 2: set(range(40)), 3: {49}}


def _collect(sampler: BPRBatchSampler, epoch: int):
    users, pos, neg = [], [], []
    for u, p, n in sampler.epoch(epoch):
        users.append(u)
        pos.append(p)
        neg.append(n)
    return torch.cat(users), torch.cat(pos), torch.cat(neg)


def test_negatives_never_collide_with_user_train_set() -> None:
    inter = _interactions()
    sampler = BPRBatchSampler(inter, N_ITEMS, batch_size=4, seed=7)

    for epoch in range(5):
        users, _, neg = _collect(sampler, epoch)
        for u, n in zip(users.tolist(), neg.tolist(), strict=True):
            assert n not in inter[u], f"epoch {epoch}: negative {n} in user {u}'s train set"


def test_every_interaction_appears_once_per_epoch() -> None:
    inter = _interactions()
    sampler = BPRBatchSampler(inter, N_ITEMS, batch_size=4, seed=7)

    users, pos, _ = _collect(sampler, epoch=0)
    seen = sorted(zip(users.tolist(), pos.tolist(), strict=True))
    expected = sorted((u, i) for u, items in inter.items() for i in items)
    assert seen == expected


def test_deterministic_given_seed_and_epoch() -> None:
    a = BPRBatchSampler(_interactions(), N_ITEMS, batch_size=4, seed=7)
    b = BPRBatchSampler(_interactions(), N_ITEMS, batch_size=4, seed=7)

    ua, pa, na = _collect(a, epoch=3)
    ub, pb, nb = _collect(b, epoch=3)
    assert torch.equal(ua, ub) and torch.equal(pa, pb) and torch.equal(na, nb)


def test_different_epochs_differ() -> None:
    sampler = BPRBatchSampler(_interactions(), N_ITEMS, batch_size=4, seed=7)

    _, _, n0 = _collect(sampler, epoch=0)
    _, _, n1 = _collect(sampler, epoch=1)
    assert not torch.equal(n0, n1)


def test_heavy_user_terminates() -> None:
    # user 2 owns 40 of 50 items: collision probability 0.8 per draw —
    # the vectorized redraw loop must still terminate quickly.
    sampler = BPRBatchSampler(_interactions(), N_ITEMS, batch_size=64, seed=1)

    users, _, neg = _collect(sampler, epoch=0)
    mask = users == 2
    assert mask.any()
    assert all(int(n) >= 40 for n in neg[mask])
