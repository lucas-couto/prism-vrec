"""Regression: the validation subsample is a pure function of sample_seed.

The ~2000-user early-stopping subsample must be identical across every
model and Optuna trial of a run so the whole hyperparameter search is
compared on the same held-out users. That invariant holds only because
the Evaluator draws it from a dedicated ``np.random.default_rng(sample_seed)``
over a sorted population (src/evaluation/protocol.py) — NOT from the global
RNGs that model init, negative sampling and dropout mutate.

These tests lock that property: they pollute the global ``random`` /
``numpy`` / ``torch`` RNGs by different amounts between two Evaluator
constructions and assert the sampled user set is unchanged. A future
refactor that switches the draw to a shared/global RNG would break them.
"""

from __future__ import annotations

import random

import numpy as np
import torch

from src.evaluation.protocol import Evaluator

_POPULATION = 500
_SAMPLE_SIZE = 50


def _population() -> tuple[dict[int, set[int]], dict[int, set[int]]]:
    """A train/test population larger than the subsample size."""
    train = {u: {u % 7} for u in range(_POPULATION)}
    test = {u: {1000 + u} for u in range(_POPULATION)}
    return train, test


def _make_evaluator(sample_seed: int) -> Evaluator:
    train, test = _population()
    return Evaluator(
        train_interactions=train,
        test_interactions=test,
        n_items=5000,
        k_values=[10],
        sample_size=_SAMPLE_SIZE,
        sample_seed=sample_seed,
    )


def _pollute_global_rngs(magnitude: int) -> None:
    """Advance every global RNG by a magnitude-dependent amount.

    Mimics different models consuming different amounts of randomness
    (weight init, negative sampling) between two subsample draws.
    """
    random.seed(magnitude)
    for _ in range(magnitude):
        random.random()
    np.random.seed(magnitude)
    np.random.rand(magnitude)
    torch.manual_seed(magnitude)
    torch.randn(magnitude)


class TestValidationSubsampleDeterminism:
    def test_same_seed_identical_despite_different_global_rng_state(self) -> None:
        _pollute_global_rngs(3)
        first = _make_evaluator(sample_seed=42)

        _pollute_global_rngs(9999)
        second = _make_evaluator(sample_seed=42)

        assert first.is_sampled and second.is_sampled
        assert len(first.test_users) == _SAMPLE_SIZE
        assert first.test_users == second.test_users

    def test_different_seeds_produce_different_subsamples(self) -> None:
        a = _make_evaluator(sample_seed=42)
        b = _make_evaluator(sample_seed=43)

        assert a.test_users != b.test_users

    def test_subsample_is_a_subset_of_the_population(self) -> None:
        evaluator = _make_evaluator(sample_seed=42)
        all_users = set(range(_POPULATION))

        assert set(evaluator.test_users).issubset(all_users)
        assert len(set(evaluator.test_users)) == _SAMPLE_SIZE
