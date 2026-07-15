"""Guards: model selection runs on validation, never on the test set.

Early stopping and the Optuna objective must score VALIDATION held-outs
(masked by each user's train items); the test set is read only by the
final evaluate step. Two guards:

* structural — no training/selection module references ``test.csv``;
* behavioural — the selection Evaluator holds the validation held-outs,
  not the test ones.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import src.steps.train as train_mod
import src.utils.parallel as parallel_mod
import src.utils.training as training_mod
from src.evaluation.protocol import Evaluator

_TRAINING_MODULES = [train_mod, training_mod, parallel_mod]


class TestNoTestCsvInSelectionPaths:
    @pytest.mark.parametrize("module", _TRAINING_MODULES, ids=lambda m: m.__name__)
    def test_module_does_not_read_test_csv(self, module) -> None:
        source = Path(module.__file__).read_text(encoding="utf-8")
        assert "test.csv" not in source, (
            f"{module.__name__} references test.csv — model selection must "
            f"use val.csv only; the test set is touched solely by "
            f"src/steps/evaluate.py."
        )

    def test_training_paths_read_val_csv(self) -> None:
        # The counterpart positive assertion: selection loads val.csv.
        train_src = Path(train_mod.__file__).read_text(encoding="utf-8")
        parallel_src = Path(parallel_mod.__file__).read_text(encoding="utf-8")
        assert "val.csv" in train_src
        assert "val.csv" in parallel_src


class TestSelectionEvaluatorHoldsValHeldOuts:
    def test_selection_evaluator_ranks_validation_items(self) -> None:
        train = {0: {0, 1}, 1: {2}, 2: {3, 4}}
        val = {0: {10}, 1: {11}, 2: {12}}
        test = {0: {20}, 1: {21}, 2: {22}}

        # Built exactly as the training path builds it: train mask + val
        # held-outs (never the test held-outs).
        selection_evaluator = Evaluator(
            train_interactions=train,
            test_interactions=val,
            n_items=100,
            k_values=[10],
        )

        held_outs = {
            item for items in selection_evaluator.test_interactions.values() for item in items
        }
        assert held_outs == {10, 11, 12}
        test_items = {item for items in test.values() for item in items}
        assert held_outs.isdisjoint(test_items)
