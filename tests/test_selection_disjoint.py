"""R6 + S3: the selection split must be per-user disjoint from train.

The selection evaluator masks every train item to ``-inf``, so a
validation held-out duplicated into train is unhittable — the user
silently scores 0 on validation, deflating the metric that drives early
stopping and hyperparameter selection.  The final-evaluation guard (A3)
already covers ``test ∩ (train ∪ val)``; these tests pin its mirror on
the selection side and the dedup that closes the entry path (duplicate
CSV rows landing in two splits).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.utils.splits import assert_holdout_disjoint


class TestAssertHoldoutDisjoint:
    def test_passes_when_disjoint(self) -> None:
        assert_holdout_disjoint({0: {1, 2}}, {0: {3}}, "ds", holdout_name="validation")

    def test_raises_on_overlap_naming_the_split(self) -> None:
        with pytest.raises(ValueError, match=r"validation split.*user=0, item=2"):
            assert_holdout_disjoint({0: {1, 2}}, {0: {2}}, "ds", holdout_name="validation")

    def test_overlap_in_another_users_history_is_fine(self) -> None:
        # Disjointness is PER USER: item 2 seen by user 1 does not
        # forbid it as user 0's held-out.
        assert_holdout_disjoint({1: {2}}, {0: {2}}, "ds")


class TestTrainSingleRunGuard:
    def test_duplicated_val_pair_fails_before_training(self) -> None:
        from src.utils.training import train_single_run

        # The guard is the first real statement: with an overlap it must
        # raise before model construction or checkpointing ever run —
        # hence the None placeholders survive.
        with pytest.raises(ValueError, match="validation split"):
            train_single_run(
                model_cls=None,
                model_name="bpr",
                n_users=2,
                n_items=5,
                visual_embeddings=None,
                train_interactions={0: {1, 2}, 1: {3}},
                selection_interactions={0: {2}},  # duplicated into train
                hyperparams={},
                config={},
                checkpoint_mgr=None,
                dataset_name="ds",
                embedding_name="none",
                device="cpu",
            )


class TestCsvSplitDeduplicates:
    def _split(self, rows: list[tuple[str, str]]):
        from src.data.example_csv import CSVDatasetProvider

        provider = CSVDatasetProvider.__new__(CSVDatasetProvider)  # skip fs-touching __init__
        provider.seed = 42
        provider.min_user_interactions = 3
        df = pd.DataFrame(rows, columns=["user_id", "item_id"])
        users = sorted(df["user_id"].unique())
        items = sorted(df["item_id"].unique())
        user2idx = {u: i for i, u in enumerate(users)}
        item2idx = {it: i for i, it in enumerate(items)}
        return provider.split(df, user2idx, item2idx)

    def test_duplicate_pair_never_lands_in_two_splits(self) -> None:
        # u0 has item i1 duplicated; without dedup the two copies can be
        # drawn into different splits (e.g. val AND train).
        rows = [
            ("u0", "i0"),
            ("u0", "i1"),
            ("u0", "i1"),
            ("u0", "i2"),
            ("u0", "i3"),
        ]

        train, val, test = self._split(rows)

        pair_sets = [
            set(map(tuple, d[["user_idx", "item_idx"]].to_numpy())) for d in (train, val, test)
        ]
        assert not (pair_sets[0] & pair_sets[1])
        assert not (pair_sets[0] & pair_sets[2])
        assert not (pair_sets[1] & pair_sets[2])
        # 4 unique pairs: 1 test + 1 val + 2 train.
        assert len(train) == 2 and len(val) == 1 and len(test) == 1

    def test_min_interactions_counts_unique_pairs(self) -> None:
        # 3 rows but only 2 unique items: below min_user_interactions=3,
        # the user must be dropped rather than split on a duplicate.
        rows = [("u0", "i0"), ("u0", "i1"), ("u0", "i1")]

        train, val, test = self._split(rows)

        assert len(train) == 0 and len(val) == 0 and len(test) == 0


class TestFinalEvaluationGuardStillWired:
    def test_load_data_raises_on_test_train_overlap(self, tmp_path: Path) -> None:
        from src.steps.evaluate import load_data

        base = tmp_path / "ds"
        base.mkdir()
        pd.DataFrame({"user_idx": [0, 0], "item_idx": [1, 2]}).to_csv(
            base / "train.csv", index=False
        )
        pd.DataFrame({"user_idx": [0], "item_idx": [3]}).to_csv(base / "val.csv", index=False)
        pd.DataFrame({"user_idx": [0], "item_idx": [2]}).to_csv(base / "test.csv", index=False)
        (base / "user2idx.json").write_text('{"u0": 0}')
        (base / "item2idx.json").write_text('{"i0": 0, "i1": 1, "i2": 2, "i3": 3}')

        with pytest.raises(ValueError, match="test split"):
            load_data(str(tmp_path), "ds")
