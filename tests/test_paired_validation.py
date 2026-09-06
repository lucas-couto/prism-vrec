"""R04 (F15 / Q17): paired statistics validate observations before testing.

Before any paired test the per-user rows must form a set of uniquely
keyed scientific observations ``(provenance, config, user)``.  The old
``_ensure_config`` dropped duplicate ``(user_id, config)`` rows keeping
the first one, whatever its value and origin, and ``pairwise_significance``
inner-joined the two configs' users away silently.  Both are reproduced
here as failing-before regressions, followed by the accepted behaviours
(identical intentional baseline duplicate, order invariance, idempotence,
declared restricted population).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation.statistical import (
    _ensure_config,
    friedman_test,
    pairwise_significance,
    per_model_summary,
)

_METRIC = "ndcg@10"


def _rows(
    configs: list[tuple[str, str]],
    n_users: int,
    *,
    condition: str | None = None,
    seed: int | None = None,
    protocol: str | None = None,
) -> list[dict]:
    rows: list[dict] = []
    for u in range(n_users):
        for i, (model, embedding) in enumerate(configs):
            row = {
                "user_id": u,
                "model_name": model,
                "embedding_name": embedding,
                _METRIC: 0.1 * (i + 1) + 0.001 * u,
            }
            if condition is not None:
                row["condition"] = condition
            if seed is not None:
                row["seed"] = seed
            if protocol is not None:
                row["protocol"] = protocol
            rows.append(row)
    return rows


def _frame(configs: list[tuple[str, str]], n_users: int = 8, **provenance) -> pd.DataFrame:
    return pd.DataFrame(_rows(configs, n_users, **provenance))


_TWO = [("vbpr", "resnet50_D128"), ("vbpr", "vit_b16_D128")]


class TestConflictingDuplicatesAreRejected:
    """The reproduction of F15: a changed value under the same key must fail."""

    def test_should_reject_same_key_with_different_value(self) -> None:
        df = pd.DataFrame(
            {
                "user_id": [7, 7],
                "model_name": ["vbpr", "vbpr"],
                "embedding_name": ["resnet50_D128", "resnet50_D128"],
                _METRIC: [0.3, 0.4],
            }
        )

        with pytest.raises(ValueError, match="conflict"):
            _ensure_config(df)

    def test_should_reject_conflicting_duplicate_inside_pairwise(self) -> None:
        df = _frame(_TWO)
        clash = df.iloc[[0]].assign(**{_METRIC: 0.99})
        df = pd.concat([df, clash], ignore_index=True)

        with pytest.raises(ValueError, match="conflict"):
            pairwise_significance(df, metric=_METRIC, n_iterations=20)

    def test_should_reject_unexplained_identical_duplicate_of_a_visual_config(self) -> None:
        # Identical values, but a visual cell is never written to two
        # condition files: this is a torn append, not an intentional share.
        df = _frame(_TWO, condition="frozen")
        df = pd.concat([df, df.iloc[[0]]], ignore_index=True)

        with pytest.raises(ValueError, match="duplicate"):
            _ensure_config(df)


class TestIntentionalBaselineDuplicate:
    def test_should_accept_identical_baseline_across_conditions_once(self) -> None:
        frozen = _frame([("bpr", "none"), ("vbpr", "resnet50_D128")], condition="frozen")
        finetuned = _frame(
            [("bpr", "none"), ("vbpr", "resnet50_finetuned_D128")], condition="finetuned"
        )
        df = pd.concat([frozen, finetuned], ignore_index=True)

        out = _ensure_config(df)

        baseline = out[out["config"] == "bpr_none"]
        assert len(baseline) == 8
        assert baseline["user_id"].is_unique

    def test_should_reject_baseline_duplicate_whose_values_differ(self) -> None:
        frozen = _frame([("bpr", "none")], condition="frozen")
        finetuned = _frame([("bpr", "none")], condition="finetuned")
        finetuned.loc[0, _METRIC] += 0.5
        df = pd.concat([frozen, finetuned], ignore_index=True)

        with pytest.raises(ValueError, match="conflict"):
            _ensure_config(df)

    def test_legacy_frame_without_condition_column_still_accepts_baseline(self) -> None:
        # Contract kept from the pre-R04 test-suite: identity and every
        # value match, so the duplicate is provably lossless.
        df = pd.DataFrame(
            {
                "user_id": [7, 7],
                "model_name": ["bpr", "bpr"],
                "embedding_name": ["none", "none"],
                _METRIC: [0.3, 0.3],
            }
        )

        assert len(_ensure_config(df)) == 1


class TestPopulationEquality:
    """The second F15 defect: pairs were inner-joined on the users present."""

    def test_should_reject_missing_user_instead_of_inner_joining(self) -> None:
        df = _frame(_TWO)
        df = df.drop(
            index=df[(df["user_id"] == 3) & (df["embedding_name"] == "vit_b16_D128")].index
        )

        with pytest.raises(ValueError, match="population"):
            pairwise_significance(df, metric=_METRIC, n_iterations=20)

    def test_should_report_which_users_are_missing_and_where(self) -> None:
        df = _frame(_TWO)
        df = df.drop(
            index=df[(df["user_id"] == 3) & (df["embedding_name"] == "vit_b16_D128")].index
        )

        with pytest.raises(ValueError) as excinfo:
            pairwise_significance(df, metric=_METRIC, n_iterations=20)

        message = str(excinfo.value)
        assert "vbpr_vit_b16_D128" in message
        assert "3" in message

    def test_should_reject_extra_user_in_one_config(self) -> None:
        df = _frame(_TWO)
        extra = pd.DataFrame(
            [{"user_id": 99, "model_name": "vbpr", "embedding_name": "resnet50_D128", _METRIC: 0.1}]
        )
        df = pd.concat([df, extra], ignore_index=True)

        with pytest.raises(ValueError, match="population"):
            pairwise_significance(df, metric=_METRIC, n_iterations=20)

    def test_friedman_should_reject_unequal_populations(self) -> None:
        three = _TWO + [("vbpr", "convnext_base_D128")]
        df = _frame(three)
        df = df.drop(
            index=df[(df["user_id"] == 0) & (df["embedding_name"] == "vit_b16_D128")].index
        )

        with pytest.raises(ValueError, match="population"):
            friedman_test(df, metric=_METRIC)

    def test_declared_intersection_reports_excluded_counts(self) -> None:
        df = _frame(_TWO)
        df = df.drop(
            index=df[(df["user_id"] == 3) & (df["embedding_name"] == "vit_b16_D128")].index
        )

        out = pairwise_significance(
            df, metric=_METRIC, n_iterations=20, population="declared_intersection"
        )

        row = out.iloc[0]
        assert row["population_policy"] == "declared_intersection"
        assert row["n_pairs"] == 7
        assert row["n_excluded_a"] == 0
        assert row["n_excluded_b"] == 1

    def test_strict_policy_rows_carry_zero_exclusions(self) -> None:
        out = pairwise_significance(_frame(_TWO), metric=_METRIC, n_iterations=20)

        assert (out["population_policy"] == "strict").all()
        assert (out["n_excluded_a"] == 0).all()
        assert (out["n_excluded_b"] == 0).all()

    def test_unknown_population_policy_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="population"):
            pairwise_significance(_frame(_TWO), metric=_METRIC, population="whatever")


class TestProvenanceHomogeneity:
    def test_should_reject_mixed_seeds(self) -> None:
        df = pd.concat([_frame(_TWO, seed=42), _frame(_TWO, seed=99)], ignore_index=True)

        with pytest.raises(ValueError, match="seed"):
            pairwise_significance(df, metric=_METRIC, n_iterations=20)

    def test_should_reject_mixed_protocols(self) -> None:
        df = pd.concat(
            [_frame([_TWO[0]], protocol="full_ranking"), _frame([_TWO[1]], protocol="sampled")],
            ignore_index=True,
        )

        with pytest.raises(ValueError, match="protocol"):
            pairwise_significance(df, metric=_METRIC, n_iterations=20)

    def test_should_reject_mixed_identity_within_a_config(self) -> None:
        # Two different checkpoints wrote rows under the same config label.
        df = _frame(_TWO).assign(n_trainable_params=1000)
        df.loc[0, "n_trainable_params"] = 2000

        with pytest.raises(ValueError, match="identity"):
            pairwise_significance(df, metric=_METRIC, n_iterations=20)

    def test_should_reject_non_finite_metric(self) -> None:
        df = _frame(_TWO)
        df.loc[2, _METRIC] = float("nan")

        with pytest.raises(ValueError, match="finite"):
            per_model_summary(df, metric=_METRIC, n_iterations=20)


class TestOrderInvarianceAndIdempotence:
    def test_reversed_rows_give_identical_pairwise_results(self) -> None:
        df = _frame(_TWO, n_users=30)

        forward = pairwise_significance(df, metric=_METRIC, n_iterations=50)
        backward = pairwise_significance(df.iloc[::-1], metric=_METRIC, n_iterations=50)

        pd.testing.assert_frame_equal(forward, backward)

    def test_reversed_rows_give_identical_summary(self) -> None:
        df = _frame(_TWO, n_users=30)

        forward = per_model_summary(df, metric=_METRIC, n_iterations=50)
        backward = per_model_summary(df.iloc[::-1], metric=_METRIC, n_iterations=50)

        pd.testing.assert_frame_equal(forward, backward)

    def test_validation_is_idempotent(self) -> None:
        frozen = _frame([("bpr", "none"), ("vbpr", "resnet50_D128")], condition="frozen")
        finetuned = _frame([("bpr", "none")], condition="finetuned")
        df = pd.concat([frozen, finetuned], ignore_index=True)

        once = _ensure_config(df)
        twice = _ensure_config(once)

        pd.testing.assert_frame_equal(
            once.reset_index(drop=True), twice.reset_index(drop=True), check_like=True
        )

    def test_shuffled_rows_do_not_change_wilcoxon_or_ci(self) -> None:
        df = _frame(_TWO, n_users=40)
        shuffled = df.sample(frac=1.0, random_state=3)

        a = pairwise_significance(df, metric=_METRIC, n_iterations=50)
        b = pairwise_significance(shuffled, metric=_METRIC, n_iterations=50)

        assert a["p_value"].iloc[0] == b["p_value"].iloc[0]
        assert np.allclose(a["diff_ci_lower"], b["diff_ci_lower"])


# --------------------------------------------------------------------------
# Per-user artifact path (paired_loader)
# --------------------------------------------------------------------------

from src.evaluation.paired_loader import UserSetMismatchError, load_paired  # noqa: E402
from src.evaluation.paired_validation import (  # noqa: E402
    DuplicateObservationError,
    ProvenanceMismatchError,
    user_population_digest,
)
from src.evaluation.persistence import CellMetadata, write_cell_artifact  # noqa: E402


def _cell_meta(recommender: str, visual: str, **overrides) -> CellMetadata:
    fields = dict(
        dataset="toy",
        visual_config=visual,
        recommender=recommender,
        seed=42,
        d=8,
        split="test",
        n_users=6,
        n_items=40,
    )
    fields.update(overrides)
    return CellMetadata(**fields)


def _cell_records(users: list[int], ranks: list[int] | None = None) -> pd.DataFrame:
    ranks = ranks if ranks is not None else [3] * len(users)
    return pd.DataFrame(
        {
            "user_id": users,
            "rank": ranks,
            "n_candidates": [39] * len(users),
            "tie_block_size": [1] * len(users),
            "top_items": [[u, u + 1] for u in users],
        }
    )


def _rewrite_meta(records_path, **fields) -> None:
    import json

    meta_path = records_path.with_name(records_path.name.replace(".csv.gz", ".meta.json"))
    meta = json.loads(meta_path.read_text())
    meta.update(fields)
    meta_path.write_text(json.dumps(meta))


class TestPairedLoaderValidation:
    def test_should_reject_repeated_user_in_a_cell(self, tmp_path) -> None:
        write_cell_artifact(_cell_records([0, 1, 1, 2]), _cell_meta("bpr", "none"), tmp_path)

        with pytest.raises(DuplicateObservationError, match="repeated user_idx"):
            load_paired(tmp_path, dataset="toy", seed=42, metric="recall", k=5)

    def test_should_still_reject_user_set_mismatch(self, tmp_path) -> None:
        write_cell_artifact(_cell_records([0, 1, 2]), _cell_meta("bpr", "none"), tmp_path)
        write_cell_artifact(_cell_records([0, 1]), _cell_meta("vbpr", "resnet"), tmp_path)

        with pytest.raises(UserSetMismatchError):
            load_paired(tmp_path, dataset="toy", seed=42, metric="recall", k=5)

    def test_should_reject_mixed_protocol_versions(self, tmp_path) -> None:
        write_cell_artifact(_cell_records([0, 1, 2]), _cell_meta("bpr", "none"), tmp_path)
        write_cell_artifact(
            _cell_records([0, 1, 2]),
            _cell_meta("vbpr", "resnet", eval_protocol_version="1.0"),
            tmp_path,
        )

        with pytest.raises(ProvenanceMismatchError, match="provenance"):
            load_paired(tmp_path, dataset="toy", seed=42, metric="recall", k=5)

    def test_should_reject_metadata_filed_under_another_dataset(self, tmp_path) -> None:
        path = write_cell_artifact(_cell_records([0, 1, 2]), _cell_meta("bpr", "none"), tmp_path)
        _rewrite_meta(path, dataset="other")

        with pytest.raises(ProvenanceMismatchError, match="dataset"):
            load_paired(tmp_path, dataset="toy", seed=42, metric="recall", k=5)

    def test_should_honour_manifest_completion_fields_when_present(self, tmp_path) -> None:
        path = write_cell_artifact(_cell_records([0, 1, 2]), _cell_meta("bpr", "none"), tmp_path)
        _rewrite_meta(path, row_count=3, expected_user_digest=user_population_digest([0, 1, 2]))

        matrix = load_paired(tmp_path, dataset="toy", seed=42, metric="recall", k=5)

        assert matrix.index.tolist() == [0, 1, 2]

    def test_should_reject_records_short_of_declared_row_count(self, tmp_path) -> None:
        path = write_cell_artifact(_cell_records([0, 1, 2]), _cell_meta("bpr", "none"), tmp_path)
        _rewrite_meta(path, row_count=4)

        with pytest.raises(ValueError, match="row_count"):
            load_paired(tmp_path, dataset="toy", seed=42, metric="recall", k=5)

    def test_should_reject_records_whose_users_differ_from_declared_digest(self, tmp_path) -> None:
        path = write_cell_artifact(_cell_records([0, 1, 2]), _cell_meta("bpr", "none"), tmp_path)
        _rewrite_meta(path, expected_user_digest=user_population_digest([0, 1, 3]))

        with pytest.raises(ProvenanceMismatchError, match="expected_user_digest"):
            load_paired(tmp_path, dataset="toy", seed=42, metric="recall", k=5)

    def test_legacy_metadata_without_completion_fields_is_accepted(self, tmp_path) -> None:
        write_cell_artifact(_cell_records([0, 1, 2]), _cell_meta("bpr", "none"), tmp_path)
        write_cell_artifact(_cell_records([2, 0, 1]), _cell_meta("vbpr", "resnet"), tmp_path)

        matrix = load_paired(tmp_path, dataset="toy", seed=42, metric="recall", k=5)

        assert set(matrix.columns) == {"bpr__none", "vbpr__resnet"}
        assert matrix.index.tolist() == [0, 1, 2]
