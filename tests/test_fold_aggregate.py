"""Fold artifact concatenation and between-fold descriptive statistics."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.evaluation.paired_loader import load_paired
from src.evaluation.persistence import CellMetadata, artifact_paths, read_cell_artifact
from src.folds import FoldAggregate, concatenate_fold_artifacts, fold_dir, write_fold_artifact

_K_VALUES = [5, 10]


def _metadata(recommender: str = "vbpr", visual_config: str = "resnet") -> CellMetadata:
    return CellMetadata(
        dataset="toy",
        visual_config=visual_config,
        recommender=recommender,
        seed=42,
        d=8,
        split="test",
        n_users=9,
        n_items=50,
    )


def _records(users: list[int], ranks: list[int]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_id": users,
            "rank": ranks,
            "n_candidates": [49] * len(users),
            "tie_block_size": [1] * len(users),
            "top_items": [[u, u + 1] for u in users],
        }
    )


# Three folds with interleaved users so that "sorted by user" differs
# from "fold order", and with ranks chosen to make hand computation easy.
_FOLDS: list[tuple[list[int], list[int]]] = [
    ([0, 3, 6], [1, 3, 20]),
    ([1, 4, 7], [2, 2, 2]),
    ([2, 5, 8], [1, 11, 40]),
]


def _write_folds(
    out_dir: Path, metadata: CellMetadata, folds=_FOLDS, *, k: int | None = None
) -> list[Path]:
    planned_k = k if k is not None else len(folds)
    return [
        write_fold_artifact(
            _records(users, ranks), metadata, i, out_dir, k=planned_k, fold_seed=100 + i
        )
        for i, (users, ranks) in enumerate(folds)
    ]


def _meta_path(records_path: Path) -> Path:
    return records_path.with_name(records_path.name.replace(".csv.gz", ".meta.json"))


class TestWriteFoldArtifact:
    def test_should_write_partial_under_folds_dir_with_fold_provenance(
        self, tmp_path: Path
    ) -> None:
        metadata = _metadata()

        path = write_fold_artifact(
            _records([0, 3], [1, 2]), metadata, 2, tmp_path, k=3, fold_seed=7
        )

        assert path.parent == fold_dir(tmp_path, 2) / "per_user" / "toy"
        assert not path.with_name(path.name.replace(".csv.gz", ".fold.json")).exists()
        meta, records = read_cell_artifact(path)
        assert meta["fold"] == {"index": 2, "k": 3, "seed": 7, "n_users": 2}
        assert {k: v for k, v in meta.items() if k != "fold"} == {
            k: v for k, v in metadata.to_dict().items() if k != "fold"
        }
        assert records["user_idx"].tolist() == [0, 3]

    def test_should_not_mutate_the_caller_metadata(self, tmp_path: Path) -> None:
        metadata = _metadata()

        write_fold_artifact(_records([0], [1]), metadata, 0, tmp_path, k=2, fold_seed=1)

        assert metadata.fold is None

    def test_should_reject_negative_fold_index(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="fold_index"):
            write_fold_artifact(_records([0], [1]), _metadata(), -1, tmp_path, k=2, fold_seed=1)

    def test_should_reject_fold_index_at_or_above_k(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="fold_index must be < k=2"):
            write_fold_artifact(_records([0], [1]), _metadata(), 2, tmp_path, k=2, fold_seed=1)


class TestConcatenateFoldArtifacts:
    def test_should_preserve_total_users_and_sort_by_user(self, tmp_path: Path) -> None:
        metadata = _metadata()
        _write_folds(tmp_path, metadata)

        path, agg = concatenate_fold_artifacts(tmp_path, metadata, k=3, k_values=_K_VALUES)

        _, records = read_cell_artifact(path)
        assert len(records) == sum(len(u) for u, _ in _FOLDS) == agg.n_users_total == 9
        assert records["user_idx"].tolist() == list(range(9))
        assert agg.n_users_per_fold == [3, 3, 3]

    def test_should_write_to_canonical_cell_location(self, tmp_path: Path) -> None:
        metadata = _metadata()
        _write_folds(tmp_path, metadata)

        path, _ = concatenate_fold_artifacts(tmp_path, metadata, k=3, k_values=_K_VALUES)

        records_path, meta_path = artifact_paths(tmp_path, metadata)
        assert path == records_path
        written = json.loads(meta_path.read_text())
        assert written["fold"] == {"k": 3, "seeds": [100, 101, 102], "n_users_per_fold": [3, 3, 3]}
        assert {k: v for k, v in written.items() if k != "fold"} == {
            k: v for k, v in metadata.to_dict().items() if k != "fold"
        }
        assert metadata.fold is None

    def test_should_keep_per_user_ranks_identical_to_partials(self, tmp_path: Path) -> None:
        metadata = _metadata()
        partial_paths = _write_folds(tmp_path, metadata)
        expected = pd.concat(
            [read_cell_artifact(p)[1] for p in partial_paths], ignore_index=True
        ).set_index("user_idx")

        path, _ = concatenate_fold_artifacts(tmp_path, metadata, k=3, k_values=_K_VALUES)

        _, records = read_cell_artifact(path)
        final = records.set_index("user_idx").loc[expected.index]
        assert final["rank"].tolist() == expected["rank"].tolist()
        assert final["n_candidates"].tolist() == expected["n_candidates"].tolist()
        assert final["tie_block_size"].tolist() == expected["tie_block_size"].tolist()
        assert final["top_items"].tolist() == expected["top_items"].tolist()

    def test_should_raise_when_user_repeats_across_folds(self, tmp_path: Path) -> None:
        metadata = _metadata()
        folds = [([0, 1], [1, 1]), ([1, 2], [1, 1])]
        _write_folds(tmp_path, metadata, folds)

        with pytest.raises(ValueError, match="user_idx \\[1\\].*fold 1"):
            concatenate_fold_artifacts(tmp_path, metadata, k=2, k_values=_K_VALUES)

    def test_should_raise_when_a_fold_is_missing(self, tmp_path: Path) -> None:
        metadata = _metadata()
        _write_folds(tmp_path, metadata, _FOLDS[:2], k=3)

        with pytest.raises(FileNotFoundError, match="fold 2"):
            concatenate_fold_artifacts(tmp_path, metadata, k=3, k_values=_K_VALUES)

    def test_should_raise_when_partial_fold_index_disagrees(self, tmp_path: Path) -> None:
        metadata = _metadata()
        paths = _write_folds(tmp_path, metadata, _FOLDS[:2])
        meta = json.loads(_meta_path(paths[1]).read_text())
        meta["fold"]["index"] = 0
        _meta_path(paths[1]).write_text(json.dumps(meta))

        with pytest.raises(ValueError, match="claims fold 0"):
            concatenate_fold_artifacts(tmp_path, metadata, k=2, k_values=_K_VALUES)

    def test_should_raise_when_partial_k_disagrees(self, tmp_path: Path) -> None:
        metadata = _metadata()
        paths = _write_folds(tmp_path, metadata, _FOLDS[:2])
        meta = json.loads(_meta_path(paths[0]).read_text())
        meta["fold"]["k"] = 5
        _meta_path(paths[0]).write_text(json.dumps(meta))

        with pytest.raises(ValueError, match="claims k=5"):
            concatenate_fold_artifacts(tmp_path, metadata, k=2, k_values=_K_VALUES)

    def test_should_raise_when_partial_lacks_fold_provenance(self, tmp_path: Path) -> None:
        metadata = _metadata()
        paths = _write_folds(tmp_path, metadata, _FOLDS[:2])
        meta = json.loads(_meta_path(paths[0]).read_text())
        meta["fold"] = None
        _meta_path(paths[0]).write_text(json.dumps(meta))

        with pytest.raises(ValueError, match="no fold provenance"):
            concatenate_fold_artifacts(tmp_path, metadata, k=2, k_values=_K_VALUES)

    def test_should_reject_k_below_two(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="k must be >= 2"):
            concatenate_fold_artifacts(tmp_path, _metadata(), k=1)


class TestBetweenFoldStatistics:
    def test_should_match_manual_mean_and_sample_std(self, tmp_path: Path) -> None:
        metadata = _metadata()
        _write_folds(tmp_path, metadata)

        _, agg = concatenate_fold_artifacts(tmp_path, metadata, k=3, k_values=_K_VALUES)

        # recall@5 per fold: fold0 ranks (1,3,20) -> 2/3; fold1 (2,2,2) -> 1; fold2 (1,11,40) -> 1/3
        recall5 = np.array([2 / 3, 1.0, 1 / 3])
        # ndcg@5: fold0 (1 + 1/log2(4) + 0)/3; fold1 3*(1/log2(3))/3; fold2 (1 + 0 + 0)/3
        ndcg5 = np.array([(1 + 0.5) / 3, 1 / np.log2(3), 1 / 3])
        assert [m["recall@5"] for m in agg.per_fold_metrics] == pytest.approx(recall5)
        assert [m["ndcg@5"] for m in agg.per_fold_metrics] == pytest.approx(ndcg5)
        assert agg.between_fold_mean["recall@5"] == pytest.approx(recall5.mean())
        assert agg.between_fold_std["recall@5"] == pytest.approx(recall5.std(ddof=1))
        assert agg.between_fold_mean["ndcg@5"] == pytest.approx(ndcg5.mean())
        assert agg.between_fold_std["ndcg@5"] == pytest.approx(ndcg5.std(ddof=1))
        # recall@10 per fold: (1,3,20) -> 2/3; (2,2,2) -> 1; (1,11,40) -> 1/3 (11 > 10)
        assert agg.between_fold_mean["recall@10"] == pytest.approx(recall5.mean())

    def test_should_only_report_recall_and_ndcg_at_requested_cutoffs(self, tmp_path: Path) -> None:
        metadata = _metadata()
        _write_folds(tmp_path, metadata)

        _, agg = concatenate_fold_artifacts(tmp_path, metadata, k=3, k_values=_K_VALUES)

        assert set(agg.between_fold_mean) == {"recall@5", "ndcg@5", "recall@10", "ndcg@10"}
        assert set(agg.between_fold_std) == set(agg.between_fold_mean)

    def test_to_dict_should_be_json_serialisable(self, tmp_path: Path) -> None:
        metadata = _metadata()
        _write_folds(tmp_path, metadata)

        _, agg = concatenate_fold_artifacts(tmp_path, metadata, k=3, k_values=_K_VALUES)

        payload = json.loads(json.dumps(agg.to_dict()))
        assert isinstance(agg, FoldAggregate)
        assert payload["n_users_total"] == 9
        assert payload["n_users_per_fold"] == [3, 3, 3]
        assert len(payload["per_fold_metrics"]) == 3


class TestStatisticalPipelineCompatibility:
    def test_paired_loader_should_consume_concatenated_cells_like_plain_cells(
        self, tmp_path: Path
    ) -> None:
        # Two systems: one from folds, one written as a plain leave-one-out cell.
        folded = _metadata(recommender="vbpr")
        _write_folds(tmp_path, folded)
        concatenate_fold_artifacts(tmp_path, folded, k=3, k_values=_K_VALUES)
        from src.evaluation.persistence import write_cell_artifact

        plain = _metadata(recommender="bpr", visual_config="none")
        write_cell_artifact(_records(list(range(9)), [4] * 9), plain, tmp_path)

        matrix = load_paired(tmp_path, dataset="toy", seed=42, metric="recall", k=5)

        assert matrix.index.tolist() == list(range(9))
        assert set(matrix.columns) == {"vbpr__resnet", "bpr__none"}
        expected_folded = {
            u: float(r <= 5) for users, ranks in _FOLDS for u, r in zip(users, ranks, strict=True)
        }
        assert matrix["vbpr__resnet"].to_dict() == expected_folded
        assert matrix["bpr__none"].tolist() == [1.0] * 9

    def test_partials_should_not_be_discovered_as_cells(self, tmp_path: Path) -> None:
        metadata = _metadata()
        _write_folds(tmp_path, metadata)
        concatenate_fold_artifacts(tmp_path, metadata, k=3, k_values=_K_VALUES)

        matrix = load_paired(tmp_path, dataset="toy", seed=42, metric="ndcg", k=10)

        assert list(matrix.columns) == ["vbpr__resnet"]
        assert len(matrix) == 9
