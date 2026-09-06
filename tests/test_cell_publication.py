"""Per-cell evaluation generations are published atomically (E06, Q14, C05).

Interruption is injected after every publication boundary (payload,
metadata, records publication, pointer).  After each one the canonical
pair must be either the previous complete generation, an incomplete new
one, or a torn pair that every reader rejects — never a silently
accepted mixture — and a repeated publication must leave exactly one
observation per cell in the summary CSV.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

import src.evaluation.persistence as persistence
from src.battery.manifest import is_cell_complete
from src.evaluation.persistence import (
    STAGE_METADATA,
    STAGE_PAYLOAD,
    STAGE_POINTER,
    STAGE_RECORDS_PUBLISHED,
    ArtifactIntegrityError,
    CellMetadata,
    ConcurrentPublicationError,
    artifact_paths,
    list_generations,
    read_cell_artifact,
    validate_cell_artifact,
    write_cell_artifact,
)
from src.steps.evaluate import _append_cell

STAGES = [STAGE_PAYLOAD, STAGE_METADATA, STAGE_RECORDS_PUBLISHED, STAGE_POINTER]


class _Interrupted(RuntimeError):
    """Stands in for a kill at a durable boundary (no cleanup runs)."""


def _meta(seed: int = 1) -> CellMetadata:
    return CellMetadata(
        dataset="ds", visual_config="resnet50", recommender="vbpr", seed=seed, d=4, split="test"
    )


def _records(ranks: list[int]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_idx": list(range(len(ranks))),
            "rank": ranks,
            "n_candidates": [10] * len(ranks),
            "tie_block_size": [1] * len(ranks),
            "top_items": [[1, 2]] * len(ranks),
        }
    )


def _interrupt_after(monkeypatch, stage: str) -> None:
    def _hook(name: str) -> None:
        if name == stage:
            raise _Interrupted(stage)

    monkeypatch.setattr(persistence, "_stage", _hook)


def _canonical_bytes(out: Path) -> tuple[bytes | None, bytes | None]:
    records, meta = artifact_paths(out, _meta())
    return (
        records.read_bytes() if records.exists() else None,
        meta.read_bytes() if meta.exists() else None,
    )


class TestFirstPublicationInterrupted:
    @pytest.mark.parametrize("stage", STAGES[:-1])
    def test_no_reader_accepts_an_incomplete_first_generation(
        self, tmp_path, monkeypatch, stage
    ) -> None:
        _interrupt_after(monkeypatch, stage)
        with pytest.raises(_Interrupted):
            write_cell_artifact(_records([1, 2, 3]), _meta(), tmp_path)
        records, _meta_path = artifact_paths(tmp_path, _meta())

        with pytest.raises(ArtifactIntegrityError):
            validate_cell_artifact(records)
        with pytest.raises(ArtifactIntegrityError):
            read_cell_artifact(records)
        assert not is_cell_complete(_cell(), tmp_path)

    def test_interruption_after_the_pointer_is_a_complete_generation(
        self, tmp_path, monkeypatch
    ) -> None:
        _interrupt_after(monkeypatch, STAGE_POINTER)
        with pytest.raises(_Interrupted):
            write_cell_artifact(_records([1, 2, 3]), _meta(), tmp_path)
        records, _ = artifact_paths(tmp_path, _meta())

        completion = validate_cell_artifact(records)
        _, df = read_cell_artifact(records)

        assert completion is not None and completion.row_count == 3
        assert list(df["rank"]) == [1, 2, 3]
        assert is_cell_complete(_cell(), tmp_path)


class TestRepublicationInterrupted:
    @pytest.mark.parametrize("stage", STAGES)
    def test_old_generation_or_torn_pair_never_a_silent_mixture(
        self, tmp_path, monkeypatch, stage
    ) -> None:
        write_cell_artifact(_records([1, 2, 3]), _meta(), tmp_path)
        before = _canonical_bytes(tmp_path)
        records, meta_path = artifact_paths(tmp_path, _meta())

        _interrupt_after(monkeypatch, stage)
        with pytest.raises(_Interrupted):
            write_cell_artifact(_records([9, 9, 9]), _meta(), tmp_path)
        monkeypatch.setattr(persistence, "_stage", lambda name: None)

        after = _canonical_bytes(tmp_path)
        if stage in (STAGE_PAYLOAD, STAGE_METADATA):
            assert after == before, "the previous generation must be untouched"
            _, df = read_cell_artifact(records)
            assert list(df["rank"]) == [1, 2, 3]
        elif stage == STAGE_RECORDS_PUBLISHED:
            # New payload, old pointer: torn — rejected by every reader.
            assert after[1] == before[1] and after[0] != before[0]
            with pytest.raises(ArtifactIntegrityError, match="torn"):
                read_cell_artifact(records)
            assert not is_cell_complete(_cell(), tmp_path)
        else:
            _, df = read_cell_artifact(records)
            assert list(df["rank"]) == [9, 9, 9]
            assert json.loads(meta_path.read_text())["row_count"] == 3

    def test_a_retry_after_any_interruption_publishes_cleanly(self, tmp_path, monkeypatch) -> None:
        write_cell_artifact(_records([1, 2, 3]), _meta(), tmp_path)
        for stage in STAGES:
            _interrupt_after(monkeypatch, stage)
            with pytest.raises(_Interrupted):
                write_cell_artifact(_records([5, 5, 5]), _meta(), tmp_path)
        monkeypatch.setattr(persistence, "_stage", lambda name: None)

        write_cell_artifact(_records([7, 7, 7]), _meta(), tmp_path)

        records, _ = artifact_paths(tmp_path, _meta())
        _, df = read_cell_artifact(records)
        assert list(df["rank"]) == [7, 7, 7]
        assert len(list_generations(records)) == 6, "generations are immutable, never deleted"


def _cell():
    from src.battery.cells import BatteryCell

    return BatteryCell(
        dataset="ds", visual_config="resnet50", recommender="vbpr", seed=1, role="search"
    )


class TestReaders:
    def test_row_count_mismatch_is_rejected(self, tmp_path) -> None:
        records, _ = artifact_paths(tmp_path, _meta())
        write_cell_artifact(_records([1, 2, 3]), _meta(), tmp_path)
        meta_path = records.with_name(records.name.replace(".csv.gz", ".meta.json"))
        meta = json.loads(meta_path.read_text())
        meta["completion"]["row_count"] = 2
        meta_path.write_text(json.dumps(meta))

        with pytest.raises(ArtifactIntegrityError, match="declares"):
            read_cell_artifact(records)

    def test_legacy_artifact_without_completion_is_identified_not_completed(self, tmp_path) -> None:
        records, meta_path = artifact_paths(tmp_path, _meta())
        records.parent.mkdir(parents=True)
        df = _records([1, 2])
        df["top_items"] = df["top_items"].map(json.dumps)
        df.to_csv(records, index=False, compression="gzip")
        meta_path.write_text(json.dumps(_meta().to_dict()))

        assert validate_cell_artifact(records) is None
        metadata, loaded = read_cell_artifact(records)
        assert len(loaded) == 2 and "completion" not in metadata
        assert not is_cell_complete(_cell(), tmp_path)

    def test_expected_user_population_is_enforced(self, tmp_path) -> None:
        with pytest.raises(ArtifactIntegrityError, match="user population"):
            write_cell_artifact(_records([1, 2]), _meta(), tmp_path, expected_users=[0, 1, 2])

    def test_completion_records_identity_and_users(self, tmp_path) -> None:
        records, _ = artifact_paths(tmp_path, _meta())
        write_cell_artifact(
            _records([1, 2]), _meta(), tmp_path, expected_users=[1, 0], identity_digest="abc"
        )
        completion = validate_cell_artifact(records)
        assert completion.identity_digest == "abc" and completion.row_count == 2
        metadata, _ = read_cell_artifact(records)
        assert metadata["row_count"] == 2 and metadata["expected_user_digest"]

    def test_concurrent_publisher_is_rejected(self, tmp_path, monkeypatch) -> None:
        records, _ = artifact_paths(tmp_path, _meta())
        records.parent.mkdir(parents=True)
        lock = persistence._CellLock(records)
        with lock, pytest.raises(ConcurrentPublicationError):
            write_cell_artifact(_records([1]), _meta(), tmp_path)


class TestSummaryRowsAreIdempotent:
    def test_republished_cell_replaces_its_rows(self, tmp_path) -> None:
        path = tmp_path / "ds_evaluation_frozen.csv"
        cell = pd.DataFrame({"user_id": [1, 2], "ndcg@10": [0.1, 0.2]})
        _append_cell(cell.assign(model_name="vbpr", embedding_name="resnet50"), path)
        _append_cell(cell.assign(model_name="bpr", embedding_name="none"), path)
        _append_cell(
            pd.DataFrame({"user_id": [1, 2], "ndcg@10": [0.9, 0.8]}).assign(
                model_name="vbpr", embedding_name="resnet50"
            ),
            path,
        )

        out = pd.read_csv(path)

        assert len(out) == 4
        vbpr = out[out["model_name"] == "vbpr"].sort_values("user_id")
        assert list(vbpr["ndcg@10"]) == [0.9, 0.8]
