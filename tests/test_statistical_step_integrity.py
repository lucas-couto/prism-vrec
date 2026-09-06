"""R05 (Q17): aggregate reports validate identity/completeness and partition outputs.

The statistical step wrote ``{dataset}_{kind}.csv`` whatever the condition
it was asked for, so ``--condition frozen`` followed by ``finetuned``
silently overwrote the first comparison; it never reconciled the cells it
tested against the evaluate step's completion record, and cross-seed
aggregation counted rows instead of seeds.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.evaluation.paired_validation import PairedValidationError
from src.steps import statistical as stat_step

_DS = "toyset"
_N_USERS = 12
_KS = [5]
_CELLS_FROZEN = [("bpr", "none"), ("vbpr", "resnet50_D8"), ("vbpr", "vit_b16_D8")]
_CELLS_FINETUNED = [
    ("bpr", "none"),
    ("vbpr", "resnet50_finetuned_D8"),
    ("vbpr", "vit_b16_finetuned_D8"),
]


def _eval_rows(cells: list[tuple[str, str]], n_users: int = _N_USERS) -> list[dict]:
    rows = []
    for u in range(n_users):
        for i, (model, embedding) in enumerate(cells):
            rows.append(
                {
                    "user_id": u,
                    "recall@5": float((u + i) % 3 == 0),
                    "ndcg@5": 0.1 * (i + 1) + 0.01 * (u % 4),
                    "dataset": _DS,
                    "model_name": model,
                    "embedding_name": embedding,
                    "protocol": "full_ranking",
                }
            )
    return rows


def _write_eval(tables: Path, target: str, rows: list[dict]) -> Path:
    tables.mkdir(parents=True, exist_ok=True)
    path = tables / f"{_DS}_evaluation_{target}.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _write_done(tables: Path, rows: list[tuple[str, str, str]]) -> None:
    pd.DataFrame(rows, columns=["target", "model_name", "embedding_name"]).to_csv(
        tables / f"{_DS}_evaluation_done.csv", index=False
    )


def _config(tmp_path: Path, **statistical) -> dict:
    stat = {
        "bootstrap": {"enabled": True, "n_iterations": 20},
        "families": ["backbone_within_model", "vs_baseline", "frozen_vs_finetuned"],
    }
    stat.update(statistical)
    return {
        "seed": 42,
        "datasets": [_DS],
        "k_values": _KS,
        "paths": {"results": str(tmp_path / "results")},
        "statistical": stat,
    }


@pytest.fixture()
def tables(tmp_path: Path) -> Path:
    tables = tmp_path / "results" / "tables"
    _write_eval(tables, "frozen", _eval_rows(_CELLS_FROZEN))
    _write_eval(tables, "finetuned", _eval_rows(_CELLS_FINETUNED))
    _write_done(
        tables,
        [("frozen", m, e) for m, e in _CELLS_FROZEN]
        + [("finetuned", m, e) for m, e in _CELLS_FINETUNED],
    )
    return tables


def _run(monkeypatch: pytest.MonkeyPatch, config: dict, condition: str) -> None:
    monkeypatch.setattr(stat_step, "load_config", lambda: config)
    stat_step.run(condition=condition)


class TestPartitionedOutputs:
    def test_frozen_and_finetuned_reports_do_not_overwrite_each_other(
        self, tables: Path, tmp_path: Path, monkeypatch
    ) -> None:
        _run(monkeypatch, _config(tmp_path), "frozen")
        frozen_pairwise = pd.read_csv(tables / f"{_DS}_frozen_pairwise.csv")

        _run(monkeypatch, _config(tmp_path), "finetuned")

        assert (tables / f"{_DS}_finetuned_pairwise.csv").exists()
        after = pd.read_csv(tables / f"{_DS}_frozen_pairwise.csv")
        pd.testing.assert_frame_equal(frozen_pairwise, after)
        assert not (tables / f"{_DS}_pairwise.csv").exists()
        assert set(after["config_a"]) <= {f"{m}_{e}" for m, e in _CELLS_FROZEN}

    def test_all_condition_gets_its_own_partition(self, tables, tmp_path, monkeypatch) -> None:
        _run(monkeypatch, _config(tmp_path), "all")

        pairwise = pd.read_csv(tables / f"{_DS}_all_pairwise.csv")
        assert (pairwise["family"] == "frozen_vs_finetuned").any()
        assert (tables / f"{_DS}_all_summary.csv").exists()

    def test_restricted_population_is_a_separate_partition(
        self, tables, tmp_path, monkeypatch
    ) -> None:
        _run(monkeypatch, _config(tmp_path, population="declared_intersection"), "frozen")

        pairwise = pd.read_csv(tables / f"{_DS}_frozen_restricted_pairwise.csv")
        assert (pairwise["population_policy"] == "declared_intersection").all()
        assert not (tables / f"{_DS}_frozen_pairwise.csv").exists()

    def test_long_format_consolidation_keeps_partitions_apart(
        self, tables, tmp_path, monkeypatch
    ) -> None:
        _run(monkeypatch, _config(tmp_path), "frozen")
        _run(monkeypatch, _config(tmp_path), "all")

        tests = pd.read_csv(tables / "statistical_tests.csv")
        assert set(tests["report_condition"]) == {"frozen", "all"}
        ci = pd.read_csv(tables / "bootstrap_ci.csv")
        assert set(ci["report_condition"]) == {"frozen", "all"}


class TestIntegrityReport:
    def test_writes_seed_and_cell_reconciliation(self, tables, tmp_path, monkeypatch) -> None:
        _run(monkeypatch, _config(tmp_path), "frozen")

        report = json.loads((tables / f"{_DS}_frozen_integrity.json").read_text())
        assert report["n_seeds_distinct"] == 1
        assert report["seed"] == 42
        assert report["population_policy"] == "strict"
        assert report["n_cells_expected"] == 3
        assert report["n_cells_completed"] == 3
        assert report["cells_missing"] == []
        assert report["cells_excluded"] == {}
        assert report["n_users_reference"] == _N_USERS
        assert report["provenance"]["protocol"] == "full_ranking"

    def test_cell_recorded_done_but_absent_from_table_is_reported_and_rejected(
        self, tables, tmp_path, monkeypatch
    ) -> None:
        _write_done(tables, [("frozen", m, e) for m, e in _CELLS_FROZEN + [("acf", "resnet50_D8")]])

        with pytest.raises(PairedValidationError, match="acf_resnet50_D8"):
            _run(monkeypatch, _config(tmp_path), "frozen")

        report = json.loads((tables / f"{_DS}_frozen_integrity.json").read_text())
        assert report["cells_missing"] == ["acf_resnet50_D8"]
        assert report["n_cells_expected"] == 4
        assert report["n_cells_completed"] == 3

    def test_incomplete_cell_is_excluded_with_reason_and_rejected_under_strict(
        self, tables, tmp_path, monkeypatch
    ) -> None:
        rows = [
            r
            for r in _eval_rows(_CELLS_FROZEN)
            if not (r["embedding_name"] == "vit_b16_D8" and r["user_id"] in (3, 4))
        ]
        _write_eval(tables, "frozen", rows)

        with pytest.raises(PairedValidationError, match="vbpr_vit_b16_D8"):
            _run(monkeypatch, _config(tmp_path), "frozen")

        report = json.loads((tables / f"{_DS}_frozen_integrity.json").read_text())
        assert "vbpr_vit_b16_D8" in report["cells_excluded"]
        assert "10 of 12" in report["cells_excluded"]["vbpr_vit_b16_D8"]

    def test_incomplete_cell_is_kept_with_reason_under_declared_intersection(
        self, tables, tmp_path, monkeypatch
    ) -> None:
        rows = [
            r
            for r in _eval_rows(_CELLS_FROZEN)
            if not (r["embedding_name"] == "vit_b16_D8" and r["user_id"] in (3, 4))
        ]
        _write_eval(tables, "frozen", rows)

        _run(monkeypatch, _config(tmp_path, population="declared_intersection"), "frozen")

        report = json.loads((tables / f"{_DS}_frozen_restricted_integrity.json").read_text())
        assert "vbpr_vit_b16_D8" in report["cells_excluded"]
        pairwise = pd.read_csv(tables / f"{_DS}_frozen_restricted_pairwise.csv")
        row = pairwise[pairwise["config_a"] == "vbpr_vit_b16_D8"].iloc[0]
        assert row["n_excluded_a"] == 2
        assert row["n_pairs"] == 10

    def test_conflicting_duplicate_rows_fail_the_step(self, tables, tmp_path, monkeypatch) -> None:
        rows = _eval_rows(_CELLS_FROZEN)
        clash = dict(rows[1], **{"ndcg@5": 0.99})
        _write_eval(tables, "frozen", rows + [clash])

        with pytest.raises(PairedValidationError, match="conflict"):
            _run(monkeypatch, _config(tmp_path), "frozen")

    def test_mixed_identity_within_a_config_fails_the_step(
        self, tables, tmp_path, monkeypatch
    ) -> None:
        rows = _eval_rows(_CELLS_FROZEN)
        for r in rows:
            r["n_trainable_params"] = 500
        rows[1]["n_trainable_params"] = 999
        _write_eval(tables, "frozen", rows)

        with pytest.raises(PairedValidationError, match="identity"):
            _run(monkeypatch, _config(tmp_path), "frozen")

    def test_without_done_marker_expected_cells_come_from_the_table(
        self, tables, tmp_path, monkeypatch
    ) -> None:
        (tables / f"{_DS}_evaluation_done.csv").unlink()

        _run(monkeypatch, _config(tmp_path), "frozen")

        report = json.loads((tables / f"{_DS}_frozen_integrity.json").read_text())
        assert report["expected_source"] == "table"
        assert report["n_cells_expected"] == 3

    def test_repeated_run_is_idempotent(self, tables, tmp_path, monkeypatch) -> None:
        _run(monkeypatch, _config(tmp_path), "frozen")
        first = pd.read_csv(tables / f"{_DS}_frozen_pairwise.csv")
        first_report = (tables / f"{_DS}_frozen_integrity.json").read_text()

        _run(monkeypatch, _config(tmp_path), "frozen")

        pd.testing.assert_frame_equal(first, pd.read_csv(tables / f"{_DS}_frozen_pairwise.csv"))
        assert (tables / f"{_DS}_frozen_integrity.json").read_text() == first_report


class TestCrossSeedDistinctSeeds:
    def _seed_dir(self, tmp_path: Path, seed: int, value: float) -> Path:
        d = tmp_path / f"results_seed{seed}"
        (d / "tables").mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            [
                {
                    "dataset": _DS,
                    "recommender": "vbpr",
                    "extractor": "resnet50",
                    "fusion": "none",
                    "condition": "frozen",
                    "metric": "ndcg",
                    "k": 10,
                    "mean": value,
                    "n_users": 100,
                }
            ]
        ).to_csv(d / "tables" / "evaluation_aggregated.csv", index=False)
        pd.DataFrame(
            [
                {
                    "dataset": _DS,
                    "metric": "ndcg",
                    "k": 10,
                    "test_type": "wilcoxon",
                    "family": "vs_baseline",
                    "group": "all",
                    "config_a": "vbpr_resnet50",
                    "config_b": "bpr_none",
                    "p_value": 0.01,
                    "corrected_p": 0.02,
                    "significant": True,
                    "diff_mean": value - 0.4,
                }
            ]
        ).to_csv(d / "tables" / "statistical_tests.csv", index=False)
        return d

    def test_duplicated_source_files_still_report_two_seeds(self, tmp_path: Path) -> None:
        from src.reporting.aggregate_seeds import (
            aggregate_evaluation,
            aggregate_statistical_tests,
        )

        d42 = self._seed_dir(tmp_path, 42, 0.5)
        d99 = self._seed_dir(tmp_path, 99, 0.6)
        dirs, seeds = [d42, d99, d42, d99], [42, 99, 42, 99]

        evaluation = aggregate_evaluation(dirs, seeds)
        tests = aggregate_statistical_tests(dirs, seeds)

        assert evaluation.iloc[0]["n_seeds"] == 2
        assert evaluation.iloc[0]["mean_across_seeds"] == pytest.approx(0.55)
        assert tests.iloc[0]["n_seeds"] == 2
        assert tests.iloc[0]["n_seeds_significant"] == 2

    def test_conflicting_rows_for_one_seed_are_rejected(self, tmp_path: Path) -> None:
        from src.reporting.aggregate_seeds import aggregate_evaluation

        d42 = self._seed_dir(tmp_path, 42, 0.5)
        other = self._seed_dir(tmp_path / "elsewhere", 42, 0.7)

        with pytest.raises(PairedValidationError, match="seed 42"):
            aggregate_evaluation([d42, other], [42, 42])

    def test_seed_is_derived_from_directory_name_when_not_given(self, tmp_path: Path) -> None:
        from src.reporting.aggregate_seeds import aggregate_evaluation

        dirs = [self._seed_dir(tmp_path, 42, 0.5), self._seed_dir(tmp_path, 99, 0.6)]

        out = aggregate_evaluation(dirs, seeds=None)

        assert out.iloc[0]["n_seeds"] == 2
