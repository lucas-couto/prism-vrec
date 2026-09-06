"""S04 — the VNPR collapse diagnostic driver runs end to end on synthetic data.

Tiny sizes, CPU, three seeds (the SPEC's minimum diagnostic budget): the
driver must produce one bounded diagnostics JSON per run, a
``summary.json`` whose per-condition counts reconcile
(expected = completed + failed, zero counted separately) and a
``report.md`` carrying the SPEC's hypothesis decision table.  The
real-data entry point is exercised on a synthetic ``processed`` layout so
the user can point it at a dataset later without a code change.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def driver():
    spec = importlib.util.spec_from_file_location(
        "vnpr_collapse_diagnostic", REPO_ROOT / "scripts" / "vnpr_collapse_diagnostic.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


TINY = [
    "--seeds",
    "0",
    "1",
    "2",
    "--learning-rates",
    "0.01",
    "--epochs",
    "2",
    "--eval-every",
    "1",
    "--batch-size",
    "32",
    "--latent-dim",
    "4",
    "--aligned-dim",
    "8",
    "--n-users",
    "24",
    "--n-items",
    "40",
    "--dv-a",
    "8",
    "--dv-b",
    "8",
]


def test_synthetic_run_writes_reconciled_summary_and_decision_table(tmp_path: Path, driver) -> None:
    out = driver.main(["--out", str(tmp_path / "exp"), *TINY])

    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    report = (out / "report.md").read_text(encoding="utf-8")

    assert set(summary["by_condition"]) == {
        "native_a",
        "native_b",
        "native_a_unit",
        "learned_mean",
        "learned_sum",
        "learned_mean_nonorm",
        "stacked_adaptive_gated",
    }
    assert len(summary["results"]) == 7 * 3
    for name, agg in summary["by_condition"].items():
        assert agg["expected"] == agg["completed"] + agg["failed"] == 3, name
        assert 0 <= agg["zero"] <= agg["completed"]
    for run in summary["results"]:
        assert run["status"] == "completed", run
        assert Path(run["diagnostics"]).exists()
        assert run["steps_applied"] == run["steps_attempted"] > 0
        assert run["validations"] == 2 and run["checkpoint_exists_last_val"] is True
    assert set(summary["decisions"]) == {key for key, *_ in driver.DECISION_ROWS}
    for _, label, nxt, forbidden in driver.DECISION_ROWS:
        assert label in report and nxt in report and forbidden in report
    assert "still unresolved" in report
    assert "synthetic" in summary["inputs"]["origin"]


def test_condition_subset_and_seed_warning(tmp_path: Path, driver) -> None:
    out = driver.main(
        [
            "--out",
            str(tmp_path / "sub"),
            *TINY[4:],
            "--seeds",
            "5",
            "--conditions",
            "native_a",
        ]
    )

    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert list(summary["by_condition"]) == ["native_a"]
    assert len(summary["results"]) == 1 and summary["results"][0]["seed"] == 5


def test_real_data_entry_point_reads_processed_layout(tmp_path: Path, driver) -> None:
    rng = np.random.default_rng(0)
    processed = tmp_path / "processed" / "ds"
    processed.mkdir(parents=True)
    rows_train, rows_val = [], []
    for u in range(16):
        items = rng.choice(30, size=5, replace=False)
        rows_train += [f"{u},{i}" for i in items[:4]]
        rows_val.append(f"{u},{items[4]}")
    (processed / "train.csv").write_text("user_idx,item_idx\n" + "\n".join(rows_train) + "\n")
    (processed / "val.csv").write_text("user_idx,item_idx\n" + "\n".join(rows_val) + "\n")
    (processed / "test.csv").write_text("user_idx,item_idx\n0,1\n")  # must never be read
    a = tmp_path / "a.npy"
    b = tmp_path / "b.npy"
    np.save(a, rng.standard_normal((30, 6)).astype("float32") * 10)
    np.save(b, rng.standard_normal((30, 4)).astype("float32") * 30)

    out = driver.main(
        [
            "--out",
            str(tmp_path / "real"),
            "--processed-dir",
            str(processed),
            "--sources",
            str(a),
            str(b),
            "--seeds",
            "0",
            "1",
            "2",
            "--learning-rates",
            "0.01",
            "--epochs",
            "1",
            "--eval-every",
            "1",
            "--batch-size",
            "16",
            "--latent-dim",
            "4",
            "--aligned-dim",
            "5",
            "--conditions",
            "native_a",
            "learned_mean",
        ]
    )

    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["inputs"] == {"origin": str(processed), "n_users": 16, "n_items": 30}
    assert "stacked_adaptive_gated" not in summary["by_condition"]  # widths differ
    assert all(r["status"] == "completed" for r in summary["results"])
