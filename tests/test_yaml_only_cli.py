"""The YAML is the only control surface for run configuration (3.0.0).

The step range, the condition, the search strategy, the protocol and the
seeds come from ``configs/*.yaml`` alone; the flags that duplicated them
were removed by the researcher's decision.  Passing one fails with
argparse's standard error plus the YAML key that replaced it; the
resolved plan is printed by ``--show-plan`` and recorded in the run
manifest.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import main


@pytest.mark.parametrize(
    ("argv", "hint"),
    [
        (["--all"], "pipeline.run_all"),
        (["--step", "train"], "pipeline.run_all: false"),
        (["--from", "train"], "pipeline.start_from"),
        (["--from", "train", "--to", "evaluate"], "pipeline.start_from"),
        (["--condition", "frozen"], "pipeline.condition"),
        (["--hp-search", "optuna"], "hp_search.strategy"),
        (["--n-trials", "3"], "hp_search.optuna.n_trials"),
        (["--eval-protocol=sampled"], "evaluation.protocol"),
        (["--seeds", "1,2"], "seeds"),
    ],
)
def test_removed_flags_fail_with_the_yaml_key(argv, hint, capsys) -> None:
    code = main.run_cli(argv)

    err = capsys.readouterr().err
    assert code == 2  # argparse's standard usage error
    assert "was removed" in err
    assert hint in err


def test_every_removed_flag_is_gone_from_the_parser() -> None:
    parser = main.build_parser()
    known = {opt for action in parser._actions for opt in action.option_strings}

    assert not known & set(main.REMOVED_FLAGS)
    for kept in ("--battery", "--folds", "--show-plan", "--inspect-pending", "--config-dir"):
        assert kept in known


def test_show_plan_prints_the_yaml_resolved_plan(monkeypatch, capsys) -> None:
    config = {
        "pipeline": {
            "run_all": False,
            "start_from": "fuse",
            "stop_at": "evaluate",
            "condition": "frozen",
        }
    }
    monkeypatch.setattr(main, "load_config", lambda *a, **k: config)

    assert main.run_cli(["--show-plan"]) == 0

    out = capsys.readouterr().out
    assert "condition='frozen' (3 steps)" in out
    assert "fuse" in out and "train" in out and "evaluate" in out
    assert "finetune" not in out.replace("evaluate_finetuning", "")


def test_run_all_true_ignores_the_range_and_both_expands_condition_steps() -> None:
    steps, condition, run_both = main._resolve_plan(
        {"pipeline": {"run_all": True, "start_from": "train", "condition": "both"}}
    )

    assert steps == list(main.STEP_ORDER)
    assert (condition, run_both) == (None, True)


def test_plan_that_filters_to_nothing_fails_loud() -> None:
    config = {
        "pipeline": {
            "run_all": False,
            "start_from": "finetune",
            "stop_at": "evaluate_finetuning",
            "condition": "frozen",
        }
    }

    with pytest.raises(ValueError, match="resolved to no steps"):
        main._resolve_plan(config)


def test_manifest_records_the_resolved_plan(tmp_path: Path, monkeypatch) -> None:
    config = {
        "seed": 1,
        "device": "cpu",
        "paths": {"results": str(tmp_path / "results")},
        "pipeline": {"run_all": False, "start_from": "download", "stop_at": "download"},
        "telemetry": {"enabled": False},
    }
    monkeypatch.setattr(main, "load_config", lambda *a, **k: config)
    monkeypatch.setitem(main.STEP_FUNCTIONS, "download", lambda: None)
    monkeypatch.chdir(tmp_path)

    assert main.run_cli([]) == 0

    runs = list((tmp_path / "results" / "runs").iterdir())
    manifest = json.loads((runs[0] / "manifest.json").read_text())
    assert manifest["plan"] == {"steps": ["download"], "condition": "both"}
