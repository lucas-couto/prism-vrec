"""The YAML is the only control surface: ``main.py`` takes NO arguments.

3.0.0 removed the flags that duplicated run configuration; 2026-09-09
removed the rest, including ``--config-dir`` (the directory is always
``configs/``).  What the command does is chosen by ``pipeline.mode``, and
the step range, condition, search strategy, protocol and seeds come from
``configs/*.yaml`` alone.  Passing any argument fails naming the key that
carries its behaviour; the resolved plan is printed by
``pipeline.mode: show_plan`` and recorded in the run manifest.
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
        (["--config-dir", "other"], "always `configs/`"),
        (["--battery"], "pipeline.mode: battery"),
        (["--show-plan"], "pipeline.mode: show_plan"),
        (["--folds"], "folds.enabled: true"),
        (["--list-datasets"], "pipeline.mode: list"),
        (["--report"], "pipeline.mode: report"),
        (["whatever"], "every knob lives in configs/*.yaml"),
    ],
)
def test_any_argument_fails_naming_the_yaml_key(argv, hint, capsys) -> None:
    code = main.run_cli(argv)

    err = capsys.readouterr().err
    assert code != 0
    assert "takes no arguments" in err
    assert hint in err


def test_there_is_no_parser_left_to_hold_a_flag() -> None:
    """Not "the flags were removed" but "there is nowhere to put one"."""
    assert not hasattr(main, "build_parser")
    for gone in ("--battery", "--folds", "--show-plan", "--config-dir", "--report"):
        assert gone in main.REMOVED_FLAGS


def test_show_plan_mode_prints_the_yaml_resolved_plan(monkeypatch, capsys) -> None:
    config = {
        "pipeline": {
            "run_all": False,
            "start_from": "fuse",
            "stop_at": "folds",
            "condition": "frozen",
            "mode": "show_plan",
        }
    }
    monkeypatch.setattr(main, "load_config", lambda *a, **k: config)

    assert main.run_cli([]) == 0

    out = capsys.readouterr().out
    assert "condition='frozen' (3 steps)" in out
    assert "fuse" in out and "train" in out and "folds" in out
    assert "evaluate" not in out.replace("evaluate_finetuning", "")
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
