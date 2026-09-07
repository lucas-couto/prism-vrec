"""``configs/zz_local.yaml`` narrows a night's run without touching tracked files.

The loader merges every ``configs/*.yaml`` alphabetically after
``default.yaml`` with the later file winning and lists replaced whole,
so an untracked ``zz_local.yaml`` (last in the order, ignored by git)
is the place for the per-night ``datasets:`` and ``pipeline:`` edits.
Documented caveat: ``pipeline.start_from`` / ``stop_at`` are ignored
while ``pipeline.run_all`` is ``true``.
"""

from __future__ import annotations

from pathlib import Path

import yaml

import main
from src.utils.config import load_config

_DEFAULT = {
    "seed": 1,
    "device": "cpu",
    "datasets": ["amazon_fashion", "amazon_women", "tradesy"],
    "pipeline": {"run_all": True, "start_from": None, "stop_at": None, "condition": "both"},
}


def _configs(tmp_path: Path, local: dict | None) -> Path:
    root = tmp_path / "configs"
    root.mkdir()
    (root / "default.yaml").write_text(yaml.safe_dump(_DEFAULT), encoding="utf-8")
    if local is not None:
        (root / "zz_local.yaml").write_text(yaml.safe_dump(local), encoding="utf-8")
    return root


def _plan(config: dict) -> list[str]:
    steps, _condition, _run_both = main._resolve_plan(config)
    return steps


def test_zz_local_replaces_the_datasets_list_and_the_pipeline_range(tmp_path) -> None:
    root = _configs(
        tmp_path,
        {
            "datasets": ["tradesy"],
            "pipeline": {"run_all": False, "start_from": None, "stop_at": "beyond_accuracy"},
        },
    )

    config = load_config(str(root))

    assert config["datasets"] == ["tradesy"]  # replaced whole, not appended
    assert config["pipeline"]["condition"] == "both"  # untouched keys survive the merge
    assert _plan(config) == main.STEP_ORDER[: main.STEP_ORDER.index("beyond_accuracy") + 1]


def test_run_all_true_ignores_stop_at(tmp_path) -> None:
    """The documented trap: with ``run_all: true`` the range keys are dead."""
    root = _configs(tmp_path, {"pipeline": {"run_all": True, "stop_at": "preprocess"}})

    config = load_config(str(root))

    assert config["pipeline"]["stop_at"] == "preprocess"
    assert _plan(config) == list(main.STEP_ORDER)


def test_without_zz_local_the_defaults_apply(tmp_path) -> None:
    config = load_config(str(_configs(tmp_path, None)))

    assert config["datasets"] == _DEFAULT["datasets"]
    assert _plan(config) == list(main.STEP_ORDER)
