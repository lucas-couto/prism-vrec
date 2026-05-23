"""Regression tests for fuse strategy-config resolution.

Pins the contract that ``_strategies_map`` reads the configured grid
from top-level ``config["strategies"]`` — the location ``load_config``
produces after merging the YAML files.
"""

from __future__ import annotations

from src.steps.fuse import _strategies_map


def test_strategies_map_returns_configured_block_for_grid_strategy() -> None:
    config = {"strategies": {"pca": {"n_components": [64, 128, 256]}}}

    result = _strategies_map(config)

    assert result.get("pca", {}) == {"n_components": [64, 128, 256]}


def test_strategies_map_empty_when_strategy_absent() -> None:
    config = {"strategies": {"mean": {}}}

    result = _strategies_map(config)

    assert result.get("pca", {}) == {}


def test_strategies_map_empty_when_no_strategies_key() -> None:
    result = _strategies_map({})

    assert result == {}
