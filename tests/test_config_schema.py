"""Validation tests for the framework configuration schema.

The schema is a safety net for plugin authors and casual users who
hand-edit the YAML configs: typos in known fields surface as
``ValidationError`` at load time instead of silently disabling a
step.  These tests pin both the *acceptance* criteria (a sane
default config validates) and the *rejection* criteria (typos and
out-of-range values fail loudly).
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.utils.config_schema import (
    CONDITION_VALUES,
    PIPELINE_STEPS,
    PipelineConfig,
    validate_config,
)


def _minimal() -> dict:
    """Return a minimal config dict that should always validate."""
    return {
        "seed": 42,
        "device": "cuda",
        "datasets": ["amazon_fashion"],
        "extractors_enabled": ["resnet50"],
    }


def test_minimal_config_validates_with_defaults_filled() -> None:
    out = validate_config(_minimal())

    assert out["seed"] == 42
    assert out["pipeline"]["condition"] == "both"
    assert out["paths"]["data_raw"] == "data/raw"
    assert out["preprocessing"]["n_min"] == 5
    assert out["finetuning"]["epochs_max"] == 15


def test_full_config_with_plugin_blocks_validates() -> None:
    """Plugin-authored YAML keys (per-recommender HP grids, per-fusion
    config blocks) must pass through without being flagged as unknown."""
    out = validate_config(
        {
            **_minimal(),
            "vbpr": {"latent_dim": [16, 32], "l2_reg": [1e-3, 1e-4]},
            "fusion": {"weighted_mean": {"alpha": [0.3, 0.5, 0.7]}},
            "extractors": {"resnet50": {"weights": "IMAGENET1K_V2"}},
            "fusion_strategies_enabled": ["mean"],
            "recommenders_enabled": ["bpr", "vbpr"],
        }
    )
    assert out["vbpr"]["latent_dim"] == [16, 32]


def test_legacy_strategies_alias_passes_through() -> None:
    """Some YAMLs use ``strategies`` instead of ``fusion`` — keep both."""
    out = validate_config({**_minimal(), "strategies": {"mean": {}}})
    assert out["strategies"] == {"mean": {}}


def test_invalid_device_raises() -> None:
    with pytest.raises(ValidationError) as excinfo:
        validate_config({**_minimal(), "device": "tpu"})
    assert "device" in str(excinfo.value)


def test_invalid_condition_raises_with_helpful_message() -> None:
    with pytest.raises(ValidationError) as excinfo:
        validate_config({**_minimal(), "pipeline": {"condition": "frozen_only"}})
    msg = str(excinfo.value)
    assert "condition" in msg


def test_negative_seed_raises() -> None:
    with pytest.raises(ValidationError):
        validate_config({**_minimal(), "seed": -1})


def test_zero_kcore_raises() -> None:
    with pytest.raises(ValidationError):
        validate_config({**_minimal(), "preprocessing": {"n_min": 0}})


def test_unknown_pipeline_key_rejected() -> None:
    """Top-level allows extras (plugin blocks), but ``pipeline``'s
    own block is strict so a typo on the orchestration knobs fails."""
    with pytest.raises(ValidationError) as excinfo:
        validate_config(
            {
                **_minimal(),
                "pipeline": {"run_all": True, "conditino": "frozen"},  # typo
            }
        )
    assert "conditino" in str(excinfo.value)


def test_unknown_paths_key_rejected() -> None:
    with pytest.raises(ValidationError):
        validate_config({**_minimal(), "paths": {"data_raww": "x"}})  # typo


def test_unknown_step_in_start_from_raises() -> None:
    with pytest.raises(ValidationError) as excinfo:
        validate_config(
            {
                **_minimal(),
                "pipeline": {"start_from": "extarct"},  # typo for 'extract'
            }
        )
    assert "extarct" in str(excinfo.value)


def test_inverted_step_range_raises() -> None:
    with pytest.raises(ValidationError) as excinfo:
        validate_config(
            {
                **_minimal(),
                "pipeline": {"start_from": "evaluate", "stop_at": "extract"},
            }
        )
    assert "cannot come after" in str(excinfo.value)


def test_negative_projection_dim_raises() -> None:
    with pytest.raises(ValidationError):
        validate_config({**_minimal(), "projection_dims": [64, -1, 256]})


def test_finetuning_invalid_lr_raises() -> None:
    with pytest.raises(ValidationError):
        validate_config(
            {
                **_minimal(),
                "finetuning": {"learning_rate": 0.0},  # must be > 0
            }
        )


def test_pipeline_config_step_constants_are_synced() -> None:
    """If main.STEP_ORDER changes, PIPELINE_STEPS must too."""
    from main import STEP_ORDER

    assert tuple(STEP_ORDER) == PIPELINE_STEPS, (
        "main.STEP_ORDER and src.utils.config_schema.PIPELINE_STEPS drifted "
        "— update one to match the other."
    )


def test_condition_constants_match_main() -> None:
    """The ``condition`` literal in PipelineConfig must match what main.py accepts."""
    annotation = PipelineConfig.model_fields["condition"].annotation
    # Literal["frozen", "finetuned", "both"] — extract the args
    import typing

    accepted = set(typing.get_args(annotation))
    assert accepted == set(CONDITION_VALUES)
