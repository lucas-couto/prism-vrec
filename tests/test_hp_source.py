"""Tests for :mod:`src.recommenders.hp_source` (hyperparameter provenance)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from src.recommenders.hp_source import (
    BEST_HYPERPARAMS_FILENAME,
    HyperparamOrigin,
    resolve_cell_hyperparams,
)

_SEARCH_CONFIG = {
    "hp_search": {"strategy": "optuna"},
    "common": {"total_dim": [64, 128], "learning_rate": [0.001, 0.01], "l2_reg": 1e-4},
}
_FIXED_CONFIG = {
    "hp_search": {"strategy": "fixed"},
    "common": {"total_dim": 64, "learning_rate": 0.001, "l2_reg": 1e-4},
}
_WINNERS = {
    "amazon_fashion": {
        "vbpr": {
            "resnet50": {
                "hyperparams": {"total_dim": 128, "latent_dim": 64, "visual_dim": 64},
                "best_metric": 0.55,
            }
        }
    }
}


def _resolve(config: dict, results_root: Path, **cell: str) -> HyperparamOrigin:
    return resolve_cell_hyperparams(
        config,
        dataset=cell.get("dataset", "amazon_fashion"),
        model_name=cell.get("model_name", "vbpr"),
        embedding_name=cell.get("embedding_name", "resnet50"),
        results_root=results_root,
    )


class TestHyperparamOrigin:
    def test_to_dict_is_a_plain_serialisable_view(self) -> None:
        origin = HyperparamOrigin("search", {"a": 1}, "ds__m__e", 0.5)

        assert json.loads(json.dumps(origin.to_dict())) == {
            "source": "search",
            "hyperparams": {"a": 1},
            "reference": "ds__m__e",
            "best_metric": 0.5,
        }


class TestResolveCellHyperparams:
    def test_fixed_strategy_returns_the_pinned_config_without_reading_disk(self, tmp_path) -> None:
        origin = _resolve(_FIXED_CONFIG, tmp_path / "does_not_exist")

        assert origin.source == "fixed"
        assert origin.reference is None
        assert origin.best_metric is None
        assert origin.hyperparams == {
            "learning_rate": 0.001,
            "l2_reg": 1e-4,
            "total_dim": 64,
            "latent_dim": 32,
            "visual_dim": 32,
        }

    def test_search_strategy_reads_existing_winners_file(self, tmp_path) -> None:
        (tmp_path / BEST_HYPERPARAMS_FILENAME).write_text(json.dumps(_WINNERS))

        origin = _resolve(_SEARCH_CONFIG, tmp_path)

        assert origin.source == "search"
        assert origin.reference == "amazon_fashion__vbpr__resnet50"
        assert origin.best_metric == 0.55
        assert origin.hyperparams == _WINNERS["amazon_fashion"]["vbpr"]["resnet50"]["hyperparams"]

    def test_missing_winners_file_is_generated_from_best_checkpoints(self, tmp_path) -> None:
        ckpt = tmp_path / "models" / "amazon_fashion" / "vbpr_resnet50_best.pt"
        ckpt.parent.mkdir(parents=True)
        torch.save({"hyperparams": {"latent_dim": 64, "visual_dim": 64}, "best_metric": 0.42}, ckpt)

        origin = _resolve(_SEARCH_CONFIG, tmp_path)

        assert origin.source == "search"
        assert origin.hyperparams == {"latent_dim": 64, "visual_dim": 64}
        assert origin.best_metric == 0.42
        assert origin.reference == "amazon_fashion__vbpr__resnet50"
        on_disk = json.loads((tmp_path / BEST_HYPERPARAMS_FILENAME).read_text())
        assert on_disk["amazon_fashion"]["vbpr"]["resnet50"]["best_metric"] == 0.42

    def test_missing_cell_raises_key_error_naming_the_cell(self, tmp_path) -> None:
        (tmp_path / BEST_HYPERPARAMS_FILENAME).write_text(json.dumps(_WINNERS))

        with pytest.raises(KeyError, match="amazon_fashion__bpr__none"):
            _resolve(_SEARCH_CONFIG, tmp_path, model_name="bpr", embedding_name="none")

    def test_grid_strategy_also_counts_as_search(self, tmp_path) -> None:
        (tmp_path / BEST_HYPERPARAMS_FILENAME).write_text(json.dumps(_WINNERS))

        origin = _resolve({**_SEARCH_CONFIG, "hp_search": {"strategy": "grid"}}, tmp_path)

        assert origin.source == "search"
