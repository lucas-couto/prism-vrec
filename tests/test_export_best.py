"""Tests for the export_best step.

Covers parsing the ``{model}_{embedding}_best.pt`` filename convention
(including recommender names that contain underscores) and the
end-to-end ``export_best_hyperparams`` flow on synthetic checkpoints.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from src.steps.export_best import (
    _parse_checkpoint_stem,
    export_best_hyperparams,
)

_KNOWN_MODELS = ["uniform_noise", "bpr", "vbpr", "avbpr", "deepstyle", "vnpr"]


class TestParseCheckpointStem:
    def test_simple_model_with_embedding(self) -> None:
        assert _parse_checkpoint_stem("vbpr_resnet50_D128", _KNOWN_MODELS) == (
            "vbpr",
            "resnet50_D128",
        )

    def test_bpr_with_none_embedding_via_underscore_suffix(self) -> None:
        assert _parse_checkpoint_stem("bpr_none", _KNOWN_MODELS) == ("bpr", "none")

    def test_model_alone_uses_none_embedding(self) -> None:
        assert _parse_checkpoint_stem("bpr", _KNOWN_MODELS) == ("bpr", "none")

    def test_multitoken_model_name_matches_longest_prefix(self) -> None:
        # uniform_noise should win over a (hypothetical) ``uniform`` prefix
        assert _parse_checkpoint_stem("uniform_noise_clip_vitb32_D128", _KNOWN_MODELS) == (
            "uniform_noise",
            "clip_vitb32_D128",
        )

    def test_unknown_model_returns_none(self) -> None:
        assert _parse_checkpoint_stem("mystery_resnet50_D128", _KNOWN_MODELS) is None


class TestExportBestHyperparams:
    def _write_ckpt(self, path: Path, hyperparams: dict, best_metric: float) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"hyperparams": hyperparams, "best_metric": best_metric}, path)

    def test_reads_pt_files_and_groups_by_dataset_model_embedding(self, tmp_path: Path) -> None:
        models_root = tmp_path / "models"
        self._write_ckpt(
            models_root / "amazon_fashion" / "bpr_none_best.pt",
            {"latent_dim": 64},
            0.4,
        )
        self._write_ckpt(
            models_root / "amazon_fashion" / "vbpr_resnet50_D128_best.pt",
            {"latent_dim": 128, "visual_dim": 64},
            0.55,
        )
        out = tmp_path / "best_hyperparams.json"

        summary = export_best_hyperparams(models_root, out)

        assert out.exists()
        on_disk = json.loads(out.read_text())
        assert summary == on_disk
        assert on_disk["amazon_fashion"]["bpr"]["none"]["hyperparams"] == {"latent_dim": 64}
        assert on_disk["amazon_fashion"]["bpr"]["none"]["best_metric"] == 0.4
        assert on_disk["amazon_fashion"]["vbpr"]["resnet50_D128"]["best_metric"] == 0.55

    def test_missing_models_dir_logs_and_returns_empty(self, tmp_path: Path) -> None:
        out = tmp_path / "best_hyperparams.json"
        assert export_best_hyperparams(tmp_path / "does_not_exist", out) == {}

    def test_unparsable_stem_is_skipped(self, tmp_path: Path) -> None:
        models_root = tmp_path / "models"
        self._write_ckpt(
            models_root / "ds" / "bpr_none_best.pt",
            {"latent_dim": 64},
            0.4,
        )
        # Stem that no registered model name prefixes
        self._write_ckpt(
            models_root / "ds" / "mystery_xyz_best.pt",
            {"x": 1},
            0.1,
        )
        out = tmp_path / "best_hyperparams.json"

        summary = export_best_hyperparams(models_root, out)
        assert list(summary["ds"].keys()) == ["bpr"]

    def test_sorts_keys_naturally(self, tmp_path: Path) -> None:
        models_root = tmp_path / "models"
        # Mixed alphabetical + numeric keys to verify natural sort
        for ds in ["b_dataset", "a_dataset"]:
            for model in ["vbpr", "bpr"]:
                self._write_ckpt(
                    models_root / ds / f"{model}_none_best.pt",
                    {"latent_dim": 32},
                    0.1,
                )
        out = tmp_path / "best_hyperparams.json"

        summary = export_best_hyperparams(models_root, out)
        assert list(summary.keys()) == ["a_dataset", "b_dataset"]
        assert list(summary["a_dataset"].keys()) == ["bpr", "vbpr"]
