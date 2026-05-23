"""Tests for the synthetic dataset provider used by the smoke profile."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.base import get_dataset_provider
from src.data.synthetic import SyntheticDatasetProvider


class TestSyntheticDatasetProvider:
    def test_registered_under_default_name(self) -> None:
        provider = get_dataset_provider("synthetic")
        assert isinstance(provider, SyntheticDatasetProvider)

    def test_save_processed_produces_canonical_layout(self, tmp_path: Path) -> None:
        provider = SyntheticDatasetProvider(
            raw_dir=tmp_path / "raw",
            n_users=10,
            n_items=20,
            n_categories=3,
            interactions_per_user=4,
        )
        processed_dir = tmp_path / "processed" / "synthetic"

        provider.save_processed(processed_dir)
        provider.extract_images(provider.raw_dir / "images")

        train = pd.read_csv(processed_dir / "train.csv")
        val = pd.read_csv(processed_dir / "val.csv")
        test = pd.read_csv(processed_dir / "test.csv")

        assert list(train.columns) == ["user_idx", "item_idx"]
        assert len(val) == 10  # one held-out item per user
        assert len(test) == 10
        # interactions_per_user=4 - 1 (val) - 1 (test) = 2 train per user.
        assert len(train) == 10 * 2

    def test_load_categories_returns_contiguous_labels(self, tmp_path: Path) -> None:
        provider = SyntheticDatasetProvider(
            raw_dir=tmp_path / "raw",
            n_users=10,
            n_items=20,
            n_categories=3,
        )
        provider.save_processed(tmp_path / "processed" / "synthetic")

        categories = provider.load_categories()
        assert categories is not None
        assert len(categories) == 20
        assert sorted(set(categories.values())) == [0, 1, 2]

    def test_num_categories(self, tmp_path: Path) -> None:
        provider = SyntheticDatasetProvider(
            raw_dir=tmp_path / "raw",
            n_users=10,
            n_items=20,
            n_categories=4,
        )
        provider.save_processed(tmp_path / "processed" / "synthetic")
        assert provider.num_categories() == 4

    def test_extract_images_creates_jpgs(self, tmp_path: Path) -> None:
        provider = SyntheticDatasetProvider(
            raw_dir=tmp_path / "raw",
            n_users=5,
            n_items=10,
            image_size=32,
        )
        image_dir = tmp_path / "raw" / "images"
        provider.extract_images(image_dir)

        files = sorted(image_dir.glob("*.jpg"))
        assert len(files) == 10
        assert {f.stem for f in files} == {str(i) for i in range(10)}

    def test_image_stems_match_item2idx_keys(self, tmp_path: Path) -> None:
        """Regression: the extract step looks up images by ``str(item_id)``
        from ``item2idx.json``; the provider's image filenames must use
        the same convention or every image silently fails to match."""
        provider = SyntheticDatasetProvider(
            raw_dir=tmp_path / "raw",
            n_users=10,
            n_items=20,
        )
        processed_dir = tmp_path / "processed" / "synthetic"
        provider.save_processed(processed_dir)
        provider.extract_images(tmp_path / "raw" / "images")

        import json

        item2idx = json.loads((processed_dir / "item2idx.json").read_text())
        image_stems = {p.stem for p in (tmp_path / "raw" / "images").glob("*.jpg")}

        for item_key in item2idx:
            assert item_key in image_stems, f"item id {item_key!r} has no image"

    def test_extract_images_is_idempotent(self, tmp_path: Path) -> None:
        provider = SyntheticDatasetProvider(raw_dir=tmp_path / "raw", n_items=5)
        image_dir = tmp_path / "raw" / "images"
        provider.extract_images(image_dir)
        first_size = (image_dir / "0.jpg").stat().st_size

        provider.extract_images(image_dir)
        second_size = (image_dir / "0.jpg").stat().st_size

        assert first_size == second_size

    def test_rejects_too_few_interactions(self, tmp_path: Path) -> None:
        import pytest

        with pytest.raises(ValueError, match="interactions_per_user"):
            SyntheticDatasetProvider(raw_dir=tmp_path / "raw", interactions_per_user=2)


class TestSmokeConfigParses:
    def test_smoke_yamls_load_against_schema(self) -> None:
        from src.utils.config import load_config

        smoke_dir = Path(__file__).resolve().parent.parent / "configs" / "smoke"
        config = load_config(str(smoke_dir))

        assert "datasets" in config
        assert config["datasets"] == ["synthetic"]
        assert config["pipeline"]["condition"] == "frozen"
        assert config["hp_search"]["strategy"] == "optuna"
        assert config["hp_search"]["optuna"]["n_trials"] == 1
        # The smoke profile keeps the bundle small.
        assert config["extractors_enabled"] == ["resnet50"]
        assert config["fusion_strategies_enabled"] == ["mean"]
        assert set(config["recommenders_enabled"]) == {"bpr", "vbpr"}
