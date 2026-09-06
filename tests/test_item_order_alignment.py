"""Feature rows follow the numeric ``item2idx`` values, not insertion order (S01 / F08).

``{"b": 1, "a": 0}`` must produce feature rows ``[a, b]``.  The extract
step used ``list(item2idx.keys())`` and ``ImageDataset`` silently dropped
items without an image, so a permuted mapping or a missing file produced
a matrix whose row ``i`` was NOT item ``i`` — undetectable by the
row-count-only validation gate.  These tests pin the canonical order,
the missing-image failure, the ordered-mapping digest persisted next to
the features and its verification in ``validate_features``.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

import src.steps.extract as extract_step
from src.extractors.base import BaseExtractor
from src.steps.extract import ImageDataset, MissingImageError, get_item_ids
from src.steps.validate_features import FeatureValidationError, validate_backbone_feature
from src.utils.item_order import (
    ItemOrderError,
    canonical_item_order,
    item_order_digest,
    load_item_order,
)

DIM = 3
#: External id -> gray level of its image.  The fake extractor's feature
#: is ``gray / 255`` broadcast over ``DIM``, so every row is checkable.
GRAY = {"alpha": 20, "beta": 60, "gamma": 100, "delta": 140, "eps": 180}


class _GrayModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = torch.nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(x.mean(dim=(1, 2, 3))[:, None].expand(-1, DIM))


class _GrayExtractor(BaseExtractor):
    """Deterministic function of the image: no weights, runs in milliseconds."""

    weights_id = "fake/gray"

    def __init__(self, device: str = "cpu") -> None:
        super().__init__(device="cpu")

    def _build_model(self):
        return _GrayModel()

    def _build_transform(self):
        from torchvision import transforms

        return transforms.ToTensor()


def _write_images(image_dir: Path, ids=GRAY) -> None:
    image_dir.mkdir(parents=True, exist_ok=True)
    for item_id, gray in ids.items():
        Image.new("RGB", (4, 4), (gray, gray, gray)).save(image_dir / f"{item_id}.png")


def _write_item2idx(processed_dir: Path, dataset: str, item2idx: dict) -> None:
    (processed_dir / dataset).mkdir(parents=True, exist_ok=True)
    (processed_dir / dataset / "item2idx.json").write_text(json.dumps(item2idx))


REVERSED = {"eps": 4, "delta": 3, "gamma": 2, "beta": 1, "alpha": 0}
RANDOM = {"gamma": 2, "alpha": 0, "eps": 4, "beta": 1, "delta": 3}
CANONICAL = ["alpha", "beta", "gamma", "delta", "eps"]


class TestCanonicalOrder:
    def test_reversed_insertion_order_yields_rows_by_index(self, tmp_path: Path) -> None:
        _write_item2idx(tmp_path, "ds", REVERSED)

        assert get_item_ids(str(tmp_path), "ds") == CANONICAL

    def test_random_insertion_order_yields_rows_by_index(self, tmp_path: Path) -> None:
        _write_item2idx(tmp_path, "ds", RANDOM)

        assert load_item_order(tmp_path, "ds") == CANONICAL

    def test_hole_is_rejected(self) -> None:
        with pytest.raises(ItemOrderError, match="hole"):
            canonical_item_order({"a": 0, "b": 2, "c": 3})

    def test_duplicate_index_is_rejected(self) -> None:
        with pytest.raises(ItemOrderError, match="more than once"):
            canonical_item_order({"a": 0, "b": 1, "c": 1})

    def test_non_integer_index_is_rejected(self) -> None:
        with pytest.raises(ItemOrderError, match="not an integer"):
            canonical_item_order({"a": "0", "b": 1})

    def test_digest_depends_on_order_not_insertion(self) -> None:
        assert item_order_digest(CANONICAL) == item_order_digest(canonical_item_order(RANDOM))
        assert item_order_digest(CANONICAL) != item_order_digest(list(reversed(CANONICAL)))


class TestMissingImage:
    def test_missing_image_raises_naming_the_item(self, tmp_path: Path) -> None:
        image_dir = tmp_path / "images"
        _write_images(image_dir, {k: v for k, v in GRAY.items() if k != "gamma"})

        with pytest.raises(MissingImageError, match="gamma"):
            ImageDataset(str(image_dir), CANONICAL)

    def test_dataset_keeps_the_given_order_when_all_images_exist(self, tmp_path: Path) -> None:
        image_dir = tmp_path / "images"
        _write_images(image_dir)

        dataset = ImageDataset(str(image_dir), CANONICAL)

        assert dataset.valid_items == CANONICAL
        assert len(dataset) == len(CANONICAL)


@pytest.fixture
def cell(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """A tiny extract cell: 5 gray images, a permuted item2idx, no real config."""
    monkeypatch.setattr(extract_step, "load_config", lambda: {})
    monkeypatch.setattr(
        extract_step,
        "resolve_dataloader_settings",
        lambda _cfg: SimpleNamespace(num_workers=0),
    )
    image_dir = tmp_path / "raw" / "ds" / "images"
    _write_images(image_dir)
    processed = tmp_path / "processed"
    _write_item2idx(processed, "ds", REVERSED)
    embeddings = tmp_path / "embeddings"
    config = {"paths": {"checkpoints": str(tmp_path / "ckpt")}}
    return SimpleNamespace(
        image_dir=image_dir, processed=processed, embeddings=embeddings, config=config
    )


def _extract(cell: SimpleNamespace, batch_size: int = 2) -> bool:
    return extract_step._extract_for_config(
        extractor_cls=_GrayExtractor,
        extractor_name="gray",
        dataset_name="ds",
        image_dir=str(cell.image_dir),
        item_ids=get_item_ids(str(cell.processed), "ds"),
        embeddings_dir=str(cell.embeddings),
        batch_size=batch_size,
        checkpoint_every=1,
        device="cpu",
        config=cell.config,
    )


def _expected_rows() -> np.ndarray:
    return np.array([[GRAY[i] / 255.0] * DIM for i in CANONICAL], dtype=np.float32)


class TestExtractCellAlignment:
    def test_rows_follow_numeric_index_under_permuted_mapping(self, cell) -> None:
        assert _extract(cell) is True

        saved = np.load(cell.embeddings / "ds" / "gray.npy")
        np.testing.assert_allclose(saved, _expected_rows(), atol=1e-6)
        meta = json.loads((cell.embeddings / "ds" / "gray.meta.json").read_text())
        assert meta["item_order"]["n_items"] == 5
        assert meta["item_order"]["digest"] == item_order_digest(CANONICAL)

    def test_validate_features_verifies_the_persisted_order(self, cell) -> None:
        _extract(cell)
        config = {"extractors": {"gray": {"raw_dim": DIM}}}

        stats = validate_backbone_feature(
            "ds", "gray", config, embeddings_dir=cell.embeddings, processed_dir=cell.processed
        )

        assert stats["alignment"] == "verified"

    def test_same_length_reordered_catalogue_is_rejected(self, cell) -> None:
        _extract(cell)
        swapped = dict(REVERSED)
        swapped["alpha"], swapped["eps"] = swapped["eps"], swapped["alpha"]
        _write_item2idx(cell.processed, "ds", swapped)
        config = {"extractors": {"gray": {"raw_dim": DIM}}}

        with pytest.raises(FeatureValidationError, match="item order"):
            validate_backbone_feature(
                "ds", "gray", config, embeddings_dir=cell.embeddings, processed_dir=cell.processed
            )

    def test_legacy_artifact_without_digest_is_reported_unverified(self, cell) -> None:
        _extract(cell)
        meta_path = cell.embeddings / "ds" / "gray.meta.json"
        meta = json.loads(meta_path.read_text())
        del meta["item_order"]
        meta_path.write_text(json.dumps(meta))
        config = {"extractors": {"gray": {"raw_dim": DIM}}}

        stats = validate_backbone_feature(
            "ds", "gray", config, embeddings_dir=cell.embeddings, processed_dir=cell.processed
        )

        assert stats["alignment"] == "unverified"

    def test_missing_image_fails_the_cell_before_any_output(self, cell) -> None:
        (cell.image_dir / "beta.png").unlink()

        with pytest.raises(MissingImageError, match="beta"):
            _extract(cell)

        assert not (cell.embeddings / "ds" / "gray.npy").exists()

    def test_broken_mapping_fails_validation_explicitly(self, cell) -> None:
        _extract(cell)
        _write_item2idx(cell.processed, "ds", {"alpha": 0, "beta": 1, "gamma": 1, "delta": 3})
        config = {"extractors": {"gray": {"raw_dim": DIM}}}

        with pytest.raises(FeatureValidationError, match="more than once"):
            validate_backbone_feature(
                "ds", "gray", config, embeddings_dir=cell.embeddings, processed_dir=cell.processed
            )
