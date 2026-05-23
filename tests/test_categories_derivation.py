"""Tests for the McAuley-taxonomy category derivation helper.

Covers the pure ``derive_categories`` function on hand-crafted item
records (bytes/str keys, missing taxonomy, level fallback, min_samples
filter) plus the CSV writer's atomicity.
"""

from __future__ import annotations

from pathlib import Path

from src.data.categories import (
    derive_categories,
    extract_taxonomy,
    write_categories_csv,
)


def _item(*path_segments: str) -> dict:
    """Build a DVBPR-shaped item dict whose taxonomy is one textual path."""
    return {"categories": [list(path_segments)]}


def _item_bytes(*path_segments: str) -> dict:
    """Same as ``_item`` but with bytes keys (some DVBPR splits use those)."""
    return {b"categories": [[seg.encode() for seg in path_segments]]}


class TestExtractTaxonomy:
    def test_returns_first_path_decoded(self) -> None:
        record = {"categories": [["Clothing", "Women", "Tops"]]}
        assert extract_taxonomy(record) == ["Clothing", "Women", "Tops"]

    def test_handles_bytes_keys_and_values(self) -> None:
        record = {b"categories": [[b"Clothing", b"Men", b"Shirts"]]}
        assert extract_taxonomy(record) == ["Clothing", "Men", "Shirts"]

    def test_returns_none_when_missing(self) -> None:
        assert extract_taxonomy({}) is None
        assert extract_taxonomy({"categories": []}) is None
        assert extract_taxonomy({"categories": [[]]}) is None

    def test_returns_none_for_non_dict(self) -> None:
        assert extract_taxonomy("not-a-dict") is None
        assert extract_taxonomy(None) is None


class TestDeriveCategories:
    def test_basic_grouping_at_level_2(self) -> None:
        items = [
            ("0", _item("Clothing", "Women", "Tops", "Tanks")),
            ("1", _item("Clothing", "Women", "Tops", "Shirts")),
            ("2", _item("Clothing", "Women", "Pants", "Jeans")),
            ("3", _item("Clothing", "Women", "Pants", "Trousers")),
        ]
        # Two items per label so min_samples=2 keeps both.
        mapping = derive_categories(items, level=2, min_samples=2)
        assert mapping is not None
        assert set(mapping.keys()) == {"0", "1", "2", "3"}
        # Two distinct labels at level 2 => two contiguous label ids.
        assert sorted(set(mapping.values())) == [0, 1]

    def test_falls_back_to_deepest_level_per_item(self) -> None:
        # One item has only two segments — should fall back to the last.
        items = [
            ("0", _item("Clothing", "Women")),
            ("1", _item("Clothing", "Women", "Tops")),
            ("2", _item("Clothing", "Women", "Tops")),
            ("3", _item("Clothing", "Women")),
        ]
        mapping = derive_categories(items, level=2, min_samples=2)
        assert mapping is not None
        assert len(mapping) == 4

    def test_min_samples_filters_small_labels(self) -> None:
        items = [
            ("0", _item("Clothing", "Women", "Tops")),
            ("1", _item("Clothing", "Women", "Tops")),
            ("2", _item("Clothing", "Women", "Tops")),
            ("3", _item("Clothing", "Women", "Pants")),  # singleton
        ]
        mapping = derive_categories(items, level=2, min_samples=2)
        assert mapping is not None
        assert "3" not in mapping  # singleton dropped

    def test_returns_none_when_no_taxonomy(self) -> None:
        items = [("0", {}), ("1", {"categories": []})]
        assert derive_categories(items) is None

    def test_returns_none_when_min_samples_eliminates_all(self) -> None:
        items = [
            ("0", _item("Clothing", "Women", "Tops")),
            ("1", _item("Clothing", "Women", "Pants")),
        ]
        assert derive_categories(items, level=2, min_samples=99) is None

    def test_bytes_records_supported(self) -> None:
        items = [
            ("0", _item_bytes("Clothing", "Women", "Tops")),
            ("1", _item_bytes("Clothing", "Women", "Tops")),
        ]
        mapping = derive_categories(items, level=2, min_samples=2)
        assert mapping == {"0": 0, "1": 0}

    def test_labels_are_contiguous_after_remap(self) -> None:
        # Crafted so the kept labels would have id gaps without remapping.
        items = [
            ("0", _item("X", "A", "Alpha")),
            ("1", _item("X", "A", "Alpha")),
            ("2", _item("X", "B", "Beta")),  # singleton, dropped
            ("3", _item("X", "C", "Gamma")),
            ("4", _item("X", "C", "Gamma")),
        ]
        mapping = derive_categories(items, level=2, min_samples=2)
        assert mapping is not None
        assert sorted(set(mapping.values())) == [0, 1]


class TestWriteCategoriesCsv:
    def test_writes_two_column_csv(self, tmp_path: Path) -> None:
        mapping = {"0": 0, "1": 1, "42": 0}
        target = tmp_path / "categories.csv"

        write_categories_csv(mapping, target)

        rows = target.read_text().strip().splitlines()
        assert rows[0] == "item_id,category_label"
        assert set(rows[1:]) == {"0,0", "1,1", "42,0"}

    def test_atomic_via_tmp_then_rename(self, tmp_path: Path) -> None:
        target = tmp_path / "categories.csv"
        write_categories_csv({"0": 0}, target)

        # No leftover .tmp after a successful write.
        assert not target.with_suffix(target.suffix + ".tmp").exists()
        assert target.exists()

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        nested = tmp_path / "a" / "b" / "categories.csv"
        write_categories_csv({"0": 0}, nested)
        assert nested.exists()
