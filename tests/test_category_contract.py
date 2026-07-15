"""Tests for the per-dataset ``expects_categories`` contract.

Covers the enforcement helper (both violation directions and both
happy paths), the ``DatasetContract`` schema field, and the enforcement
routed through a registered provider's ``load_categories`` — the exact
silent signal the contract guards.
"""

from __future__ import annotations

import pytest

from src.data.base import DatasetProvider, register_dataset_provider
from src.data.categories import CategoryContractError, enforce_category_contract
from src.utils.config_schema import DatasetContract, validate_config


class _FakeProvider(DatasetProvider):
    """Minimal provider whose category presence is injected in the test."""

    def __init__(self, name: str, categories: dict[str, int] | None) -> None:
        super().__init__(name=name)
        self._categories = categories

    def download(self) -> None:  # pragma: no cover - unused
        raise NotImplementedError

    def save_processed(self, processed_dir) -> None:  # pragma: no cover - unused
        raise NotImplementedError

    def extract_images(self, image_dir) -> None:  # pragma: no cover - unused
        raise NotImplementedError

    def load_categories(self) -> dict[str, int] | None:
        return self._categories


class TestEnforceCategoryContractViolations:
    def test_expects_false_but_data_has_categories_raises(self) -> None:
        categories = {"0": 0, "1": 1, "2": 0}

        with pytest.raises(CategoryContractError, match="expects_categories=false"):
            enforce_category_contract("tradesy", expects_categories=False, categories=categories)

    def test_expects_true_but_data_has_no_categories_raises(self) -> None:
        with pytest.raises(CategoryContractError, match="expects_categories=true"):
            enforce_category_contract("amazon_fashion", expects_categories=True, categories=None)

    def test_expects_true_but_empty_mapping_raises(self) -> None:
        with pytest.raises(CategoryContractError, match="expects_categories=true"):
            enforce_category_contract("amazon_men", expects_categories=True, categories={})

    def test_message_names_the_dataset(self) -> None:
        with pytest.raises(CategoryContractError, match="tradesy"):
            enforce_category_contract("tradesy", expects_categories=False, categories={"0": 0})


class TestEnforceCategoryContractHappyPaths:
    def test_expects_true_with_categories_passes(self) -> None:
        enforce_category_contract(
            "amazon_fashion", expects_categories=True, categories={"0": 0, "1": 1}
        )

    def test_expects_false_without_categories_passes(self) -> None:
        enforce_category_contract("tradesy", expects_categories=False, categories=None)

    def test_expects_false_with_empty_mapping_passes(self) -> None:
        # An empty mapping carries no labels, so it satisfies expects=false.
        enforce_category_contract("tradesy", expects_categories=False, categories={})


class TestEnforcementThroughProvider:
    """Faithful to the real call site: enforce on ``load_categories()``."""

    def test_false_contract_with_labelled_provider_raises(self) -> None:
        register_dataset_provider(
            "fake_labelled", lambda: _FakeProvider("fake_labelled", {"0": 0, "1": 1})
        )
        provider = _FakeProvider("fake_labelled", {"0": 0, "1": 1})

        with pytest.raises(CategoryContractError):
            enforce_category_contract(
                "fake_labelled", expects_categories=False, categories=provider.load_categories()
            )

    def test_true_contract_with_unlabelled_provider_raises(self) -> None:
        provider = _FakeProvider("fake_unlabelled", None)

        with pytest.raises(CategoryContractError):
            enforce_category_contract(
                "fake_unlabelled", expects_categories=True, categories=provider.load_categories()
            )


class TestDatasetContractSchema:
    def test_field_parses_from_config(self) -> None:
        cfg = validate_config(
            {
                "datasets": ["tradesy"],
                "dataset_contracts": {"tradesy": {"expects_categories": False}},
            }
        )
        assert cfg["dataset_contracts"]["tradesy"]["expects_categories"] is False

    def test_unknown_key_in_contract_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            DatasetContract(expects_categories=True, typo=1)  # type: ignore[call-arg]

    def test_missing_field_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            DatasetContract()  # type: ignore[call-arg]

    def test_default_is_empty_and_optional(self) -> None:
        cfg = validate_config({"datasets": ["amazon_fashion"]})
        assert cfg["dataset_contracts"] == {}
