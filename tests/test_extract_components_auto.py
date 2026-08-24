"""Auto-enabling of component extraction from ``recommenders_enabled``.

``src.steps.extract._components_needed`` must turn component extraction
on whenever an enabled recommender declares ``requires_components``
(e.g. ACF), so the train step can never hit
``EnabledRecommenderHasNoCellsError`` merely because the
``extract_components`` flag was left off.
"""

from __future__ import annotations

from src.steps.extract import _components_needed


class TestComponentsNeeded:
    def test_explicit_true_wins_without_any_recommender(self):
        config = {"extract_components": True, "recommenders_enabled": []}

        assert _components_needed(config) is True

    def test_acf_enabled_auto_enables_components(self):
        config = {"extract_components": False, "recommenders_enabled": ["bpr", "acf"]}

        assert _components_needed(config) is True

    def test_no_component_recommender_and_flag_off_stays_off(self):
        config = {
            "extract_components": False,
            "recommenders_enabled": ["bpr", "vbpr", "vnpr", "deepstyle"],
        }

        assert _components_needed(config) is False

    def test_missing_keys_default_to_off(self):
        assert _components_needed({}) is False

    def test_unregistered_recommender_name_is_skipped(self):
        config = {"recommenders_enabled": ["definitely_not_registered"]}

        assert _components_needed(config) is False
