"""R02 — one canonical path from raw suggestions to the effective configuration.

``effective_hyperparams`` is the single expansion every producer uses
(grid points, fixed configuration, Optuna samples) and every consumer
re-applies (winner export, replay): it fills the declared single-valued
defaults, derives the per-paper dimension split from ``total_dim`` and
is idempotent, so re-applying it to an already effective configuration
never double-expands or overrides a dimension.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from src.recommenders import iter_specs
from src.recommenders.hp_search import (
    EffectiveHyperparamsError,
    effective_hyperparams,
    get_fixed_hyperparams,
    get_hyperparam_grid,
    resolve_dimensions,
    sample_hyperparams,
)
from src.recommenders.hp_source import (
    HyperparamOrigin,
    WinnerResolutionError,
    hyperparam_sources,
    resolve_cell_hyperparams,
    resolve_replay_hyperparams,
)

BUILTIN = ("bpr", "vbpr", "vnpr", "deepstyle", "avbpr", "acf")


def _config(**overrides) -> dict:
    cfg = {
        "hp_search": {"strategy": "grid"},
        "common": {"total_dim": [64, 128], "learning_rate": [0.001, 0.01], "l2_reg": [0.0001]},
        "vnpr": {"dropout": 0.0},
        "acf": {"att_hidden": [64, 128], "max_history": [50]},
        "avbpr": {"att_hidden": [64, 128]},
    }
    cfg.update(overrides)
    return cfg


class _RecordingTrial:
    def suggest_categorical(self, name, choices):
        return choices[0]

    def suggest_int(self, name, low, high, **_):
        return low

    def suggest_float(self, name, low, high, **_):
        return low


class TestEffectiveHyperparams:
    def test_fills_single_valued_defaults_and_derives_the_dimension_split(self) -> None:
        cfg = _config()

        effective = effective_hyperparams("vbpr", {"total_dim": 128, "learning_rate": 0.01}, cfg)

        assert effective == {
            "total_dim": 128,
            "learning_rate": 0.01,
            "l2_reg": 0.0001,
            "latent_dim": 64,
            "visual_dim": 64,
        }

    @pytest.mark.parametrize("model", BUILTIN)
    def test_every_registered_model_derives_its_own_dimensions(self, model) -> None:
        cfg = _config()

        effective = effective_hyperparams(model, {"total_dim": 128, "learning_rate": 0.01}, cfg)

        for key, value in resolve_dimensions(model, 128).items():
            assert effective[key] == value

    @pytest.mark.parametrize("model", BUILTIN)
    def test_is_idempotent_for_every_registered_model(self, model) -> None:
        cfg = _config()
        once = effective_hyperparams(model, {"total_dim": 64, "learning_rate": 0.001}, cfg)

        assert effective_hyperparams(model, once, cfg) == once

    def test_rejects_a_direct_dimension_that_contradicts_the_budget(self) -> None:
        cfg = _config()

        with pytest.raises(EffectiveHyperparamsError, match="latent_dim"):
            effective_hyperparams("vbpr", {"total_dim": 128, "latent_dim": 128}, cfg)

    def test_leaves_multi_valued_unsuggested_keys_absent(self) -> None:
        cfg = _config()

        effective = effective_hyperparams("acf", {"total_dim": 64, "att_hidden": 64}, cfg)

        assert "learning_rate" not in effective  # never chosen: not fabricated
        assert effective["max_history"] == 50  # single-valued: pinned

    def test_without_a_budget_the_legacy_dimensions_pass_through(self) -> None:
        cfg = {"common": {"latent_dim": [16], "visual_dim": [8], "learning_rate": [0.01]}}

        effective = effective_hyperparams("vbpr", {"learning_rate": 0.01}, cfg)

        assert effective["latent_dim"] == 16 and effective["visual_dim"] == 8


class TestProducersShareTheCanonicalPath:
    def test_grid_points_are_already_effective(self) -> None:
        cfg = _config()
        for model in BUILTIN:
            for point in get_hyperparam_grid(model, cfg):
                assert effective_hyperparams(model, point, cfg) == point

    def test_fixed_configuration_is_already_effective(self) -> None:
        cfg = _config(
            common={"total_dim": 64, "learning_rate": 0.01, "l2_reg": 1e-4},
            acf={"att_hidden": 64, "max_history": 50},
            avbpr={"att_hidden": 64},
        )
        for model in BUILTIN:
            fixed = get_fixed_hyperparams(model, cfg)
            assert effective_hyperparams(model, fixed, cfg) == fixed

    def test_optuna_sample_from_an_hp_space_carries_the_pinned_defaults(self) -> None:
        cfg = _config(
            acf={
                "att_hidden": [64, 128],
                "max_history": [50],
                "hp_space": {
                    "total_dim": {"type": "categorical", "choices": [64, 128]},
                    "learning_rate": {"type": "categorical", "choices": [0.001, 0.01]},
                },
            }
        )

        sampled = sample_hyperparams(_RecordingTrial(), "acf", cfg)

        assert sampled["l2_reg"] == 0.0001
        assert sampled["max_history"] == 50
        assert "att_hidden" not in sampled  # multi-valued and not in the space
        assert sampled["latent_dim"] == 64 and sampled["visual_dim"] == 64
        assert effective_hyperparams("acf", sampled, cfg) == sampled


class TestHyperparamSources:
    def test_labels_every_key_as_suggested_default_or_derived(self) -> None:
        cfg = _config()
        raw = {"total_dim": 128, "learning_rate": 0.01}
        effective = effective_hyperparams("vbpr", raw, cfg)

        assert hyperparam_sources(raw, effective) == {
            "total_dim": "suggested",
            "learning_rate": "suggested",
            "l2_reg": "default",
            "latent_dim": "derived",
            "visual_dim": "derived",
        }


def _winner(root: Path, model: str, embedding: str, hyperparams: dict, metric: float) -> Path:
    path = root / "models" / "ds" / f"{model}_{embedding}_best.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": {},
            "hyperparams": hyperparams,
            "best_metric": metric,
            "n_users": 3,
            "n_items": 5,
            "selection_fingerprint": "fp",
        },
        path,
    )
    return path


class TestResolveReplayHyperparams:
    @pytest.mark.parametrize("model", BUILTIN)
    def test_replays_every_registered_model_from_its_winner(self, tmp_path, model) -> None:
        cfg = _config()
        embedding = "none" if model == "bpr" else "resnet50"
        winner = get_hyperparam_grid(model, cfg)[-1]
        _winner(tmp_path / "search", model, embedding, winner, 0.4)

        origin = resolve_replay_hyperparams(
            cfg,
            dataset="ds",
            model_name=model,
            embedding_name=embedding,
            search_results_root=tmp_path / "search",
            search_seed=1,
        )

        assert origin.source == "search"
        assert origin.hyperparams == winner
        assert origin.best_metric == 0.4
        assert origin.reference == f"ds__{model}__{embedding}"
        assert origin.provenance["strategy"] == "grid"
        assert origin.provenance["search_seed"] == 1
        assert json.loads(json.dumps(origin.to_dict())) == origin.to_dict()

    def test_a_legacy_winner_without_pinned_defaults_is_completed_not_double_expanded(
        self, tmp_path
    ) -> None:
        cfg = _config()
        _winner(
            tmp_path / "search", "vbpr", "resnet50", {"total_dim": 64, "learning_rate": 0.01}, 0.1
        )

        origin = resolve_replay_hyperparams(
            cfg,
            dataset="ds",
            model_name="vbpr",
            embedding_name="resnet50",
            search_results_root=tmp_path / "search",
            search_seed=1,
        )

        assert origin.hyperparams == {
            "total_dim": 64,
            "learning_rate": 0.01,
            "l2_reg": 0.0001,
            "latent_dim": 32,
            "visual_dim": 32,
        }
        assert origin.provenance["sources"]["l2_reg"] == "default"

    def test_missing_winner_artifact_fails(self, tmp_path) -> None:
        with pytest.raises(WinnerResolutionError, match="vbpr_resnet50_best.pt"):
            resolve_replay_hyperparams(
                _config(),
                dataset="ds",
                model_name="vbpr",
                embedding_name="resnet50",
                search_results_root=tmp_path / "search",
                search_seed=1,
            )

    def test_optuna_replay_without_storage_cannot_find_a_study(self, tmp_path) -> None:
        cfg = _config(hp_search={"strategy": "optuna", "optuna": {"storage": None}})
        _winner(tmp_path / "search", "vbpr", "resnet50", get_hyperparam_grid("vbpr", cfg)[0], 0.1)

        with pytest.raises(WinnerResolutionError, match="storage"):
            resolve_replay_hyperparams(
                cfg,
                dataset="ds",
                model_name="vbpr",
                embedding_name="resnet50",
                search_results_root=tmp_path / "search",
                search_seed=1,
            )

    def test_fixed_strategy_ignores_search_artifacts(self, tmp_path) -> None:
        cfg = _config(
            hp_search={"strategy": "fixed"},
            common={"total_dim": 64, "learning_rate": 0.01, "l2_reg": 1e-4},
        )

        origin = resolve_replay_hyperparams(
            cfg,
            dataset="ds",
            model_name="vbpr",
            embedding_name="resnet50",
            search_results_root=tmp_path / "missing",
            search_seed=1,
        )

        assert origin.source == "fixed"
        assert origin.hyperparams == get_fixed_hyperparams("vbpr", cfg)


class TestResolveCellHyperparamsProvenance:
    def test_search_origin_from_the_winners_file_is_effective_and_labelled(self, tmp_path) -> None:
        cfg = _config()
        winners = {
            "ds": {"vbpr": {"resnet50": {"hyperparams": {"total_dim": 64}, "best_metric": 0.2}}}
        }
        (tmp_path / "best_hyperparams.json").write_text(json.dumps(winners))

        origin = resolve_cell_hyperparams(
            cfg,
            dataset="ds",
            model_name="vbpr",
            embedding_name="resnet50",
            results_root=tmp_path,
        )

        assert origin.hyperparams["latent_dim"] == 32
        assert origin.suggestion == {"total_dim": 64}
        assert origin.provenance["sources"]["latent_dim"] == "derived"

    def test_to_dict_keeps_the_legacy_fields_and_adds_the_new_ones(self) -> None:
        origin = HyperparamOrigin("search", {"a": 1}, "ds__m__e", 0.5)

        assert origin.to_dict() == {
            "source": "search",
            "hyperparams": {"a": 1},
            "reference": "ds__m__e",
            "best_metric": 0.5,
            "suggestion": None,
            "provenance": None,
        }


def test_builtin_list_matches_the_registry() -> None:
    assert set(BUILTIN) <= {spec.name for spec in iter_specs()}
