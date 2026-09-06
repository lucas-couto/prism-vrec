"""S02 — pre-fusion normalisation in the non-learned online sidecar path.

Finding F10: ``load_embedding`` stacked the components of an equal-dim
online sidecar (``adaptive_gated`` over PCA-aligned or pre-aligned
sources) exactly as stored, ignoring the sidecar's ``normalize`` flag.
Every offline equal-dim strategy and the learned-alignment module L2
normalise each source before the element-wise operation (protocol
§10.3), so the online non-learned representation was the only one built
from un-normalised sources: a ``[3, 3]`` row kept its norm ``sqrt(18)``.

These tests pin the corrected recipe: each source is normalised exactly
once, before fusion, with the same zero-vector rule as the offline
:func:`l2_normalize` (zero rows stay zero); ``normalize=False`` keeps
magnitudes; the lazy stacked source mirrors the eager array; and the
stacked slices are byte-for-byte the inputs the offline strategies
consume, so every equal-dim strategy agrees offline vs online on the
same aligned inputs.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from src.data.feature_source import ArrayFeatureSource, StackedFeatureSource
from src.fusions import get_fusion_strategy, load_embedding
from src.fusions.online import (
    SIDECAR_RECIPE_VERSION,
    LearnedAlignmentFusion,
    StackedSources,
)
from src.fusions.strategies import fuse_concat, l2_normalize
from src.recommenders.vbpr import VBPR

N_ITEMS, DIM = 12, 5

EQUAL_DIM_OPS = (
    ("mean", {}),
    ("sum", {}),
    ("prod", {}),
    ("max_pool", {}),
    ("weighted_mean", {"weights": [0.3, 0.7]}),
    ("softmax_weighted", {"logits": [1.0, 0.0]}),
    ("sigmoid_gated", {"logits": [0.4, -0.4]}),
)


def _sources(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Two equal-dim sources of very different scale, with one zero row each."""
    rng = np.random.default_rng(seed)
    a = (rng.standard_normal((N_ITEMS, DIM)) * 10.0).astype(np.float32)
    b = (rng.standard_normal((N_ITEMS, DIM)) * 0.1).astype(np.float32)
    a[3] = 0.0
    b[7] = 0.0
    a[0] = 3.0  # the F10 row: norm sqrt(9 * DIM), never 1 before the fix
    return a, b


def _write_sidecar(tmp_path: Path, payload: dict, name: str = "hybrid_adaptive_gated.json") -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


@pytest.fixture
def sidecar_dir(tmp_path: Path) -> Path:
    a, b = _sources()
    np.save(tmp_path / "a.npy", a)
    np.save(tmp_path / "b.npy", b)
    return tmp_path


def _payload(normalize: bool | None, **extra) -> dict:
    payload = {
        "strategy": "adaptive_gated",
        "online": True,
        "alignment": "pca",
        "components": ["a.npy", "b.npy"],
        **extra,
    }
    if normalize is not None:
        payload["normalize"] = normalize
    return payload


class TestEagerStack:
    def test_f10_source_row_is_unit_norm_after_the_fix(self, sidecar_dir: Path) -> None:
        path = _write_sidecar(sidecar_dir, _payload(True))

        stacked = load_embedding(path)

        norms = np.linalg.norm(stacked[0], axis=-1)
        np.testing.assert_allclose(norms, [1.0, 1.0], rtol=1e-6)

    def test_each_source_is_normalised_once_and_zero_rows_stay_finite_zero(
        self, sidecar_dir: Path
    ) -> None:
        a, b = _sources()
        path = _write_sidecar(sidecar_dir, _payload(True))

        stacked = load_embedding(path)

        assert stacked.shape == (N_ITEMS, 2, DIM)
        np.testing.assert_allclose(stacked[:, 0, :], l2_normalize(a), rtol=1e-6)
        np.testing.assert_allclose(stacked[:, 1, :], l2_normalize(b), rtol=1e-6)
        assert np.isfinite(stacked).all()
        assert not stacked[3, 0].any() and not stacked[7, 1].any()
        nonzero = np.linalg.norm(stacked, axis=-1)[np.linalg.norm(stacked, axis=-1) > 0]
        np.testing.assert_allclose(nonzero, 1.0, rtol=1e-6)

    def test_normalize_false_preserves_source_magnitudes(self, sidecar_dir: Path) -> None:
        a, b = _sources()
        path = _write_sidecar(sidecar_dir, _payload(False))

        stacked = load_embedding(path)

        np.testing.assert_array_equal(stacked, np.stack([a, b], axis=1))

    def test_missing_flag_defaults_to_the_declared_protocol_default(
        self, sidecar_dir: Path
    ) -> None:
        """No ``normalize`` key: the loader applies the §10.3 default (true)."""
        path = _write_sidecar(sidecar_dir, _payload(None))

        stacked = load_embedding(path)

        np.testing.assert_allclose(np.linalg.norm(stacked[0], axis=-1), [1.0, 1.0], rtol=1e-6)

    def test_dtype_is_kept(self, sidecar_dir: Path) -> None:
        path = _write_sidecar(sidecar_dir, _payload(True))

        assert load_embedding(path).dtype == np.float32


class TestRecipeIdentity:
    def test_loaded_stack_carries_the_loader_recipe_version(self, sidecar_dir: Path) -> None:
        path = _write_sidecar(sidecar_dir, _payload(True, recipe_version=SIDECAR_RECIPE_VERSION))

        stacked = load_embedding(path)
        lazy = load_embedding(path, lazy=True)

        assert isinstance(stacked, StackedSources)
        assert stacked.recipe_version == SIDECAR_RECIPE_VERSION == 2
        assert stacked.sidecar_recipe_version == SIDECAR_RECIPE_VERSION
        assert stacked.normalize is True
        assert (lazy.recipe_version, lazy.sidecar_recipe_version) == (2, 2)
        assert lazy.normalize is True

    def test_legacy_sidecar_is_identified_not_guessed(self, sidecar_dir: Path) -> None:
        """A sidecar written before the fix has no version: reported as ``None``."""
        path = _write_sidecar(sidecar_dir, _payload(True))

        stacked = load_embedding(path)
        lazy = load_embedding(path, lazy=True)

        assert stacked.sidecar_recipe_version is None
        assert lazy.sidecar_recipe_version is None
        assert stacked.recipe_version == lazy.recipe_version == SIDECAR_RECIPE_VERSION

    def test_learned_sidecar_is_untouched_by_the_fix(self, sidecar_dir: Path) -> None:
        a, b = _sources()
        payload = _payload(True, alignment="learned", strategy="mean", dim=3)
        path = _write_sidecar(sidecar_dir, payload, "hybrid_mean_learned_D3.json")

        ragged = load_embedding(path)

        # Raw native rows: the learned module normalises AFTER its linear map.
        np.testing.assert_array_equal(np.asarray(ragged), np.concatenate([a, b], axis=1))

    def test_fuse_step_stamps_the_recipe_version_on_non_learned_sidecars(
        self, tmp_path: Path
    ) -> None:
        from src.steps.fuse import _collect_fusion_tasks

        emb_dir = tmp_path / "emb"
        (emb_dir / "ds").mkdir(parents=True)
        for name in ("a", "b"):
            np.save(emb_dir / "ds" / f"{name}_p8.npy", np.ones((4, 8), dtype=np.float32))
        processed = tmp_path / "processed" / "ds"
        processed.mkdir(parents=True)
        (processed / "train.csv").write_text("user_idx,item_idx\n0,0\n1,1\n", encoding="utf-8")

        tasks = _collect_fusion_tasks(
            "ds",
            str(emb_dir),
            str(tmp_path / "processed"),
            ["a_p8", "b_p8"],
            {},
            True,
            {"adaptive_gated"},
            "learned",
            8,
            variant_token="_p8",
            pre_aligned=True,
        )

        [task] = tasks
        assert task["sidecar_payload"]["recipe_version"] == SIDECAR_RECIPE_VERSION
        assert task["sidecar_payload"]["normalize"] is True


class TestOfflineOnlineParity:
    @pytest.mark.parametrize(("strategy", "kwargs"), EQUAL_DIM_OPS)
    @pytest.mark.parametrize("normalize", [True, False])
    def test_stacked_slices_are_the_offline_inputs_for_every_strategy(
        self, sidecar_dir: Path, strategy: str, kwargs: dict, normalize: bool
    ) -> None:
        """Offline ``fuse_X([a, b], normalize)`` == ``fuse_X`` of the online slices."""
        a, b = _sources()
        path = _write_sidecar(sidecar_dir, _payload(normalize))
        fuse = get_fusion_strategy(strategy, **kwargs)

        stacked = load_embedding(path)
        offline = fuse([a, b], normalize=normalize)
        online = fuse([stacked[:, 0, :], stacked[:, 1, :]], normalize=False)

        np.testing.assert_allclose(online, offline, rtol=1e-6, atol=1e-7)

    @pytest.mark.parametrize("normalize", [True, False])
    def test_concat_parity(self, sidecar_dir: Path, normalize: bool) -> None:
        a, b = _sources()
        path = _write_sidecar(sidecar_dir, _payload(normalize))

        stacked = load_embedding(path)

        np.testing.assert_allclose(
            stacked.reshape(N_ITEMS, 2 * DIM), fuse_concat([a, b], normalize=normalize), rtol=1e-6
        )

    @pytest.mark.parametrize(("strategy", "kwargs"), EQUAL_DIM_OPS)
    @pytest.mark.parametrize("normalize", [True, False])
    def test_learned_module_with_identity_projections_matches_offline(
        self, strategy: str, kwargs: dict, normalize: bool
    ) -> None:
        """Same aligned inputs, learned path: normalise after the (identity) map."""
        a, b = _sources()
        module = LearnedAlignmentFusion([DIM, DIM], DIM, strategy, normalize=normalize, **kwargs)
        with torch.no_grad():
            for proj in module.projections:
                proj.weight.copy_(torch.eye(DIM))
                proj.bias.zero_()
        fuse = get_fusion_strategy(strategy, **kwargs)

        with torch.no_grad():
            online = module(torch.from_numpy(np.concatenate([a, b], axis=1))).numpy()
        offline = fuse([a, b], normalize=normalize)

        np.testing.assert_allclose(online, offline, rtol=1e-5, atol=1e-6)

    def test_recommender_applies_no_second_normalisation(self, sidecar_dir: Path) -> None:
        """adaptive_gated at init is the plain mean of the (already unit) sources."""
        a, b = _sources()
        path = _write_sidecar(sidecar_dir, _payload(True))
        stacked = load_embedding(path)
        torch.manual_seed(0)
        model = VBPR(3, N_ITEMS, stacked, {"latent_dim": 2, "visual_dim": 2}).eval()

        with torch.no_grad():
            fused = model._resolve_visual(torch.arange(N_ITEMS)).numpy()

        expected = 0.5 * (l2_normalize(a) + l2_normalize(b))
        np.testing.assert_allclose(fused, expected, rtol=1e-6, atol=1e-7)


class TestLazyMirror:
    @pytest.mark.parametrize("normalize", [True, False])
    def test_lazy_rows_equal_the_eager_stack(self, sidecar_dir: Path, normalize: bool) -> None:
        path = _write_sidecar(sidecar_dir, _payload(normalize))

        eager = load_embedding(path)
        lazy = load_embedding(path, lazy=True)

        ids = np.array([3, 0, 7, 0, 11])
        np.testing.assert_array_equal(lazy.read_rows(ids), np.asarray(eager)[ids])
        np.testing.assert_array_equal(lazy.read_rows(np.arange(N_ITEMS)), np.asarray(eager))

    def test_stacked_source_default_keeps_rows_as_stored(self) -> None:
        a, b = _sources()

        source = StackedFeatureSource([ArrayFeatureSource(a), ArrayFeatureSource(b)])

        assert source.normalize is False
        np.testing.assert_array_equal(source.read_rows(np.array([0])), np.stack([a, b], 1)[[0]])

    def test_stacked_source_normalises_per_source_per_row(self) -> None:
        a, b = _sources()

        source = StackedFeatureSource(
            [ArrayFeatureSource(a), ArrayFeatureSource(b)], normalize=True
        )
        rows = source.read_rows(np.array([0, 3, 7]))

        np.testing.assert_allclose(rows[:, 0, :], l2_normalize(a)[[0, 3, 7]], rtol=1e-6)
        np.testing.assert_allclose(rows[:, 1, :], l2_normalize(b)[[0, 3, 7]], rtol=1e-6)
        assert rows.dtype == np.float32
        assert source.read_rows(np.array([], dtype=np.int64)).shape == (0, 2, DIM)

    def test_normalising_a_component_stack_is_refused(self) -> None:
        comp = ArrayFeatureSource(np.ones((4, 3, 2), dtype=np.float16))

        with pytest.raises(ValueError, match="2-D"):
            StackedFeatureSource([comp, comp], normalize=True)
