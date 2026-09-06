"""Canonical scientific identity (C02 v2) — normalisation, digests, mutation matrix (E03/E04).

The identity of an experiment must change when any scientific ingredient
changes (seed, hyperparameters, split, item mapping, feature content,
normalisation recipe, protocol, condition, fold) and must NOT change when
identical content is moved to another path or when a mapping is written
in a different insertion order.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.utils.identity import (
    EVALUATION_SPLITS,
    SELECTION_SPLITS,
    IdentityError,
    canonical_digest,
    canonical_json,
    clear_identity_cache,
    experiment_identity,
    feature_digest,
    mapping_digest,
    resolve_data_identity,
    selection_scope_digest,
)

N_USERS, N_ITEMS, DIM = 4, 6, 3


def _write_dataset(root: Path, *, item_order: list[str] | None = None) -> Path:
    base = root / "processed" / "ds"
    base.mkdir(parents=True)
    train = [(u, (u + i) % N_ITEMS) for u in range(N_USERS) for i in range(2)]
    val = [(u, (u + 2) % N_ITEMS) for u in range(N_USERS)]
    test = [(u, (u + 3) % N_ITEMS) for u in range(N_USERS)]
    for name, rows in (("train", train), ("val", val), ("test", test)):
        pd.DataFrame(rows, columns=["user_idx", "item_idx"]).to_csv(
            base / f"{name}.csv", index=False
        )
    (base / "user2idx.json").write_text(json.dumps({str(u): u for u in range(N_USERS)}))
    ids = item_order or [str(i) for i in range(N_ITEMS)]
    (base / "item2idx.json").write_text(json.dumps({ids[i]: i for i in range(N_ITEMS)}))
    return root / "processed"


def _write_features(root: Path, *, seed: int = 0) -> Path:
    emb = root / "embeddings" / "ds"
    emb.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    np.save(emb / "resnet50.npy", rng.standard_normal((N_ITEMS, DIM)).astype(np.float32))
    np.save(emb / "vit_b16.npy", rng.standard_normal((N_ITEMS, DIM + 1)).astype(np.float32))
    return emb


def _write_sidecar(emb: Path, name: str, *, normalize: bool = True) -> Path:
    sidecar = emb / name
    sidecar.write_text(
        json.dumps(
            {
                "strategy": "mean",
                "online": True,
                "alignment": "learned",
                "dim": 2,
                "components": ["resnet50.npy", "vit_b16.npy"],
                "normalize": normalize,
                "fusion_kwargs": {},
            }
        )
    )
    return sidecar


@pytest.fixture(autouse=True)
def _fresh_cache():
    clear_identity_cache()
    yield
    clear_identity_cache()


class TestCanonicalJson:
    def test_should_sort_keys_and_use_compact_separators(self) -> None:
        assert canonical_json({"b": 1, "a": [1, 2]}) == '{"a":[1,2],"b":1}'

    def test_should_keep_int_and_float_distinct(self) -> None:
        assert canonical_json({"x": 1}) != canonical_json({"x": 1.0})

    def test_should_keep_bool_distinct_from_int(self) -> None:
        assert canonical_json({"x": True}) != canonical_json({"x": 1})

    def test_should_convert_numpy_scalars_arrays_tuples_and_paths(self) -> None:
        payload = {
            "i": np.int64(3),
            "f": np.float32(0.5),
            "a": np.array([1, 2]),
            "t": (1, 2),
            "p": Path("a/b"),
        }
        assert canonical_json(payload) == '{"a":[1,2],"f":0.5,"i":3,"p":"a/b","t":[1,2]}'

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
    def test_should_reject_non_finite_numbers(self, bad: float) -> None:
        with pytest.raises(IdentityError, match="non-finite"):
            canonical_json({"x": bad})

    def test_should_reject_non_string_keys_and_unknown_types(self) -> None:
        with pytest.raises(IdentityError, match="non-string key"):
            canonical_json({1: "a"})
        with pytest.raises(IdentityError, match="unsupported value type"):
            canonical_json({"x": object()})

    def test_digest_is_sha256_of_the_canonical_text(self) -> None:
        import hashlib

        payload = {"k": [1.5, "v"]}
        expected = hashlib.sha256(canonical_json(payload).encode()).hexdigest()
        assert canonical_digest(payload) == expected


class TestMappingDigest:
    def test_should_ignore_insertion_order(self) -> None:
        assert mapping_digest({"b": 1, "a": 0}) == mapping_digest({"a": 0, "b": 1})

    def test_should_change_when_an_index_is_reassigned(self) -> None:
        assert mapping_digest({"a": 1, "b": 0}) != mapping_digest({"a": 0, "b": 1})

    def test_should_reject_holes(self) -> None:
        with pytest.raises(IdentityError):
            mapping_digest({"a": 0, "b": 2})


class TestFeatureDigest:
    def test_sidecar_digest_recurses_into_components(self, tmp_path) -> None:
        emb = _write_features(tmp_path)
        sidecar = _write_sidecar(emb, "hybrid_mean_learned_D2.json")
        before = feature_digest(sidecar)

        clear_identity_cache()
        np.save(emb / "vit_b16.npy", np.ones((N_ITEMS, DIM + 1), dtype=np.float32))

        assert feature_digest(sidecar) != before

    def test_sidecar_digest_changes_with_the_normalisation_recipe(self, tmp_path) -> None:
        emb = _write_features(tmp_path)
        a = feature_digest(_write_sidecar(emb, "a.json", normalize=True))
        b = feature_digest(_write_sidecar(emb, "b.json", normalize=False))
        assert a != b

    def test_missing_component_is_an_error_not_a_guess(self, tmp_path) -> None:
        emb = _write_features(tmp_path)
        sidecar = _write_sidecar(emb, "hybrid.json")
        (emb / "vit_b16.npy").unlink()
        with pytest.raises(IdentityError, match="missing"):
            feature_digest(sidecar)


def _identity(processed: Path, feature: Path | None, **overrides) -> dict:
    data = resolve_data_identity(processed, "ds", feature, splits=SELECTION_SPLITS)
    fields = {
        "data": data,
        "model_name": "vbpr",
        "implementation": "impl",
        "hyperparams": {"learning_rate": 0.01, "latent_dim": 4},
        "selection_budget": {"epochs": 2, "patience": 1},
        "seed": 1,
        "protocol": {"candidates": "full_ranking", "k_values": [10]},
        "condition": "frozen",
        "fold": None,
    }
    fields.update(overrides)
    return experiment_identity(**fields)


class TestMutationMatrix:
    """Change one ingredient at a time; identical content elsewhere matches."""

    def test_identical_content_under_another_root_has_the_same_identity(self, tmp_path) -> None:
        processed = _write_dataset(tmp_path / "a")
        feature = _write_features(tmp_path / "a") / "resnet50.npy"
        moved = tmp_path / "elsewhere"
        shutil.copytree(tmp_path / "a", moved)

        original = canonical_digest(_identity(processed, feature))
        relocated = canonical_digest(
            _identity(moved / "processed", moved / "embeddings" / "ds" / "resnet50.npy")
        )

        assert original == relocated

    def test_item_mapping_insertion_order_does_not_change_the_identity(self, tmp_path) -> None:
        a = _write_dataset(tmp_path / "a")
        b = tmp_path / "b" / "processed" / "ds"
        shutil.copytree(a / "ds", b)
        mapping = json.loads((b / "item2idx.json").read_text())
        (b / "item2idx.json").write_text(json.dumps(dict(reversed(list(mapping.items())))))

        assert canonical_digest(_identity(a, None)) == canonical_digest(_identity(b.parent, None))

    @pytest.mark.parametrize(
        "mutation",
        ["seed", "hyperparams", "protocol", "condition", "fold", "selection_budget"],
    )
    def test_changing_a_scalar_ingredient_changes_the_identity(self, tmp_path, mutation) -> None:
        processed = _write_dataset(tmp_path)
        base = canonical_digest(_identity(processed, None))
        changed = {
            "seed": {"seed": 2},
            "hyperparams": {"hyperparams": {"learning_rate": 0.01, "latent_dim": 8}},
            "protocol": {"protocol": {"candidates": "full_ranking", "k_values": [20]}},
            "condition": {"condition": "finetuned"},
            "fold": {"fold": {"index": 0, "k": 5, "partition_seed": 7}},
            "selection_budget": {"selection_budget": {"epochs": 3, "patience": 1}},
        }[mutation]

        assert canonical_digest(_identity(processed, None, **changed)) != base

    def test_changing_the_split_changes_the_identity(self, tmp_path) -> None:
        processed = _write_dataset(tmp_path)
        base = canonical_digest(_identity(processed, None))
        val = processed / "ds" / "val.csv"
        val.write_text(val.read_text().replace("\n0,", "\n1,", 1))
        clear_identity_cache()

        assert canonical_digest(_identity(processed, None)) != base

    def test_changing_the_item_mapping_changes_the_identity(self, tmp_path) -> None:
        a = _write_dataset(tmp_path / "a")
        b = _write_dataset(tmp_path / "b", item_order=[str(i) for i in reversed(range(N_ITEMS))])
        assert canonical_digest(_identity(a, None)) != canonical_digest(_identity(b, None))

    def test_changing_feature_content_changes_the_identity(self, tmp_path) -> None:
        processed = _write_dataset(tmp_path)
        feature = _write_features(tmp_path, seed=0) / "resnet50.npy"
        base = canonical_digest(_identity(processed, feature))
        clear_identity_cache()
        _write_features(tmp_path, seed=1)

        assert canonical_digest(_identity(processed, feature)) != base

    def test_evaluation_identity_includes_the_test_split(self, tmp_path) -> None:
        processed = _write_dataset(tmp_path)
        before = resolve_data_identity(processed, "ds", None, splits=EVALUATION_SPLITS)
        test = processed / "ds" / "test.csv"
        test.write_text(test.read_text().replace("\n0,", "\n1,", 1))
        clear_identity_cache()
        after = resolve_data_identity(processed, "ds", None, splits=EVALUATION_SPLITS)
        selection = resolve_data_identity(processed, "ds", None, splits=SELECTION_SPLITS)

        assert before.split_digest != after.split_digest
        assert selection.split_digest != after.split_digest

    def test_selection_scope_ignores_hyperparameters_only(self, tmp_path) -> None:
        processed = _write_dataset(tmp_path)
        base = _identity(processed, None)
        other_hp = _identity(processed, None, hyperparams={"learning_rate": 0.1, "latent_dim": 4})
        other_seed = _identity(processed, None, seed=9)

        assert selection_scope_digest(base) == selection_scope_digest(other_hp)
        assert selection_scope_digest(base) != selection_scope_digest(other_seed)

    def test_unresolved_data_is_explicit(self) -> None:
        payload = experiment_identity(
            data=None,
            model_name="bpr",
            implementation="impl",
            hyperparams={},
            selection_budget=None,
            seed=1,
            protocol={},
            condition="frozen",
        )
        assert payload["data_identity"] == "unresolved"
        assert payload["schema_version"] == 2
