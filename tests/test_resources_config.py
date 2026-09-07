"""``configs/resources.yaml`` is the single source of computational limits.

The resolver validates every field at startup and names the offending
key; the removed keys (``hp_search.workers``, a top-level ``dataloader:``
block, the flat M05 names) fail with a pointer to their replacement;
every consumer reads the resolved value at its boundary; and the block
is execution metadata: recorded in the manifest, absent from the
scientific identity.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

import src.steps.train as train_mod
from src.recommenders.bpr import BPR
from src.steps import fuse as fuse_mod
from src.utils import memory as memory_mod
from src.utils.config_schema import validate_config
from src.utils.dataloader import autotune, resolve_dataloader_settings
from src.utils.identity import canonical_digest
from src.utils.parallel import TrainingJob, _WorkerContext
from src.utils.resources import VRAM_SHARE_ENV, ResourcesConfig, resolve_resources
from src.utils.training import resolve_training_identity

GB = 1024**3
SHIPPED = Path("configs/resources.yaml")


def _shipped() -> dict:
    return yaml.safe_load(SHIPPED.read_text(encoding="utf-8"))


class TestDefaultsAndShippedFile:
    def test_missing_block_resolves_to_the_documented_defaults(self) -> None:
        resolved = resolve_resources(None, env={})

        assert resolved == ResourcesConfig()
        assert resolved.gpu.vram_share == 0.5
        assert resolved.gpu.ranking_vram_share == 0.125
        assert resolved.host.budget_bytes is None
        assert resolved.host.headroom_bytes == resolved.host.reserved_bytes == 4 * GB
        assert (resolved.workers.training, resolved.workers.fusion) == (1, 1)
        assert resolved.workers.dataloader is None
        assert (resolved.features.residency, resolved.features.item_block) == ("dense", 8192)

    def test_shipped_file_matches_the_approved_schema(self) -> None:
        resolved = resolve_resources(_shipped(), env={})

        assert resolved.gpu.vram_share == 0.95  # unattended nights; 0.5 is the code default
        assert resolved.gpu.ranking_vram_share == 0.125
        assert resolved.host.budget_bytes is None
        assert resolved.workers.training == 1
        assert resolved.workers.fusion == 1
        # The researcher's pins that used to live in default.yaml -> dataloader.
        assert resolved.workers.dataloader == 10
        assert resolved.dataloader.prefetch_factor == 6
        assert resolved.dataloader.batch_size == 192
        assert resolved.features.item_block == 8192

    def test_shipped_configs_directory_loads_and_carries_the_block(self, monkeypatch) -> None:
        from src.utils.config import load_config

        monkeypatch.delenv(VRAM_SHARE_ENV, raising=False)
        merged = load_config("configs")

        assert resolve_resources(merged, env={}) == resolve_resources(_shipped(), env={})
        assert "dataloader" not in merged
        assert "workers" not in merged["hp_search"]

    def test_payload_spells_auto_for_unpinned_values(self) -> None:
        payload = ResourcesConfig().to_payload()

        assert payload["workers"]["dataloader"] == "auto"
        assert payload["dataloader"] == {"prefetch_factor": "auto", "batch_size": "auto"}
        json.dumps(payload)


class TestValidationMatrix:
    @pytest.mark.parametrize(
        ("block", "key"),
        [
            ({"gpu": {"vram_share": 0}}, "resources.gpu.vram_share"),
            ({"gpu": {"vram_share": 1.5}}, "resources.gpu.vram_share"),
            ({"gpu": {"vram_share": -0.1}}, "resources.gpu.vram_share"),
            ({"gpu": {"vram_share": math.nan}}, "resources.gpu.vram_share"),
            ({"gpu": {"vram_share": math.inf}}, "resources.gpu.vram_share"),
            ({"gpu": {"vram_share": True}}, "resources.gpu.vram_share"),
            ({"gpu": {"vram_share": "0.5"}}, "resources.gpu.vram_share"),
            ({"gpu": {"ranking_vram_share": 0}}, "resources.gpu.ranking_vram_share"),
            ({"gpu": {"ranking_vram_share": 2}}, "resources.gpu.ranking_vram_share"),
            ({"host": {"budget_bytes": -1}}, "resources.host.budget_bytes"),
            ({"host": {"budget_bytes": 1.5}}, "resources.host.budget_bytes"),
            ({"host": {"budget_bytes": True}}, "resources.host.budget_bytes"),
            ({"host": {"headroom_bytes": -1}}, "resources.host.headroom_bytes"),
            ({"host": {"headroom_bytes": math.nan}}, "resources.host.headroom_bytes"),
            ({"host": {"headroom_bytes": None}}, "resources.host.headroom_bytes"),
            ({"host": {"reserved_bytes": "4g"}}, "resources.host.reserved_bytes"),
            ({"workers": {"training": -1}}, "resources.workers.training"),
            ({"workers": {"training": 2.0}}, "resources.workers.training"),
            ({"workers": {"fusion": 0}}, "resources.workers.fusion"),
            ({"workers": {"dataloader": -1}}, "resources.workers.dataloader"),
            ({"workers": {"dataloader": "many"}}, "resources.workers.dataloader"),
            ({"dataloader": {"prefetch_factor": 0}}, "resources.dataloader.prefetch_factor"),
            ({"dataloader": {"batch_size": 0.5}}, "resources.dataloader.batch_size"),
            ({"features": {"residency": "sometimes"}}, "resources.features.residency"),
            ({"features": {"item_block": 0}}, "resources.features.item_block"),
            ({"features": {"item_block": math.inf}}, "resources.features.item_block"),
        ],
    )
    def test_bad_values_fail_naming_the_key(self, block, key) -> None:
        with pytest.raises(ValueError, match=key.replace(".", r"\.")):
            resolve_resources({"resources": block}, env={})

    @pytest.mark.parametrize(
        ("block", "match"),
        [
            ({"cpu": {}}, "resources has unknown keys: \\['cpu'\\]"),
            ({"gpu": {"vram": 0.5}}, "resources.gpu has unknown keys: \\['vram'\\]"),
            ({"workers": {"train": 1}}, "resources.workers has unknown keys"),
            ({"features": {"block": 1}}, "resources.features has unknown keys"),
        ],
    )
    def test_unknown_keys_fail(self, block, match) -> None:
        with pytest.raises(ValueError, match=match):
            resolve_resources({"resources": block}, env={})

    @pytest.mark.parametrize(
        ("block", "replacement"),
        [
            ({"host_budget_bytes": None}, "resources.host.budget_bytes"),
            ({"headroom_bytes": 1}, "resources.host.headroom_bytes"),
            ({"max_workers": 2}, "resources.workers.training"),
            ({"feature_residency": "dense"}, "resources.features.residency"),
        ],
    )
    def test_flat_legacy_keys_fail_naming_the_nested_key(self, block, replacement) -> None:
        with pytest.raises(ValueError, match=replacement.replace(".", r"\.")):
            resolve_resources({"resources": block}, env={})

    def test_non_mapping_block_and_section_fail(self) -> None:
        with pytest.raises(ValueError, match="resources must be a mapping"):
            resolve_resources({"resources": [1]}, env={})
        with pytest.raises(ValueError, match="resources.gpu must be a mapping"):
            resolve_resources({"resources": {"gpu": 0.5}}, env={})

    def test_hp_search_workers_is_rejected_with_a_pointer(self) -> None:
        with pytest.raises(ValueError, match="resources.workers.training"):
            resolve_resources({"hp_search": {"workers": 1}}, env={})

    def test_top_level_dataloader_block_is_rejected_with_a_pointer(self) -> None:
        with pytest.raises(ValueError, match="resources.dataloader"):
            resolve_resources({"dataloader": {"num_workers": 4}}, env={})

    def test_null_sections_mean_defaults(self) -> None:
        resolved = resolve_resources({"resources": {"gpu": None, "host": None}}, env={})

        assert resolved == ResourcesConfig()


class TestSchemaBoundary:
    """``validate_config`` (every ``load_config``) fails before any step runs."""

    def _minimal(self, **extra) -> dict:
        return {"seed": 1, "device": "cpu", "datasets": ["x"], **extra}

    def test_invalid_block_fails_config_validation(self) -> None:
        with pytest.raises(ValidationError, match="resources.gpu.vram_share"):
            validate_config(self._minimal(resources={"gpu": {"vram_share": 3}}))

    def test_hp_search_workers_fails_config_validation(self) -> None:
        with pytest.raises(ValidationError, match="resources.workers.training"):
            validate_config(self._minimal(hp_search={"strategy": "grid", "workers": 1}))

    def test_top_level_dataloader_fails_config_validation(self) -> None:
        with pytest.raises(ValidationError, match="resources.workers.dataloader"):
            validate_config(self._minimal(dataloader={"num_workers": 4}))

    def test_valid_block_survives_validation_verbatim(self) -> None:
        block = {"workers": {"training": 2}, "features": {"item_block": 4096}}

        out = validate_config(self._minimal(resources=block))

        assert out["resources"] == block
        assert "workers" not in out["hp_search"]


class TestEnvOverridePrecedence:
    def test_env_beats_the_yaml_and_the_yaml_beats_the_default(self) -> None:
        yaml_only = resolve_resources({"resources": {"gpu": {"vram_share": 0.3}}}, env={})
        with_env = resolve_resources(
            {"resources": {"gpu": {"vram_share": 0.3}}}, env={VRAM_SHARE_ENV: "0.9"}
        )

        assert yaml_only.gpu.vram_share == 0.3
        assert with_env.gpu.vram_share == 0.9

    def test_env_override_is_validated_like_the_key(self) -> None:
        with pytest.raises(ValueError, match=VRAM_SHARE_ENV):
            resolve_resources({}, env={VRAM_SHARE_ENV: "1.2"})


class TestConsumers:
    """Every reader observes the resolved value at its own boundary."""

    def test_fusion_pool_size_comes_from_workers_fusion(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(fuse_mod, "available_cpus", lambda: 16)
        monkeypatch.setattr(memory_mod, "memory_budget_bytes", lambda: 64 * GB)
        pending = [{"strategy_name": "concat", "emb_list_paths": [], "sidecar_payload": {}}] * 8

        three = resolve_resources({"resources": {"workers": {"fusion": 3}}}, env={})
        assert fuse_mod._plan_fusion_workers(pending, three) == 3
        assert fuse_mod._plan_fusion_workers(pending, ResourcesConfig()) == 1

    def test_fusion_reserve_comes_from_host_reserved_bytes(self, monkeypatch) -> None:
        seen: dict = {}
        monkeypatch.setattr(fuse_mod, "available_cpus", lambda: 16)
        monkeypatch.setattr(fuse_mod, "plan_pool_workers", lambda **kw: seen.update(kw) or 1)
        pending = [{"strategy_name": "concat", "emb_list_paths": [], "sidecar_payload": {}}]
        resources = resolve_resources({"resources": {"host": {"reserved_bytes": 7 * GB}}}, env={})

        fuse_mod._plan_fusion_workers(pending, resources)

        assert seen["reserve_bytes"] == 7 * GB

    def test_training_worker_count_comes_from_workers_training(self, tmp_path, monkeypatch) -> None:
        cfg = {
            "seed": 1,
            "device": "cpu",
            "datasets": ["synthetic"],
            "recommenders_enabled": ["bpr"],
            "hp_search": {"strategy": "grid", "optuna": {}},
            "common": {"total_dim": 8, "learning_rate": 0.001, "l2_reg": 1e-4},
            "paths": {
                "data_processed": str(tmp_path / "p"),
                "embeddings": str(tmp_path / "e"),
                "results": str(tmp_path / "r"),
            },
            "resources": {"workers": {"training": 3}},
        }
        seen: dict = {}
        monkeypatch.setattr(train_mod, "load_config", lambda: cfg)
        monkeypatch.setattr(train_mod, "assert_dimension_parity", lambda c: None)
        monkeypatch.setattr(
            "src.recommenders.hp_budget.assert_uniform_budget", lambda c: None, raising=False
        )
        monkeypatch.setattr(
            "src.recommenders.hp_budget.resolve_hp_budget", lambda c, d: None, raising=False
        )
        monkeypatch.setattr(
            "src.steps.validate_features.gate_dataset_features",
            lambda *a, **k: None,
            raising=False,
        )
        monkeypatch.setattr(train_mod, "_run_grid", lambda *a, **k: seen.update(k))

        train_mod.run("frozen")

        assert seen == {"workers": 3, "sequential": False}

    def test_train_step_refuses_the_removed_hp_search_workers_key(self, monkeypatch) -> None:
        cfg = {
            "seed": 1,
            "datasets": ["synthetic"],
            "recommenders_enabled": ["bpr"],
            "hp_search": {"strategy": "grid", "workers": 1},
        }
        monkeypatch.setattr(train_mod, "load_config", lambda: cfg)
        monkeypatch.setattr(train_mod, "assert_dimension_parity", lambda c: None)
        monkeypatch.setattr(
            "src.recommenders.hp_budget.assert_uniform_budget", lambda c: None, raising=False
        )
        monkeypatch.setattr(
            "src.recommenders.hp_budget.resolve_hp_budget", lambda c, d: None, raising=False
        )
        monkeypatch.setattr(
            "src.steps.validate_features.gate_dataset_features",
            lambda *a, **k: None,
            raising=False,
        )

        with pytest.raises(ValueError, match="resources.workers.training"):
            train_mod.run("frozen")

    def test_ranking_share_used_by_the_worker_planner(self, tmp_path) -> None:
        import logging

        cfg = {
            "paths": {"checkpoints": str(tmp_path / "ckpt")},
            "resources": {"gpu": {"ranking_vram_share": 0.25}},
        }
        context = _WorkerContext(1, logging.getLogger("resources_test"), cfg)
        context._worker_vram = 8 * GB
        job = TrainingJob(
            dataset_name="d",
            model_name="bpr",
            embedding_name="none",
            hyperparams={},
            n_users=1,
            n_items=1,
            embeddings_path=None,
            processed_dir=str(tmp_path),
            device="cpu",
        )

        assert context._ranking_budget(job) == 2 * GB
        job.retry_count = 1
        assert context._ranking_budget(job) == GB

    def test_item_block_installed_on_a_model(self) -> None:
        model = BPR(n_users=3, n_items=5, config={"latent_dim": 2})
        assert model._LAZY_ITEM_BLOCK == 8192

        block = resolve_resources({"resources": {"features": {"item_block": 512}}}, env={})
        model.configure_item_block(block.features.item_block)

        assert model._LAZY_ITEM_BLOCK == 512
        assert BPR._LAZY_ITEM_BLOCK == 8192  # class default untouched
        with pytest.raises(ValueError, match="item_block"):
            model.configure_item_block(0)

    def test_ledger_charges_lazy_jobs_per_item_block(self) -> None:
        small = train_mod._resident_feature_bytes(10**9, 16, lazy=True, item_block=1000)
        large = train_mod._resident_feature_bytes(10**9, 16, lazy=True, item_block=8192)

        assert small == 1000 * 16 * 4 * 2
        assert large == 8192 * 16 * 4 * 2

    def test_dataloader_workers_come_from_workers_dataloader(self, monkeypatch) -> None:
        import src.utils.dataloader as dl

        autotune.cache_clear()
        monkeypatch.setattr(dl, "available_cpus", lambda: 16)
        monkeypatch.setattr(dl, "_memory_budget_bytes", lambda: 64 * GB)
        try:
            pinned = resolve_dataloader_settings(
                {
                    "resources": {
                        "workers": {"dataloader": 3},
                        "dataloader": {"prefetch_factor": 2, "batch_size": 64},
                    }
                }
            )
            auto = resolve_dataloader_settings({"resources": {"workers": {"dataloader": "auto"}}})
        finally:
            autotune.cache_clear()

        assert (pinned.num_workers, pinned.prefetch_factor, pinned.batch_size) == (3, 2, 64)
        assert (auto.num_workers, auto.prefetch_factor, auto.batch_size) == (12, 8, 256)


class TestExecutionMetadata:
    def _identity(self, resources: dict) -> str:
        config = {
            "seed": 3,
            "common": {"epochs": 5, "batch_size": 8},
            "resources": resources,
        }
        payload = resolve_training_identity(
            model_cls=BPR,
            model_name="bpr",
            dataset_name="d",
            embedding_name="none",
            hyperparams={"latent_dim": 4},
            config=config,
        )
        assert "resources" not in json.dumps(payload)
        return canonical_digest(payload)

    def test_identity_digest_is_invariant_to_the_resources_block(self) -> None:
        base = self._identity({})
        changed = self._identity(
            {
                "gpu": {"vram_share": 0.9, "ranking_vram_share": 0.5},
                "host": {"budget_bytes": GB, "headroom_bytes": 0, "reserved_bytes": 0},
                "workers": {"training": 4, "fusion": 8, "dataloader": 2},
                "dataloader": {"prefetch_factor": 1, "batch_size": 1},
                "features": {"residency": "lazy", "item_block": 1},
            }
        )

        assert base == changed

    def test_manifest_records_the_resolved_block(self, tmp_path, monkeypatch) -> None:
        from src.utils.manifest import start_run

        monkeypatch.setenv(VRAM_SHARE_ENV, "0.75")
        snapshot = {
            "seed": 1,
            "device": "cpu",
            "resources": {"workers": {"training": 2}, "features": {"item_block": 4096}},
        }

        run_dir = start_run(snapshot, results_root=tmp_path / "runs")

        manifest = json.loads((run_dir / "manifest.json").read_text())
        recorded = manifest["resources"]
        assert recorded["gpu"]["vram_share"] == 0.75
        assert recorded["workers"] == {"training": 2, "fusion": 1, "dataloader": "auto"}
        assert recorded["features"] == {"residency": "dense", "item_block": 4096}
        assert manifest["dataloader_autotune"]["yaml_overrides"] == {}

    def test_budget_above_the_cgroup_limit_logs_a_warning(self, monkeypatch) -> None:
        warnings: list[str] = []
        monkeypatch.setattr(memory_mod.logger, "warning", lambda msg, *a: warnings.append(msg % a))
        monkeypatch.setattr(
            memory_mod, "_read_int_file", lambda p: 8 * GB if "memory.max" in str(p) else None
        )
        over = resolve_resources({"resources": {"host": {"budget_bytes": 16 * GB}}}, env={})
        under = resolve_resources({"resources": {"host": {"budget_bytes": 4 * GB}}}, env={})

        assert memory_mod.warn_if_budget_exceeds_cgroup(over) == 8 * GB
        assert memory_mod.warn_if_budget_exceeds_cgroup(under) == 8 * GB
        assert memory_mod.warn_if_budget_exceeds_cgroup(ResourcesConfig()) == 8 * GB

        assert len(warnings) == 1
        assert "resources.host.budget_bytes" in warnings[0]

    def test_no_cgroup_limit_means_no_warning(self, monkeypatch) -> None:
        warnings: list[str] = []
        monkeypatch.setattr(memory_mod.logger, "warning", lambda msg, *a: warnings.append(msg))
        monkeypatch.setattr(memory_mod, "_read_int_file", lambda p: None)
        over = resolve_resources({"resources": {"host": {"budget_bytes": 10**15}}}, env={})

        assert memory_mod.warn_if_budget_exceeds_cgroup(over) is None
        assert warnings == []
