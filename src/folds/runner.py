"""User-level K-fold cross-validation runner (the ``evaluate`` step under ``folds.enabled``).

Protocol (Rendle et al., UAI 2009, §6.2 — repeated splits with the
hyperparameter search done once and kept constant):

1. Users of every dataset are partitioned into ``folds.k`` mutually
   exclusive folds (:mod:`src.folds.partition`).
2. For every battery cell and every fold ``i``:
   * the fold's users leave the training set; the model is trained on
     the other ``k - 1`` folds with the cell's FROZEN hyperparameters
     (:mod:`src.recommenders.hp_source`: the prior search's winner or
     the fixed config values) under seed ``folds.seed + i``;
   * the held-out users are folded in from their profile with every
     non-user parameter frozen (:mod:`src.folds.foldin`);
   * they are evaluated on their single target item, producing a partial
     per-user artifact (:mod:`src.folds.aggregate`).
3. The ``k`` partial artifacts are concatenated into the cell's canonical
   per-user artifact, so the paired statistics keep the user as the
   unit; between-fold mean/std are recorded as descriptive variability.

Folds and seeds are distinct variance sources: each fold runs under its
own seed, so the reported variability is COMBINED (partition +
optimisation) and the manifest says so.
"""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path
from typing import Any

import torch

from src.battery.cells import BatteryCell, enumerate_cells
from src.battery.manifest import BatteryManifest
from src.evaluation.persistence import (
    ArtifactIntegrityError,
    CellMetadata,
    artifact_paths,
    validate_cell_artifact,
)
from src.folds.aggregate import concatenate_fold_artifacts, write_fold_artifact
from src.folds.foldin import FoldInConfig, fold_in_users
from src.folds.partition import FoldPlan, FoldSplit, build_fold_plan, fold_split
from src.folds.splits_io import load_split_frames
from src.recommenders.hp_search import assert_dimension_parity
from src.recommenders.hp_source import HyperparamOrigin, resolve_cell_hyperparams
from src.utils.identity import (
    EVALUATION_SPLITS,
    IdentityError,
    canonical_digest,
    resolve_data_identity,
)
from src.utils.logging import get_logger
from src.utils.resources import resolve_resources
from src.utils.timing import note_skipped_cell, time_cell

logger = get_logger(__name__)

#: Manifest note attached to every cell (1.5).
VARIABILITY_NOTE = (
    "between-fold variability is combined: user partition + optimisation seed "
    "(fold i runs under folds.seed + i)"
)


def manifest_path(results_dir: str | Path) -> Path:
    return Path(results_dir) / "folds" / "manifest.json"


def _fold_paths(config: dict, fold_index: int) -> dict:
    """Seed-isolated results/checkpoint roots for one fold."""
    paths = config["paths"]
    return {
        **paths,
        "results": f"{paths['results']}_fold{fold_index}",
        "checkpoints": f"{paths.get('checkpoints', 'checkpoints')}_fold{fold_index}",
    }


def _fold_config(config: dict, fold_index: int) -> dict:
    cfg = copy.deepcopy(config)
    cfg["seed"] = int(config["folds"]["seed"]) + fold_index
    cfg["paths"] = _fold_paths(config, fold_index)
    return cfg


def fold_plan_digest(plan: FoldPlan, dataset: str, processed_dir: str) -> str:
    """Identity of a fold plan: k, partition seed, profile rule, the exact
    user assignment and the split / item-mapping content it was built from (E07).

    Recorded as the cell artifact's ``config_hash``; a concatenated fold
    artifact is complete for the current configuration only when its
    recorded digest equals this one.
    """
    try:
        data = resolve_data_identity(processed_dir, dataset, None, splits=EVALUATION_SPLITS)
        split, mapping = data.split_digest, data.item_mapping_digest
    except IdentityError as exc:
        logger.warning("%s: split identity unresolved for the fold plan (%s).", dataset, exc)
        split, mapping = None, None
    return canonical_digest(
        {
            "schema_version": 1,
            "k": int(plan.k),
            "seed": int(plan.seed),
            "min_profile": int(plan.min_profile),
            "assignment": sorted((int(u), int(f)) for u, f in plan.assignment.items()),
            "split_digest": split,
            "item_mapping_digest": mapping,
        }
    )


def _dataset_plan(config: dict, dataset: str, processed_dir: str) -> tuple[FoldPlan, tuple]:
    train, val, test, n_users, n_items = load_split_frames(processed_dir, dataset)
    folds_cfg = config["folds"]
    plan = build_fold_plan(
        train,
        val,
        test,
        n_users=n_users,
        k=int(folds_cfg["k"]),
        seed=int(folds_cfg["seed"]),
        min_profile=int(folds_cfg["min_profile"]),
    )
    return plan, (train, val, test, n_users, n_items)


def _embedding_path(embeddings_dir: str, dataset: str, visual_config: str) -> str | None:
    from src.battery.execute import _embedding_path as resolve

    return resolve(embeddings_dir, dataset, visual_config)


def _load_visual(emb_path: str | None, *, lazy: bool = False):
    if emb_path is None:
        return None
    from src.fusions import load_embedding

    return load_embedding(emb_path, lazy=lazy)


def _ctor_kwargs(model_cls: type, split: FoldSplit, dataset: str, processed_dir: str) -> dict:
    kwargs: dict = {}
    if getattr(model_cls, "wants_history", False):
        kwargs["train_interactions"] = split.train_interactions
    if getattr(model_cls, "wants_categories", False):
        from src.data.categories import item_category_array

        kwargs["item_categories"] = item_category_array(dataset, processed_dir)
    return kwargs


def _train_fold_model(
    cell: BatteryCell,
    split: FoldSplit,
    origin: HyperparamOrigin,
    cfg: dict,
    *,
    n_users: int,
    n_items: int,
    visual,
    device: str,
    fold_index: int,
    k: int,
) -> float:
    """Train the cell on the fold's training users with frozen hyperparameters."""
    from src.recommenders import get_recommender_class
    from src.steps.train import identity_context_for
    from src.utils.checkpoint import CheckpointManager
    from src.utils.training import train_single_run

    model_cls = get_recommender_class(cell.recommender)
    processed_dir = cfg["paths"]["data_processed"]
    kwargs = _ctor_kwargs(model_cls, split, cell.dataset, processed_dir)
    fold = {
        "index": int(fold_index),
        "k": int(k),
        "partition_seed": int(cfg["folds"]["seed"]),
        "min_profile": int(cfg["folds"]["min_profile"]),
    }
    return train_single_run(
        model_cls=model_cls,
        model_name=cell.recommender,
        n_users=n_users,
        n_items=n_items,
        visual_embeddings=visual,
        train_interactions=split.train_interactions,
        selection_interactions=split.selection_interactions,
        hyperparams=origin.hyperparams,
        config=cfg,
        checkpoint_mgr=CheckpointManager(cfg["paths"]["checkpoints"]),
        dataset_name=cell.dataset,
        embedding_name=cell.visual_config,
        device=device,
        item_categories=kwargs.get("item_categories"),
        log_context=f"fold={fold_index + 1}/{k}",
        identity_context=identity_context_for(
            processed_dir,
            cell.dataset,
            cell.visual_config,
            _embedding_path(cfg["paths"]["embeddings"], cell.dataset, cell.visual_config),
            fold=fold,
        ),
    )


def _load_best_model(
    cell: BatteryCell,
    split: FoldSplit,
    cfg: dict,
    *,
    n_users: int,
    n_items: int,
    visual,
    device: str,
):
    """Rebuild the fold's best checkpoint as a live model."""
    from src.recommenders import get_recommender_class
    from src.utils.checkpoint import load_best_checkpoint

    model_cls = get_recommender_class(cell.recommender)
    path = (
        Path(cfg["paths"]["results"])
        / "models"
        / cell.dataset
        / f"{cell.recommender}_{cell.visual_config}_best.pt"
    )
    # Validating reader (I02): an absent / truncated / legacy winner raises
    # BestCheckpointError instead of surfacing from deep inside torch.
    saved = load_best_checkpoint(path, map_location=device)
    model_config = {**saved["hyperparams"], "history_seed": int(cfg["seed"])}
    model_config.setdefault("l2_reg", 0.0001)
    kwargs = _ctor_kwargs(model_cls, split, cell.dataset, cfg["paths"]["data_processed"])
    model = model_cls(
        n_users=n_users,
        n_items=n_items,
        visual_embeddings=visual,
        config=model_config,
        **kwargs,
    ).to(device)
    model.configure_item_block(resolve_resources(cfg).features.item_block)
    model.load_state_dict(saved["model_state"])
    return model, saved["hyperparams"]


def _fold_in_config(config: dict, hyperparams: dict, seed: int) -> FoldInConfig:
    fi = config["folds"]["fold_in"]
    common = config.get("common", {})
    return FoldInConfig(
        epochs=int(fi["epochs"]),
        learning_rate=float(fi.get("learning_rate") or hyperparams["learning_rate"]),
        batch_size=int(fi.get("batch_size") or common.get("batch_size", 4096)),
        seed=seed,
    )


def _evaluate_fold(
    model,
    split: FoldSplit,
    cfg: dict,
    metadata: CellMetadata,
    fold_index: int,
    out_dir: Path,
    device,
    *,
    k: int,
) -> Path:
    """Rank every held-out user's single target against the catalogue."""
    from src.steps.evaluate import build_evaluator

    evaluator = build_evaluator(
        cfg, split.profile_interactions, split.target_interactions, metadata.n_items
    )
    if evaluator.protocol != "full_ranking":
        raise RuntimeError("K-fold evaluation requires evaluation.protocol = full_ranking.")
    model.eval()
    _, records = evaluator.evaluate_with_records(model, device=device)
    return write_fold_artifact(
        records, metadata, fold_index, out_dir, k=k, fold_seed=int(cfg["seed"])
    )


def run_cell_folds(
    cell: BatteryCell,
    config: dict,
    plan: FoldPlan,
    frames: tuple,
    *,
    results_dir: str | Path,
    device: str,
) -> dict:
    """Run every fold of one cell and concatenate the artifacts."""
    train, val, test, n_users, n_items = frames
    origin = resolve_cell_hyperparams(
        config,
        dataset=cell.dataset,
        model_name=cell.recommender,
        embedding_name=cell.visual_config,
        results_root=results_dir,
    )
    from src.steps.train import lazy_features_for

    emb_path = _embedding_path(config["paths"]["embeddings"], cell.dataset, cell.visual_config)
    visual = _load_visual(emb_path, lazy=lazy_features_for(config, emb_path))
    artifact_root = Path(results_dir)  # artifact_paths appends per_user/<dataset>
    metadata = CellMetadata(
        dataset=cell.dataset,
        visual_config=cell.visual_config,
        recommender=cell.recommender,
        seed=int(config["folds"]["seed"]),
        d=int(origin.hyperparams.get("latent_dim", 0)),
        split="test",
        n_users=n_users,
        n_items=n_items,
        config_hash=fold_plan_digest(plan, cell.dataset, config["paths"]["data_processed"]),
    )

    fold_entries: list[dict] = []
    for fold_index in range(plan.k):
        cfg = _fold_config(config, fold_index)
        split = fold_split(plan, fold_index, train, val, test, dataset_name=cell.dataset)
        started = time.perf_counter()
        logger.info(
            "Fold %d/%d of %s: training on %d users, folding in %d held-out users",
            fold_index + 1,
            plan.k,
            cell.key(),
            len(split.train_interactions),
            len(split.test_users),
        )
        best_val = _train_fold_model(
            cell,
            split,
            origin,
            cfg,
            n_users=n_users,
            n_items=n_items,
            visual=visual,
            device=device,
            fold_index=fold_index,
            k=plan.k,
        )
        model, hyperparams = _load_best_model(
            cell, split, cfg, n_users=n_users, n_items=n_items, visual=visual, device=device
        )
        report = fold_in_users(
            model,
            split.profile_interactions,
            _fold_in_config(config, hyperparams, cfg["seed"]),
            n_items=n_items,
            device=device,
        )
        _evaluate_fold(model, split, cfg, metadata, fold_index, artifact_root, device, k=plan.k)
        fold_entries.append(
            {
                "fold": fold_index,
                "seed": cfg["seed"],
                "n_test_users": len(split.test_users),
                "best_val_metric": float(best_val),
                "fold_in": report.__dict__,
                "duration_seconds": round(time.perf_counter() - started, 3),
            }
        )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    _, aggregate = concatenate_fold_artifacts(
        artifact_root, metadata, plan.k, k_values=list(config.get("k_values", [5, 10, 20]))
    )
    return {
        "hyperparam_origin": origin.to_dict(),
        "partition": plan.summary(),
        "folds": fold_entries,
        "aggregate": aggregate.to_dict(),
        "variability": VARIABILITY_NOTE,
    }


def _cell_done(
    cell: BatteryCell,
    metadata_seed: int,
    results_dir: Path,
    *,
    k: int | None = None,
    plan_digest: str | None = None,
) -> bool:
    """Whether the cell's concatenated fold artifact is complete for THIS plan (E07).

    Requires a validated generation (payload matches its completion
    pointer), K-fold provenance of the concatenated shape, and — when
    given — the requested ``k``, the fold seeds ``seed + i`` and the fold
    plan digest (partition seed, profile rule, assignment, split content).
    Existence alone, a legacy artifact or a leave-one-out artifact at the
    same path never count.
    """
    meta = CellMetadata(
        dataset=cell.dataset,
        visual_config=cell.visual_config,
        recommender=cell.recommender,
        seed=metadata_seed,
        d=0,
        split="test",
        n_users=0,
        n_items=0,
    )
    records_path, meta_path = artifact_paths(results_dir, meta)
    if not (records_path.exists() and meta_path.exists()):
        return False
    try:
        if validate_cell_artifact(records_path) is None:
            return False
        written = json.loads(meta_path.read_text(encoding="utf-8"))
    except (ArtifactIntegrityError, OSError, ValueError) as exc:
        logger.warning("%s: fold artifact not accepted (%s).", cell.key(), exc)
        return False
    reason = _fold_provenance_mismatch(written, metadata_seed, k=k, plan_digest=plan_digest)
    if reason:
        logger.warning("%s: fold artifact present but %s; re-running.", cell.key(), reason)
        return False
    return True


def _fold_provenance_mismatch(
    written: dict, metadata_seed: int, *, k: int | None, plan_digest: str | None
) -> str | None:
    provenance = written.get("fold")
    if not isinstance(provenance, dict) or "k" not in provenance or "index" in provenance:
        return "it carries no concatenated K-fold provenance"
    if k is not None and int(provenance["k"]) != int(k):
        return f"it was built with k={provenance['k']}, not k={k}"
    if k is not None and [int(s) for s in provenance.get("seeds", [])] != [
        metadata_seed + i for i in range(int(k))
    ]:
        return f"its fold seeds {provenance.get('seeds')} differ from seed {metadata_seed} + i"
    if plan_digest is not None and written.get("config_hash") != plan_digest:
        return "its fold plan (partition seed / profile rule / splits) differs"
    return None


def run_folds(
    config: dict,
    results_dir: str | Path,
    *,
    processed_dir: str | None = None,
    embeddings_dir: str | None = None,
    execute: Any = None,
) -> BatteryManifest:
    """Run the K-fold protocol over every battery cell (resumable).

    ``execute`` (tests) replaces :func:`run_cell_folds` with the same
    signature ``(cell, config, plan, frames, results_dir=..., device=...)``.
    """
    folds_cfg = config.get("folds") or {}
    if not folds_cfg.get("enabled", False):
        raise RuntimeError(
            "K-fold run requested but configs/default.yaml -> folds.enabled is false; "
            "the evaluate step runs the single split while it is false."
        )
    from src.recommenders.hp_budget import assert_uniform_budget
    from src.utils.device import resolve_device

    assert_uniform_budget(config)
    assert_dimension_parity(config)

    processed_dir = processed_dir or config["paths"]["data_processed"]
    embeddings_dir = embeddings_dir or config["paths"]["embeddings"]
    results_dir = Path(results_dir)
    device = resolve_device(config["device"])
    runner = execute or run_cell_folds

    cells = [
        c
        for c in enumerate_cells(config, processed_dir=processed_dir, embeddings_dir=embeddings_dir)
        if c.role == "search"
    ]
    manifest = BatteryManifest.load(manifest_path(results_dir))
    manifest.sync_cells(cells)
    manifest.save()

    plans: dict[str, tuple[FoldPlan, tuple]] = {}
    digests: dict[str, str] = {}
    for cell in cells:
        if cell.dataset not in plans:
            plans[cell.dataset] = _dataset_plan(config, cell.dataset, processed_dir)
            digests[cell.dataset] = fold_plan_digest(
                plans[cell.dataset][0], cell.dataset, processed_dir
            )
            logger.info("Fold plan %s: %s", cell.dataset, plans[cell.dataset][0].summary())
        key = cell.key()
        # A ``done`` manifest state is not enough: the artifact itself must
        # validate for the CURRENT k / partition seed / profile rule / splits.
        if _cell_done(
            cell,
            int(folds_cfg["seed"]),
            results_dir,
            k=int(folds_cfg["k"]),
            plan_digest=digests[cell.dataset],
        ):
            manifest.set_state(key, "done", note="fold artifact already present")
            manifest.save()
            note_skipped_cell()
            continue
        if manifest.state_of(key) == "done":
            logger.warning("%s: manifest says done but no valid fold artifact; re-running.", key)
        manifest.set_state(key, "running")
        manifest.save()
        started = time.perf_counter()
        plan, frames = plans[cell.dataset]
        model_key = f"{cell.recommender}_{cell.visual_config}"
        try:
            with time_cell("evaluate", dataset=cell.dataset, model_key=model_key, fold_k=plan.k):
                extra = runner(cell, config, plan, frames, results_dir=results_dir, device=device)
            manifest.set_state(
                key,
                "done",
                duration_seconds=round(time.perf_counter() - started, 3),
                error=None,
                **(extra or {}),
            )
        except Exception as exc:  # noqa: BLE001 — isolate per-cell failures
            logger.error("Fold cell failed: %s (%s)", key, exc)
            manifest.set_state(
                key,
                "failed",
                duration_seconds=round(time.perf_counter() - started, 3),
                error=str(exc),
            )
        manifest.save()
    summary = manifest.summary()
    if any(v for k, v in summary.items() if k != "done"):
        logger.error("K-fold run INCOMPLETE: %s", summary)
    else:
        logger.info("K-fold run finished: %s", summary)
    return manifest
