"""Step 05, Recommender hyperparameter search.

Two strategies are supported, selected by
``configs/recommenders.yaml -> hp_search.strategy``:

* ``grid`` (default), Cartesian product over the lists declared
  per recommender, dispatched in parallel via
  :class:`TrainingOrchestrator`.
* ``optuna``, Bayesian search via :mod:`optuna`, sequential within
  each ``(dataset, model, embedding)`` cell with median-pruner
  stopping bad trials early.  Cells are processed one after another
  (no inter-cell parallelism in the MVP).

Both backends share the same per-trial entry point, so the actual
training loop in :mod:`src.utils.training` is unchanged.
"""

from __future__ import annotations

import json
from itertools import product
from pathlib import Path

from src.recommenders import (
    get_recommender_spec,
    is_registered,
    iter_specs,
    registered_recommender_names,
)
from src.recommenders.hp_search import (
    CellKey,
    create_study,
    get_strategy,
    sample_hyperparams,
)
from src.utils.checkpoint import CheckpointManager
from src.utils.config import load_config
from src.utils.device import resolve_device
from src.utils.logging import get_logger
from src.utils.parallel import TrainingJob, TrainingOrchestrator
from src.utils.seed import set_seed

logger = get_logger(__name__)


def get_embedding_files(
    embeddings_dir: str,
    dataset_name: str,
    dim_filter: list[str] | None = None,
) -> list[str]:
    """List embedding stems for a dataset, optionally filtered by dim.

    Includes both ``.npy`` files (offline embeddings + offline fusions)
    and ``.json`` sidecars (online fusions like ``adaptive_gated``).
    The stem is what the train step uses to identify the embedding;
    ``load_embedding`` resolves the actual on-disk path at load time.
    """
    emb_dir = Path(embeddings_dir) / dataset_name
    if not emb_dir.exists():
        return []
    names = [f.stem for f in sorted(emb_dir.glob("*.npy"))]
    names.extend(f.stem for f in sorted(emb_dir.glob("hybrid_*.json")))
    names = sorted(set(names))
    if dim_filter:
        # Component artifacts end in "_comp" after the dim token
        # (``<extractor>_D<dim>_comp``); accept that suffixed form too so
        # the dim filter does not silently drop them.
        names = [
            n for n in names if any(n.endswith(d) or n.endswith(f"{d}_comp") for d in dim_filter)
        ]
    return names


def is_component_artifact(stem: str) -> bool:
    """Whether an embedding stem is a 3-D per-item component artifact.

    Component artifacts (``<extractor>_D<dim>_comp``) feed models that
    declare ``requires_components`` (e.g. ACF).  They are routed only to
    those models and excluded from the pooled-embedding pool consumed by
    every other recommender, so existing models' job lists are unchanged.
    """
    return stem.endswith("_comp")


def _resolve_embedding_path(embeddings_dir: str, dataset_name: str, stem: str) -> str | None:
    """Map a stem to either ``<stem>.npy`` or ``<stem>.json`` on disk."""
    base = Path(embeddings_dir) / dataset_name
    npy = base / f"{stem}.npy"
    if npy.exists():
        return str(npy)
    sidecar = base / f"{stem}.json"
    if sidecar.exists():
        return str(sidecar)
    return None


def get_hyperparam_grid(model_name: str, config: dict) -> list[dict]:
    """Cartesian product of grid-search hyperparameters for a recommender.

    Reads the model's :class:`RecommenderSpec` to decide which keys to
    include, there are no model-specific branches here.  Plugin authors
    declare their hyperparameters via ``extra_hyperparam_keys`` /
    ``uses_visual_dim`` when registering.
    """
    spec = get_recommender_spec(model_name)
    common = config.get("common", {})
    model_specific = config.get(model_name, {})

    params: dict = {
        "latent_dim": common.get("latent_dim", [64]),
        "learning_rate": common.get("learning_rate", [0.001]),
        "l2_reg": common.get("l2_reg", [0.0001]),
    }
    if spec.uses_visual_dim:
        params["visual_dim"] = common.get("visual_dim", [64])
    for key in spec.extra_hyperparam_keys:
        if key in model_specific:
            params[key] = model_specific[key]

    keys = list(params.keys())
    values = [params[k] if isinstance(params[k], list) else [params[k]] for k in keys]
    return [dict(zip(keys, combo, strict=False)) for combo in product(*values)]


def build_job_list(
    condition: str,
    config: dict,
    processed_dir: str,
    embeddings_dir: str,
    device: str,
) -> list[TrainingJob]:
    """Return the list of pending training jobs for the given condition."""
    datasets = config.get("datasets", [])
    checkpoint_mgr = CheckpointManager()
    jobs: list[TrainingJob] = []

    dim_filter = config.get("embedding_dims", [])

    enabled = config.get("recommenders_enabled")
    if enabled is None or not enabled:
        logger.warning(
            "recommenders_enabled is missing or empty in configs/recommenders.yaml, "
            "no training jobs will be scheduled. Add e.g. recommenders_enabled: "
            "[bpr, vbpr] to enable them. Registered recommenders: %s",
            ", ".join(registered_recommender_names()),
        )
        model_names: list[str] = []
    else:
        unknown = [m for m in enabled if not is_registered(m)]
        if unknown:
            logger.warning(
                "recommenders_enabled lists unregistered models (skipped): %s. "
                "Registered recommenders: %s",
                ", ".join(sorted(unknown)),
                ", ".join(registered_recommender_names()),
            )
        # Iterate in (priority, name) order so cheaper models train first.
        enabled_set = set(enabled)
        model_names = [s.name for s in iter_specs() if s.name in enabled_set]

    for dataset_name in datasets:
        all_embs = get_embedding_files(embeddings_dir, dataset_name, dim_filter or None)
        if condition == "frozen":
            embedding_names = [e for e in all_embs if "_finetuned" not in e]
        else:
            embedding_names = [e for e in all_embs if "_finetuned" in e]

        with open(Path(processed_dir) / dataset_name / "user2idx.json") as f:
            n_users = len(json.load(f))
        with open(Path(processed_dir) / dataset_name / "item2idx.json") as f:
            n_items = len(json.load(f))

        for model_name in model_names:
            spec = get_recommender_spec(model_name)
            if not spec.requires_visual:
                # Models that ignore visual features (e.g. plain BPR) only
                # run in the frozen condition with embedding_name="none".
                sources = ["none"] if condition == "frozen" else []
            else:
                sources = [
                    e
                    for e in embedding_names
                    if is_component_artifact(e) == spec.requires_components
                ]

            for emb_name in sources:
                experiment_key = f"{dataset_name}_{emb_name}_{model_name}"
                completed = checkpoint_mgr.load_grid_search_progress(experiment_key)
                completed_hashes = {json.dumps(c["hyperparams"], sort_keys=True) for c in completed}

                grid = get_hyperparam_grid(model_name, config)

                if emb_name == "none":
                    emb_path = None
                else:
                    emb_path = _resolve_embedding_path(
                        embeddings_dir,
                        dataset_name,
                        emb_name,
                    )
                    if emb_path is None:
                        continue

                for hp in grid:
                    if json.dumps(hp, sort_keys=True) in completed_hashes:
                        continue

                    jobs.append(
                        TrainingJob(
                            dataset_name=dataset_name,
                            model_name=model_name,
                            embedding_name=emb_name,
                            hyperparams=hp,
                            n_users=n_users,
                            n_items=n_items,
                            embeddings_path=emb_path,
                            processed_dir=processed_dir,
                            device=device,
                            priority=spec.priority,
                        )
                    )

    return jobs


def run(condition: str = "frozen", workers: int = 0, sequential: bool = False) -> None:
    """Dispatch the hyperparameter search for the given condition.

    Parameters
    ----------
    condition:
        ``"frozen"`` or ``"finetuned"``, selects which embedding files
        are eligible for the search.
    workers:
        Number of parallel workers (``0`` = auto-detect via VRAM).
        Honoured by the ``grid`` strategy only; ``optuna`` runs each
        cell sequentially.
    sequential:
        Force a single worker regardless of ``workers``.
    """
    if condition not in {"frozen", "finetuned"}:
        raise ValueError(f"condition must be 'frozen' or 'finetuned', got {condition!r}")

    config = load_config()
    set_seed(config["seed"])

    if not config.get("datasets"):
        logger.info("train step skipped: datasets list is empty in configs/default.yaml.")
        return
    if not config.get("recommenders_enabled"):
        logger.info(
            "train step skipped: recommenders_enabled is empty in configs/recommenders.yaml.",
        )
        return

    logger.info("Condition: %s", condition)

    startup_mgr = CheckpointManager()
    removed = startup_mgr.clear_all_training_checkpoints()
    if removed > 0:
        logger.info("Cleared %d stale training checkpoint(s) at startup", removed)

    strategy = get_strategy(config)
    logger.info("Hyperparameter-search strategy: %s", strategy)

    if strategy == "optuna":
        _run_optuna(condition, config)
    else:
        _run_grid(condition, config, workers=workers, sequential=sequential)


def _run_grid(
    condition: str,
    config: dict,
    *,
    workers: int,
    sequential: bool,
) -> None:
    """Original Cartesian grid behaviour, dispatched via the orchestrator."""
    device = resolve_device(config["device"])
    processed_dir = config["paths"]["data_processed"]
    embeddings_dir = config["paths"]["embeddings"]

    jobs = build_job_list(condition, config, processed_dir, embeddings_dir, device)

    if not jobs:
        logger.info("No pending jobs. All experiments already completed.")
        return

    logger.info("Total pending jobs: %d", len(jobs))

    n_workers = 1 if sequential else workers
    orchestrator = TrainingOrchestrator(
        n_workers=n_workers,
        device=device,
        log_dir="logs",
    )

    results = orchestrator.run(jobs)

    ok = sum(1 for r in results if r.get("status") == "ok")
    logger.info("Training complete: %d/%d experiments succeeded.", ok, len(jobs))


def _legit_trial_count(study) -> int:
    """Number of legitimate HPO outcomes (COMPLETE + PRUNED) in *study*.

    ``len(study.trials)`` also counts FAIL trials (infra crashes such as
    a corrupt-embedding load) and stale RUNNING trials (process killed
    mid-trial). Counting those toward ``n_trials`` truncated or skipped
    the search for affected cells. Only COMPLETE and PRUNED are real
    search outcomes that may consume the trial budget.
    """
    return sum(1 for t in study.trials if t.state.name in ("COMPLETE", "PRUNED"))


def _run_optuna(condition: str, config: dict) -> None:
    """Per-cell Optuna search with median pruning.

    For each ``(dataset, model, embedding)`` cell we create (or load)
    an Optuna study and run ``hp_search.optuna.n_trials`` trials.  The
    training loop reports the validation metric after every
    ``eval_every_epochs`` epochs; pruned trials raise and are skipped
    without persisting a checkpoint.
    """
    import optuna

    device = resolve_device(config["device"])
    processed_dir = config["paths"]["data_processed"]
    embeddings_dir = config["paths"]["embeddings"]
    optuna_cfg = config["hp_search"]["optuna"]
    n_trials = int(optuna_cfg["n_trials"])
    timeout = optuna_cfg.get("timeout_seconds")

    cells = _list_cells(condition, config, processed_dir, embeddings_dir)
    logger.info("Optuna cells to process: %d (n_trials=%d)", len(cells), n_trials)

    for cell, n_users, n_items, emb_path in cells:
        logger.info("=== Optuna cell: %s ===", cell.study_name())
        study = create_study(cell, config)

        def _objective(trial, _cell=cell, _users=n_users, _items=n_items, _emb=emb_path):
            hp = sample_hyperparams(trial, _cell.model_name, config)
            return _train_one_optuna_trial(
                cell=_cell,
                hyperparams=hp,
                n_users=_users,
                n_items=_items,
                embeddings_path=_emb,
                processed_dir=processed_dir,
                device=device,
                config=config,
                trial=trial,
            )

        try:
            existing = _legit_trial_count(study)
            remaining = max(0, n_trials - existing)
            if remaining == 0:
                logger.info(
                    "  cell %s: already has %d legit trials >= n_trials=%d, skipping",
                    cell.study_name(),
                    existing,
                    n_trials,
                )
            else:
                study.optimize(
                    _objective,
                    n_trials=remaining,
                    timeout=timeout,
                    gc_after_trial=True,
                    show_progress_bar=False,
                )
        except KeyboardInterrupt:
            logger.warning("Optuna study interrupted by user.")
            raise

        completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        pruned = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
        logger.info(
            "  cell %s: %d completed, %d pruned. best_value=%.4f best_params=%s",
            cell.study_name(),
            len(completed),
            len(pruned),
            study.best_value if completed else 0.0,
            study.best_params if completed else {},
        )


def _list_cells(
    condition: str,
    config: dict,
    processed_dir: str,
    embeddings_dir: str,
) -> list[tuple[CellKey, int, int, str | None]]:
    """Enumerate every ``(dataset, model, embedding)`` cell to optimise.

    Mirrors the iteration in :func:`build_job_list` but stops at the
    cell granularity (no per-HP enumeration).
    """
    out: list[tuple[CellKey, int, int, str | None]] = []
    enabled = set(config.get("recommenders_enabled") or [])
    model_names = [s.name for s in iter_specs() if s.name in enabled]
    dim_filter = config.get("embedding_dims", [])

    for dataset_name in config.get("datasets", []):
        all_embs = get_embedding_files(embeddings_dir, dataset_name, dim_filter or None)
        if condition == "frozen":
            embedding_names = [e for e in all_embs if "_finetuned" not in e]
        else:
            embedding_names = [e for e in all_embs if "_finetuned" in e]

        with open(Path(processed_dir) / dataset_name / "user2idx.json") as f:
            n_users = len(json.load(f))
        with open(Path(processed_dir) / dataset_name / "item2idx.json") as f:
            n_items = len(json.load(f))

        for model_name in model_names:
            spec = get_recommender_spec(model_name)
            if not spec.requires_visual:
                sources = ["none"] if condition == "frozen" else []
            else:
                sources = [
                    e
                    for e in embedding_names
                    if is_component_artifact(e) == spec.requires_components
                ]
            for emb_name in sources:
                if emb_name == "none":
                    emb_path = None
                else:
                    emb_path = _resolve_embedding_path(
                        embeddings_dir,
                        dataset_name,
                        emb_name,
                    )
                    if emb_path is None:
                        continue
                out.append(
                    (
                        CellKey(dataset_name, model_name, emb_name),
                        n_users,
                        n_items,
                        emb_path,
                    )
                )
    return out


def _train_one_optuna_trial(
    *,
    cell: CellKey,
    hyperparams: dict,
    n_users: int,
    n_items: int,
    embeddings_path: str | None,
    processed_dir: str,
    device: str,
    config: dict,
    trial,
) -> float:
    """Single trial entry point: load data, train one model, return metric."""
    from src.fusions import load_embedding
    from src.recommenders import get_recommender_class
    from src.utils.training import train_single_run

    dataset_name = cell.dataset_name
    train_path = Path(processed_dir) / dataset_name / "train.csv"
    test_path = Path(processed_dir) / dataset_name / "test.csv"

    import pandas as pd

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    train_interactions: dict = {}
    for u, i in zip(train_df["user_idx"], train_df["item_idx"], strict=False):
        train_interactions.setdefault(int(u), set()).add(int(i))
    test_interactions: dict = {}
    for u, i in zip(test_df["user_idx"], test_df["item_idx"], strict=False):
        test_interactions.setdefault(int(u), set()).add(int(i))

    visual_embeddings = None
    if embeddings_path is not None:
        visual_embeddings = load_embedding(embeddings_path)

    model_cls = get_recommender_class(cell.model_name)
    checkpoint_mgr = CheckpointManager()

    return train_single_run(
        model_cls=model_cls,
        model_name=cell.model_name,
        n_users=n_users,
        n_items=n_items,
        visual_embeddings=visual_embeddings,
        train_interactions=train_interactions,
        test_interactions=test_interactions,
        hyperparams=hyperparams,
        config=config,
        checkpoint_mgr=checkpoint_mgr,
        dataset_name=cell.dataset_name,
        embedding_name=cell.embedding_name,
        device=device,
        optuna_trial=trial,
    )
