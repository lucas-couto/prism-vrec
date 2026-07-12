"""Step 05, Recommender hyperparameter search.

Two strategies are supported, selected by
``configs/recommenders.yaml -> hp_search.strategy``:

* ``grid`` (default), Cartesian product over the lists declared
  per recommender, dispatched in parallel via
  :class:`TrainingOrchestrator`.
* ``optuna``, Bayesian search via :mod:`optuna`, sequential within
  each ``(dataset, model, embedding)`` cell with median-pruner
  stopping bad trials early.  Independent cells are dispatched to a
  small pool of worker processes (B7); trials inside a cell stay
  sequential so the TPE sampler always conditions on every previous
  trial of its own study.

Both backends share the same per-trial entry point, so the actual
training loop in :mod:`src.utils.training` is unchanged.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
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
from src.utils.artifact_names import (
    FUSION_PREFIX,
    is_component_artifact,
    is_finetuned_artifact,
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

    ``dim_filter`` only applies to fusion artifacts (``hybrid_*``), whose
    names carry an explicit alignment-dim token; single-extractor
    artifacts are native-dim and carry no dim token, so they always pass.
    """
    emb_dir = Path(embeddings_dir) / dataset_name
    if not emb_dir.exists():
        return []
    names = [f.stem for f in sorted(emb_dir.glob("*.npy"))]
    names.extend(f.stem for f in sorted(emb_dir.glob("hybrid_*.json")))
    names = sorted(set(names))
    if dim_filter:
        names = [
            n
            for n in names
            if not n.startswith(FUSION_PREFIX)
            or any(n.endswith(d) or n.endswith(f"{d}_comp") for d in dim_filter)
        ]
    return names


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


@dataclass(frozen=True)
class _Cell:
    """One ``(dataset, model, embedding)`` unit of work with its metadata."""

    dataset_name: str
    model_name: str
    spec: object
    embedding_name: str
    embedding_path: str | None
    n_users: int
    n_items: int


def _resolve_model_names(config: dict) -> list[str]:
    """Registered, enabled recommender names in (priority, name) order."""
    enabled = set(config.get("recommenders_enabled") or [])
    return [s.name for s in iter_specs() if s.name in enabled]


def _iter_cells(
    condition: str,
    config: dict,
    processed_dir: str,
    embeddings_dir: str,
    model_names: list[str],
) -> Iterator[_Cell]:
    """Yield every eligible training cell for *condition*.

    Single source of truth for cell eligibility (dataset enumeration,
    frozen/fine-tuned embedding filtering, per-model visual/component
    source routing, embedding-path resolution).  Both the grid backend
    (:func:`build_job_list`) and the Optuna backend (:func:`_list_cells`)
    consume this so their notions of "which cells exist" can never drift.
    """
    dim_filter = config.get("embedding_dims", [])

    for dataset_name in config.get("datasets", []):
        all_embs = get_embedding_files(embeddings_dir, dataset_name, dim_filter or None)
        if condition == "frozen":
            embedding_names = [e for e in all_embs if not is_finetuned_artifact(e)]
        else:
            embedding_names = [e for e in all_embs if is_finetuned_artifact(e)]

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
                if emb_name == "none":
                    emb_path = None
                else:
                    emb_path = _resolve_embedding_path(embeddings_dir, dataset_name, emb_name)
                    if emb_path is None:
                        continue
                yield _Cell(
                    dataset_name=dataset_name,
                    model_name=model_name,
                    spec=spec,
                    embedding_name=emb_name,
                    embedding_path=emb_path,
                    n_users=n_users,
                    n_items=n_items,
                )


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
    checkpoint_mgr = CheckpointManager()
    jobs: list[TrainingJob] = []

    enabled = config.get("recommenders_enabled")
    if enabled is None or not enabled:
        logger.warning(
            "recommenders_enabled is missing or empty in configs/recommenders.yaml, "
            "no training jobs will be scheduled. Add e.g. recommenders_enabled: "
            "[bpr, vbpr] to enable them. Registered recommenders: %s",
            ", ".join(registered_recommender_names()),
        )
        return jobs

    unknown = [m for m in enabled if not is_registered(m)]
    if unknown:
        logger.warning(
            "recommenders_enabled lists unregistered models (skipped): %s. "
            "Registered recommenders: %s",
            ", ".join(sorted(unknown)),
            ", ".join(registered_recommender_names()),
        )
    # Iterate in (priority, name) order so cheaper models train first.
    model_names = _resolve_model_names(config)

    for cell in _iter_cells(condition, config, processed_dir, embeddings_dir, model_names):
        experiment_key = f"{cell.dataset_name}_{cell.embedding_name}_{cell.model_name}"
        completed = checkpoint_mgr.load_grid_search_progress(experiment_key)
        completed_hashes = {json.dumps(c["hyperparams"], sort_keys=True) for c in completed}

        for hp in get_hyperparam_grid(cell.model_name, config):
            if json.dumps(hp, sort_keys=True) in completed_hashes:
                continue

            jobs.append(
                TrainingJob(
                    dataset_name=cell.dataset_name,
                    model_name=cell.model_name,
                    embedding_name=cell.embedding_name,
                    hyperparams=hp,
                    n_users=cell.n_users,
                    n_items=cell.n_items,
                    embeddings_path=cell.embedding_path,
                    processed_dir=processed_dir,
                    device=device,
                    priority=cell.spec.priority,
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
        Grid parallelises over jobs; ``optuna`` parallelises over
        cells (capped at 3 — see :func:`_resolve_optuna_workers`).
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
        _run_optuna(condition, config, workers=workers, sequential=sequential)
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


def _resolve_optuna_workers(workers: int, device: str, n_cells: int) -> int:
    """Worker count for inter-cell Optuna parallelism.

    Reuses the VRAM heuristic of the grid orchestrator but caps the pool
    at 3: an Optuna worker holds a full study (data + model + evaluator)
    for the whole cell, and 3 concurrent training processes is the
    empirically verified ceiling on the reference 24 GB pod.  Never more
    workers than cells.
    """
    from src.utils.parallel import detect_max_workers

    n = detect_max_workers(device) if workers <= 0 else workers
    return max(1, min(n, 3, n_cells))


def _optimize_one_cell(
    cell: CellKey,
    n_users: int,
    n_items: int,
    emb_path: str | None,
    *,
    config: dict,
    processed_dir: str,
    device: str,
    log=logger,
) -> dict:
    """Create/load the study for *cell* and run its remaining trials.

    Runs in the parent (sequential mode) or inside a worker process
    (parallel mode); the study is always created in the executing
    process so in-memory storage never crosses a process boundary.
    """
    import optuna

    optuna_cfg = config["hp_search"]["optuna"]
    n_trials = int(optuna_cfg["n_trials"])
    timeout = optuna_cfg.get("timeout_seconds")

    log.info("=== Optuna cell: %s ===", cell.study_name())
    study = create_study(cell, config)

    def _objective(trial):
        hp = sample_hyperparams(trial, cell.model_name, config)
        return _train_one_optuna_trial(
            cell=cell,
            hyperparams=hp,
            n_users=n_users,
            n_items=n_items,
            embeddings_path=emb_path,
            processed_dir=processed_dir,
            device=device,
            config=config,
            trial=trial,
        )

    existing = _legit_trial_count(study)
    remaining = max(0, n_trials - existing)
    if remaining == 0:
        log.info(
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

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    summary = {
        "cell": cell.study_name(),
        "status": "ok",
        "completed": len(completed),
        "pruned": len(pruned),
        "best_value": study.best_value if completed else 0.0,
        "best_params": study.best_params if completed else {},
    }
    log.info(
        "  cell %s: %d completed, %d pruned. best_value=%.4f best_params=%s",
        summary["cell"],
        summary["completed"],
        summary["pruned"],
        summary["best_value"],
        summary["best_params"],
    )
    return summary


def _optuna_cell_worker(
    worker_id: int,
    cell_queue,
    result_queue,
    n_workers: int,
    config: dict,
    processed_dir: str,
    device: str,
) -> None:
    """Worker process: pulls whole cells and runs their studies.

    Mirrors :func:`src.utils.parallel._worker_fn` (memory fraction,
    isolation of failures per unit of work) at cell granularity.
    """
    from queue import Empty as _Empty

    import torch as _torch

    from src.utils.logging import get_logger as _get_logger

    wlog = _get_logger(f"optuna_worker_{worker_id}")

    if _torch.cuda.is_available() and n_workers > 1:
        fraction = min(0.95, 1.0 / n_workers + 0.05)
        _torch.cuda.set_per_process_memory_fraction(fraction)

    while True:
        try:
            item = cell_queue.get(timeout=5)
        except _Empty:
            break
        if item is None:
            break

        cell, n_users, n_items, emb_path = item
        try:
            summary = _optimize_one_cell(
                cell,
                n_users,
                n_items,
                emb_path,
                config=config,
                processed_dir=processed_dir,
                device=device,
                log=wlog,
            )
            result_queue.put(summary)
        except Exception as exc:  # noqa: BLE001, isolate failures per cell
            wlog.error("  Error on cell %s: %s", cell.study_name(), exc, exc_info=True)
            result_queue.put(
                {"cell": cell.study_name(), "status": "error", "error": str(exc)},
            )


def _run_optuna(
    condition: str,
    config: dict,
    *,
    workers: int = 0,
    sequential: bool = False,
) -> None:
    """Per-cell Optuna search with median pruning (parallel across cells).

    For each ``(dataset, model, embedding)`` cell we create (or load)
    an Optuna study and run ``hp_search.optuna.n_trials`` trials.
    Cells are independent studies, so they are dispatched to worker
    processes (B7); trials WITHIN a cell remain sequential, keeping the
    TPE sampler conditioned on every prior trial of its study.  Cells
    whose studies already hold ``n_trials`` legitimate outcomes are
    skipped, so a killed run resumes where it stopped (requires a
    persistent ``hp_search.optuna.storage``).
    """
    device = resolve_device(config["device"])
    processed_dir = config["paths"]["data_processed"]
    embeddings_dir = config["paths"]["embeddings"]
    n_trials = int(config["hp_search"]["optuna"]["n_trials"])

    cells = _list_cells(condition, config, processed_dir, embeddings_dir)
    logger.info("Optuna cells to process: %d (n_trials=%d)", len(cells), n_trials)
    if not cells:
        return

    n_workers = 1 if sequential else _resolve_optuna_workers(workers, device, len(cells))

    if n_workers == 1:
        try:
            for cell, n_users, n_items, emb_path in cells:
                _optimize_one_cell(
                    cell,
                    n_users,
                    n_items,
                    emb_path,
                    config=config,
                    processed_dir=processed_dir,
                    device=device,
                )
        except KeyboardInterrupt:
            logger.warning("Optuna study interrupted by user.")
            raise
        return

    if config["hp_search"]["optuna"].get("storage") is None:
        logger.warning(
            "hp_search.optuna.storage is null: studies live in worker memory "
            "and completed-cell skip will not survive a restart. Set a "
            "sqlite storage for resumable parallel search.",
        )

    import torch.multiprocessing as mp

    logger.info("Optuna inter-cell parallelism: %d workers", n_workers)
    ctx = mp.get_context("spawn")
    cell_queue = ctx.Queue()
    result_queue = ctx.Queue()
    for item in cells:
        cell_queue.put(item)
    for _ in range(n_workers):
        cell_queue.put(None)

    procs = []
    for i in range(n_workers):
        p = ctx.Process(
            target=_optuna_cell_worker,
            args=(i, cell_queue, result_queue, n_workers, config, processed_dir, device),
            daemon=True,
        )
        p.start()
        procs.append(p)

    results: list[dict] = []
    while len(results) < len(cells):
        try:
            results.append(result_queue.get(timeout=30))
        except Exception:  # noqa: BLE001, queue.Empty from a spawn context
            if not any(p.is_alive() for p in procs):
                logger.warning("All Optuna workers exited early.")
                break
    for p in procs:
        p.join(timeout=30)

    ok = sum(1 for r in results if r.get("status") == "ok")
    logger.info("Optuna search complete: %d/%d cells succeeded.", ok, len(cells))
    for r in results:
        if r.get("status") != "ok":
            logger.error("  cell %s failed: %s", r.get("cell"), r.get("error"))


def _list_cells(
    condition: str,
    config: dict,
    processed_dir: str,
    embeddings_dir: str,
) -> list[tuple[CellKey, int, int, str | None]]:
    """Enumerate every ``(dataset, model, embedding)`` cell to optimise.

    Shares :func:`_iter_cells` with :func:`build_job_list` but stops at
    the cell granularity (no per-HP enumeration).
    """
    model_names = _resolve_model_names(config)
    return [
        (
            CellKey(cell.dataset_name, cell.model_name, cell.embedding_name),
            cell.n_users,
            cell.n_items,
            cell.embedding_path,
        )
        for cell in _iter_cells(condition, config, processed_dir, embeddings_dir, model_names)
    ]


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

    item_categories = None
    if getattr(model_cls, "wants_categories", False):
        from src.data.categories import item_category_array

        item_categories = item_category_array(dataset_name, processed_dir)

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
        item_categories=item_categories,
    )
