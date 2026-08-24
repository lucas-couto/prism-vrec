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

from tqdm import tqdm

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
    is_projected_artifact,
)
from src.utils.checkpoint import CheckpointManager
from src.utils.config import load_config
from src.utils.device import resolve_device
from src.utils.logging import get_logger
from src.utils.parallel import TrainingJob, TrainingOrchestrator
from src.utils.seed import set_seed

logger = get_logger(__name__)


EMBEDDING_VARIANTS = ("native", "projected", "both")


def _resolve_embedding_variants(config: dict) -> str:
    """Read ``embedding_variants`` from the recommender config.

    Selects which of the two artifact families the recommenders are
    trained on when a fixed projection is configured in
    ``configs/extractors.yaml``: the backbones' native features, their
    fixed-dim projections, or both side by side.
    """
    value = str(config.get("embedding_variants", "both"))
    if value not in EMBEDDING_VARIANTS:
        raise ValueError(
            f"embedding_variants must be one of {list(EMBEDDING_VARIANTS)}, got {value!r}"
        )
    return value


def filter_by_variant(names: list[str], variant: str) -> list[str]:
    """Keep the embedding names belonging to *variant*.

    ``"none"`` — the pseudo-embedding of the non-visual baselines — is
    never filtered out: it belongs to no artifact family, and dropping
    it would silently remove plain BPR from a projected-only battery.
    """
    if variant == "both":
        return names
    want_projected = variant == "projected"
    return [n for n in names if n == "none" or is_projected_artifact(n) == want_projected]


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
    variant = _resolve_embedding_variants(config)

    for dataset_name in config.get("datasets", []):
        all_embs = get_embedding_files(embeddings_dir, dataset_name, dim_filter or None)
        all_embs = filter_by_variant(all_embs, variant)
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


class EnabledRecommenderHasNoCellsError(RuntimeError):
    """An enabled recommender enumerated zero training cells (audit D5).

    Silently dropping a model out of the comparison (e.g. ACF enabled but
    no ``*_comp.npy`` artifact exists) would bias the battery without any
    signal; enumeration therefore fails loud instead.
    """


def assert_enabled_recommenders_have_cells(
    cell_counts: dict[str, int],
    condition: str,
) -> None:
    """Fail loud when an enabled recommender silently drops out (D5).

    Args:
        cell_counts: Per enabled (and registered) recommender, the number
            of ``(dataset, model, embedding)`` cells enumerated for it.
        condition: ``"frozen"`` or ``"finetuned"`` — non-visual models
            are legitimately absent from the finetuned condition.

    Raises:
        EnabledRecommenderHasNoCellsError: When a recommender has zero
            cells while other recommenders enumerated at least one and
            the emptiness is not condition-expected.
    """
    if not any(cell_counts.values()):
        # Nothing at all to train for this condition (e.g. no finetuned
        # artifacts yet): the existing "no pending jobs" paths report it.
        return
    for model_name, count in cell_counts.items():
        if count > 0:
            continue
        spec = get_recommender_spec(model_name)
        if not spec.requires_visual and condition != "frozen":
            # Feature-blind baselines (plain BPR) only run frozen.
            continue
        if spec.requires_components:
            reason = (
                "it requires component embeddings and no *_comp.npy artifact "
                "matched the enabled datasets/filters — run an extractor that "
                "emits component embeddings, or disable the recommender"
            )
        else:
            reason = (
                "no embedding artifact matched the enabled datasets/filters "
                f"(condition={condition!r}, embedding_variants, embedding_dims)"
            )
        raise EnabledRecommenderHasNoCellsError(
            f"recommender {model_name!r} is enabled but enumerated 0 training "
            f"cells for condition {condition!r}: {reason}. It would silently "
            "drop out of the comparison."
        )


def _cell_counts(
    condition: str,
    config: dict,
    processed_dir: str,
    embeddings_dir: str,
) -> dict[str, int]:
    """Cells per enabled model, BEFORE completed-job filtering (D5 guard)."""
    model_names = _resolve_model_names(config)
    counts = {name: 0 for name in model_names}
    for cell in _iter_cells(condition, config, processed_dir, embeddings_dir, model_names):
        counts[cell.model_name] += 1
    return counts


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

    # Fairness guard-rail (Task H): no recommender may declare its own
    # protocol budget — the budget is shared per dataset. Fail before
    # training rather than confound the comparison silently.
    from src.recommenders.hp_budget import assert_uniform_budget

    assert_uniform_budget(config)

    # Feature sanity gate (Task G): fail loud before burning battery time
    # on a corrupt matrix. Validates every backbone + fused .npy consumed.
    from src.steps.validate_features import gate_dataset_features

    gate_dataset_features(
        config.get("datasets", []),
        config,
        embeddings_dir=config["paths"]["embeddings"],
        processed_dir=config["paths"]["data_processed"],
    )

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
    from src.recommenders.hp_budget import grid_budget_message

    device = resolve_device(config["device"])
    processed_dir = config["paths"]["data_processed"]
    embeddings_dir = config["paths"]["embeddings"]

    # D1: the grid backend spends one selection shot per config, so
    # unequal per-model spaces are unequal budgets. Cannot be fixed
    # silently (spaces are legitimate per-model choices) — warn loud.
    grid_sizes = {
        name: len(get_hyperparam_grid(name, config)) for name in _resolve_model_names(config)
    }
    budget_warning = grid_budget_message(grid_sizes)
    if budget_warning:
        logger.warning(budget_warning)

    # D5: an enabled recommender with zero cells must fail, not vanish.
    assert_enabled_recommenders_have_cells(
        _cell_counts(condition, config, processed_dir, embeddings_dir),
        condition,
    )

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
        per_worker_bytes=_estimate_worker_bytes(jobs, processed_dir),
    )

    results = orchestrator.run(jobs)

    ok = sum(1 for r in results if r.get("status") == "ok")
    logger.info("Training complete: %d/%d experiments succeeded.", ok, len(jobs))


#: Host RAM a training worker needs on top of its data: the Python
#: interpreter, the imported torch stack and the process's CUDA context.
_WORKER_BASE_BYTES = 1536 * 1024**2

#: Interaction dicts (``{user: set(items)}``) are far larger in memory
#: than the CSV they come from — boxed ints inside per-user sets.  This
#: multiplier converts the on-disk CSV size into a resident estimate.
_INTERACTIONS_MEMORY_FACTOR = 40


def _estimate_worker_bytes(jobs: list[TrainingJob], processed_dir: str) -> int:
    """Estimate the host RAM one training worker holds.

    Workers are spawned, not forked, so nothing is shared: each one
    caches the full visual embedding matrix plus the train/val
    interaction dicts for the datasets it touches.  The estimate takes
    the worst case across *jobs* (largest embedding file, largest
    interaction file) so the pool is sized for the heaviest cell rather
    than the average one.

    Returns ``0`` when nothing can be measured, which
    :func:`src.utils.parallel.detect_max_workers` reads as "unknown"
    and leaves the VRAM heuristic untouched.
    """

    def _size(path: str | Path) -> int:
        try:
            return Path(path).stat().st_size
        except OSError:
            return 0

    emb_bytes = max(
        (_size(job.embeddings_path) for job in jobs if job.embeddings_path),
        default=0,
    )
    inter_bytes = max(
        (
            _size(Path(processed_dir) / job.dataset_name / "train.csv")
            + _size(Path(processed_dir) / job.dataset_name / "val.csv")
            for job in jobs
        ),
        default=0,
    )
    if emb_bytes == 0 and inter_bytes == 0:
        return 0
    return _WORKER_BASE_BYTES + emb_bytes + inter_bytes * _INTERACTIONS_MEMORY_FACTOR


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

    from src.recommenders.hp_budget import resolve_hp_budget

    optuna_cfg = config["hp_search"]["optuna"]
    # Single source of the protocol budget, shared by every recommender of
    # this dataset (Task H); per-dataset override via ``hp_budget:``.
    n_trials = int(resolve_hp_budget(config, cell.dataset_name)["n_trials"])
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

    # D5: an enabled recommender with zero cells must fail, not vanish.
    counts = {name: 0 for name in _resolve_model_names(config)}
    for cell_key, _n_users, _n_items, _emb_path in cells:
        counts[cell_key.model_name] += 1
    assert_enabled_recommenders_have_cells(counts, condition)

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
    total = len(cells)
    # Live cell-level battery bar (done/total, %, elapsed<ETA, rate).
    # Parent-side observability only — never touches worker computation.
    # Renders in place under a TTY (compose ``tty: true``); auto-quiet
    # off a TTY via ``disable=None``.
    with tqdm(total=total, desc="Training (Optuna cells)", unit="cell", disable=None) as pbar:
        while len(results) < total:
            try:
                results.append(result_queue.get(timeout=30))
            except Exception:  # noqa: BLE001, queue.Empty from a spawn context
                if not any(p.is_alive() for p in procs):
                    logger.warning("All Optuna workers exited early.")
                    break
                continue
            pbar.update(1)
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
    trial=None,
) -> float:
    """Single trial entry point: load data, train one model, return metric."""
    from src.fusions import load_embedding
    from src.recommenders import get_recommender_class
    from src.utils.training import train_single_run

    dataset_name = cell.dataset_name
    train_path = Path(processed_dir) / dataset_name / "train.csv"
    # Model selection (early stopping + the Optuna objective) runs on
    # VALIDATION users, masking each user's TRAIN items. The test set is
    # never read during training/selection — it is touched only by the
    # final evaluate step. Mirrors the grid worker (src/utils/parallel.py).
    val_path = Path(processed_dir) / dataset_name / "val.csv"

    import pandas as pd

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)

    train_interactions: dict = {}
    for u, i in zip(train_df["user_idx"], train_df["item_idx"], strict=False):
        train_interactions.setdefault(int(u), set()).add(int(i))
    val_interactions: dict = {}
    for u, i in zip(val_df["user_idx"], val_df["item_idx"], strict=False):
        val_interactions.setdefault(int(u), set()).add(int(i))

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
        selection_interactions=val_interactions,
        hyperparams=hyperparams,
        config=config,
        checkpoint_mgr=checkpoint_mgr,
        dataset_name=cell.dataset_name,
        embedding_name=cell.embedding_name,
        device=device,
        optuna_trial=trial,
        item_categories=item_categories,
    )


def train_replay(
    *,
    cell: CellKey,
    hyperparams: dict,
    n_users: int,
    n_items: int,
    embeddings_path: str | None,
    processed_dir: str,
    device: str,
    config: dict,
) -> float:
    """D2 replay: train ONE fixed config (no search), early stopping on val.

    The clean "train with a given config" entry point the battery runner
    (Task I) invokes to replicate a search's best config on the
    non-primary seeds.  Validation early stopping is active (via
    ``train_single_run``); ``config['seed']`` selects the seed.  Returns
    the best validation metric.  Orchestration across seeds is Task I's.
    """
    return _train_one_optuna_trial(
        cell=cell,
        hyperparams=hyperparams,
        n_users=n_users,
        n_items=n_items,
        embeddings_path=embeddings_path,
        processed_dir=processed_dir,
        device=device,
        config=config,
        trial=None,
    )
