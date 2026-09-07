"""Parallel training orchestrator for recommendation models.

Manages a pool of GPU worker processes to train multiple recommender
models simultaneously.  Automatically detects available VRAM and sizes
the pool accordingly.

Every submitted job is tracked by the parent in a :class:`_JobRegistry`
and ends in exactly one terminal :class:`JobOutcome` (``succeeded`` /
``failed`` / ``cancelled``).  A worker process that exits before it
publishes a result is accounted for from its exit status and its last
assignment; queue emptiness is never used as a completion signal.
"""

from __future__ import annotations

import hashlib
import json
import queue
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from queue import Empty
from typing import Any

import torch
import torch.multiprocessing as mp

from src.utils.atomic_io import atomic_write
from src.utils.logging import get_logger
from src.utils.memory import AdmissionPlan, available_cpus, plan_pool_workers
from src.utils.resources import ResourcesConfig, resolve_resources

logger = get_logger(__name__)

#: Factor the ranking budget is multiplied by per OOM retry.  Halving
#: halves the user-batch, which is what actually overflowed: the ranking
#: buffers scale with ``batch x n_items``, not with the model.
_OOM_SHRINK_PER_RETRY = 0.5

#: Retries before a job is declared unrecoverable.  At the third attempt
#: the budget is a quarter of the original.  A job therefore runs at
#: most ``MAX_OOM_RETRIES + 1`` times, on the sequential and on the
#: parallel path alike; only ``torch.cuda.OutOfMemoryError`` is retried.
MAX_OOM_RETRIES = 2

#: Seconds the parent waits on the result queue before it checks its
#: workers for unexpected exits.  Progress is logged at most every
#: ``_PROGRESS_LOG_S`` seconds regardless of this poll.
_RESULT_POLL_S = 5.0
_PROGRESS_LOG_S = 30.0

#: Value of a worker's slot in the shared assignment table when it is
#: not running any job.
_NO_ASSIGNMENT = -1

#: Host bytes assumed per worker when the caller cannot estimate the
#: footprint (M06): the interpreter + torch stack + CUDA context of a
#: worker (``_WORKER_BASE_BYTES`` in the train step) plus room for one
#: modest dataset.  An unknown footprint is never treated as free -- the
#: pool used to be sized from VRAM alone, i.e. against infinite host RAM.
UNKNOWN_WORKER_FOOTPRINT_BYTES = 2 * 1024**3

#: Terminal outcome statuses (C03).
OUTCOME_SUCCEEDED = "succeeded"
OUTCOME_FAILED = "failed"
OUTCOME_CANCELLED = "cancelled"

#: Type alias for the per-job execution hook the worker loop calls.
JobRunner = Callable[["TrainingJob"], float]


class SingleSlotCache:
    """A cache that holds exactly one entry, evicting on every miss.

    The training worker is long-lived (thousands of jobs) and each value
    it caches is a whole dataset or a whole ``(n_items, D)`` float32
    embedding matrix -- a learned-alignment sidecar for amazon_fashion
    concatenates to ``(166270, 2816)``, i.e. 1.9 GB.  An unbounded dict
    therefore accumulated every fusion of every dataset until the
    container cgroup OOM-killed the worker mid-run.

    Jobs are emitted dataset -> model -> embedding, so a single slot
    still serves the whole hyperparameter grid of a cell from cache; a
    miss on a cell boundary costs one npy read instead of a dead run.
    The previous value is dropped *before* the new one is built, so the
    peak is one entry, not two.
    """

    def __init__(self) -> None:
        self._key: str | None = None
        self._value: object | None = None

    def get_or_load(self, key: str, loader):
        """Return the cached value for *key*, calling *loader* on a miss.

        @param key - Identity of the entry (dataset name or artefact path).
        @param loader - Zero-argument callable building the value on a miss.
        @returns The cached or freshly loaded value.
        """
        if self._key == key:
            return self._value

        self._key = None
        self._value = None

        value = loader()
        self._key = key
        self._value = value
        return value


try:
    import fcntl

    def _lock_file(f):
        fcntl.flock(f, fcntl.LOCK_EX)

    def _unlock_file(f):
        fcntl.flock(f, fcntl.LOCK_UN)
except ImportError:
    # Windows fallback: no locking (single-machine, low contention)
    def _lock_file(f):
        pass

    def _unlock_file(f):
        pass


@dataclass
class TrainingJob:
    """Single training job to be executed by a worker.

    Heavy data (interactions, embeddings, config) are NOT stored here.
    Workers load them from disk using the path/name references.

    ``submit_index`` is the job's position in the orchestrator's
    submission order; the parent uses it to recover which job a worker
    held when the worker died.  It is assigned by the orchestrator and
    is not part of the job identity.

    ``data_identity`` is the content identity of the job's dataset and
    feature artifact (:meth:`src.utils.identity.DataIdentity.to_payload`),
    resolved once by the parent so every worker binds its checkpoints
    and grid progress to the same digests (E03/E04).  ``None`` means the
    parent did not resolve it; the worker then records the identity as
    unresolved rather than guessing one.
    """

    dataset_name: str
    model_name: str
    embedding_name: str
    hyperparams: dict
    n_users: int
    n_items: int
    embeddings_path: str | None
    processed_dir: str
    device: str
    priority: int = 0
    retry_count: int = 0
    submit_index: int = _NO_ASSIGNMENT
    data_identity: dict | None = None
    #: Read the feature artifact through bounded row access (M01/M02)
    #: instead of a resident matrix; decided by the admission planner.
    lazy_features: bool = False

    @property
    def job_id(self) -> str:
        # hashlib, not hash(): built-in str hashing is salted per process
        # (PYTHONHASHSEED), so spawned workers would compute a different id
        # than the parent and OOM-retry matching would silently never fire.
        digest = hashlib.md5(
            json.dumps(self.hyperparams, sort_keys=True).encode("utf-8")
        ).hexdigest()
        return f"{self.dataset_name}_{self.embedding_name}_{self.model_name}_{digest[:6]}"


@dataclass(frozen=True)
class JobOutcome:
    """Terminal outcome of one submitted job (internal record, C03).

    ``attempt_count`` counts every execution of the job, OOM retries
    included.  ``error_type`` is the exception class name for
    ``failed`` outcomes, or a symbolic reason (``WorkerExit``,
    ``PoolExited``) when no exception reached the parent.
    """

    job_id: str
    attempt_id: str
    status: str
    attempt_count: int
    error_type: str | None = None
    error_message: str | None = None
    best_metric: float | None = None
    identity: dict[str, Any] = field(default_factory=dict)

    def to_result(self) -> dict:
        """Render the outcome in the orchestrator's public result shape.

        ``status`` keeps the historical values (``ok`` / ``oom`` /
        ``error``) so existing callers counting ``status == "ok"`` are
        unchanged; ``cancelled`` is new.  ``outcome``, ``attempts`` and
        ``error_type`` are additive.
        """
        base = {
            "job_id": self.job_id,
            "outcome": self.status,
            "attempts": self.attempt_count,
        }
        if self.status == OUTCOME_SUCCEEDED:
            return {**base, "status": "ok", "best_metric": self.best_metric}
        if self.status == OUTCOME_CANCELLED:
            return {**base, "status": "cancelled", "error": self.error_message}
        status = "oom" if self.error_type == "OutOfMemoryError" else "error"
        return {
            **base,
            "status": status,
            "error": self.error_message,
            "error_type": self.error_type,
        }


def _job_identity(job: TrainingJob) -> dict[str, Any]:
    return {
        "dataset_name": job.dataset_name,
        "model_name": job.model_name,
        "embedding_name": job.embedding_name,
        "hyperparams": dict(job.hyperparams),
    }


class _JobRegistry:
    """Parent-side ledger: every submitted job gets one terminal outcome.

    States: ``open`` (submitted, no outcome yet), ``retry_pending`` (an
    OOM attempt was recorded and a retry is owed) and terminal
    (:class:`JobOutcome`).  Terminal outcomes are immutable: a second
    message for the same job -- a duplicate delivery, or a stale
    result arriving after the parent already failed the job from its
    worker's exit status -- is ignored and logged.
    """

    def __init__(self, jobs: list[TrainingJob]) -> None:
        self._jobs: dict[str, TrainingJob] = {}
        self._order: list[str] = []
        self._outcomes: dict[str, JobOutcome] = {}
        self._retry_pending: dict[str, TrainingJob] = {}
        for index, job in enumerate(jobs):
            if job.job_id in self._jobs:
                raise ValueError(f"duplicate job submitted: {job.job_id}")
            job.submit_index = index
            self._jobs[job.job_id] = job
            self._order.append(job.job_id)

    # -- queries ---------------------------------------------------------

    @property
    def jobs(self) -> list[TrainingJob]:
        return [self._jobs[job_id] for job_id in self._order]

    def job_at(self, submit_index: int) -> TrainingJob | None:
        if 0 <= submit_index < len(self._order):
            return self._jobs[self._order[submit_index]]
        return None

    def is_terminal(self, job_id: str) -> bool:
        return job_id in self._outcomes

    def open_ids(self) -> list[str]:
        """Jobs with neither a terminal outcome nor a pending retry."""
        return [
            job_id
            for job_id in self._order
            if job_id not in self._outcomes and job_id not in self._retry_pending
        ]

    def take_retries(self) -> list[TrainingJob]:
        """Hand out the jobs owed a retry, moving them back to ``open``."""
        retries = list(self._retry_pending.values())
        self._retry_pending.clear()
        return retries

    def outcomes(self) -> list[JobOutcome]:
        return [self._outcomes[job_id] for job_id in self._order if job_id in self._outcomes]

    def results(self) -> list[dict]:
        return [outcome.to_result() for outcome in self.outcomes()]

    # -- transitions -----------------------------------------------------

    def record(self, message: dict) -> bool:
        """Apply one worker message; return False when it was ignored."""
        job_id = message.get("job_id")
        job = self._jobs.get(job_id)
        if job is None:
            logger.warning("Ignoring result for unknown job %s", job_id)
            return False
        if job_id in self._outcomes or job_id in self._retry_pending:
            logger.warning(
                "Ignoring duplicate result for %s (status=%s): already %s",
                job_id,
                message.get("status"),
                "terminal" if job_id in self._outcomes else "retry-pending",
            )
            return False

        attempt = int(message.get("attempt", job.retry_count + 1))
        status = message.get("status")
        if status == "ok":
            self._finish(
                job,
                attempt,
                OUTCOME_SUCCEEDED,
                best_metric=message.get("best_metric"),
            )
        elif status == "oom":
            self._record_oom(job, attempt, message)
        else:
            self._finish(
                job,
                attempt,
                OUTCOME_FAILED,
                error_type=message.get("error_type") or "Exception",
                error_message=message.get("error"),
            )
        return True

    def _record_oom(self, job: TrainingJob, attempt: int, message: dict) -> None:
        if job.retry_count < MAX_OOM_RETRIES:
            job.retry_count += 1
            self._retry_pending[job.job_id] = job
            return
        self._finish(
            job,
            attempt,
            OUTCOME_FAILED,
            error_type="OutOfMemoryError",
            error_message=message.get("error")
            or f"CUDA out of memory on every attempt ({attempt} attempts)",
        )

    def fail(self, job_id: str, *, error_type: str, error_message: str) -> None:
        """Terminate an open or retry-pending job as ``failed``."""
        job = self._jobs[job_id]
        if job_id in self._outcomes:
            return
        self._retry_pending.pop(job_id, None)
        self._finish(
            job,
            job.retry_count + 1,
            OUTCOME_FAILED,
            error_type=error_type,
            error_message=error_message,
        )

    def cancel_open(self, reason: str) -> list[str]:
        """Terminate every open job as ``cancelled``; return their ids."""
        cancelled = self.open_ids()
        for job_id in cancelled:
            job = self._jobs[job_id]
            self._finish(
                job,
                job.retry_count,
                OUTCOME_CANCELLED,
                error_type="PoolExited",
                error_message=reason,
            )
        return cancelled

    def _finish(self, job: TrainingJob, attempt: int, status: str, **fields: Any) -> None:
        self._outcomes[job.job_id] = JobOutcome(
            job_id=job.job_id,
            attempt_id=f"{job.job_id}#{attempt}",
            status=status,
            attempt_count=attempt,
            identity=_job_identity(job),
            **fields,
        )


def detect_max_workers(
    device: str = "cuda", per_worker_bytes: int = 0, *, reserve_bytes: int
) -> int:
    """Estimate how many training workers fit in GPU VRAM *and* host RAM.

    Uses a simple heuristic based on total VRAM rather than dummy-model
    profiling, because real datasets (100K+ items) use far more memory
    than any small dummy can predict.

    VRAM is only half the constraint: every spawned worker keeps its own
    copy of the interaction dicts and the visual embedding matrix in
    host RAM (``spawn`` shares nothing), so a pool sized purely from
    VRAM can exhaust system memory instead.  When *per_worker_bytes* is
    given, the host-memory budget lowers the count accordingly; the
    default of ``0`` means "unknown", which is charged the conservative
    :data:`UNKNOWN_WORKER_FOOTPRINT_BYTES` per worker (M06) instead of
    being read as "no host memory needed".  *reserve_bytes*
    (``resources.host.reserved_bytes``) is withheld from the host budget.
    """
    if per_worker_bytes <= 0:
        per_worker_bytes = UNKNOWN_WORKER_FOOTPRINT_BYTES
    if device == "cpu" or not torch.cuda.is_available():
        cpu_cap = max(1, available_cpus() - 1)
        return plan_pool_workers(
            per_worker_bytes=per_worker_bytes,
            hard_cap=cpu_cap,
            reserve_bytes=reserve_bytes,
            label="training pool",
        )

    try:
        # Free VRAM, not total: the orchestrating process (and anything
        # else on the GPU) may still hold several GB from an earlier step,
        # and workers sized from the total then OOM against each other.
        torch.cuda.empty_cache()
        free_bytes, total_bytes = torch.cuda.mem_get_info(0)
        free_mb = free_bytes / (1024 * 1024)
        total_mb = total_bytes / (1024 * 1024)
    except Exception as exc:
        logger.warning("VRAM detection failed (%s), defaulting to 2.", exc)
        return 2

    # ~4 GB per worker: real models with 100K+ items need dedicated GPU bandwidth
    mb_per_worker = 4096
    margin_mb = 1024
    available_mb = free_mb - margin_mb
    n_workers = max(1, int(available_mb / mb_per_worker))
    n_workers = min(n_workers, max(1, available_cpus() - 1))

    logger.info(
        "VRAM: free=%.0f MB of %.0f MB, ~%d MB/worker, margin=%d MB → %d workers",
        free_mb,
        total_mb,
        mb_per_worker,
        margin_mb,
        n_workers,
    )
    return plan_pool_workers(
        per_worker_bytes=per_worker_bytes,
        hard_cap=n_workers,
        reserve_bytes=reserve_bytes,
        label="training pool",
    )


def _locked_append_grid_progress(path: Path, entry: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(".lock")
    with open(lock_path, "w") as lf:
        _lock_file(lf)
        try:
            existing = []
            if path.exists():
                with open(path) as f:
                    existing = json.load(f)
            existing.append(entry)
            # fsync + retried replace (networked-FS dirent lag); the
            # surrounding flock already serialises the read-modify-write.
            atomic_write(
                lambda tmp: Path(tmp).write_text(json.dumps(existing, indent=2)),
                path,
            )
        finally:
            _unlock_file(lf)


class _WorkerContext:
    """Per-process training state: config, checkpoints and the one-slot caches.

    Built lazily by :func:`_worker_fn` the first time a real job runs,
    so a worker driven by an injected ``job_runner`` (tests) never
    touches the configuration directory or the checkpoint root.

    ``config`` is the parent's RESOLVED configuration snapshot (custom
    ``--config-dir``, multi-seed override, CLI overrides such as
    ``--hp-search`` / ``--n-trials``, seed and result/checkpoint roots).
    A spawned process starts with fresh module globals, so re-reading
    the YAML here would silently drop every one of those (audit F05);
    the snapshot travels through the process arguments instead.  The
    disk fallback exists only for callers that predate the snapshot and
    is logged as such.
    """

    def __init__(self, n_workers: int, wlog, config: dict | None = None) -> None:
        from src.utils.checkpoint import CheckpointManager

        self._wlog = wlog
        if config is None:
            from src.utils.config import load_config

            wlog.warning(
                "worker received no resolved configuration snapshot; reloading the "
                "YAML defaults from disk (CLI/seed/config-dir overrides are NOT applied)."
            )
            config = load_config()
        self._config = config
        self._resources: ResourcesConfig = resolve_resources(config)
        self._worker_vram = _probe_worker_vram(
            n_workers, wlog, vram_share=self._resources.gpu.vram_share
        )
        self._checkpoint_mgr = CheckpointManager(checkpoint_root(config))
        # One slot each: an unbounded cache here is what OOM-killed the
        # worker mid-run.  See :class:`SingleSlotCache`.
        self._data_cache = SingleSlotCache()
        self._emb_cache = SingleSlotCache()

    def _read_data(self, processed_dir: str, dataset_name: str):
        import pandas as pd

        base = Path(processed_dir) / dataset_name
        train_df = pd.read_csv(base / "train.csv")
        val_df = pd.read_csv(base / "val.csv")
        with open(base / "user2idx.json") as f:
            n_users = len(json.load(f))
        with open(base / "item2idx.json") as f:
            n_items = len(json.load(f))
        train_inter: dict[int, set[int]] = {}
        for _, row in train_df.iterrows():
            u, i = int(row["user_idx"]), int(row["item_idx"])
            train_inter.setdefault(u, set()).add(i)
        val_inter: dict[int, set[int]] = {}
        for _, row in val_df.iterrows():
            u, i = int(row["user_idx"]), int(row["item_idx"])
            val_inter.setdefault(u, set()).add(i)

        # Item→category indices for wants_categories models (DeepStyle).
        # Built once per dataset per worker; None when the dataset ships
        # no labels (DeepStyle then degenerates to VBPR by design).
        from src.data.categories import item_category_array

        item_cats = item_category_array(dataset_name, processed_dir)

        return (n_users, n_items, train_inter, val_inter, item_cats)

    def _load_data(self, processed_dir: str, dataset_name: str):
        return self._data_cache.get_or_load(
            dataset_name,
            lambda: self._read_data(processed_dir, dataset_name),
        )

    def _load_embeddings(self, path: str | None, *, lazy: bool = False):
        # ``load_embedding`` transparently handles online-fusion
        # sidecars: a ``.json`` path expands to a stacked
        # ``(n_items, M, D)`` array, while ``.npy`` paths load directly.
        # ``lazy`` (admission-decided, M05) returns a bounded source.
        if path is None:
            return None
        from src.fusions import load_embedding

        key = f"{path}#lazy" if lazy else path
        return self._emb_cache.get_or_load(key, lambda p=path: load_embedding(p, lazy=lazy))

    def _ranking_budget(self, job: TrainingJob) -> int | None:
        # Each OOM retry halves the ranking budget, which halves the
        # user-batch the evaluator can afford.  Without this the job
        # came back byte-for-byte identical and OOM'd again.  The share
        # (``resources.gpu.ranking_vram_share``) bounds the kernel length
        # of one ranking batch; the remainder of the allowance covers the
        # model, its tables, the optimiser state and the autograd graph.
        if not self._worker_vram:
            return None
        share = self._resources.gpu.ranking_vram_share
        budget = int(self._worker_vram * share * _OOM_SHRINK_PER_RETRY**job.retry_count)
        if job.retry_count:
            self._wlog.info(
                "  Retry %d for %s: ranking budget %.2f GB",
                job.retry_count,
                job.job_id,
                budget / 1024**3,
            )
        return budget

    def run(self, job: TrainingJob) -> float:
        """Train *job* and return its best validation metric."""
        from src.recommenders import get_recommender_class
        from src.utils.identity import build_identity_context, canonical_digest, condition_of
        from src.utils.training import resolve_training_identity, train_single_run

        torch.cuda.empty_cache()
        model_cls = get_recommender_class(job.model_name)
        n_users, n_items, train_inter, val_inter, item_cats = self._load_data(
            job.processed_dir,
            job.dataset_name,
        )
        visual_emb = self._load_embeddings(job.embeddings_path, lazy=job.lazy_features)
        # The parent resolved the data identity once per cell; the worker
        # binds its checkpoints and grid progress to the same digests.
        identity_context = build_identity_context(
            job.data_identity, condition=condition_of(job.embedding_name)
        )
        identity_digest = canonical_digest(
            resolve_training_identity(
                model_cls=model_cls,
                model_name=job.model_name,
                dataset_name=job.dataset_name,
                embedding_name=job.embedding_name,
                hyperparams=job.hyperparams,
                config=self._config,
                identity_context=identity_context,
            )
        )

        best_val = train_single_run(
            model_cls=model_cls,
            model_name=job.model_name,
            n_users=n_users,
            n_items=n_items,
            visual_embeddings=visual_emb,
            train_interactions=train_inter,
            selection_interactions=val_inter,
            hyperparams=job.hyperparams,
            config=self._config,
            checkpoint_mgr=self._checkpoint_mgr,
            dataset_name=job.dataset_name,
            embedding_name=job.embedding_name,
            device=job.device,
            item_categories=item_cats,
            ranking_budget_bytes=self._ranking_budget(job),
            identity_context=identity_context,
        )

        experiment_key = f"{job.dataset_name}_{job.embedding_name}_{job.model_name}"
        # Same root the parent's ``build_job_list`` reads completed work
        # from; a hard-coded ``checkpoints/`` here diverged from a
        # seed-suffixed or custom root and the skip never fired.
        gs_path = grid_progress_path(self._checkpoint_mgr, experiment_key)
        _locked_append_grid_progress(
            gs_path,
            {
                "hyperparams": job.hyperparams,
                "best_metric": best_val,
                "identity_digest": identity_digest,
            },
        )

        run_id = self._checkpoint_mgr.get_run_id(
            job.dataset_name,
            job.embedding_name,
            job.model_name,
            job.hyperparams,
        )
        self._checkpoint_mgr.clear_training_checkpoint(run_id)
        return best_val


def _probe_worker_vram(n_workers: int, wlog, *, vram_share: float) -> int:
    """Cap this process's VRAM and return the byte allowance (0 = unknown)."""
    # The per-process cap and the ranking budget derived from it are the
    # same decision seen from two sides: torch enforces the cap, and the
    # evaluator has to size its (batch x n_items) buffers to fit inside
    # it.  ``set_per_process_memory_fraction`` is invisible to
    # ``get_device_properties``, so the number has to travel by hand.
    if not torch.cuda.is_available():
        return 0
    # Capped in BOTH cases -- skipping the cap for n == 1 is what
    # let a single worker claim all 16 GB of the display GPU.
    from src.utils.device import cap_process_vram

    fraction = cap_process_vram(n_workers, vram_share=vram_share)
    try:
        total = torch.cuda.get_device_properties(0).total_memory
        return int(total * fraction)
    except Exception as exc:  # noqa: BLE001 — probing must not kill the worker
        wlog.warning("VRAM probe failed (%s); evaluator will size itself.", exc)
        return 0


def checkpoint_root(config: dict) -> str:
    """The checkpoint root of a resolved configuration (``paths.checkpoints``)."""
    return str((config.get("paths") or {}).get("checkpoints", "checkpoints"))


def grid_progress_path(checkpoint_mgr, experiment_key: str) -> Path:
    """Grid-progress file of *experiment_key* under the manager's root.

    Mirrors ``CheckpointManager.load_grid_search_progress`` so the
    writer (worker) and the reader (parent) can never disagree on the
    directory.
    """
    return Path(checkpoint_mgr.checkpoint_dir) / "grid_search" / f"{experiment_key}.json"


def _worker_fn(
    worker_id: int,
    job_queue,
    result_queue,
    n_workers: int,
    log_dir: str,
    job_runner: JobRunner | None = None,
    assignment=None,
    config: dict | None = None,
) -> None:
    """Worker process: pulls jobs from queue, trains, reports results.

    Every job produces exactly one message: ``ok``, ``oom`` (only for
    ``torch.cuda.OutOfMemoryError``; the parent decides whether to
    retry) or ``error`` (any other exception -- never retried).  While
    a job runs, ``assignment[worker_id]`` holds its ``submit_index`` so
    the parent can fail it from the exit status if this process dies
    before the message is published.  ``job_runner`` replaces the real
    training call (fault-injection tests); ``None`` uses
    :class:`_WorkerContext` built on ``config``, the parent's resolved
    configuration snapshot.
    """
    project_root = str(Path(__file__).resolve().parent.parent.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from src.utils.logging import get_logger as _get_logger

    wlog = _get_logger(f"worker_{worker_id}", log_dir=log_dir)
    runner: JobRunner | None = job_runner

    while True:
        try:
            job: TrainingJob | None = job_queue.get(timeout=5)
        except Empty:
            break
        if job is None:
            break

        if runner is None:
            runner = _WorkerContext(n_workers, wlog, config).run

        hp_str = " ".join(f"{k}={v}" for k, v in sorted(job.hyperparams.items()))
        wlog.info(
            "Starting: %s × %s × %s | %s",
            job.model_name,
            job.embedding_name,
            job.dataset_name,
            hp_str,
        )
        if assignment is not None:
            assignment[worker_id] = job.submit_index
        result_queue.put(_run_one_job(job, runner, wlog))
        if assignment is not None:
            assignment[worker_id] = _NO_ASSIGNMENT


def _run_one_job(job: TrainingJob, runner: JobRunner, wlog) -> dict:
    """Execute one attempt of *job* and build its result message."""
    attempt = job.retry_count + 1
    base = {"job_id": job.job_id, "attempt": attempt}
    try:
        best_val = runner(job)
    except torch.cuda.OutOfMemoryError as exc:
        torch.cuda.empty_cache()
        wlog.warning("  OOM on %s (attempt %d)", job.job_id, attempt)
        return {**base, "status": "oom", "retry_count": job.retry_count, "error": str(exc)}
    except Exception as exc:  # noqa: BLE001 — isolate failures per job
        wlog.error("  Error on %s: %s", job.job_id, exc, exc_info=True)
        return {
            **base,
            "status": "error",
            "error": str(exc),
            "error_type": type(exc).__name__,
        }
    wlog.info("  Done: best_metric=%.4f", best_val)
    return {**base, "status": "ok", "best_metric": best_val}


class TrainingOrchestrator:
    """Manages parallel training of recommendation models."""

    def __init__(
        self,
        n_workers: int = 0,
        device: str = "cuda",
        log_dir: str = "logs",
        per_worker_bytes: int = 0,
        *,
        job_runner: JobRunner | None = None,
        config: dict | None = None,
        admission: AdmissionPlan | None = None,
    ) -> None:
        """Size the pool.

        *per_worker_bytes* is the caller's estimate of the host RAM one
        worker holds (interaction dicts + visual embeddings + the CUDA
        context).  It only applies to the auto-detected count: an
        explicit *n_workers* is honoured verbatim, because pinning the
        pool is how a researcher overrides the heuristic -- except that
        an *admission* plan (M05/M06) is enforced: a pinned count above
        the admitted one is clamped with a warning, and a plan that
        admits nothing refuses to build a pool.

        *job_runner* replaces the per-job training call inside every
        worker (fault-injection tests).  It must be picklable for the
        spawned pool; ``None`` runs the real training.

        *config* is the parent's resolved configuration snapshot, handed
        to every worker so none of them reloads the YAML defaults.
        """
        self.device = device
        self.log_dir = log_dir
        reserve = resolve_resources(config).host.reserved_bytes
        self.n_workers = (
            detect_max_workers(device, per_worker_bytes, reserve_bytes=reserve)
            if n_workers <= 0
            else n_workers
        )
        self.admission = admission
        if admission is not None:
            self.n_workers = _enforce_admission(self.n_workers, admission)
        self._job_runner = job_runner
        self._config = config
        logger.info("Training orchestrator: %d workers", self.n_workers)

    def run(self, jobs: list[TrainingJob]) -> list[dict]:
        """Run every job and return one result dict per submitted job."""
        if not jobs:
            return []

        jobs.sort(key=lambda j: (j.priority, j.dataset_name, j.embedding_name))

        if self.n_workers == 1 or self.device == "cpu":
            return self._run_sequential(jobs)
        return self._run_parallel(jobs)

    # -- sequential ------------------------------------------------------

    def _run_sequential(self, jobs: list[TrainingJob]) -> list[dict]:
        logger.info("Running %d jobs sequentially.", len(jobs))
        registry = _JobRegistry(jobs)
        self._run_sequential_into(registry, jobs)
        return registry.results()

    def _run_sequential_into(self, registry: _JobRegistry, batch: list[TrainingJob]) -> None:
        """Run *batch* in this process, retrying OOM jobs until exhaustion.

        The worker loop is in-process, so every message is already in
        the result queue when it returns; the drain reconciles the
        batch against the registry instead of trusting emptiness.
        """
        while batch:
            job_queue: queue.Queue = queue.Queue()
            result_queue: queue.Queue = queue.Queue()
            for job in batch:
                job_queue.put(job)
            job_queue.put(None)
            _worker_fn(
                0, job_queue, result_queue, 1, self.log_dir, self._job_runner, config=self._config
            )
            self._drain_sequential(registry, batch, result_queue)
            batch = registry.take_retries()

    @staticmethod
    def _drain_sequential(registry: _JobRegistry, batch: list[TrainingJob], result_queue) -> None:
        while True:
            try:
                message = result_queue.get_nowait()
            except Empty:
                break
            registry.record(message)
        for job in batch:
            if job.job_id in registry.open_ids():
                registry.fail(
                    job.job_id,
                    error_type="NoResult",
                    error_message="worker loop returned without publishing a result",
                )

    # -- parallel --------------------------------------------------------

    def _run_parallel(self, jobs: list[TrainingJob]) -> list[dict]:
        logger.info("Running %d jobs with %d workers.", len(jobs), self.n_workers)
        registry = _JobRegistry(jobs)
        start_time = time.time()

        ctx = mp.get_context("spawn")
        job_queue = ctx.Queue()
        result_queue = ctx.Queue()
        assignment = ctx.Array("i", [_NO_ASSIGNMENT] * self.n_workers)

        for job in registry.jobs:
            job_queue.put(job)
        for _ in range(self.n_workers):
            job_queue.put(None)

        workers = [
            ctx.Process(
                target=_worker_fn,
                args=(i, job_queue, result_queue, self.n_workers, self.log_dir),
                kwargs={
                    "job_runner": self._job_runner,
                    "assignment": assignment,
                    "config": self._config,
                },
                daemon=True,
            )
            for i in range(self.n_workers)
        ]
        for worker in workers:
            worker.start()

        self._collect(registry, workers, assignment, result_queue, start_time)

        for worker in workers:
            worker.join(timeout=30)

        retries = registry.take_retries()
        if retries:
            logger.info("Retrying %d OOM jobs sequentially...", len(retries))
            self._run_sequential_into(registry, retries)

        results = registry.results()
        elapsed_h = (time.time() - start_time) / 3600
        ok = sum(1 for r in results if r.get("status") == "ok")
        logger.info("Done: %d/%d succeeded in %.1f h.", ok, len(jobs), elapsed_h)
        return results

    def _collect(self, registry, workers, assignment, result_queue, start_time) -> None:
        """Consume results until every job is terminal or retry-pending."""
        total = len(registry.jobs)
        reaped: set[int] = set()
        last_log = start_time
        while registry.open_ids():
            try:
                message = result_queue.get(timeout=_RESULT_POLL_S)
            except Empty:
                self._reap_dead_workers(registry, workers, assignment, reaped)
                if not any(w.is_alive() for w in workers):
                    _drain_nowait(result_queue, registry)
                    self._reap_dead_workers(registry, workers, assignment, reaped)
                    cancelled = registry.cancel_open("worker pool exited before the job started")
                    if cancelled:
                        logger.error("All workers exited; %d jobs cancelled.", len(cancelled))
                    break
            else:
                registry.record(message)
            last_log = self._maybe_log_progress(registry, workers, total, start_time, last_log)

    @staticmethod
    def _reap_dead_workers(registry, workers, assignment, reaped: set[int]) -> None:
        for worker_id, worker in enumerate(workers):
            if worker_id in reaped or worker.is_alive():
                continue
            reaped.add(worker_id)
            job = registry.job_at(assignment[worker_id])
            if job is None or registry.is_terminal(job.job_id):
                continue
            message = (
                f"worker {worker_id} exited with code {worker.exitcode} "
                f"before publishing a result for attempt {job.retry_count + 1}"
            )
            logger.error("%s: %s", job.job_id, message)
            registry.fail(job.job_id, error_type="WorkerExit", error_message=message)

    @staticmethod
    def _maybe_log_progress(registry, workers, total, start_time, last_log) -> float:
        now = time.time()
        if now - last_log < _PROGRESS_LOG_S:
            return last_log
        completed = len(registry.outcomes())
        elapsed = now - start_time
        eta_h = (elapsed / max(completed, 1)) * (total - completed) / 3600
        logger.info(
            "Progress: %d/%d (%.1f%%) | %d workers | ETA: ~%.1f h",
            completed,
            total,
            100 * completed / total,
            sum(1 for w in workers if w.is_alive()),
            eta_h,
        )
        return now


def _enforce_admission(n_workers: int, admission: AdmissionPlan) -> int:
    """Clamp a pool size to what the resolved host budget admits (M06)."""
    if not admission.admitted or admission.n_workers < 1:
        from src.utils.memory import AdmissionError

        raise AdmissionError(f"no worker admitted against the host budget: {admission.reason}")
    if n_workers > admission.n_workers:
        logger.warning(
            "Training orchestrator: %d workers requested but the host budget admits %d "
            "(%s); using %d.",
            n_workers,
            admission.n_workers,
            admission.reason,
            admission.n_workers,
        )
        return admission.n_workers
    return n_workers


def _drain_nowait(result_queue, registry: _JobRegistry) -> None:
    """Record whatever is already queued without waiting for more."""
    while True:
        try:
            message = result_queue.get(timeout=0.1)
        except Empty:
            return
        registry.record(message)
