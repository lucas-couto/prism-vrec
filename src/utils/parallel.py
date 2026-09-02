"""Parallel training orchestrator for recommendation models.

Manages a pool of GPU worker processes to train multiple recommender
models simultaneously.  Automatically detects available VRAM and sizes
the pool accordingly.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from queue import Empty

import torch
import torch.multiprocessing as mp

from src.utils.atomic_io import atomic_write
from src.utils.logging import get_logger
from src.utils.memory import plan_pool_workers

logger = get_logger(__name__)

#: Share of a worker's GPU allowance the validation ranking may hold.
#: The remainder covers the model, its embedding tables, the optimiser
#: state and the autograd graph.
_RANKING_VRAM_SHARE = 0.5

#: Factor the ranking budget is multiplied by per OOM retry.  Halving
#: halves the user-batch, which is what actually overflowed: the ranking
#: buffers scale with ``batch x n_items``, not with the model.
_OOM_SHRINK_PER_RETRY = 0.5

#: Retries before a job is declared unrecoverable.  At the third attempt
#: the budget is a quarter of the original.
MAX_OOM_RETRIES = 2


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

    @property
    def job_id(self) -> str:
        # hashlib, not hash(): built-in str hashing is salted per process
        # (PYTHONHASHSEED), so spawned workers would compute a different id
        # than the parent and OOM-retry matching would silently never fire.
        digest = hashlib.md5(
            json.dumps(self.hyperparams, sort_keys=True).encode("utf-8")
        ).hexdigest()
        return f"{self.dataset_name}_{self.embedding_name}_{self.model_name}_{digest[:6]}"


def detect_max_workers(device: str = "cuda", per_worker_bytes: int = 0) -> int:
    """Estimate how many training workers fit in GPU VRAM *and* host RAM.

    Uses a simple heuristic based on total VRAM rather than dummy-model
    profiling, because real datasets (100K+ items) use far more memory
    than any small dummy can predict.

    VRAM is only half the constraint: every spawned worker keeps its own
    copy of the interaction dicts and the visual embedding matrix in
    host RAM (``spawn`` shares nothing), so a pool sized purely from
    VRAM can exhaust system memory instead.  When *per_worker_bytes* is
    given, the host-memory budget lowers the count accordingly; the
    default of ``0`` means "unknown", which preserves the VRAM-only
    behaviour for callers that cannot estimate the footprint.
    """
    if device == "cpu" or not torch.cuda.is_available():
        cpu_cap = max(1, (os.cpu_count() or 4) - 1)
        return plan_pool_workers(
            per_worker_bytes=per_worker_bytes,
            hard_cap=cpu_cap,
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
    n_workers = min(n_workers, max(1, (os.cpu_count() or 4) - 1))

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


def _worker_fn(
    worker_id: int,
    job_queue: mp.Queue,
    result_queue: mp.Queue,
    n_workers: int,
    log_dir: str,
) -> None:
    """Worker process: pulls jobs from queue, trains, reports results."""
    project_root = str(Path(__file__).resolve().parent.parent.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    import json

    import numpy as np
    import pandas as pd

    from src.recommenders import get_recommender_class
    from src.utils.checkpoint import CheckpointManager
    from src.utils.config import load_config
    from src.utils.logging import get_logger as _get_logger
    from src.utils.training import train_single_run

    wlog = _get_logger(f"worker_{worker_id}", log_dir=log_dir)

    # The per-process cap and the ranking budget derived from it are the
    # same decision seen from two sides: torch enforces the cap, and the
    # evaluator has to size its (batch x n_items) buffers to fit inside
    # it.  ``set_per_process_memory_fraction`` is invisible to
    # ``get_device_properties``, so the number has to travel by hand.
    worker_vram = 0
    if torch.cuda.is_available():
        # 0.90/n keeps the sum of caps below the card (see train.py).
        fraction = 0.90 / n_workers if n_workers > 1 else 1.0
        if n_workers > 1:
            torch.cuda.set_per_process_memory_fraction(fraction)
        try:
            total = torch.cuda.get_device_properties(0).total_memory
            worker_vram = int(total * fraction)
        except Exception as exc:  # noqa: BLE001 — probing must not kill the worker
            wlog.warning("VRAM probe failed (%s); evaluator will size itself.", exc)

    checkpoint_mgr = CheckpointManager()
    config = load_config()

    _data_cache: dict[str, tuple] = {}
    _emb_cache: dict[str, np.ndarray] = {}

    def _load_data(processed_dir: str, dataset_name: str):
        if dataset_name in _data_cache:
            return _data_cache[dataset_name]
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

        result = (n_users, n_items, train_inter, val_inter, item_cats)
        _data_cache[dataset_name] = result
        return result

    while True:
        try:
            job: TrainingJob | None = job_queue.get(timeout=5)
        except Empty:
            break

        if job is None:
            break

        hp_str = " ".join(f"{k}={v}" for k, v in sorted(job.hyperparams.items()))
        wlog.info(
            "Starting: %s × %s × %s | %s",
            job.model_name,
            job.embedding_name,
            job.dataset_name,
            hp_str,
        )

        try:
            torch.cuda.empty_cache()
            model_cls = get_recommender_class(job.model_name)

            n_users, n_items, train_inter, val_inter, item_cats = _load_data(
                job.processed_dir,
                job.dataset_name,
            )

            # ``load_embedding`` transparently handles online-fusion
            # sidecars: a ``.json`` path expands to a stacked
            # ``(n_items, M, D)`` array, while ``.npy`` paths load directly.
            from src.fusions import load_embedding

            visual_emb = None
            if job.embeddings_path is not None:
                if job.embeddings_path not in _emb_cache:
                    _emb_cache[job.embeddings_path] = load_embedding(job.embeddings_path)
                visual_emb = _emb_cache[job.embeddings_path]

            # Each OOM retry halves the ranking budget, which halves the
            # user-batch the evaluator can afford.  Without this the job
            # came back byte-for-byte identical and OOM'd again.
            ranking_budget = (
                int(worker_vram * _RANKING_VRAM_SHARE * _OOM_SHRINK_PER_RETRY**job.retry_count)
                if worker_vram
                else None
            )
            if job.retry_count:
                wlog.info(
                    "  Retry %d for %s: ranking budget %.2f GB",
                    job.retry_count,
                    job.job_id,
                    (ranking_budget or 0) / 1024**3,
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
                config=config,
                checkpoint_mgr=checkpoint_mgr,
                dataset_name=job.dataset_name,
                embedding_name=job.embedding_name,
                device=job.device,
                item_categories=item_cats,
                ranking_budget_bytes=ranking_budget,
            )

            experiment_key = f"{job.dataset_name}_{job.embedding_name}_{job.model_name}"
            gs_path = Path("checkpoints/grid_search") / f"{experiment_key}.json"
            _locked_append_grid_progress(
                gs_path,
                {"hyperparams": job.hyperparams, "best_metric": best_val},
            )

            run_id = checkpoint_mgr.get_run_id(
                job.dataset_name,
                job.embedding_name,
                job.model_name,
                job.hyperparams,
            )
            checkpoint_mgr.clear_training_checkpoint(run_id)

            result_queue.put({"job_id": job.job_id, "status": "ok", "best_metric": best_val})
            wlog.info("  Done: best_metric=%.4f", best_val)

        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            wlog.warning("  OOM on %s", job.job_id)
            result_queue.put(
                {
                    "job_id": job.job_id,
                    "status": "oom",
                    "retry_count": job.retry_count,
                }
            )

        except Exception as exc:
            wlog.error("  Error on %s: %s", job.job_id, exc, exc_info=True)
            result_queue.put({"job_id": job.job_id, "status": "error", "error": str(exc)})


class TrainingOrchestrator:
    """Manages parallel training of recommendation models."""

    def __init__(
        self,
        n_workers: int = 0,
        device: str = "cuda",
        log_dir: str = "logs",
        per_worker_bytes: int = 0,
    ) -> None:
        """Size the pool.

        *per_worker_bytes* is the caller's estimate of the host RAM one
        worker holds (interaction dicts + visual embeddings + the CUDA
        context).  It only applies to the auto-detected count: an
        explicit *n_workers* is honoured verbatim, because pinning the
        pool is how a researcher overrides the heuristic.
        """
        self.device = device
        self.log_dir = log_dir
        self.n_workers = (
            detect_max_workers(device, per_worker_bytes) if n_workers <= 0 else n_workers
        )
        logger.info("Training orchestrator: %d workers", self.n_workers)

    def run(self, jobs: list[TrainingJob]) -> list[dict]:
        if not jobs:
            return []

        jobs.sort(key=lambda j: (j.priority, j.dataset_name, j.embedding_name))

        if self.n_workers == 1 or self.device == "cpu":
            return self._run_sequential(jobs)
        return self._run_parallel(jobs)

    def _run_sequential(self, jobs: list[TrainingJob]) -> list[dict]:
        logger.info("Running %d jobs sequentially.", len(jobs))
        job_queue = mp.Queue()
        result_queue = mp.Queue()
        for job in jobs:
            job_queue.put(job)
        job_queue.put(None)
        _worker_fn(0, job_queue, result_queue, 1, self.log_dir)
        results = []
        while not result_queue.empty():
            results.append(result_queue.get())
        return results

    def _run_parallel(self, jobs: list[TrainingJob]) -> list[dict]:
        logger.info("Running %d jobs with %d workers.", len(jobs), self.n_workers)

        ctx = mp.get_context("spawn")
        job_queue = ctx.Queue()
        result_queue = ctx.Queue()

        for job in jobs:
            job_queue.put(job)
        for _ in range(self.n_workers):
            job_queue.put(None)

        workers = []
        for i in range(self.n_workers):
            p = ctx.Process(
                target=_worker_fn,
                args=(i, job_queue, result_queue, self.n_workers, self.log_dir),
                daemon=True,
            )
            p.start()
            workers.append(p)

        results = []
        completed = 0
        total = len(jobs)
        oom_retry: list[TrainingJob] = []
        start_time = time.time()
        last_log_time = start_time

        while completed < total:
            try:
                result = result_queue.get(timeout=30)
            except Empty:
                alive = sum(1 for w in workers if w.is_alive())
                if alive == 0:
                    logger.warning("All workers exited.")
                    break
                elapsed = time.time() - start_time
                eta_h = (elapsed / max(completed, 1)) * (total - completed) / 3600
                logger.info(
                    "Progress: %d/%d (%.1f%%) | %d workers | ETA: ~%.1f h",
                    completed,
                    total,
                    100 * completed / total,
                    alive,
                    eta_h,
                )
                last_log_time = time.time()
                continue

            completed += 1

            now = time.time()
            if now - last_log_time >= 30:
                elapsed = now - start_time
                eta_h = (elapsed / completed) * (total - completed) / 3600
                alive = sum(1 for w in workers if w.is_alive())
                logger.info(
                    "Progress: %d/%d (%.1f%%) | %d workers | ETA: ~%.1f h",
                    completed,
                    total,
                    100 * completed / total,
                    alive,
                    eta_h,
                )
                last_log_time = now

            if result["status"] == "oom":
                retry_count = result.get("retry_count", 0) + 1
                if retry_count <= MAX_OOM_RETRIES:
                    for job in jobs:
                        if job.job_id == result["job_id"]:
                            job.retry_count = retry_count
                            oom_retry.append(job)
                            break
                else:
                    logger.error("Unrecoverable OOM: %s", result["job_id"])
            else:
                results.append(result)

        for w in workers:
            w.join(timeout=30)

        if oom_retry:
            logger.info("Retrying %d OOM jobs sequentially...", len(oom_retry))
            results.extend(self._run_sequential(oom_retry))

        elapsed_h = (time.time() - start_time) / 3600
        ok = sum(1 for r in results if r.get("status") == "ok")
        logger.info("Done: %d/%d succeeded in %.1f h.", ok, total, elapsed_h)

        return results
