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

logger = get_logger(__name__)


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


def detect_max_workers(device: str = "cuda") -> int:
    """Estimate how many training workers fit in GPU VRAM.

    Uses a simple heuristic based on total VRAM rather than dummy-model
    profiling, because real datasets (100K+ items) use far more memory
    than any small dummy can predict.
    """
    if device == "cpu" or not torch.cuda.is_available():
        return max(1, (os.cpu_count() or 4) - 1)

    try:
        total_mb = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
    except Exception as exc:
        logger.warning("VRAM detection failed (%s), defaulting to 2.", exc)
        return 2

    # ~4 GB per worker: real models with 100K+ items need dedicated GPU bandwidth
    mb_per_worker = 4096
    margin_mb = 1024
    available_mb = total_mb - margin_mb
    n_workers = max(1, int(available_mb / mb_per_worker))
    n_workers = min(n_workers, max(1, (os.cpu_count() or 4) - 1))

    logger.info(
        "VRAM: total=%.0f MB, ~%d MB/worker, margin=%d MB → %d workers",
        total_mb,
        mb_per_worker,
        margin_mb,
        n_workers,
    )
    return n_workers


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

    if torch.cuda.is_available() and n_workers > 1:
        fraction = min(0.95, 1.0 / n_workers + 0.05)
        torch.cuda.set_per_process_memory_fraction(fraction)

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
        result = (n_users, n_items, train_inter, val_inter)
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

            n_users, n_items, train_inter, val_inter = _load_data(
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

            best_val = train_single_run(
                model_cls=model_cls,
                model_name=job.model_name,
                n_users=n_users,
                n_items=n_items,
                visual_embeddings=visual_emb,
                train_interactions=train_inter,
                test_interactions=val_inter,
                hyperparams=job.hyperparams,
                config=config,
                checkpoint_mgr=checkpoint_mgr,
                dataset_name=job.dataset_name,
                embedding_name=job.embedding_name,
                device=job.device,
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
    ) -> None:
        self.device = device
        self.log_dir = log_dir
        self.n_workers = detect_max_workers(device) if n_workers <= 0 else n_workers
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
                if retry_count <= 2:
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
