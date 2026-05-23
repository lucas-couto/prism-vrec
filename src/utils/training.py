"""Single training run logic, extracted for use by parallel workers."""

import fcntl
import hashlib
import json
import multiprocessing
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from src.evaluation.protocol import Evaluator
from src.utils.amp_compat import cuda_autocast, get_grad_scaler
from src.utils.checkpoint import (
    CheckpointManager,
    capture_rng_states,
    restore_rng_states,
)
from src.utils.seed import set_seed


def _derive_job_seed(
    base_seed: int,
    dataset_name: str,
    model_name: str,
    embedding_name: str,
    hyperparams: dict,
) -> int:
    """Derive a deterministic seed from job identity.

    Ensures that the same (dataset, model, embedding, hyperparams) tuple
    always starts from the same PRNG state, independently of the parallel
    execution order. Different tuples get uncorrelated seeds.
    """
    key = json.dumps(
        {
            "dataset": dataset_name,
            "model": model_name,
            "embedding": embedding_name,
            "hyperparams": hyperparams,
        },
        sort_keys=True,
    )
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:8]
    return (int(digest, 16) ^ base_seed) & 0x7FFFFFFF


class BPRDataset(Dataset):
    """Dataset para amostragem BPR (user, pos_item, neg_item)."""

    def __init__(self, train_interactions: dict, n_items: int):
        self.n_items = n_items
        self.interactions = []
        self.user_items = train_interactions
        for user, items in train_interactions.items():
            for item in items:
                self.interactions.append((user, item))

    def __len__(self):
        return len(self.interactions)

    def __getitem__(self, idx):
        user, pos_item = self.interactions[idx]
        neg_item = random.randint(0, self.n_items - 1)
        while neg_item in self.user_items[user]:
            neg_item = random.randint(0, self.n_items - 1)
        return user, pos_item, neg_item


def _best_effort_resume_checkpoint(checkpoint_mgr, logger, **kwargs) -> None:
    """Save the per-trial resume checkpoint, tolerating I/O failures.

    The resume checkpoint only lets a killed trial continue instead of
    restarting; it never affects the trial's result (metrics come from
    evaluation, best weights are saved separately by ``_save_best_model``).
    On some networked filesystems a freshly written temp file can be
    invisible to the subsequent rename even after fsync+retry, raising
    ``OSError`` — that must not propagate and kill the whole run. Log and
    continue.  Non-``OSError`` failures are real bugs and still surface.
    """
    try:
        checkpoint_mgr.save_training_checkpoint(**kwargs)
    except OSError as exc:
        logger.warning(
            "resume checkpoint save failed (non-fatal, trial continues): %s",
            exc,
        )


def _save_best_model(
    model,
    hyperparams: dict,
    metric: float,
    n_users: int,
    n_items: int,
    dataset_name: str,
    model_name: str,
    embedding_name: str,
    results_root: str | Path = "results",
) -> None:
    """Save model weights only if metric beats the existing best on disk."""
    best_model_path = (
        Path(results_root) / "models" / dataset_name / f"{model_name}_{embedding_name}_best.pt"
    )
    best_model_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = best_model_path.with_suffix(".lock")

    with open(lock_path, "w") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        try:
            # Corrupted files left behind by SIGKILL during a prior save
            # are treated as absent (will be overwritten).
            if best_model_path.exists():
                try:
                    existing = torch.load(best_model_path, map_location="cpu", weights_only=False)
                    if existing.get("best_metric", 0.0) >= metric:
                        return
                except (RuntimeError, EOFError, OSError):
                    pass

            # Atomic save: write a temp file in the same directory, then
            # rename into place. Prevents leaving a half-written file
            # behind if the process is killed mid-write.
            tmp_path = best_model_path.with_suffix(".pt.tmp")
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "hyperparams": hyperparams,
                    "best_metric": metric,
                    "n_users": n_users,
                    "n_items": n_items,
                },
                tmp_path,
            )
            tmp_path.rename(best_model_path)
        finally:
            fcntl.flock(lf, fcntl.LOCK_UN)


def train_single_run(
    model_cls,
    model_name: str,
    n_users: int,
    n_items: int,
    visual_embeddings,
    train_interactions: dict,
    test_interactions: dict,
    hyperparams: dict,
    config: dict,
    checkpoint_mgr: CheckpointManager,
    dataset_name: str,
    embedding_name: str,
    device: str,
    optuna_trial=None,
) -> float:
    """Train a single model with one hyperparameter configuration.

    Returns the best validation metric achieved.

    Parameters
    ----------
    optuna_trial:
        Optional ``optuna.Trial``.  When supplied, the validation
        metric is reported every ``eval_every_epochs`` and the loop
        raises :class:`optuna.TrialPruned` whenever the configured
        Optuna pruner decides the trial is not promising.
    """
    from src.utils.logging import get_logger

    logger = get_logger(f"train_{model_name}")

    run_id = checkpoint_mgr.get_run_id(dataset_name, embedding_name, model_name, hyperparams)
    epochs = config.get("common", {}).get("epochs", 100)
    batch_size = config.get("common", {}).get("batch_size", 4096)
    patience = config.get("common", {}).get("early_stopping_patience", 10)
    es_metric = config.get("common", {}).get("early_stopping_metric", "ndcg@10")
    eval_every_epochs = config.get("common", {}).get("eval_every_epochs", 10)
    eval_sample_size = config.get("common", {}).get("eval_sample_size")
    base_seed = config.get("seed", 42)
    eval_sample_seed = base_seed

    # Reset all PRNGs deterministically per job so parallel execution order
    # does not affect results. Checkpoint-based resume (below) restores the
    # exact RNG state from the last completed epoch instead.
    job_seed = _derive_job_seed(
        base_seed,
        dataset_name,
        model_name,
        embedding_name,
        hyperparams,
    )
    set_seed(job_seed)

    model_config = {**hyperparams, "l2_reg": hyperparams.get("l2_reg", 0.0001)}

    model = model_cls(
        n_users=n_users,
        n_items=n_items,
        visual_embeddings=visual_embeddings,
        config=model_config,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams["learning_rate"])

    start_epoch = 0
    best_metric = 0.0
    epochs_without_improvement = 0

    ckpt = checkpoint_mgr.load_training_checkpoint(run_id)
    if ckpt is not None:
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = ckpt["epoch"] + 1
        best_metric = ckpt["best_metric"]
        epochs_without_improvement = ckpt.get("epochs_without_improvement", 0)
        if "rng_states" in ckpt:
            restore_rng_states(ckpt["rng_states"])

    train_dataset = BPRDataset(train_interactions, n_items)
    use_cuda = device != "cpu" and torch.cuda.is_available()
    is_daemon = getattr(multiprocessing.current_process(), "daemon", False)
    # Workers are respawned every epoch on purpose: BPRDataset.__getitem__
    # uses Python's `random` module for negative sampling, and respawning
    # resets each worker's RNG state so the same epoch always sees the
    # same sequence of negative samples.  Switching to persistent_workers
    # would change that sequence and break consistency with previously
    # completed runs.
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0 if is_daemon else 2,
        pin_memory=use_cuda,
    )

    scaler = get_grad_scaler(enabled=use_cuda)
    evaluator = Evaluator(
        train_interactions,
        test_interactions,
        n_items,
        k_values=[10],
        sample_size=eval_sample_size,
        sample_seed=eval_sample_seed,
    )

    loss_device = torch.device(device) if use_cuda else torch.device("cpu")

    try:
        for epoch in range(start_epoch, epochs):
            model.train()
            # Accumulate loss as a GPU tensor and sync to CPU only once per
            # epoch to avoid per-batch GPU↔CPU stalls from .item() calls.
            total_loss = torch.zeros((), device=loss_device)
            n_batches = 0

            for users, pos_items, neg_items in train_loader:
                users = users.to(device, non_blocking=True)
                pos_items = pos_items.to(device, non_blocking=True)
                neg_items = neg_items.to(device, non_blocking=True)

                with cuda_autocast(enabled=use_cuda):
                    score_pos, score_neg = model(users, pos_items, neg_items)
                    loss = model.bpr_loss(score_pos, score_neg)

                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                total_loss += loss.detach()
                n_batches += 1

            avg_loss = (total_loss / max(n_batches, 1)).item()
            logger.debug("epoch=%d avg_loss=%.6f", epoch, avg_loss)

            if (epoch + 1) % eval_every_epochs == 0 or epoch == epochs - 1:
                metrics = evaluator.evaluate(model, device=device)
                current_metric = metrics.get(es_metric, 0.0)

                if current_metric > best_metric:
                    best_metric = current_metric
                    epochs_without_improvement = 0
                    _save_best_model(
                        model,
                        hyperparams,
                        current_metric,
                        n_users,
                        n_items,
                        dataset_name,
                        model_name,
                        embedding_name,
                        results_root=config.get("paths", {}).get("results", "results"),
                    )
                else:
                    epochs_without_improvement += eval_every_epochs

                if optuna_trial is not None:
                    optuna_trial.report(current_metric, step=epoch)
                    if optuna_trial.should_prune():
                        import optuna  # noqa: WPS433

                        raise optuna.TrialPruned()

                if epochs_without_improvement >= patience:
                    break

            _best_effort_resume_checkpoint(
                checkpoint_mgr,
                logger,
                run_id=run_id,
                epoch=epoch,
                model_state=model.state_dict(),
                optimizer_state=optimizer.state_dict(),
                best_metric=best_metric,
                epochs_without_improvement=epochs_without_improvement,
                rng_states=capture_rng_states(),
            )

        return best_metric
    finally:
        # Clean up the per-trial resume checkpoint on every exit path
        # (normal completion, early stopping, Optuna prune, exception).
        # Without this, checkpoints/training/ grows unboundedly: hundreds
        # of MB per trial × thousands of trials per pipeline run. Trials
        # already complete have their best weights in results/models/
        # and their metrics in optuna.db, so the resume state is dead
        # weight.
        checkpoint_mgr.clear_training_checkpoint(run_id)
