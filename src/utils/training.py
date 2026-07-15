"""Single training run logic, extracted for use by parallel workers."""

import fcntl
import hashlib
import json
import time
from pathlib import Path

import torch

from src.evaluation.protocol import Evaluator
from src.utils.amp_compat import cuda_autocast, get_grad_scaler
from src.utils.atomic_io import atomic_write
from src.utils.checkpoint import (
    CheckpointManager,
    capture_rng_states,
    restore_rng_states,
)
from src.utils.logging import get_logger
from src.utils.seed import set_seed

logger = get_logger(__name__)


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


class BPRBatchSampler:
    """Vectorized in-process BPR triple batching (v2 training protocol).

    Replaces the former ``Dataset`` + ``DataLoader`` pair whose
    ``__getitem__`` rejection-sampled ONE negative at a time in pure
    Python — with ``num_workers=0`` in the parallel regime, every
    4096-sample batch was generated serially on the CPU, dominating
    the epoch time of shallow models (efficiency-audit bottleneck #4).

    Per epoch (deterministic given ``(seed, epoch)``): the interaction
    list is shuffled with a dedicated ``torch.Generator``; negatives are
    drawn per batch in bulk and collisions with the user's training set
    are re-drawn vectorized (membership via ``torch.isin`` on the sorted
    ``user * n_items + item`` key array — a C++ binary search, no Python
    loop).  This CHANGES the negative-sample sequence relative to v1.x
    (accepted: the v2 protocol re-runs every battery).
    """

    def __init__(self, train_interactions: dict, n_items: int, batch_size: int, seed: int):
        self.n_items = n_items
        self.batch_size = batch_size
        self.seed = seed

        users: list[int] = []
        items: list[int] = []
        for user, item_set in train_interactions.items():
            for item in item_set:
                users.append(user)
                items.append(item)
        self.users = torch.tensor(users, dtype=torch.long)
        self.pos_items = torch.tensor(items, dtype=torch.long)
        # Sorted composite keys for vectorized membership tests.
        self._keys = torch.sort(self.users * n_items + self.pos_items).values

    def __len__(self) -> int:
        return self.users.shape[0]

    def n_batches(self) -> int:
        return (len(self) + self.batch_size - 1) // self.batch_size

    def epoch(self, epoch: int):
        """Yield ``(users, pos, neg)`` batches for *epoch*, deterministically.

        The WHOLE epoch is drawn in one vectorized pass (single shuffle,
        single bulk negative draw, collision redraws over the shrinking
        collision set) and then sliced into batches — per-batch tensor-op
        launch overhead would otherwise dominate at small batch counts.
        """
        generator = torch.Generator()
        generator.manual_seed((self.seed * 1_000_003 + epoch) & 0x7FFF_FFFF_FFFF_FFFF)
        perm = torch.randperm(len(self), generator=generator)

        users = self.users[perm]
        pos = self.pos_items[perm]
        neg = torch.randint(0, self.n_items, (len(self),), generator=generator)
        user_keys = users * self.n_items
        collides = torch.isin(user_keys + neg, self._keys)
        while bool(collides.any()):
            n_bad = int(collides.sum())
            redraw = torch.randint(0, self.n_items, (n_bad,), generator=generator)
            neg[collides] = redraw
            still = torch.isin(user_keys[collides] + redraw, self._keys)
            new_collides = torch.zeros_like(collides)
            new_collides[collides] = still
            collides = new_collides

        for start in range(0, len(self), self.batch_size):
            end = start + self.batch_size
            yield users[start:end], pos[start:end], neg[start:end]


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
                except (RuntimeError, EOFError, OSError) as exc:
                    logger.warning(
                        "existing best model %s is unreadable (%r); overwriting it.",
                        best_model_path,
                        exc,
                    )

            payload = {
                "model_state": model.state_dict(),
                "hyperparams": hyperparams,
                "best_metric": metric,
                "n_users": n_users,
                "n_items": n_items,
            }
            # atomic_write adds fsync + retried replace on top of the
            # tmp+rename pattern (networked-FS dirent lag).
            atomic_write(lambda tmp, p=payload: torch.save(p, tmp), best_model_path)
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
    item_categories=None,
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

    # Only models that declare ``wants_history`` accept the
    # ``train_interactions`` keyword (e.g. ACF item-level attention), and
    # only ``wants_categories`` models (DeepStyle) receive the item→category
    # index array; every other recommender keeps the original 4-argument
    # constructor untouched.
    ctor_kwargs: dict = {}
    if getattr(model_cls, "wants_history", False):
        ctor_kwargs["train_interactions"] = train_interactions
    if getattr(model_cls, "wants_categories", False):
        ctor_kwargs["item_categories"] = item_categories

    model = model_cls(
        n_users=n_users,
        n_items=n_items,
        visual_embeddings=visual_embeddings,
        config=model_config,
        **ctor_kwargs,
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

    use_cuda = device != "cpu" and torch.cuda.is_available()
    # In-process vectorized sampler: no DataLoader workers to spawn, no
    # per-sample Python rejection loop.  Seeded from the job seed so the
    # negative sequence is deterministic and independent of parallel
    # execution order (each epoch draws from Generator(seed, epoch)).
    sampler = BPRBatchSampler(train_interactions, n_items, batch_size, seed=job_seed)

    scaler = get_grad_scaler(enabled=use_cuda)
    evaluator = Evaluator(
        train_interactions,
        test_interactions,
        n_items,
        k_values=[10],
        sample_size=eval_sample_size,
        sample_seed=eval_sample_seed,
        tiebreak_seed=base_seed,
    )

    loss_device = torch.device(device) if use_cuda else torch.device("cpu")

    try:
        for epoch in range(start_epoch, epochs):
            model.train()
            # Accumulate loss as a GPU tensor and sync to CPU only once per
            # epoch to avoid per-batch GPU↔CPU stalls from .item() calls.
            total_loss = torch.zeros((), device=loss_device)
            n_batches = 0

            train_t0 = time.perf_counter()
            for users, pos_items, neg_items in sampler.epoch(epoch):
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
            train_seconds = time.perf_counter() - train_t0
            logger.debug("epoch=%d avg_loss=%.6f", epoch, avg_loss)

            if (epoch + 1) % eval_every_epochs == 0 or epoch == epochs - 1:
                eval_t0 = time.perf_counter()
                metrics = evaluator.evaluate(model, device=device)
                eval_seconds = time.perf_counter() - eval_t0
                # Train-vs-eval split per model: the number the efficiency
                # audit could not answer without instrumentation.
                logger.info(
                    "timing model=%s epoch=%d train_s=%.2f eval_s=%.2f",
                    model_name,
                    epoch,
                    train_seconds,
                    eval_seconds,
                )
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
