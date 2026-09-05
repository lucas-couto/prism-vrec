"""Generic fine-tuner for visual extractors via category classification."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.finetuning.checkpoint import (
    best_weights_path,
    finetune_resume_identity,
    load_best_weights,
    load_resume_envelope,
    save_best_weights,
    save_resume_envelope,
    split_state_dict,
)
from src.utils import flops, telemetry
from src.utils.amp_compat import cuda_autocast, get_grad_scaler
from src.utils.checkpoint import capture_rng_states, restore_rng_states
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class FineTuningResult:
    """Output of :meth:`FineTuner.train`.

    Carries the trained model alongside its decomposed state so callers
    can persist the backbone and the classification head separately
    without having to re-introspect the module hierarchy.
    """

    model: nn.Module
    backbone_state: dict
    head_state: dict
    in_features: int
    n_classes: int
    best_val_acc: float
    epochs_trained: int
    early_stopped: bool


class FineTuner:
    """Fine-tunes a visual extractor backbone on category classification.

    The trainer is *extractor-agnostic*: it does not know anything about
    the architecture being fine-tuned beyond the contract documented on
    :class:`src.extractors.base.BaseExtractor`.  Specifically, the
    backbone must expose a submodule named ``projection`` whose
    ``in_features`` matches the pooled-feature size of the network — the
    trainer replaces that projection with a fresh classification head
    and uses the supplied *unfreeze_prefixes* to decide which other
    submodules to keep trainable.

    Parameters
    ----------
    backbone:
        The :class:`nn.Module` backbone from the extractor.  Its
        ``projection`` layer will be replaced with a classification head.
    extractor_name:
        Plain name used in log messages.  Has no functional effect — the
        trainer no longer looks anything up by name.
    n_classes:
        Number of category classes.
    unfreeze_prefixes:
        Module-name prefixes that should remain trainable.  Empty list
        keeps the backbone frozen and trains only the classification
        head.
    device:
        Torch device.
    config:
        Fine-tuning hyperparameters dict.
    """

    def __init__(
        self,
        backbone: nn.Module,
        extractor_name: str,
        n_classes: int,
        unfreeze_prefixes: list[str],
        device: str | torch.device,
        config: dict,
        in_features: int | None = None,
    ) -> None:
        self.device = torch.device(device)
        self.extractor_name = extractor_name
        self.n_classes = n_classes
        self.unfreeze_prefixes = list(unfreeze_prefixes)
        self.config = config

        self.model = backbone.to(self.device)

        # v2 backbones default projection to nn.Identity (extraction emits
        # the native feature), so the head size must be given explicitly
        # (the extractor's probed native_dim). Legacy Linear/Sequential
        # projections are still introspected for backward compatibility.
        if in_features is None:
            proj = self.model.projection
            if isinstance(proj, nn.Sequential):
                in_features = proj[0].in_features
            elif isinstance(proj, nn.Linear):
                in_features = proj.in_features
            else:
                raise ValueError(
                    "FineTuner needs in_features when the backbone projection "
                    f"is {type(proj).__name__} (pass extractor.native_dim)."
                )
        self._proj_in_features = in_features

        self.model.projection = nn.Linear(in_features, n_classes).to(self.device)

        for param in self.model.parameters():
            param.requires_grad = False

        for name, param in self.model.named_parameters():
            if self._is_unfrozen_name(name):
                param.requires_grad = True

        # Always unfreeze the classification head
        for param in self.model.projection.parameters():
            param.requires_grad = True

        # BatchNorm modules OUTSIDE the unfrozen prefixes must not update
        # their running statistics: ``model.train()`` alone would let
        # every BN layer re-estimate running_mean/var on the fine-tuning
        # data even though its affine weights are frozen, silently
        # rewriting the "frozen" part of the backbone (measured drift on
        # LeViT-256 stem BN: >12 sigma after a single epoch).  BN-dense
        # backbones (LeViT, ResNet, CoAtNet) are corrupted the most;
        # LayerNorm backbones are unaffected (LN has no running stats).
        self._frozen_norms = [
            module
            for name, module in self.model.named_modules()
            if isinstance(module, _BatchNorm)
            and module.track_running_stats
            and not self._is_unfrozen_name(name)
        ]
        if self._frozen_norms:
            logger.info(
                "FineTuner: %d frozen BatchNorm layers pinned to eval mode "
                "(running stats preserved).",
                len(self._frozen_norms),
            )

        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        logger.info(
            "FineTuner: %s, %d classes, %d/%d params trainable (%.1f%%)",
            extractor_name,
            n_classes,
            trainable,
            total,
            100 * trainable / total,
        )

    def _is_unfrozen_name(self, name: str) -> bool:
        """Whether a parameter/module qualified name is in the trainable set."""
        return any(name.startswith(p) for p in self.unfreeze_prefixes) or "projection" in name

    def _set_train_mode(self) -> None:
        """Enter train mode while keeping frozen BatchNorm layers in eval.

        Must be used instead of a bare ``self.model.train()`` — see the
        ``_frozen_norms`` comment in ``__init__``.
        """
        self.model.train()
        for module in self._frozen_norms:
            module.eval()

    def _account_flops(self, images: torch.Tensor, *, training: bool = False) -> None:
        """Attribute this batch's compute to the run's telemetry counters.

        Training batches are charged ``TRAINING_MULTIPLIER x`` the
        forward cost to account for the backward pass; validation
        batches are charged the forward only.
        """
        key = f"{self.extractor_name}::finetune::{tuple(images.shape[1:])}"
        flops.calibrate(key, self.model, images[:1])
        n = int(images.shape[0])
        flops.record(key, n, training=training)
        telemetry.add_items(n)

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        checkpoint_path: str | Path | None = None,
    ) -> FineTuningResult:
        """Run fine-tuning and return the trained model + decomposed state.

        The classification head stays attached to ``self.model`` after
        training (so the post-hoc evaluator can reload it), and the
        returned :class:`FineTuningResult` carries the backbone and head
        state_dicts already split for downstream persistence.
        """
        missing = [
            key
            for key in ("learning_rate", "weight_decay", "epochs_max", "patience")
            if key not in self.config
        ]
        if missing:
            # A typo'd key in configs/finetuning.yaml would otherwise
            # silently train with hidden defaults while the researcher
            # believes their configured hyperparameters were used.
            logger.warning(
                "Fine-tuning config is missing %s; using built-in defaults for them.",
                ", ".join(missing),
            )
        lr = self.config.get("learning_rate", 1e-4)
        weight_decay = self.config.get("weight_decay", 1e-4)
        epochs_max = self.config.get("epochs_max", 15)
        patience = self.config.get("patience", 5)

        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=lr,
            weight_decay=weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs_max)
        criterion = nn.CrossEntropyLoss()
        use_amp = self.device.type == "cuda"
        scaler = get_grad_scaler(enabled=use_amp)

        identity = finetune_resume_identity(
            extractor_name=self.extractor_name,
            n_classes=self.n_classes,
            in_features=self._proj_in_features,
            unfreeze_prefixes=self.unfreeze_prefixes,
            config=self.config,
            use_amp=use_amp,
        )

        best_acc = 0.0
        best_state: dict | None = None
        best_epoch: int | None = None
        best_ref: dict | None = None
        epochs_no_improve = 0
        start_epoch = 0
        last_epoch = -1
        early_stopped = False

        if checkpoint_path is not None and Path(checkpoint_path).exists():
            resumed = self._load_resume(
                Path(checkpoint_path),
                identity=identity,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
            )
            start_epoch = resumed["epoch"] + 1
            best_acc = float(resumed["best_metric"])
            best_epoch = resumed["best_epoch"]
            best_ref = resumed["best_ref"]
            best_state = resumed["best_state"]
            epochs_no_improve = resumed["epochs_no_improve"]

        for epoch in range(start_epoch, epochs_max):
            self._set_train_mode()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for images, labels in tqdm(
                train_loader,
                desc=f"FT epoch {epoch + 1}/{epochs_max}",
                leave=False,
            ):
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                with cuda_autocast(enabled=use_amp):
                    logits = self.model(images)
                    loss = criterion(logits, labels)
                self._account_flops(images, training=True)

                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                train_loss += loss.item() * len(labels)
                train_correct += (logits.argmax(1) == labels).sum().item()
                train_total += len(labels)

            scheduler.step()

            val_acc = self._validate(val_loader, criterion, use_amp)
            train_acc = train_correct / max(train_total, 1)

            logger.info(
                "  Epoch %d/%d: train_acc=%.4f, val_acc=%.4f (best=%.4f)",
                epoch + 1,
                epochs_max,
                train_acc,
                val_acc,
                best_acc,
            )

            last_epoch = epoch
            # First observation wins (even 0.0 — Q06), strict improvement
            # afterwards; a tie keeps the earlier best and counts against
            # patience.
            if best_state is None or val_acc > best_acc:
                best_acc = val_acc
                best_epoch = epoch
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                epochs_no_improve = 0
                if checkpoint_path is not None:
                    # Committed and digest-bound BEFORE the envelope below
                    # references it.
                    best_ref = save_best_weights(
                        best_weights_path(checkpoint_path),
                        best_state,
                        identity=identity,
                        epoch=epoch,
                        val_acc=val_acc,
                    )
            else:
                epochs_no_improve += 1

            if checkpoint_path is not None:
                save_resume_envelope(
                    checkpoint_path,
                    {
                        "identity": identity,
                        "model_state": self.model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "scheduler_state": scheduler.state_dict(),
                        "scaler_state": scaler.state_dict(),
                        "rng_states": capture_rng_states(),
                        "epoch": epoch,
                        "has_valid_observation": best_state is not None,
                        "best_metric": best_acc,
                        "best_epoch": best_epoch,
                        "best_ref": best_ref,
                        "epochs_no_improve": epochs_no_improve,
                    },
                )

            if epochs_no_improve >= patience:
                logger.info("  Early stopping at epoch %d", epoch + 1)
                early_stopped = True
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        # Clean up the resume checkpoint and its best-weights sibling (the
        # persistent fine-tuned weights are saved by the caller via
        # ``src.finetuning.checkpoint``).
        if checkpoint_path is not None:
            for stale in (Path(checkpoint_path), best_weights_path(checkpoint_path)):
                if stale.exists():
                    stale.unlink()

        backbone_state, head_state = split_state_dict(self.model.state_dict())
        epochs_trained = last_epoch + 1 if last_epoch >= 0 else 0

        logger.info("  Fine-tuning complete: best_val_acc=%.4f", best_acc)
        return FineTuningResult(
            model=self.model,
            backbone_state=backbone_state,
            head_state=head_state,
            in_features=self._proj_in_features,
            n_classes=self.n_classes,
            best_val_acc=best_acc,
            epochs_trained=epochs_trained,
            early_stopped=early_stopped,
        )

    def _load_resume(
        self,
        ckpt_file: Path,
        *,
        identity: str,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        scaler,
    ) -> dict:
        """Validate and load a v2 resume envelope into the live objects.

        Envelope, identity and the referenced best-weights file are
        checked BEFORE anything is mutated.  The historical best is
        loaded into memory (CPU) so the end of training can return it
        even when no later epoch improves; it is never restored into the
        live model here — that would change the training trajectory.
        """
        ckpt = load_resume_envelope(ckpt_file, identity=identity)
        best_state = None
        if ckpt["has_valid_observation"]:
            best_state = load_best_weights(
                ckpt["best_ref"], identity=identity, source=f"resume {ckpt_file}"
            )
        self.model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        if scaler.is_enabled():
            scaler.load_state_dict(ckpt["scaler_state"])
        # Same augmentation / shuffle sequence as an uninterrupted run
        # (bit-identical resume).
        restore_rng_states(ckpt["rng_states"])
        logger.info(
            "  Resumed fine-tuning from epoch %d (best_acc=%.4f at epoch %s)",
            ckpt["epoch"] + 1,
            float(ckpt["best_metric"]),
            ckpt["best_epoch"],
        )
        return {**ckpt, "best_state": best_state}

    def _validate(
        self,
        val_loader: DataLoader,
        criterion: nn.Module,
        use_amp: bool,
    ) -> float:
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                with cuda_autocast(enabled=use_amp):
                    logits = self.model(images)
                self._account_flops(images)
                correct += (logits.argmax(1) == labels).sum().item()
                total += len(labels)
        return correct / max(total, 1)
