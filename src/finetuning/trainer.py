"""Generic fine-tuner for visual extractors via category classification."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.finetuning.checkpoint import split_state_dict
from src.utils.amp_compat import cuda_autocast, get_grad_scaler
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
    ) -> None:
        self.device = torch.device(device)
        self.extractor_name = extractor_name
        self.n_classes = n_classes
        self.unfreeze_prefixes = list(unfreeze_prefixes)
        self.config = config

        self.model = backbone.to(self.device)

        proj = self.model.projection
        if isinstance(proj, nn.Sequential):
            in_features = proj[0].in_features
        else:
            in_features = proj.in_features
        self._proj_in_features = in_features

        self.model.projection = nn.Linear(in_features, n_classes).to(self.device)

        for param in self.model.parameters():
            param.requires_grad = False

        for name, param in self.model.named_parameters():
            if any(name.startswith(p) for p in self.unfreeze_prefixes) or "projection" in name:
                param.requires_grad = True

        # Always unfreeze the classification head
        for param in self.model.projection.parameters():
            param.requires_grad = True

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

        best_acc = 0.0
        best_state = None
        epochs_no_improve = 0
        start_epoch = 0
        last_epoch = -1
        early_stopped = False

        if checkpoint_path is not None:
            ckpt_file = Path(checkpoint_path)
            if ckpt_file.exists():
                ckpt = torch.load(ckpt_file, map_location="cpu", weights_only=False)
                self.model.load_state_dict(ckpt["model_state"])
                optimizer.load_state_dict(ckpt["optimizer_state"])
                scheduler.load_state_dict(ckpt["scheduler_state"])
                start_epoch = ckpt["epoch"] + 1
                best_acc = ckpt["best_acc"]
                epochs_no_improve = ckpt.get("epochs_no_improve", 0)
                logger.info(
                    "  Resumed fine-tuning from epoch %d (best_acc=%.4f)",
                    start_epoch,
                    best_acc,
                )

        for epoch in range(start_epoch, epochs_max):
            self.model.train()
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
            if val_acc > best_acc:
                best_acc = val_acc
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if checkpoint_path is not None:
                tmp = Path(checkpoint_path).with_suffix(".tmp")
                torch.save(
                    {
                        "model_state": self.model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "scheduler_state": scheduler.state_dict(),
                        "epoch": epoch,
                        "best_acc": best_acc,
                        "epochs_no_improve": epochs_no_improve,
                    },
                    tmp,
                )
                tmp.rename(checkpoint_path)

            if epochs_no_improve >= patience:
                logger.info("  Early stopping at epoch %d", epoch + 1)
                early_stopped = True
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        # Clean up the resume checkpoint (the persistent fine-tuned weights
        # are saved by the caller via ``src.finetuning.checkpoint``).
        if checkpoint_path is not None and Path(checkpoint_path).exists():
            Path(checkpoint_path).unlink()

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
                correct += (logits.argmax(1) == labels).sum().item()
                total += len(labels)
        return correct / max(total, 1)
