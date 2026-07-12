"""Post-hoc evaluation of fine-tuned extractors on the category task.

Loads a fine-tuning checkpoint (backbone + classification head + metadata),
deterministically reconstructs the validation split using the seed stored
in the checkpoint metadata, runs a forward pass on the val images and
computes a richer metric set than what is logged during training:

* top-1 / top-5 accuracy
* macro-averaged F1, precision, recall
* weighted F1
* per-class precision / recall / F1 / support
* confusion matrix
* mean cross-entropy loss

The evaluator is read-only: it never trains, never overwrites checkpoints
and never modifies the disk state of the rest of the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader

from src.data.base import get_dataset_provider
from src.extractors import get_extractor_class
from src.finetuning.checkpoint import HEAD_PREFIX, load_finetuned
from src.finetuning.dataset import CategoryDataset
from src.utils.amp_compat import cuda_autocast
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class EvaluationReport:
    """Structured output of :meth:`FineTuningEvaluator.evaluate`."""

    extractor_name: str
    dataset_name: str
    n_classes: int
    n_val_samples: int
    metrics: dict[str, Any]
    training_summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "extractor": self.extractor_name,
            "dataset": self.dataset_name,
            "n_classes": self.n_classes,
            "n_val_samples": self.n_val_samples,
            "metrics": self.metrics,
            "training": self.training_summary,
        }


class CheckpointMissingHeadError(RuntimeError):
    """Raised when the checkpoint was saved without a classification head.

    Legacy checkpoints (flat state_dict, pre-v2 format) cannot be evaluated
    because the head was discarded before saving.  Re-run the fine-tuning
    step with the current code to produce a v2 checkpoint.
    """


class FineTuningEvaluator:
    """Recompute classification metrics for a single fine-tuned extractor.

    Parameters
    ----------
    extractor_name:
        Registered extractor name (must match the checkpoint).
    dataset_name:
        Dataset name; used to fetch the same categories that produced the
        original train/val split.
    image_dir:
        Directory holding the per-item JPEGs used during training.
    checkpoint_path:
        Path to the fine-tuning checkpoint (v2 format).
    device:
        Torch device.
    batch_size:
        Forward-pass batch size.
    num_workers:
        DataLoader workers.
    """

    def __init__(
        self,
        extractor_name: str,
        dataset_name: str,
        image_dir: str | Path,
        checkpoint_path: str | Path,
        device: str = "cuda",
        batch_size: int = 256,
        num_workers: int = 8,
    ) -> None:
        self.extractor_name = extractor_name
        self.dataset_name = dataset_name
        self.image_dir = str(image_dir)
        self.checkpoint_path = Path(checkpoint_path)
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.num_workers = num_workers

    def evaluate(self) -> EvaluationReport:
        backbone_state, head_state, metadata = load_finetuned(self.checkpoint_path)
        if head_state is None or metadata is None:
            raise CheckpointMissingHeadError(
                f"Checkpoint {self.checkpoint_path} is in the legacy flat format "
                "(no classification head). Re-run the fine-tune step to produce "
                "a v2 checkpoint."
            )

        n_classes = int(metadata["n_classes"])
        in_features = int(metadata["in_features"])
        if "split_seed" not in metadata:
            # A wrong seed silently reconstructs a different val split
            # than the one used at training time, corrupting the metrics.
            logger.warning(
                "Checkpoint metadata for %s/%s has no split_seed; assuming 42. "
                "Reported metrics are only valid if training used the same seed.",
                self.extractor_name,
                self.dataset_name,
            )
        split_seed = int(metadata.get("split_seed", 42))

        provider = get_dataset_provider(self.dataset_name)
        categories = provider.load_categories()
        if categories is None:
            raise ValueError(
                f"Dataset {self.dataset_name!r} has no category labels; "
                "post-hoc evaluation requires the labelled training set."
            )

        _, val_cats = CategoryDataset.stratified_split(categories, seed=split_seed)

        extractor_cls = get_extractor_class(self.extractor_name)
        extractor = extractor_cls(device=str(self.device), output_dim=in_features)
        model = extractor.model
        # Replace the freshly-built projection with a classification head
        # that matches the saved one before loading weights.
        model.projection = nn.Linear(in_features, n_classes).to(self.device)

        full_state = dict(backbone_state)
        full_state.update(head_state)
        missing, unexpected = model.load_state_dict(full_state, strict=False)
        if missing:
            # Missing keys mean parts of the network would evaluate with
            # randomly initialised weights, producing plausible-looking
            # but wrong accuracy numbers. Fail instead of reporting them.
            raise RuntimeError(
                f"Checkpoint for {self.extractor_name}/{self.dataset_name} is "
                f"missing {len(missing)} state-dict keys (e.g. {missing[:5]}); "
                "the extractor code likely drifted from the checkpoint format."
            )
        if unexpected:
            logger.warning(
                "Checkpoint has %d unexpected keys when loading %s/%s: %s",
                len(unexpected),
                self.extractor_name,
                self.dataset_name,
                unexpected[:5],
            )

        val_ds = CategoryDataset(
            self.image_dir,
            val_cats,
            transform=extractor.transform,
            augment=False,
        )
        if len(val_ds) == 0:
            raise RuntimeError(
                f"Empty validation set for {self.extractor_name}/{self.dataset_name}: "
                "check that the image directory matches the one used at training."
            )

        use_cuda = self.device.type == "cuda"
        loader = DataLoader(
            val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=use_cuda,
            persistent_workers=self.num_workers > 0,
        )

        metrics = self._forward_and_score(model, loader, n_classes, use_cuda)
        training_summary = {
            "best_val_acc_during_training": float(metadata.get("best_val_acc", 0.0)),
            "epochs_trained": int(metadata.get("epochs_trained", 0)),
            "early_stopped": bool(metadata.get("early_stopped", False)),
            "split_seed": split_seed,
            "format_version": str(metadata.get("format_version", "v2")),
        }

        return EvaluationReport(
            extractor_name=self.extractor_name,
            dataset_name=self.dataset_name,
            n_classes=n_classes,
            n_val_samples=len(val_ds),
            metrics=metrics,
            training_summary=training_summary,
        )

    def _forward_and_score(
        self,
        model: nn.Module,
        loader: DataLoader,
        n_classes: int,
        use_cuda: bool,
    ) -> dict[str, Any]:
        model.eval()
        criterion = nn.CrossEntropyLoss(reduction="sum")

        topk = min(5, n_classes)
        total = 0
        top1_correct = 0
        topk_correct = 0
        loss_sum = 0.0

        all_preds: list[np.ndarray] = []
        all_labels: list[np.ndarray] = []

        with torch.no_grad():
            for images, labels in loader:
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                with cuda_autocast(enabled=use_cuda):
                    logits = model(images)
                    loss = criterion(logits.float(), labels)

                loss_sum += loss.item()
                preds = logits.argmax(dim=1)
                top1_correct += int((preds == labels).sum().item())
                if topk > 1:
                    topk_pred = logits.topk(topk, dim=1).indices
                    topk_correct += int(topk_pred.eq(labels.unsqueeze(1)).any(dim=1).sum().item())
                else:
                    topk_correct = top1_correct

                total += int(labels.size(0))
                all_preds.append(preds.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        preds_arr = np.concatenate(all_preds) if all_preds else np.empty(0, dtype=np.int64)
        labels_arr = np.concatenate(all_labels) if all_labels else np.empty(0, dtype=np.int64)

        labels_full = list(range(n_classes))
        per_class_p, per_class_r, per_class_f1, support = precision_recall_fscore_support(
            labels_arr,
            preds_arr,
            labels=labels_full,
            zero_division=0,
        )
        macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
            labels_arr,
            preds_arr,
            labels=labels_full,
            average="macro",
            zero_division=0,
        )
        weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
            labels_arr,
            preds_arr,
            labels=labels_full,
            average="weighted",
            zero_division=0,
        )
        cm = confusion_matrix(labels_arr, preds_arr, labels=labels_full).tolist()

        return {
            "top1_acc": top1_correct / total if total else 0.0,
            f"top{topk}_acc": topk_correct / total if total else 0.0,
            "loss": loss_sum / total if total else 0.0,
            "macro_precision": float(macro_p),
            "macro_recall": float(macro_r),
            "macro_f1": float(macro_f1),
            "weighted_precision": float(weighted_p),
            "weighted_recall": float(weighted_r),
            "weighted_f1": float(weighted_f1),
            "per_class": {
                str(c): {
                    "precision": float(per_class_p[i]),
                    "recall": float(per_class_r[i]),
                    "f1": float(per_class_f1[i]),
                    "support": int(support[i]),
                }
                for i, c in enumerate(labels_full)
            },
            "confusion_matrix": cm,
        }


__all__ = [
    "CheckpointMissingHeadError",
    "EvaluationReport",
    "FineTuningEvaluator",
    "HEAD_PREFIX",
]
