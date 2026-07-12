"""Step 03b, Fine-tune extractors via category classification.

For each ``(extractor, dataset)`` combination this step:
1. Loads the original pretrained backbone via the extractor registry.
2. Replaces the projection head with a category-classification head.
3. Fine-tunes the unfrozen layers on the dataset's category labels.
4. Re-extracts embeddings at every projection dim configured in
   ``configs/extractors.yaml``.

Datasets without category labels (Tradesy) reuse the fine-tuned
backbone of ``ft_config["tradesy_transfer_from"]`` (default
``amazon_fashion``).

Failures are isolated per ``(extractor, dataset)`` so a single OOM /
layer-name mismatch does not abort the entire queue.
"""

from __future__ import annotations

import gc
import traceback
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data import dvbpr  # noqa: F401
from src.data.base import get_dataset_provider
from src.extractors import get_extractor_class, is_registered
from src.finetuning.checkpoint import (
    FineTuningMetadata,
    load_finetuned,
    save_finetuned,
)
from src.finetuning.dataset import CategoryDataset
from src.finetuning.trainer import FineTuner
from src.steps.extract import ImageDataset, get_item_ids
from src.utils.config import load_config
from src.utils.dataloader import resolve_dataloader_settings
from src.utils.device import resolve_device
from src.utils.logging import get_logger
from src.utils.seed import set_seed
from src.utils.timing import time_cell

logger = get_logger(__name__)


def _finetune_and_extract(
    extractor_name: str,
    dataset_name: str,
    image_dir: str,
    categories: dict,
    n_classes: int,
    projection_dims: list[int],
    embeddings_dir: str,
    ft_config: dict,
    device: str,
    split_seed: int,
    backbone_weights_path: Path | None = None,
) -> None:
    """Fine-tune one extractor and extract embeddings at every projection dim."""

    all_exist = all(
        (Path(embeddings_dir) / dataset_name / f"{extractor_name}_finetuned_D{dim}.npy").exists()
        for dim in projection_dims
    )
    if all_exist:
        logger.info("  %s finetuned embeddings already exist, skipping.", extractor_name)
        return

    weights_path = Path(f"checkpoints/finetuning/{extractor_name}_{dataset_name}_best.pt")
    weights_path.parent.mkdir(parents=True, exist_ok=True)

    extractor_cls = get_extractor_class(extractor_name)

    if backbone_weights_path is not None and backbone_weights_path.exists():
        logger.info("  Loading transferred weights from %s", backbone_weights_path)
        backbone_state, _, _ = load_finetuned(backbone_weights_path)
    elif weights_path.exists():
        logger.info("  Loading existing fine-tuned weights from %s", weights_path)
        backbone_state, _, _ = load_finetuned(weights_path)
    else:
        logger.info(
            "  Fine-tuning %s on %s (%d classes)...",
            extractor_name,
            dataset_name,
            n_classes,
        )

        extractor = extractor_cls(device=device, output_dim=projection_dims[0])
        backbone = extractor.model

        train_cats, val_cats = CategoryDataset.stratified_split(
            categories,
            seed=split_seed,
        )

        train_ds = CategoryDataset(
            image_dir, train_cats, transform=extractor.transform, augment=True
        )
        val_ds = CategoryDataset(image_dir, val_cats, transform=extractor.transform, augment=False)

        if len(train_ds) == 0 or len(val_ds) == 0:
            # Fail loudly: an empty loader would run 0 batches per epoch,
            # early-stop at val_acc=0.0 and save untouched weights labeled
            # as a fine-tuned model, silently poisoning re-extraction.
            raise RuntimeError(
                f"Empty fine-tuning dataset for '{dataset_name}' "
                f"(train={len(train_ds)}, val={len(val_ds)} items) — "
                f"check image_dir {image_dir}."
            )

        batch_size = int(ft_config.get("batch_size", 128))
        use_cuda = device != "cpu" and torch.cuda.is_available()
        loader_settings = resolve_dataloader_settings(load_config())
        num_workers = loader_settings.num_workers
        prefetch = loader_settings.prefetch_factor

        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=use_cuda,
            persistent_workers=num_workers > 0,
            prefetch_factor=prefetch if num_workers > 0 else None,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=use_cuda,
            persistent_workers=num_workers > 0,
            prefetch_factor=prefetch if num_workers > 0 else None,
        )

        ckpt_path = f"checkpoints/finetuning/{extractor_name}_{dataset_name}_ckpt.pt"
        finetuner = FineTuner(
            backbone=backbone,
            extractor_name=extractor_name,
            n_classes=n_classes,
            unfreeze_prefixes=list(extractor_cls.unfreeze_prefixes),
            device=device,
            config=ft_config,
        )
        result = finetuner.train(
            train_loader,
            val_loader,
            checkpoint_path=ckpt_path,
        )

        metadata = FineTuningMetadata(
            extractor_name=extractor_name,
            dataset_name=dataset_name,
            n_classes=result.n_classes,
            in_features=result.in_features,
            best_val_acc=result.best_val_acc,
            epochs_trained=result.epochs_trained,
            early_stopped=result.early_stopped,
            split_seed=split_seed,
        )
        save_finetuned(
            weights_path,
            result.backbone_state,
            result.head_state,
            metadata,
        )
        backbone_state = result.backbone_state
        logger.info("  Saved fine-tuned weights to %s", weights_path)

        # Drop everything the fine-tuner needed (model, optimiser state,
        # autograd graph, head, train/val loaders, the FineTuner itself,
        # the result dataclass holding ``result.model``) before we walk
        # projection_dims and spin up a fresh extractor per dim.  Without
        # this, Python keeps strong references in the enclosing scope
        # and peak memory doubles when the re-extract dataloader queues
        # fill up, which can OOM-kill the worker pool on small-RAM hosts.
        # ``backbone_state`` is the only thing we still need.
        del extractor, backbone, finetuner, train_loader, val_loader, result
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    for dim in projection_dims:
        output_path = Path(embeddings_dir) / dataset_name / f"{extractor_name}_finetuned_D{dim}.npy"
        if output_path.exists():
            logger.info("  %s finetuned D=%d: already exists, skipping.", extractor_name, dim)
            continue

        logger.info("  Extracting %s finetuned D=%d...", extractor_name, dim)

        extractor = extractor_cls(device=device, output_dim=dim)

        current_state = extractor.model.state_dict()
        ft_state_filtered = {
            k: v
            for k, v in backbone_state.items()
            if k in current_state and "projection" not in k and v.shape == current_state[k].shape
        }
        current_state.update(ft_state_filtered)
        extractor.model.load_state_dict(current_state)

        item_ids = get_item_ids(
            load_config()["paths"]["data_processed"],
            dataset_name,
        )
        dataset = ImageDataset(image_dir, item_ids, transform=extractor.transform)
        ext_settings = resolve_dataloader_settings(load_config())
        ext_workers = ext_settings.num_workers
        ext_prefetch = ext_settings.prefetch_factor
        ext_batch = ext_settings.batch_size
        dataloader = DataLoader(
            dataset,
            batch_size=ext_batch,
            shuffle=False,
            num_workers=ext_workers,
            pin_memory=True,
            persistent_workers=ext_workers > 0,
            prefetch_factor=ext_prefetch if ext_workers > 0 else None,
        )

        ckpt_path = f"checkpoints/extraction/{dataset_name}_{extractor_name}_finetuned_D{dim}"
        Path(ckpt_path).parent.mkdir(parents=True, exist_ok=True)

        embeddings, extracted_ids = extractor.extract_batch(
            dataloader,
            checkpoint_path=ckpt_path,
            save_every=500,
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        extractor.save(embeddings, extracted_ids, str(output_path))
        logger.info("  %s finetuned D=%d: saved (%s)", extractor_name, dim, embeddings.shape)


def run() -> None:
    """Fine-tune every configured extractor on every dataset.

    Failures are isolated per ``(extractor, dataset)`` cell so the queue
    survives an OOM, a missing layer name, or a HuggingFace Hub timeout.
    """
    config = load_config()
    set_seed(config["seed"])

    device = resolve_device(config["device"])
    split_seed = int(config["seed"])
    raw_dir = config["paths"]["data_raw"]
    embeddings_dir = config["paths"]["embeddings"]
    projection_dims = config.get("projection_dims", [64, 128, 256])
    datasets = config.get("datasets", [])
    ft_config = config.get("finetuning", {})
    ft_extractors = list(ft_config.get("extractors", []))
    tradesy_transfer_from = ft_config.get("tradesy_transfer_from", "amazon_fashion")

    enabled = config.get("extractors_enabled") or []
    if not enabled:
        logger.info(
            "finetune step skipped: extractors_enabled is empty in configs/extractors.yaml.",
        )
        return
    if not datasets:
        logger.info("finetune step skipped: datasets list is empty in configs/default.yaml.")
        return
    if not ft_extractors:
        logger.info(
            "finetune step skipped: finetuning.extractors is empty in "
            "configs/finetuning.yaml, set it to the names of the extractors "
            "you want fine-tuned, or remove the step from the pipeline.",
        )
        return

    enabled_set = set(enabled)
    dropped = [name for name in ft_extractors if name not in enabled_set]
    if dropped:
        logger.info(
            "Fine-tune list trimmed by extractors_enabled (skipped): %s",
            ", ".join(dropped),
        )
    ft_extractors = [name for name in ft_extractors if name in enabled_set]
    if not ft_extractors:
        logger.info(
            "finetune step skipped: every name in finetuning.extractors is "
            "absent from extractors_enabled.",
        )
        return

    succeeded: list[str] = []
    failed: list[tuple[str, str]] = []

    for dataset_name in datasets:
        logger.info("=== Fine-tuning for %s ===", dataset_name)

        image_dir = f"{raw_dir}/{dataset_name}/images"

        provider = get_dataset_provider(dataset_name)
        categories = provider.load_categories()
        n_classes = provider.num_categories()
        has_categories = categories is not None and n_classes > 0

        for ext_name in ft_extractors:
            job_label = f"{ext_name} × {dataset_name}"

            if not is_registered(ext_name):
                logger.warning("  Extractor %s not found, skipping.", ext_name)
                failed.append((job_label, "extractor not registered"))
                continue

            try:
                with time_cell("finetune", dataset=dataset_name, extractor=ext_name):
                    if has_categories:
                        _finetune_and_extract(
                            extractor_name=ext_name,
                            dataset_name=dataset_name,
                            image_dir=image_dir,
                            categories=categories,
                            n_classes=n_classes,
                            projection_dims=projection_dims,
                            embeddings_dir=embeddings_dir,
                            ft_config=ft_config,
                            device=device,
                            split_seed=split_seed,
                        )
                    else:
                        transfer_weights = Path(
                            f"checkpoints/finetuning/{ext_name}_{tradesy_transfer_from}_best.pt"
                        )
                        if not transfer_weights.exists():
                            logger.warning(
                                "  %s: no categories and no transfer weights from %s, skipping.",
                                dataset_name,
                                tradesy_transfer_from,
                            )
                            failed.append(
                                (
                                    job_label,
                                    f"missing transfer weights from {tradesy_transfer_from}",
                                ),
                            )
                            continue

                        logger.info(
                            "  %s: transferring fine-tuned weights from %s",
                            dataset_name,
                            tradesy_transfer_from,
                        )
                        _finetune_and_extract(
                            extractor_name=ext_name,
                            dataset_name=dataset_name,
                            image_dir=image_dir,
                            categories={},
                            n_classes=0,
                            projection_dims=projection_dims,
                            embeddings_dir=embeddings_dir,
                            ft_config=ft_config,
                            device=device,
                            split_seed=split_seed,
                            backbone_weights_path=transfer_weights,
                        )

                    succeeded.append(job_label)
            except KeyboardInterrupt:
                raise
            except Exception as exc:  # noqa: BLE001, we want to swallow everything else
                logger.error(
                    "  FAILED: %s, %s: %s\n%s",
                    job_label,
                    type(exc).__name__,
                    exc,
                    traceback.format_exc(),
                )
                failed.append((job_label, f"{type(exc).__name__}: {exc}"))

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    logger.info("Fine-tuning complete. succeeded=%d failed=%d", len(succeeded), len(failed))
    if failed:
        logger.warning("Failed jobs:")
        for job_label, err in failed:
            logger.warning("  - %s, %s", job_label, err)
