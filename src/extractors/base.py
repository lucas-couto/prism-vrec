import abc
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from src.utils.amp_compat import cuda_autocast
from src.utils.atomic_io import atomic_np_save
from src.utils.logging import get_logger

logger = get_logger(__name__)


class BaseExtractor(abc.ABC):
    """Abstract base class for visual feature extractors.

    Plugin-author hooks
    -------------------
    Subclasses are expected to override these two methods and *may*
    override the optional class attributes below:

    * :meth:`_build_model` — return the trainable :class:`nn.Module`.  Its
      last submodule must be named ``projection`` and end in a layer whose
      ``in_features`` matches the backbone's pooled-feature size; the
      fine-tuner replaces this layer with a classification head.
    * :meth:`_build_transform` — return the image transform pipeline used
      both at extraction and fine-tuning time.

    Class attributes
    ----------------
    ``unfreeze_prefixes`` (list of str)
        Module-name prefixes that should remain trainable during
        fine-tuning.  Empty list (default) means *only* the freshly added
        classification head is trained.  Each prefix is matched with
        ``startswith`` against ``named_parameters()`` of the backbone.
    """

    #: Names (or name prefixes) of the backbone submodules that should
    #: remain trainable when the extractor is fine-tuned.  Override in
    #: subclasses that want their backbone partially unfrozen.
    unfreeze_prefixes: list[str] = []

    #: ``True`` when the extractor can emit per-item *component* features
    #: (the spatial feature-map cells / patch tokens before global
    #: pooling) of shape ``(M, output_dim)``.  Subclasses that override
    #: :meth:`_forward_components` set this to ``True``; the pooled output
    #: path is unaffected.
    supports_components: bool = False

    def __init__(self, device: str, output_dim: int):
        self.device = torch.device(device)
        self.output_dim = output_dim
        self.model = None
        self.transform = None

    @abc.abstractmethod
    def _build_model(self):
        """Build and return the feature extraction model."""
        ...

    @abc.abstractmethod
    def _build_transform(self):
        """Build and return the image transform pipeline."""
        ...

    def extract(self, image) -> np.ndarray:
        """Extract embedding from a single PIL image.

        Parameters
        ----------
        image : PIL.Image
            Input image.

        Returns
        -------
        np.ndarray
            1-D embedding of shape ``(output_dim,)``.
        """
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad(), cuda_autocast(enabled=self.device.type == "cuda"):
            embedding = self.model(tensor)
        return embedding.float().squeeze(0).cpu().numpy()

    def extract_batch(
        self,
        dataloader,
        checkpoint_path: str | None = None,
        save_every: int = 500,
    ) -> tuple[np.ndarray, list]:
        """Extract embeddings from an entire dataloader with checkpoint support.

        The dataloader is expected to yield ``(images, item_ids)`` tuples where
        *images* is a batch of PIL images or pre-transformed tensors and
        *item_ids* is a list/tuple of corresponding identifiers.

        Parameters
        ----------
        dataloader : torch.utils.data.DataLoader
            Dataloader that yields ``(images, item_ids)`` pairs.  If the
            images are already tensors they are used directly; otherwise the
            extractor's ``transform`` is applied.
        checkpoint_path : str, optional
            Path to a ``.pt`` file used for saving / resuming partial progress.
        save_every : int
            Save a partial checkpoint every *save_every* batches.

        Returns
        -------
        tuple[np.ndarray, list]
            ``(embeddings, item_ids)`` where *embeddings* has shape
            ``(N, output_dim)`` and *item_ids* is a flat list of length *N*.
        """
        all_embeddings: list[np.ndarray] = []
        all_item_ids: list = []
        start_batch = 0

        if checkpoint_path is not None:
            ckpt_file = Path(checkpoint_path)
            if ckpt_file.exists():
                ckpt = torch.load(ckpt_file, map_location="cpu", weights_only=False)
                all_embeddings = [ckpt["embeddings"]]
                all_item_ids = ckpt["item_ids"]
                start_batch = ckpt["last_batch_index"] + 1
                logger.info(
                    "Resuming from checkpoint: %d items, starting at batch %d",
                    len(all_item_ids),
                    start_batch,
                )

        use_amp = self.device.type == "cuda"
        self.model.eval()
        with torch.no_grad():
            for batch_idx, (images, item_ids) in enumerate(
                tqdm(dataloader, desc="Extracting features")
            ):
                if batch_idx < start_batch:
                    continue

                if not isinstance(images, torch.Tensor):
                    images = torch.stack([self.transform(img) for img in images])

                images = images.to(self.device)
                with cuda_autocast(enabled=use_amp):
                    embeddings = self.model(images)
                embeddings = embeddings.float().cpu().numpy()

                all_embeddings.append(embeddings)
                if isinstance(item_ids, torch.Tensor):
                    all_item_ids.extend(item_ids.tolist())
                else:
                    all_item_ids.extend(list(item_ids))

                if checkpoint_path is not None and (batch_idx + 1) % save_every == 0:
                    partial_emb = np.concatenate(all_embeddings, axis=0)
                    torch.save(
                        {
                            "embeddings": partial_emb,
                            "item_ids": all_item_ids,
                            "last_batch_index": batch_idx,
                        },
                        checkpoint_path,
                    )
                    logger.info(
                        "Checkpoint saved at batch %d (%d items)", batch_idx, len(all_item_ids)
                    )

        if len(all_embeddings) == 0:
            final_embeddings = np.empty((0, self.output_dim), dtype=np.float32)
        else:
            final_embeddings = np.concatenate(all_embeddings, axis=0)

        if checkpoint_path is not None:
            final_path = Path(checkpoint_path)
            torch.save(
                {
                    "embeddings": final_embeddings,
                    "item_ids": all_item_ids,
                    "last_batch_index": -1,
                },
                final_path,
            )

        return final_embeddings, all_item_ids

    def _forward_components(self, images: torch.Tensor) -> torch.Tensor:
        """Return per-item component features ``(B, M, output_dim)``.

        Override in subclasses that expose pre-pool features (spatial
        feature-map cells or patch tokens) and set
        :attr:`supports_components` to ``True``.  Components are passed
        through the SAME trainable ``projection`` as the pooled path, so
        the last dimension is ``output_dim``.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not expose component features.",
        )

    def extract_components_batch(
        self,
        dataloader,
        checkpoint_path: str | None = None,
        save_every: int = 500,
    ) -> tuple[np.ndarray, list]:
        """Extract component features for a dataloader, stacking ``(N, M, output_dim)``.

        Mirrors :meth:`extract_batch` (same checkpoint/resume semantics)
        but uses :meth:`_forward_components` instead of the pooled model.
        """
        all_components: list[np.ndarray] = []
        all_item_ids: list = []
        start_batch = 0

        if checkpoint_path is not None and Path(checkpoint_path).exists():
            ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            all_components = [ckpt["components"]]
            all_item_ids = ckpt["item_ids"]
            start_batch = ckpt["last_batch_index"] + 1

        use_amp = self.device.type == "cuda"
        self.model.eval()
        with torch.no_grad():
            for batch_idx, (images, item_ids) in enumerate(
                tqdm(dataloader, desc="Extracting components")
            ):
                if batch_idx < start_batch:
                    continue
                if not isinstance(images, torch.Tensor):
                    images = torch.stack([self.transform(img) for img in images])
                images = images.to(self.device)
                with cuda_autocast(enabled=use_amp):
                    components = self._forward_components(images)
                all_components.append(components.float().cpu().numpy())
                if isinstance(item_ids, torch.Tensor):
                    all_item_ids.extend(item_ids.tolist())
                else:
                    all_item_ids.extend(list(item_ids))

                if checkpoint_path is not None and (batch_idx + 1) % save_every == 0:
                    torch.save(
                        {
                            "components": np.concatenate(all_components, axis=0),
                            "item_ids": all_item_ids,
                            "last_batch_index": batch_idx,
                        },
                        checkpoint_path,
                    )

        if len(all_components) == 0:
            final = np.empty((0, 0, self.output_dim), dtype=np.float32)
        else:
            final = np.concatenate(all_components, axis=0)

        if checkpoint_path is not None:
            torch.save(
                {"components": final, "item_ids": all_item_ids, "last_batch_index": -1},
                checkpoint_path,
            )
        return final, all_item_ids

    def save_components(self, components: np.ndarray, item_ids: list, path: str):
        """Save 3-D component features as ``<path>.npy`` + ``<path>_ids.json``.

        ``components`` has shape ``(N, M, output_dim)``.
        """
        base = Path(path)
        base.parent.mkdir(parents=True, exist_ok=True)
        npy_path = base.with_suffix(".npy")
        json_path = base.with_name(base.stem + "_ids.json")
        atomic_np_save(components, npy_path)
        with open(json_path, "w") as f:
            json.dump(item_ids, f)
        logger.info(
            "Saved %d component features to %s (%s)", len(item_ids), npy_path, components.shape
        )

    def save(self, embeddings: np.ndarray, item_ids: list, path: str):
        """Save embeddings as ``.npy`` and item_ids as ``.json``.

        Parameters
        ----------
        embeddings : np.ndarray
            Matrix of shape ``(N, output_dim)``.
        item_ids : list
            List of item identifiers of length *N*.
        path : str
            Base path (without extension).  Two files are written:
            ``<path>.npy`` and ``<path>_ids.json``.
        """
        base = Path(path)
        base.parent.mkdir(parents=True, exist_ok=True)

        npy_path = base.with_suffix(".npy")
        json_path = base.with_name(base.stem + "_ids.json")

        atomic_np_save(embeddings, npy_path)
        with open(json_path, "w") as f:
            json.dump(item_ids, f)

        logger.info("Saved %d embeddings to %s", len(item_ids), npy_path)
        logger.info("Saved item ids to %s", json_path)
