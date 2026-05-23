import abc
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from src.utils.amp_compat import cuda_autocast
from src.utils.atomic_io import atomic_np_save


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
                print(
                    f"Resuming from checkpoint: {len(all_item_ids)} items, "
                    f"starting at batch {start_batch}"
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
                    print(f"Checkpoint saved at batch {batch_idx} ({len(all_item_ids)} items)")

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

        print(f"Saved {len(item_ids)} embeddings to {npy_path}")
        print(f"Saved item ids to {json_path}")
