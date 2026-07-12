import abc
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from tqdm import tqdm

from src.utils.amp_compat import cuda_autocast
from src.utils.atomic_io import atomic_np_save
from src.utils.logging import get_logger

logger = get_logger(__name__)

#: ImageNet channel statistics (used by backbones whose canonical recipe
#: is ImageNet normalisation, e.g. DINOv2).
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _imagenet_transform(
    size: int = 224,
    interpolation: transforms.InterpolationMode = transforms.InterpolationMode.BILINEAR,
) -> transforms.Compose:
    """Generic resize + ImageNet normalisation pipeline.

    .. warning::
        This is a convenience fallback for plugin authors only.  A
        pretrained backbone's preprocessing recipe (resolution, resize/
        crop, interpolation, normalisation) is part of the model just
        like its weights — applying the wrong recipe silently degrades
        features.  Every built-in extractor resolves its canonical
        transform from the library that ships the weights (see
        :func:`timm_canonical_transform` and the per-extractor
        ``_build_transform`` overrides); do the same for your plugin.
    """
    return transforms.Compose(
        [
            transforms.Resize((size, size), interpolation=interpolation),
            transforms.ToTensor(),
            transforms.Normalize(mean=list(IMAGENET_MEAN), std=list(IMAGENET_STD)),
        ]
    )


def timm_canonical_transform(timm_model) -> transforms.Compose:
    """Resolve the canonical eval transform for a timm model instance.

    Uses ``timm.data.resolve_model_data_config`` so the recipe (mean/std,
    interpolation, crop_pct, input size) is read from the *checkpoint's*
    pretrained config rather than hardcoded — normalisation differs
    between tags of the same architecture (e.g. ``vit_base_patch16_224``
    ``augreg2`` uses mean/std 0.5, not ImageNet).
    """
    import timm

    data_config = timm.data.resolve_model_data_config(timm_model)
    return timm.data.create_transform(**data_config, is_training=False)


class HFProcessorTransform:
    """Adapter turning a HuggingFace image processor into a transform.

    Wraps ``AutoImageProcessor`` output into the plain ``(C, H, W)``
    tensor the extraction pipeline expects.  A class (not a lambda) so
    DataLoader workers can pickle it.
    """

    def __init__(self, processor) -> None:
        self.processor = processor

    def __call__(self, image) -> torch.Tensor:
        return self.processor(image, return_tensors="pt")["pixel_values"].squeeze(0)

    def __repr__(self) -> str:
        return f"HFProcessorTransform({type(self.processor).__name__})"


class BaseExtractor(abc.ABC):  # noqa: B024 — template base: subclasses set backbone_cls
    """Abstract base class for visual feature extractors.

    v2 contract: extractors emit features at the backbone's **native**
    dimensionality (ResNet-50 → 2048, ViT-B/16 → 768, ...).  No
    projection, pooling change or truncation is applied at extraction
    time — the learned projection ``E`` inside each recommender maps the
    native feature to the common latent dimension ``d``.

    Plugin-author hooks
    -------------------
    * :meth:`_build_model` — return the :class:`nn.Module`.  Its last
      submodule must be named ``projection`` and default to
      :class:`nn.Identity`; the fine-tuner swaps it for a classification
      head, and extraction keeps it as identity so saved features stay
      native.
    * :meth:`_build_transform` — return the image transform pipeline.
      Must be the **canonical** recipe of the pretrained weights in use
      (resolve it from the shipping library; see
      :func:`timm_canonical_transform`), never a generic shared recipe.

    Class attributes
    ----------------
    ``unfreeze_prefixes`` (list of str)
        Module-name prefixes that should remain trainable during
        fine-tuning.  Empty list (default) means *only* the freshly added
        classification head is trained.  Each prefix is matched with
        ``startswith`` against ``named_parameters()`` of the backbone.
    ``extraction_point`` (str)
        Human-readable description of where in the backbone the feature
        is taken (e.g. ``"avgpool"``, ``"CLS token"``).  Recorded in the
        artifact metadata.
    ``weights_id`` (str)
        Exact identifier of the pretrained weights in use (library +
        checkpoint tag/revision).  Recorded in the artifact metadata.
    """

    #: Names (or name prefixes) of the backbone submodules that should
    #: remain trainable when the extractor is fine-tuned.  Override in
    #: subclasses that want their backbone partially unfrozen.
    unfreeze_prefixes: list[str] = []

    #: ``True`` when the extractor can emit per-item *component* features
    #: (the spatial feature-map cells / patch tokens before global
    #: pooling) of shape ``(M, native_dim)``.  Subclasses that override
    #: :meth:`_forward_components` set this to ``True``; the pooled output
    #: path is unaffected.
    supports_components: bool = False

    #: Backbone :class:`nn.Module` class instantiated by the default
    #: :meth:`_build_model` as ``backbone_cls()``.  The backbone's last
    #: submodule must be named ``projection`` (see the class docstring).
    #: A subclass may instead override :meth:`_build_model`.
    backbone_cls: type[nn.Module] | None = None

    #: Where the feature is taken from (metadata; override per subclass).
    extraction_point: str = "unspecified"

    #: Exact pretrained-weights identifier (metadata; override per subclass).
    weights_id: str = "unspecified"

    def __init__(self, device: str = "cuda"):
        self.device = torch.device(device)
        self.model = self._build_model()
        self.transform = self._build_transform()
        # The native output dimensionality is READ from the model with a
        # probe forward — never hand-written.  Hardcoded dims are exactly
        # how silent errors like "LeViT-256 outputs 256" creep in.
        self.native_dim = self._probe_native_dim()

    def _build_model(self) -> nn.Module:
        """Instantiate ``backbone_cls`` on the target device in eval mode.

        Subclasses that use a non-trivial construction path (e.g. an
        ``open_clip`` preprocess) override this instead of setting
        ``backbone_cls``.
        """
        if self.backbone_cls is None:
            raise NotImplementedError(
                f"{type(self).__name__} must set backbone_cls or override _build_model()."
            )
        model = self.backbone_cls()
        model = model.to(self.device)
        model.eval()
        return model

    @abc.abstractmethod
    def _build_transform(self):
        """Return the canonical transform of the pretrained weights in use.

        Resolve it from the library that ships the weights (torchvision
        ``weights.transforms()``, :func:`timm_canonical_transform`,
        ``AutoImageProcessor``, open_clip's ``preprocess``) instead of
        writing a ``Compose`` by hand — the recipe is part of the model.
        """

    def _probe_native_dim(self) -> int:
        """Read the native output dim from the model via a probe forward.

        Runs the real transform on a dummy image so the probe input has
        exactly the shape the pipeline will produce.
        """
        from PIL import Image

        dummy = Image.new("RGB", (256, 256))
        tensor = self.transform(dummy).unsqueeze(0).to(self.device)
        with torch.no_grad():
            out = self.model(tensor)
        if out.dim() != 2:
            raise RuntimeError(
                f"{type(self).__name__}: probe forward returned shape "
                f"{tuple(out.shape)}; expected a pooled (1, D) feature."
            )
        return int(out.shape[1])

    def metadata(self) -> dict:
        """Artifact metadata persisted next to every saved feature file."""
        input_size = None
        try:
            from PIL import Image

            probe = self.transform(Image.new("RGB", (256, 256)))
            input_size = list(probe.shape)
        except Exception:  # noqa: BLE001 — metadata is best-effort descriptive
            pass
        return {
            "extractor": type(self).__name__,
            "native_dim": self.native_dim,
            "extraction_point": self.extraction_point,
            "weights_id": self.weights_id,
            "input_shape": input_size,
            "transform": repr(self.transform),
        }

    def extract(self, image) -> np.ndarray:
        """Extract embedding from a single PIL image.

        Parameters
        ----------
        image : PIL.Image
            Input image.

        Returns
        -------
        np.ndarray
            1-D embedding of shape ``(native_dim,)``.
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
            ``(N, native_dim)`` and *item_ids* is a flat list of length *N*.
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
            final_embeddings = np.empty((0, self.native_dim), dtype=np.float32)
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
        """Return per-item component features ``(B, M, native_dim)``.

        Default: delegate to the backbone's ``forward_components`` when
        the extractor advertises ``supports_components = True`` (every
        built-in backbone exposes that method).  Extractors that do not
        expose components raise, and those with a non-standard component
        path may override this method.  Components pass through the SAME
        trainable ``projection`` as the pooled path, so the last
        dimension is ``native_dim``.
        """
        if not self.supports_components:
            raise NotImplementedError(
                f"{type(self).__name__} does not expose component features.",
            )
        return self.model.forward_components(images)

    def extract_components_batch(
        self,
        dataloader,
        checkpoint_path: str | None = None,
        save_every: int = 500,
    ) -> tuple[np.ndarray, list]:
        """Extract component features for a dataloader, stacking ``(N, M, native_dim)``.

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
            final = np.empty((0, 0, self.native_dim), dtype=np.float32)
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

        ``components`` has shape ``(N, M, native_dim)``.
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
            Matrix of shape ``(N, native_dim)``.
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
