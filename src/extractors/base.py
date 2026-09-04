import abc
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from tqdm import tqdm

from src.extractors.components import pool_components
from src.utils import flops, telemetry
from src.utils.amp_compat import cuda_autocast
from src.utils.atomic_io import atomic_np_save, atomic_write
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


def _dataset_len(dataloader) -> int:
    dataset = getattr(dataloader, "dataset", None)
    if dataset is None:
        raise ValueError(
            "streaming extraction needs len(dataloader.dataset) to preallocate "
            "the on-disk matrix; pass checkpoint_path=None for loaders without a dataset."
        )
    return len(dataset)


def _resume_state(part_path: Path, progress_path: Path, n_total: int):
    """Reopen a part file + progress sidecar; restart clean when they don't match."""
    if not (part_path.exists() and progress_path.exists()):
        part_path.unlink(missing_ok=True)
        progress_path.unlink(missing_ok=True)
        return None, 0, 0, []
    try:
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        candidate = np.lib.format.open_memmap(part_path, mode="r+")
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        logger.warning("Corrupt extraction checkpoint %s (%s); restarting", part_path, exc)
        part_path.unlink(missing_ok=True)
        progress_path.unlink(missing_ok=True)
        return None, 0, 0, []
    if candidate.shape[0] != n_total or progress.get("n_total") != n_total:
        del candidate  # catalogue changed under the part file: restart clean
        part_path.unlink()
        progress_path.unlink()
        return None, 0, 0, []
    logger.info(
        "  resume: %d/%d rows already on disk (%s)", progress["rows_done"], n_total, part_path.name
    )
    return (
        candidate,
        progress["last_batch_index"] + 1,
        progress["rows_done"],
        list(progress["item_ids"]),
    )


def _finalise_array(array: np.ndarray, npy_path: Path) -> None:
    """Persist *array* as ``npy_path``.

    A read-mode memmap over a streaming ``.part.npy`` is finalised by
    atomic rename (same filesystem) or a streamed copy (cross-filesystem)
    — never materialised in RAM.  Anything else goes through
    :func:`atomic_np_save`.
    """
    source = getattr(array, "filename", None)
    if source is None or not Path(source).name.endswith(".part.npy"):
        atomic_np_save(array, npy_path)
        return
    source = Path(source)
    del array  # drop the mapping before moving the file under it
    try:
        source.replace(npy_path)
    except OSError:
        atomic_np_save(np.load(source, mmap_mode="r"), npy_path)
        source.unlink()
    progress = source.with_name(source.name.replace(".part.npy", ".progress.json"))
    progress.unlink(missing_ok=True)


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
    #: Optional ``g``: pool the native ``√M × √M`` region map to ``g × g``
    #: before saving components (see :mod:`src.extractors.components`).
    #: Set by the extract step from ``component_grid`` in the config;
    #: ``None`` keeps the native regions.
    component_grid: int | None = None

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

    def _account_flops(
        self,
        images: torch.Tensor,
        *,
        forward=None,
        tag: str = "pooled",
        training: bool = False,
    ) -> None:
        """Attribute this batch's compute to the run's telemetry counters.

        The first call for a given ``(extractor, tag, input shape)``
        measures one sample under the dispatch counter; every later call
        is a multiplication.  Keying on ``shape[1:]`` rather than the full
        shape keeps a trailing partial batch from forcing a second
        calibration, since the cached value is already per-sample.
        """
        key = f"{type(self).__name__}::{tag}::{tuple(images.shape[1:])}"
        flops.calibrate(key, forward or self.model, images[:1])
        n = int(images.shape[0])
        flops.record(key, n, training=training)
        telemetry.add_items(n)

    def extract_batch(
        self,
        dataloader,
        checkpoint_path: str | None = None,
        save_every: int = 500,
    ) -> tuple[np.ndarray, list]:
        """Extract pooled embeddings ``(N, native_dim)`` from a dataloader.

        The dataloader is expected to yield ``(images, item_ids)`` tuples where
        *images* is a batch of PIL images or pre-transformed tensors and
        *item_ids* is a list/tuple of corresponding identifiers.

        With a *checkpoint_path* the catalogue is NEVER accumulated in RAM:
        batches stream into an fp32 ``<checkpoint_path>.part.npy`` memmap
        with resume progress in ``<checkpoint_path>.progress.json`` (same
        mechanism as :meth:`extract_components_batch`).  The former
        accumulate-and-pickle checkpoint held the whole matrix in RAM three
        times over at every save and OOM-killed the container on large
        catalogues.  The returned array is a read-mode memmap; hand it to
        :meth:`save`, which finalises by rename when possible.

        Parameters
        ----------
        dataloader : torch.utils.data.DataLoader
            Dataloader that yields ``(images, item_ids)`` pairs.
        checkpoint_path : str, optional
            Base path for the streaming part file and progress sidecar.
        save_every : int
            Flush progress every *save_every* batches.

        Returns
        -------
        tuple[np.ndarray, list]
            ``(embeddings, item_ids)`` with *embeddings* of shape ``(N, native_dim)``.
        """
        if checkpoint_path is None:
            return self._collect_in_memory(
                self._iter_batches(
                    dataloader, 0, self.model, self._account_flops, "Extracting features"
                ),
                empty_shape=(0, self.native_dim),
                dtype=np.float32,
            )
        legacy = Path(checkpoint_path)
        if legacy.is_file():
            # Pre-streaming pickle checkpoint (possibly truncated by an OOM
            # kill mid-save): it is neither readable nor needed any more.
            logger.warning("Discarding legacy pooled checkpoint %s", legacy)
            legacy.unlink()
        return self._extract_streaming(
            lambda start: self._iter_batches(
                dataloader, start, self.model, self._account_flops, "Extracting features"
            ),
            dataloader,
            checkpoint_path,
            save_every,
            dtype=np.float32,
            empty_shape=(0, self.native_dim),
        )

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

        Component matrices are ~50-500x the pooled ones (resnet50 on
        amazon_fashion: 49 x 2048 per item = 67 GB fp32), so with a
        *checkpoint_path* batches stream into an fp16
        ``<checkpoint_path>.part.npy`` memmap with resume progress in
        ``<checkpoint_path>.progress.json``.  Peak host memory is one
        batch, regardless of catalogue size.  The returned array is a
        read-mode memmap over the part file; hand it to
        :meth:`save_components`, which finalises by rename.

        Streaming needs the total row count upfront
        (``len(dataloader.dataset)``).  Without a *checkpoint_path* the
        in-memory path is used — fine for tests and small plugin
        catalogues, fatal for real ones.
        """
        if checkpoint_path is None:
            return self._extract_components_in_memory(dataloader)
        return self._extract_components_streaming(dataloader, checkpoint_path, save_every)

    def _account_component_flops(self, images: torch.Tensor) -> None:
        self._account_flops(images, forward=self._forward_components, tag="components")

    def _iter_batches(self, dataloader, start_batch: int, forward, account, desc: str):
        """Yield ``(batch_idx, features_fp32, ids)`` from *start_batch* on.

        *forward* runs under autocast; *account(images)* attributes the
        batch's FLOPs afterwards, outside autocast (calibration must see
        the same dtype dispatch as before the streaming refactor).
        """
        use_amp = self.device.type == "cuda"
        self.model.eval()
        with torch.no_grad():
            for batch_idx, (images, item_ids) in enumerate(tqdm(dataloader, desc=desc)):
                if batch_idx < start_batch:
                    continue
                if not isinstance(images, torch.Tensor):
                    images = torch.stack([self.transform(img) for img in images])
                images = images.to(self.device)
                with cuda_autocast(enabled=use_amp):
                    features = forward(images)
                account(images)
                ids = item_ids.tolist() if isinstance(item_ids, torch.Tensor) else list(item_ids)
                yield batch_idx, features.float().cpu().numpy(), ids

    def _iter_component_batches(self, dataloader, start_batch: int):
        """Yield ``(batch_idx, components_fp16, ids)`` from *start_batch* on."""
        gen = self._iter_batches(
            dataloader,
            start_batch,
            self._forward_components,
            self._account_component_flops,
            "Extracting components",
        )
        for batch_idx, comp, ids in gen:
            if self.component_grid is not None:
                comp = pool_components(torch.from_numpy(comp), self.component_grid).numpy()
            yield batch_idx, comp.astype(np.float16), ids

    @staticmethod
    def _collect_in_memory(batches, empty_shape: tuple, dtype) -> tuple[np.ndarray, list]:
        chunks: list[np.ndarray] = []
        all_item_ids: list = []
        for _, feats, ids in batches:
            chunks.append(feats.astype(dtype, copy=False))
            all_item_ids.extend(ids)
        if not chunks:
            return np.empty(empty_shape, dtype=dtype), all_item_ids
        return np.concatenate(chunks, axis=0), all_item_ids

    def _extract_components_in_memory(self, dataloader) -> tuple[np.ndarray, list]:
        """Accumulate-in-RAM path (no checkpoint, small catalogues only)."""
        return self._collect_in_memory(
            self._iter_component_batches(dataloader, start_batch=0),
            empty_shape=(0, 0, self.native_dim),
            dtype=np.float16,
        )

    def _extract_components_streaming(
        self, dataloader, checkpoint_path: str, save_every: int
    ) -> tuple[np.ndarray, list]:
        """Stream component batches into an fp16 on-disk memmap (resumable)."""
        return self._extract_streaming(
            lambda start: self._iter_component_batches(dataloader, start),
            dataloader,
            checkpoint_path,
            save_every,
            dtype=np.float16,
            empty_shape=(0, 0, self.native_dim),
        )

    def _extract_streaming(
        self,
        batch_factory,
        dataloader,
        checkpoint_path: str,
        save_every: int,
        dtype,
        empty_shape: tuple,
    ) -> tuple[np.ndarray, list]:
        """Stream ``(batch_idx, features, ids)`` into an on-disk memmap (resumable).

        *batch_factory(start_batch)* returns the batch iterator resumed at
        *start_batch* (see :meth:`_iter_batches`).
        """
        n_total = _dataset_len(dataloader)
        part_path = Path(f"{checkpoint_path}.part.npy")
        progress_path = Path(f"{checkpoint_path}.progress.json")
        part_path.parent.mkdir(parents=True, exist_ok=True)

        memmap, start_batch, row, all_item_ids = _resume_state(part_path, progress_path, n_total)

        def _save_progress(last_batch_index: int) -> None:
            payload = json.dumps(
                {
                    "last_batch_index": last_batch_index,
                    "rows_done": row,
                    "n_total": n_total,
                    "item_ids": all_item_ids,
                }
            )
            atomic_write(
                lambda tmp: Path(tmp).write_text(payload, encoding="utf-8"),
                progress_path,
            )

        for batch_idx, feats, ids in batch_factory(start_batch):
            if memmap is None:
                memmap = np.lib.format.open_memmap(
                    part_path, mode="w+", dtype=dtype, shape=(n_total, *feats.shape[1:])
                )
            memmap[row : row + feats.shape[0]] = feats
            row += feats.shape[0]
            all_item_ids.extend(ids)
            if (batch_idx + 1) % save_every == 0:
                memmap.flush()
                _save_progress(batch_idx)

        if memmap is None:
            return np.empty(empty_shape, dtype=dtype), all_item_ids
        memmap.flush()
        _save_progress(-1)
        del memmap  # release the write mapping before reopening read-only
        return np.load(part_path, mmap_mode="r"), all_item_ids

    def save_components(self, components: np.ndarray, item_ids: list, path: str):
        """Save 3-D component features as ``<path>.npy`` + ``<path>_ids.json``.

        ``components`` has shape ``(N, M, native_dim)``.  When it is the
        read-mode memmap produced by the streaming extraction (a
        ``.part.npy`` on the SAME filesystem as the destination), the
        data is finalised by an atomic rename — never rewritten, so the
        catalogue-sized matrix costs no second pass and no double disk.
        """
        base = Path(path)
        base.parent.mkdir(parents=True, exist_ok=True)
        npy_path = base.with_suffix(".npy")
        json_path = base.with_name(base.stem + "_ids.json")

        shape = tuple(components.shape)
        _finalise_array(components, npy_path)

        with open(json_path, "w") as f:
            json.dump(item_ids, f)
        logger.info("Saved %d component features to %s (%s)", len(item_ids), npy_path, shape)

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

        _finalise_array(embeddings, npy_path)
        with open(json_path, "w") as f:
            json.dump(item_ids, f)

        logger.info("Saved %d embeddings to %s", len(item_ids), npy_path)
        logger.info("Saved item ids to %s", json_path)
