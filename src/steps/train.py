"""Step 05, Recommender hyperparameter search.

Three strategies are supported, selected by
``configs/recommenders.yaml -> hp_search.strategy``:

* ``grid`` (default), Cartesian product over the lists declared
  per recommender, dispatched in parallel via
  :class:`TrainingOrchestrator`.
* ``optuna``, Bayesian search via :mod:`optuna`, sequential within
  each ``(dataset, model, embedding)`` cell with median-pruner
  stopping bad trials early.  Independent cells are dispatched to a
  small pool of worker processes (B7); trials inside a cell stay
  sequential so the TPE sampler always conditions on every previous
  trial of its own study.

* ``fixed``, no search: every hyperparameter is pinned to a single
  value in the YAML and each cell is trained exactly once through
  :func:`train_replay` (no Optuna study, no storage).  This is the
  K-fold cross-validation path, where the configuration was chosen
  beforehand.

All backends share the same per-trial entry point, so the actual
training loop in :mod:`src.utils.training` is unchanged.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm

from src.recommenders import (
    get_recommender_spec,
    is_registered,
    iter_specs,
    registered_recommender_names,
)
from src.recommenders.hp_search import (
    CellKey,
    assert_dimension_parity,
    create_study,
    get_fixed_hyperparams,
    get_hyperparam_grid,
    get_strategy,
    sample_hyperparams,
)
from src.utils.artifact_names import (
    FUSION_PREFIX,
    is_component_artifact,
    is_finetuned_artifact,
    is_projected_artifact,
)
from src.utils.checkpoint import CheckpointManager
from src.utils.config import load_config
from src.utils.device import resolve_device
from src.utils.identity import (
    PROVENANCE_SUFFIX,
    SELECTION_SPLITS,
    DataIdentity,
    IdentityError,
    build_identity_context,
    canonical_digest,
    resolve_data_identity,
)
from src.utils.logging import get_logger
from src.utils.memory import (
    AdmissionPlan,
    admit_workers,
    available_cpus,
    choose_lazy_features,
    estimate_model_state_bytes,
    npy_payload_bytes,
    resolve_host_budget,
)
from src.utils.parallel import TrainingJob, TrainingOrchestrator, checkpoint_root
from src.utils.resources import resolve_resources
from src.utils.seed import set_seed

logger = get_logger(__name__)


EMBEDDING_VARIANTS = ("native", "projected", "both")


def _resolve_embedding_variants(config: dict) -> str:
    """Read ``embedding_variants`` from the recommender config.

    Selects which of the two artifact families the recommenders are
    trained on when a fixed projection is configured in
    ``configs/extractors.yaml``: the backbones' native features, their
    fixed-dim projections, or both side by side.
    """
    value = str(config.get("embedding_variants", "both"))
    if value not in EMBEDDING_VARIANTS:
        raise ValueError(
            f"embedding_variants must be one of {list(EMBEDDING_VARIANTS)}, got {value!r}"
        )
    return value


def filter_by_variant(names: list[str], variant: str) -> list[str]:
    """Keep the embedding names belonging to *variant*.

    ``"none"`` — the pseudo-embedding of the non-visual baselines — is
    never filtered out: it belongs to no artifact family, and dropping
    it would silently remove plain BPR from a projected-only battery.
    """
    if variant == "both":
        return names
    want_projected = variant == "projected"
    return [n for n in names if n == "none" or is_projected_artifact(n) == want_projected]


def get_embedding_files(
    embeddings_dir: str,
    dataset_name: str,
    dim_filter: list[str] | None = None,
) -> list[str]:
    """List embedding stems for a dataset, optionally filtered by dim.

    Includes both ``.npy`` files (offline embeddings + offline fusions)
    and ``.json`` sidecars (online fusions like ``adaptive_gated``).
    The stem is what the train step uses to identify the embedding;
    ``load_embedding`` resolves the actual on-disk path at load time.

    ``dim_filter`` only applies to fusion artifacts (``hybrid_*``), whose
    names carry an explicit alignment-dim token; single-extractor
    artifacts are native-dim and carry no dim token, so they always pass.
    """
    emb_dir = Path(embeddings_dir) / dataset_name
    if not emb_dir.exists():
        return []
    names = [f.stem for f in sorted(emb_dir.glob("*.npy"))]
    # ``<artifact>.provenance.json`` (E05) sits next to every fusion
    # output, including the ``hybrid_*.json`` sidecars, so a bare glob
    # would turn ``hybrid_x.json.provenance.json`` into a phantom
    # embedding ``hybrid_x.json.provenance`` whose jobs can only fail.
    names.extend(
        f.stem
        for f in sorted(emb_dir.glob("hybrid_*.json"))
        if not f.name.endswith(PROVENANCE_SUFFIX)
    )
    names = sorted(set(names))
    if dim_filter:
        names = [
            n
            for n in names
            if not n.startswith(FUSION_PREFIX)
            or any(n.endswith(d) or n.endswith(f"{d}_comp") for d in dim_filter)
        ]
    return names


def _resolve_embedding_path(embeddings_dir: str, dataset_name: str, stem: str) -> str | None:
    """Map a stem to either ``<stem>.npy`` or ``<stem>.json`` on disk."""
    base = Path(embeddings_dir) / dataset_name
    npy = base / f"{stem}.npy"
    if npy.exists():
        return str(npy)
    sidecar = base / f"{stem}.json"
    if sidecar.exists():
        return str(sidecar)
    return None


@dataclass(frozen=True)
class _Cell:
    """One ``(dataset, model, embedding)`` unit of work with its metadata."""

    dataset_name: str
    model_name: str
    spec: object
    embedding_name: str
    embedding_path: str | None
    n_users: int
    n_items: int


def _resolve_model_names(config: dict) -> list[str]:
    """Registered, enabled recommender names in (priority, name) order."""
    enabled = set(config.get("recommenders_enabled") or [])
    return [s.name for s in iter_specs() if s.name in enabled]


def filter_by_enabled_fusions(names: list[str], config: dict) -> list[str]:
    """Drop ``hybrid_*`` stems whose strategy is not currently enabled.

    The embeddings directory accumulates fusion artifacts across runs;
    without this filter, disabling a strategy in
    ``configs/fusion.yaml -> fusion_strategies_enabled`` stopped the
    fuse step from WRITING it but the train step still picked the
    on-disk artifact up as a cell.  The config is the source of truth:
    an empty/absent list keeps no fusion cells.  Stems whose strategy
    cannot be parsed against the registered names are excluded loudly.
    """
    from src.fusions.registry import registered_fusion_strategies
    from src.utils.artifact_names import fusion_strategy_of

    enabled = set(config.get("fusion_strategies_enabled") or [])
    known = registered_fusion_strategies()
    kept: list[str] = []
    for name in names:
        if not name.startswith(FUSION_PREFIX):
            kept.append(name)
            continue
        strategy = fusion_strategy_of(name, known)
        if strategy is None:
            logger.warning(
                "embedding %r looks like a fusion artifact but matches no "
                "registered strategy; excluded from training.",
                name,
            )
            continue
        if strategy in enabled:
            kept.append(name)
    n_dropped = len(names) - len(kept)
    if n_dropped:
        logger.info(
            "fusion filter: %d hybrid artifact(s) on disk excluded "
            "(strategy not in fusion_strategies_enabled).",
            n_dropped,
        )
    return kept


def filter_by_enabled_extractors(names: list[str], config: dict) -> list[str]:
    """Drop single-extractor stems whose backbone is not currently enabled.

    Mirror of :func:`filter_by_enabled_fusions` for the other artifact
    family: the embeddings directory accumulates every backbone ever
    extracted, so shrinking ``extractors_enabled`` stopped the EXTRACT
    step but the train step still picked the on-disk artifacts up as
    cells.  Matching is longest-first against the registered extractor
    names, so a stem like ``cvt_13_p128`` (projection token) or
    ``resnet50_finetuned`` (condition marker) resolves to its backbone.
    ``none`` and ``hybrid_*`` stems pass through untouched — the
    baseline belongs to no backbone and fusions are governed by
    :func:`filter_by_enabled_fusions`.
    """
    from src.extractors import registered_extractor_names

    enabled = set(config.get("extractors_enabled") or [])
    known = sorted(registered_extractor_names(), key=len, reverse=True)
    kept: list[str] = []
    for name in names:
        if name == "none" or name.startswith(FUSION_PREFIX):
            kept.append(name)
            continue
        base = next((k for k in known if name == k or name.startswith(f"{k}_")), None)
        if base is None:
            logger.warning(
                "embedding %r matches no registered extractor; excluded from training.",
                name,
            )
            continue
        if base in enabled:
            kept.append(name)
    n_dropped = len(names) - len(kept)
    if n_dropped:
        logger.info(
            "extractor filter: %d artifact(s) on disk excluded "
            "(backbone not in extractors_enabled).",
            n_dropped,
        )
    return kept


def _iter_cells(
    condition: str,
    config: dict,
    processed_dir: str,
    embeddings_dir: str,
    model_names: list[str],
) -> Iterator[_Cell]:
    """Yield every eligible training cell for *condition*.

    Single source of truth for cell eligibility (dataset enumeration,
    frozen/fine-tuned embedding filtering, per-model visual/component
    source routing, embedding-path resolution).  Both the grid backend
    (:func:`build_job_list`) and the Optuna backend (:func:`_list_cells`)
    consume this so their notions of "which cells exist" can never drift.
    """
    dim_filter = config.get("embedding_dims", [])
    variant = _resolve_embedding_variants(config)

    for dataset_name in config.get("datasets", []):
        all_embs = get_embedding_files(embeddings_dir, dataset_name, dim_filter or None)
        all_embs = filter_by_variant(all_embs, variant)
        all_embs = filter_by_enabled_fusions(all_embs, config)
        all_embs = filter_by_enabled_extractors(all_embs, config)
        if condition == "frozen":
            embedding_names = [e for e in all_embs if not is_finetuned_artifact(e)]
        else:
            embedding_names = [e for e in all_embs if is_finetuned_artifact(e)]

        with open(Path(processed_dir) / dataset_name / "user2idx.json") as f:
            n_users = len(json.load(f))
        with open(Path(processed_dir) / dataset_name / "item2idx.json") as f:
            n_items = len(json.load(f))

        for model_name in model_names:
            spec = get_recommender_spec(model_name)
            if not spec.requires_visual:
                # Models that ignore visual features (e.g. plain BPR) only
                # run in the frozen condition with embedding_name="none".
                sources = ["none"] if condition == "frozen" else []
            else:
                sources = [
                    e
                    for e in embedding_names
                    if is_component_artifact(e) == spec.requires_components
                ]

            for emb_name in sources:
                if emb_name == "none":
                    emb_path = None
                else:
                    emb_path = _resolve_embedding_path(embeddings_dir, dataset_name, emb_name)
                    if emb_path is None:
                        continue
                yield _Cell(
                    dataset_name=dataset_name,
                    model_name=model_name,
                    spec=spec,
                    embedding_name=emb_name,
                    embedding_path=emb_path,
                    n_users=n_users,
                    n_items=n_items,
                )


class EnabledRecommenderHasNoCellsError(RuntimeError):
    """An enabled recommender enumerated zero training cells (audit D5).

    Silently dropping a model out of the comparison (e.g. ACF enabled but
    no ``*_comp.npy`` artifact exists) would bias the battery without any
    signal; enumeration therefore fails loud instead.
    """


class TrainingJobsFailedError(RuntimeError):
    """Required training work failed or was never accounted for (audit F04).

    Raised by the grid and parallel-Optuna backends after every unit of
    work has been collected, so the completed cells keep their artifacts
    on disk while ``main.py`` still exits non-zero and the run manifest
    records the failure.  ``failures`` lists one dict per non-successful
    unit (``id``, ``status``, ``error``); ``total`` is the number
    submitted.
    """

    #: Failures quoted verbatim in the message; the rest are summarised.
    _QUOTED = 10

    def __init__(self, failures: list[dict], total: int, *, unit: str) -> None:
        self.failures = failures
        self.total = total
        self.unit = unit
        super().__init__(self._summary())

    def _summary(self) -> str:
        counts: dict[str, int] = {}
        for failure in self.failures:
            counts[failure["status"]] = counts.get(failure["status"], 0) + 1
        breakdown = ", ".join(f"{n} {status}" for status, n in sorted(counts.items()))
        quoted = "; ".join(
            f"{f['id']}: {f['status']} ({f.get('error') or 'no detail'})"
            for f in self.failures[: self._QUOTED]
        )
        more = len(self.failures) - self._QUOTED
        tail = f"; ... {more} more" if more > 0 else ""
        return (
            f"{len(self.failures)} of {self.total} {self.unit}s did not succeed "
            f"({breakdown}): {quoted}{tail}"
        )


def _raise_if_work_failed(
    results: list[dict],
    expected_ids: list[str],
    *,
    id_key: str,
    unit: str,
) -> None:
    """Reconcile *results* against *expected_ids*; raise on any shortfall.

    A unit counts as failed when its result status is not ``ok`` and as
    ``unaccounted`` when no result carries its id at all -- a worker
    that died without publishing, or a message lost with it.  Duplicate
    results for one id are tolerated (first wins).  Success is exactly
    ``submitted == succeeded``; a log line is never a substitute for
    the exception this raises.
    """
    seen: dict[str, dict] = {}
    for result in results:
        seen.setdefault(str(result.get(id_key)), result)

    failures: list[dict] = []
    for unit_id in expected_ids:
        result = seen.get(unit_id)
        if result is None:
            failures.append({"id": unit_id, "status": "unaccounted", "error": None})
        elif result.get("status") != "ok":
            failures.append(
                {"id": unit_id, "status": result.get("status"), "error": result.get("error")}
            )
    if not failures:
        return
    for failure in failures:
        logger.error("  %s %s: %s (%s)", unit, failure["id"], failure["status"], failure["error"])
    raise TrainingJobsFailedError(failures, len(expected_ids), unit=unit)


def assert_enabled_recommenders_have_cells(
    cell_counts: dict[str, int],
    condition: str,
) -> None:
    """Fail loud when an enabled recommender silently drops out (D5).

    Args:
        cell_counts: Per enabled (and registered) recommender, the number
            of ``(dataset, model, embedding)`` cells enumerated for it.
        condition: ``"frozen"`` or ``"finetuned"`` — non-visual models
            are legitimately absent from the finetuned condition.

    Raises:
        EnabledRecommenderHasNoCellsError: When a recommender has zero
            cells while other recommenders enumerated at least one and
            the emptiness is not condition-expected.
    """
    if not any(cell_counts.values()):
        # Nothing at all to train for this condition (e.g. no finetuned
        # artifacts yet): the existing "no pending jobs" paths report it.
        return
    for model_name, count in cell_counts.items():
        if count > 0:
            continue
        spec = get_recommender_spec(model_name)
        if not spec.requires_visual and condition != "frozen":
            # Feature-blind baselines (plain BPR) only run frozen.
            continue
        if spec.requires_components:
            reason = (
                "it requires component embeddings and no *_comp.npy artifact "
                "matched the enabled datasets/filters — run an extractor that "
                "emits component embeddings, or disable the recommender"
            )
        else:
            reason = (
                "no embedding artifact matched the enabled datasets/filters "
                f"(condition={condition!r}, embedding_variants, embedding_dims)"
            )
        raise EnabledRecommenderHasNoCellsError(
            f"recommender {model_name!r} is enabled but enumerated 0 training "
            f"cells for condition {condition!r}: {reason}. It would silently "
            "drop out of the comparison."
        )


def _cell_counts(
    condition: str,
    config: dict,
    processed_dir: str,
    embeddings_dir: str,
) -> dict[str, int]:
    """Cells per enabled model, BEFORE completed-job filtering (D5 guard)."""
    model_names = _resolve_model_names(config)
    counts = {name: 0 for name in model_names}
    for cell in _iter_cells(condition, config, processed_dir, embeddings_dir, model_names):
        counts[cell.model_name] += 1
    return counts


def build_job_list(
    condition: str,
    config: dict,
    processed_dir: str,
    embeddings_dir: str,
    device: str,
) -> list[TrainingJob]:
    """Return the list of pending training jobs for the given condition.

    Every job carries the content identity of its dataset and feature
    artifact (resolved once per cell here, in the parent), and completed
    work is read from the configuration's checkpoint root rather than
    the repository default.
    """
    checkpoint_mgr = CheckpointManager(checkpoint_root(config))
    jobs: list[TrainingJob] = []
    identities: dict[tuple[str, str | None], DataIdentity | None] = {}

    enabled = config.get("recommenders_enabled")
    if enabled is None or not enabled:
        logger.warning(
            "recommenders_enabled is missing or empty in configs/recommenders.yaml, "
            "no training jobs will be scheduled. Add e.g. recommenders_enabled: "
            "[bpr, vbpr] to enable them. Registered recommenders: %s",
            ", ".join(registered_recommender_names()),
        )
        return jobs

    unknown = [m for m in enabled if not is_registered(m)]
    if unknown:
        logger.warning(
            "recommenders_enabled lists unregistered models (skipped): %s. "
            "Registered recommenders: %s",
            ", ".join(sorted(unknown)),
            ", ".join(registered_recommender_names()),
        )
    # Iterate in (priority, name) order so cheaper models train first.
    model_names = _resolve_model_names(config)

    for cell in _iter_cells(condition, config, processed_dir, embeddings_dir, model_names):
        experiment_key = f"{cell.dataset_name}_{cell.embedding_name}_{cell.model_name}"
        completed = checkpoint_mgr.load_grid_search_progress(experiment_key)
        data_identity = _cell_data_identity(identities, cell, processed_dir)
        context = build_identity_context(data_identity, condition=condition)
        completed_digests = _completed_identity_digests(completed, experiment_key)

        for hp in get_hyperparam_grid(cell.model_name, config):
            digest = _job_identity_digest(cell, hp, config, context)
            if digest in completed_digests:
                continue

            jobs.append(
                TrainingJob(
                    dataset_name=cell.dataset_name,
                    model_name=cell.model_name,
                    embedding_name=cell.embedding_name,
                    hyperparams=hp,
                    n_users=cell.n_users,
                    n_items=cell.n_items,
                    embeddings_path=cell.embedding_path,
                    processed_dir=processed_dir,
                    device=device,
                    priority=cell.spec.priority,
                    data_identity=data_identity.to_payload() if data_identity else None,
                )
            )

    return jobs


def _completed_identity_digests(completed: list[dict], experiment_key: str) -> set[str]:
    """Identity digests of the grid entries that may be reused (E04, Q13).

    A grid-progress entry is reusable only when it carries the identity
    digest of the run that produced it; entries written before the field
    existed match nothing — they are legacy, identified explicitly, and
    the configuration is trained again rather than assumed complete.
    """
    digests = {str(c["identity_digest"]) for c in completed if c.get("identity_digest")}
    n_legacy = len(completed) - len(digests)
    if n_legacy:
        logger.warning(
            "%s: %d grid-progress entr%s carry no identity digest (legacy); not reused.",
            experiment_key,
            n_legacy,
            "y" if n_legacy == 1 else "ies",
        )
    return digests


def _job_identity_digest(cell: _Cell, hyperparams: dict, config: dict, context: dict) -> str:
    """Digest of the C02 identity a job with *hyperparams* will train under."""
    from src.recommenders import get_recommender_class
    from src.utils.training import resolve_training_identity

    return canonical_digest(
        resolve_training_identity(
            model_cls=get_recommender_class(cell.model_name),
            model_name=cell.model_name,
            dataset_name=cell.dataset_name,
            embedding_name=cell.embedding_name,
            hyperparams=hyperparams,
            config=config,
            identity_context=context,
        )
    )


def identity_context_for(
    processed_dir: str,
    dataset_name: str,
    embedding_name: str,
    embeddings_path: str | None,
    *,
    fold: dict | None = None,
) -> dict:
    """Identity context of one training call (Optuna trial, replay, fold).

    Resolves the data identity through the process-level digest cache;
    unreadable inputs leave it unresolved (warned), never guessed.
    """
    from src.utils.identity import condition_of

    try:
        data = resolve_data_identity(
            processed_dir, dataset_name, embeddings_path, splits=SELECTION_SPLITS
        )
    except IdentityError as exc:
        logger.warning("%s/%s: data identity unresolved (%s).", dataset_name, embedding_name, exc)
        data = None
    return build_identity_context(data, condition=condition_of(embedding_name), fold=fold)


def _cell_data_identity(
    cache: dict[tuple[str, str | None], DataIdentity | None],
    cell: _Cell,
    processed_dir: str,
) -> DataIdentity | None:
    """Content identity of *cell*'s dataset + feature, memoised per cell key.

    ``None`` when the split files or the artifact cannot be read: the
    job is then recorded with an *unresolved* identity (never a guessed
    one) and the training call fails on the missing input itself.
    """
    key = (cell.dataset_name, cell.embedding_path)
    if key not in cache:
        try:
            cache[key] = resolve_data_identity(
                processed_dir,
                cell.dataset_name,
                cell.embedding_path,
                splits=SELECTION_SPLITS,
            )
        except IdentityError as exc:
            logger.warning(
                "%s/%s: data identity unresolved (%s); jobs of this cell carry no content digests.",
                cell.dataset_name,
                cell.embedding_name,
                exc,
            )
            cache[key] = None
    return cache[key]


def run(condition: str = "frozen", workers: int = 0, sequential: bool = False) -> None:
    """Dispatch the hyperparameter search for the given condition.

    Parameters
    ----------
    condition:
        ``"frozen"`` or ``"finetuned"``, selects which embedding files
        are eligible for the search.
    workers:
        Number of parallel workers (``0`` = auto-detect via VRAM).
        Grid parallelises over jobs; ``optuna`` parallelises over
        cells (capped at 3 — see :func:`_resolve_optuna_workers`).
    sequential:
        Force a single worker regardless of ``workers``.
    """
    if condition not in {"frozen", "finetuned"}:
        raise ValueError(f"condition must be 'frozen' or 'finetuned', got {condition!r}")

    config = load_config()
    set_seed(config["seed"])
    # Computational limits, validated before any gate runs (a removed
    # key such as hp_search.workers fails here, naming its replacement).
    resources = resolve_resources(config)

    if not config.get("datasets"):
        logger.info("train step skipped: datasets list is empty in configs/default.yaml.")
        return
    if not config.get("recommenders_enabled"):
        logger.info(
            "train step skipped: recommenders_enabled is empty in configs/recommenders.yaml.",
        )
        return

    logger.info("Condition: %s", condition)

    # Fairness guard-rail (Task H): no recommender may declare its own
    # protocol budget — the budget is shared per dataset. Fail before
    # training rather than confound the comparison silently.
    from src.recommenders.hp_budget import assert_uniform_budget, resolve_hp_budget

    assert_uniform_budget(config)
    # Resolve every dataset's budget up front so an unsupported selection
    # metric or a malformed value fails before any cell trains (R03).
    for dataset_name in config.get("datasets", []):
        resolve_hp_budget(config, dataset_name)
    # Dimension-parity guard-rail: every recommender draws its dimensions
    # from the shared common.total_dim budget (H1 controls capacity).
    assert_dimension_parity(config)

    # Feature sanity gate (Task G): fail loud before burning battery time
    # on a corrupt matrix. Validates every backbone + fused .npy consumed.
    from src.steps.validate_features import gate_dataset_features

    gate_dataset_features(
        config.get("datasets", []),
        config,
        embeddings_dir=config["paths"]["embeddings"],
        processed_dir=config["paths"]["data_processed"],
    )

    # No global cleanup of ``checkpoints/training/`` here (E04): a resume
    # envelope belongs to the run whose identity it carries, and
    # ``train_single_run`` validates that identity before reuse.  Deleting
    # every file at startup destroyed another run's resumable state.
    strategy = get_strategy(config)
    logger.info("Hyperparameter-search strategy: %s", strategy)

    # resources.workers.training is the default; an explicit CLI value
    # still wins.  workers=1 disables the process pool entirely; 0 = auto.
    effective_workers = workers or resources.workers.training
    if strategy == "optuna":
        _run_optuna(condition, config, workers=effective_workers, sequential=sequential)
    elif strategy == "fixed":
        _run_fixed(condition, config, workers=effective_workers, sequential=sequential)
    else:
        _run_grid(condition, config, workers=effective_workers, sequential=sequential)


def _run_fixed(
    condition: str,
    config: dict,
    *,
    workers: int,
    sequential: bool,
) -> None:
    """Train every cell exactly once with the pinned hyperparameters.

    ``hp_search.strategy: fixed`` performs no search: each
    ``(dataset, model, embedding)`` cell is trained a single time via
    :func:`train_replay` with :func:`get_fixed_hyperparams`, so no
    Optuna study is created or read and the ``battery.db`` storage is
    never opened.  The pinned configuration is resolved for every
    enabled recommender *before* any training starts, so a multi-valued
    key fails loud up front instead of after hours of work.

    NOTE: cells run sequentially in the calling process.  ``workers``
    and ``sequential`` are accepted for signature parity with the other
    backends but do not enable a worker pool yet; with one cell per
    configuration the pool would buy little, and a single process keeps
    the GPU budget identical to the pinned ``resources.workers.training: 1``.
    """
    device = resolve_device(config["device"])
    processed_dir = config["paths"]["data_processed"]
    embeddings_dir = config["paths"]["embeddings"]

    # Fail loud before enumerating anything: every enabled recommender
    # must resolve to exactly one configuration under ``fixed``.
    pinned = {name: get_fixed_hyperparams(name, config) for name in _resolve_model_names(config)}

    cells = _list_cells(condition, config, processed_dir, embeddings_dir)
    if workers > 1 and not sequential:
        logger.info("fixed strategy runs cells sequentially; workers=%d ignored.", workers)
    logger.info("Fixed-hyperparameter cells to train: %d (one run each)", len(cells))

    # D5: an enabled recommender with zero cells must fail, not vanish.
    counts = {name: 0 for name in pinned}
    for cell_key, _n_users, _n_items, _emb_path in cells:
        counts[cell_key.model_name] += 1
    assert_enabled_recommenders_have_cells(counts, condition)

    for cell, n_users, n_items, emb_path in tqdm(
        cells, desc="Training (fixed cells)", unit="cell", disable=None
    ):
        logger.info(
            "=== Fixed cell: %s  hyperparams=%s ===", cell.study_name(), pinned[cell.model_name]
        )
        metric = train_replay(
            cell=cell,
            hyperparams=pinned[cell.model_name],
            n_users=n_users,
            n_items=n_items,
            embeddings_path=emb_path,
            processed_dir=processed_dir,
            device=device,
            config=config,
        )
        logger.info("  cell %s: val metric=%.4f", cell.study_name(), metric)


def _run_grid(
    condition: str,
    config: dict,
    *,
    workers: int,
    sequential: bool,
) -> None:
    """Original Cartesian grid behaviour, dispatched via the orchestrator."""
    from src.recommenders.hp_budget import grid_budget_message

    device = resolve_device(config["device"])
    processed_dir = config["paths"]["data_processed"]
    embeddings_dir = config["paths"]["embeddings"]

    # D1: the grid backend spends one selection shot per config, so
    # unequal per-model spaces are unequal budgets. Cannot be fixed
    # silently (spaces are legitimate per-model choices) — warn loud.
    grid_sizes = {
        name: len(get_hyperparam_grid(name, config)) for name in _resolve_model_names(config)
    }
    budget_warning = grid_budget_message(grid_sizes)
    if budget_warning:
        logger.warning(budget_warning)

    # D5: an enabled recommender with zero cells must fail, not vanish.
    assert_enabled_recommenders_have_cells(
        _cell_counts(condition, config, processed_dir, embeddings_dir),
        condition,
    )

    jobs = build_job_list(condition, config, processed_dir, embeddings_dir, device)

    if not jobs:
        logger.info("No pending jobs. All experiments already completed.")
        return

    logger.info("Total pending jobs: %d", len(jobs))

    # M05: admit jobs against the resolved host budget BEFORE launching
    # anything.  A job whose declared minimum exceeds the budget is
    # refused once (a failed outcome), never launched repeatedly.
    n_workers = 1 if sequential else workers
    admitted, refused, plan = plan_training_admission(
        jobs, processed_dir, config, requested_workers=n_workers
    )
    results = [_refused_result(job, reason) for job, reason in refused]
    if admitted:
        # The resolved configuration travels with the pool: a spawned
        # worker must never rebuild it from the YAML defaults (F05).
        orchestrator = TrainingOrchestrator(
            n_workers=plan.n_workers if not sequential else 1,
            device=device,
            log_dir="logs",
            per_worker_bytes=plan.per_worker_bytes,
            config=config,
            admission=plan,
        )
        results.extend(orchestrator.run(admitted))

    ok = sum(1 for r in results if r.get("status") == "ok")
    logger.info("Training complete: %d/%d experiments succeeded.", ok, len(jobs))
    # Completed jobs already wrote their checkpoints and grid progress;
    # the exception only denies the step (and the run) a success marker.
    _raise_if_work_failed(results, [j.job_id for j in jobs], id_key="job_id", unit="job")


#: Host RAM a training worker needs on top of its data: the Python
#: interpreter, the imported torch stack and the process's CUDA context.
_WORKER_BASE_BYTES = 1536 * 1024**2

#: Interaction dicts (``{user: set(items)}``) are far larger in memory
#: than the CSV they come from — boxed ints inside per-user sets.  This
#: multiplier converts the on-disk CSV size into a resident estimate.
_INTERACTIONS_MEMORY_FACTOR = 40

#: Blocks of ``resources.features.item_block`` rows charged to a lazy job
#: (source rows on the host plus their cast/transfer copy).
_LAZY_BLOCKS_CHARGED = 2


@dataclass(frozen=True)
class JobMemoryEstimate:
    """Ledger of one training job's host footprint (M05, SPEC "Memory ledger")."""

    feature_bytes: int
    model_bytes: int
    interactions_bytes: int
    ranking_bytes: int
    base_bytes: int = _WORKER_BASE_BYTES

    @property
    def total(self) -> int:
        return (
            self.base_bytes
            + self.feature_bytes
            + self.model_bytes
            + self.interactions_bytes
            + self.ranking_bytes
        )


def _file_size(path: str | Path) -> int:
    try:
        return Path(path).stat().st_size
    except OSError:
        return 0


def feature_payload_bytes(path: str | Path | None) -> tuple[int, int]:
    """``(raw payload bytes, visual width)`` of a feature artifact.

    A ``.npy`` is measured from its header; an online-fusion sidecar sums
    the payloads of its components (its own JSON size says nothing about
    what a worker loads).  Unreadable inputs count as ``(0, 0)``.
    """
    if path is None:
        return 0, 0
    file = Path(path)
    try:
        if file.suffix == ".json":
            sidecar = json.loads(file.read_text(encoding="utf-8"))
            parts = [npy_payload_bytes(file.parent / c) for c in sidecar.get("components", [])]
            return sum(b for b, _ in parts), sum(w for _, w in parts)
        return npy_payload_bytes(file)
    except (OSError, ValueError, KeyError):
        return 0, 0


def _resident_feature_bytes(payload: int, width: int, *, lazy: bool, item_block: int) -> int:
    if not lazy:
        return payload
    return min(payload, item_block * width * 4 * _LAZY_BLOCKS_CHARGED)


def estimate_job_bytes(
    job: TrainingJob, processed_dir: str, config: dict, *, lazy: bool
) -> JobMemoryEstimate:
    """Host bytes one worker needs for *job*: payload, model/optimizer, data, ranking."""
    from src.evaluation.protocol import host_ranking_bytes

    payload, width = feature_payload_bytes(job.embeddings_path)
    item_block = resolve_resources(config).features.item_block
    hp = job.hyperparams
    total_dim = (
        hp.get("total_dim") or hp.get("latent_dim") or config.get("common", {}).get("total_dim")
    )
    inter = _file_size(Path(processed_dir) / job.dataset_name / "train.csv") + _file_size(
        Path(processed_dir) / job.dataset_name / "val.csv"
    )
    return JobMemoryEstimate(
        feature_bytes=_resident_feature_bytes(payload, width, lazy=lazy, item_block=item_block),
        model_bytes=estimate_model_state_bytes(
            job.n_users, job.n_items, total_dim, visual_dim=width
        ),
        interactions_bytes=inter * _INTERACTIONS_MEMORY_FACTOR,
        ranking_bytes=host_ranking_bytes(job.n_items),
    )


def plan_training_admission(
    jobs: list[TrainingJob],
    processed_dir: str,
    config: dict,
    *,
    requested_workers: int,
) -> tuple[list[TrainingJob], list[tuple[TrainingJob, str]], AdmissionPlan]:
    """Split *jobs* into admitted / refused and size the pool (M05, Q10).

    Each job is charged its feature payload (source bytes, not sidecar
    size), model + optimizer state, interaction dicts and the host
    ranking workspace, on top of the worker base.  Under
    ``resources.features.residency: auto`` a job whose dense payload does
    not fit is switched to lazy reads before the verdict.  Jobs whose
    declared minimum still exceeds ``budget - headroom`` are refused; the
    pool is sized so the aggregate commitment of the admitted jobs'
    heaviest estimate fits, capped by the requested/auto worker count
    and the CPU quota.
    """
    resources = resolve_resources(config)
    budget = resolve_host_budget(config)
    residency = resources.features.residency
    headroom = resources.host.headroom_bytes
    usable = max(0, budget.limit_bytes - headroom)
    admitted: list[TrainingJob] = []
    refused: list[tuple[TrainingJob, str]] = []
    heaviest = 0
    for job in jobs:
        payload, _ = feature_payload_bytes(job.embeddings_path)
        job.lazy_features = choose_lazy_features(residency, payload, usable)
        estimate = estimate_job_bytes(job, processed_dir, config, lazy=job.lazy_features)
        if estimate.total > usable:
            refused.append((job, _refusal_reason(estimate, usable, budget.source)))
            continue
        heaviest = max(heaviest, estimate.total)
        admitted.append(job)
    hard_cap = requested_workers if requested_workers > 0 else max(1, available_cpus() - 1)
    plan = admit_workers(
        heaviest,
        hard_cap=hard_cap,
        budget=budget,
        headroom_bytes=headroom,
        label="training pool",
    )
    logger.info(
        "Admission: %d job(s) admitted, %d refused; budget %.2f GB (%s), headroom %.2f GB, "
        "heaviest job %.2f GB, workers %d, feature residency %s.",
        len(admitted),
        len(refused),
        budget.limit_bytes / 1024**3,
        budget.source,
        headroom / 1024**3,
        heaviest / 1024**3,
        plan.n_workers,
        residency,
    )
    for job, reason in refused:
        logger.error("  refused %s: %s", job.job_id, reason)
    return admitted, refused, plan


def _refusal_reason(estimate: JobMemoryEstimate, usable: int, source: str) -> str:
    gb = 1024**3
    return (
        f"declared minimum {estimate.total / gb:.2f} GB (features {estimate.feature_bytes / gb:.2f}, "
        f"model+optimizer {estimate.model_bytes / gb:.2f}, interactions "
        f"{estimate.interactions_bytes / gb:.2f}, ranking {estimate.ranking_bytes / gb:.2f}, base "
        f"{estimate.base_bytes / gb:.2f}) exceeds the usable host budget {usable / gb:.2f} GB "
        f"({source}); not launched"
    )


def _refused_result(job: TrainingJob, reason: str) -> dict:
    return {
        "job_id": job.job_id,
        "outcome": "failed",
        "attempts": 0,
        "status": "error",
        "error": reason,
        "error_type": "AdmissionRefused",
    }


def _estimate_worker_bytes(jobs: list[TrainingJob], processed_dir: str) -> int:
    """Estimate the host RAM one training worker holds (heaviest job).

    Kept for callers of the historical helper; the admission planner
    above is what sizes the pool.  Returns ``0`` when nothing can be
    measured (``detect_max_workers`` reads that as "unknown").
    """
    estimates = [estimate_job_bytes(job, processed_dir, {}, lazy=job.lazy_features) for job in jobs]
    measured = [e.total for e in estimates if e.feature_bytes or e.interactions_bytes]
    return max(measured, default=0)


def lazy_features_for(config: dict, embeddings_path: str | Path | None) -> bool:
    """Whether a single (non-pooled) training/evaluation call should read lazily.

    ``resources.features.residency``: ``dense`` (default) keeps today's
    resident matrices, ``lazy`` always reads bounded rows, ``auto``
    switches when the dense payload would not fit the usable budget.
    """
    if embeddings_path is None:
        return False
    resources = resolve_resources(config)
    residency = resources.features.residency
    if residency == "dense":
        return False
    budget = resolve_host_budget(config)
    payload, _ = feature_payload_bytes(embeddings_path)
    usable = max(0, budget.limit_bytes - resources.host.headroom_bytes)
    return choose_lazy_features(residency, payload, usable)


def _legit_trial_count(study) -> int:
    """Number of legitimate HPO outcomes (COMPLETE + PRUNED) in *study*.

    ``len(study.trials)`` also counts FAIL trials (infra crashes such as
    a corrupt-embedding load) and stale RUNNING trials (process killed
    mid-trial). Counting those toward ``n_trials`` truncated or skipped
    the search for affected cells. Only COMPLETE and PRUNED are real
    search outcomes that may consume the trial budget.
    """
    return sum(1 for t in study.trials if t.state.name in ("COMPLETE", "PRUNED"))


def _resolve_optuna_workers(workers: int, device: str, n_cells: int, *, reserve_bytes: int) -> int:
    """Worker count for inter-cell Optuna parallelism.

    Reuses the VRAM heuristic of the grid orchestrator but caps the pool
    at 3: an Optuna worker holds a full study (data + model + evaluator)
    for the whole cell, and 3 concurrent training processes is the
    empirically verified ceiling on the reference 24 GB pod.  Never more
    workers than cells.  *reserve_bytes* is ``resources.host.reserved_bytes``.
    """
    from src.utils.parallel import detect_max_workers

    n = detect_max_workers(device, reserve_bytes=reserve_bytes) if workers <= 0 else workers
    return max(1, min(n, 3, n_cells))


def _optimize_one_cell(
    cell: CellKey,
    n_users: int,
    n_items: int,
    emb_path: str | None,
    *,
    config: dict,
    processed_dir: str,
    device: str,
    log=logger,
    ranking_budget_bytes: int | None = None,
) -> dict:
    """Create/load the study for *cell* and run its remaining trials.

    Runs in the parent (sequential mode) or inside a worker process
    (parallel mode); the study is always created in the executing
    process so in-memory storage never crosses a process boundary.
    """
    import optuna

    from src.recommenders.hp_budget import resolve_hp_budget

    optuna_cfg = config["hp_search"]["optuna"]
    # Single source of the protocol budget, shared by every recommender of
    # this dataset (Task H); per-dataset override via ``hp_budget:``.
    n_trials = int(resolve_hp_budget(config, cell.dataset_name)["n_trials"])
    timeout = optuna_cfg.get("timeout_seconds")

    log.info("=== Optuna cell: %s ===", cell.study_name())
    study = create_study(cell, config)

    def _objective(trial):
        hp = sample_hyperparams(trial, cell.model_name, config)
        return _train_one_optuna_trial(
            cell=cell,
            hyperparams=hp,
            n_users=n_users,
            n_items=n_items,
            embeddings_path=emb_path,
            processed_dir=processed_dir,
            device=device,
            config=config,
            trial=trial,
            ranking_budget_bytes=ranking_budget_bytes,
        )

    existing = _legit_trial_count(study)
    remaining = max(0, n_trials - existing)
    if remaining == 0:
        log.info(
            "  cell %s: already has %d legit trials >= n_trials=%d, skipping",
            cell.study_name(),
            existing,
            n_trials,
        )
    else:
        study.optimize(
            _objective,
            n_trials=remaining,
            timeout=timeout,
            gc_after_trial=True,
            show_progress_bar=False,
        )

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    summary = {
        "cell": cell.study_name(),
        "status": "ok",
        "completed": len(completed),
        "pruned": len(pruned),
        "best_value": study.best_value if completed else 0.0,
        "best_params": study.best_params if completed else {},
    }
    log.info(
        "  cell %s: %d completed, %d pruned. best_value=%.4f best_params=%s",
        summary["cell"],
        summary["completed"],
        summary["pruned"],
        summary["best_value"],
        summary["best_params"],
    )
    return summary


#: OOM retries per Optuna cell; each retry shrinks the ranking budget.
_OOM_MAX_RETRIES = 2
_OOM_SHRINK = 0.6


def _optimize_cell_with_oom_retry(
    cell: CellKey,
    n_users: int,
    n_items: int,
    emb_path: str | None,
    *,
    config: dict,
    processed_dir: str,
    device: str,
    log=logger,
) -> dict:
    """Run one Optuna cell, retrying CUDA OOM with a shrinking budget.

    Mirrors the grid orchestrator's OOM handling at cell granularity:
    a transient OOM (fragmentation, a sibling worker peaking) must not
    cost the battery a whole cell.  Each retry empties the CUDA cache
    and shrinks the explicit ranking budget by ``_OOM_SHRINK`` starting
    from the allowance-derived default.
    """
    import torch

    budget: int | None = None
    for attempt in range(_OOM_MAX_RETRIES + 1):
        try:
            return _optimize_one_cell(
                cell,
                n_users,
                n_items,
                emb_path,
                config=config,
                processed_dir=processed_dir,
                device=device,
                log=log,
                ranking_budget_bytes=budget,
            )
        except torch.OutOfMemoryError:
            if attempt == _OOM_MAX_RETRIES:
                raise
            torch.cuda.empty_cache()
            if budget is None:
                from src.evaluation.protocol import default_ranking_budget

                budget = default_ranking_budget(torch.device(device))
            budget = max(1, int(budget * _OOM_SHRINK))
            log.warning(
                "  OOM on cell %s (attempt %d/%d) — retrying with ranking budget %.2f GB",
                cell.study_name(),
                attempt + 1,
                _OOM_MAX_RETRIES,
                budget / 1024**3,
            )
    raise AssertionError("unreachable")  # pragma: no cover


def _optuna_cell_worker(
    worker_id: int,
    cell_queue,
    result_queue,
    n_workers: int,
    config: dict,
    processed_dir: str,
    device: str,
) -> None:
    """Worker process: pulls whole cells and runs their studies.

    Mirrors :func:`src.utils.parallel._worker_fn` (memory fraction,
    isolation of failures per unit of work) at cell granularity.
    """
    from queue import Empty as _Empty

    from src.utils.device import cap_process_vram
    from src.utils.logging import get_logger as _get_logger

    wlog = _get_logger(f"optuna_worker_{worker_id}")

    # Capped for any n: the sum of the pool's caps stays below the card
    # (a CUDA context lives outside the cap, on top of it), and a lone
    # worker still leaves the display its headroom.
    cap_process_vram(n_workers, vram_share=resolve_resources(config).gpu.vram_share)

    while True:
        try:
            item = cell_queue.get(timeout=5)
        except _Empty:
            break
        if item is None:
            break

        cell, n_users, n_items, emb_path = item
        try:
            summary = _optimize_cell_with_oom_retry(
                cell,
                n_users,
                n_items,
                emb_path,
                config=config,
                processed_dir=processed_dir,
                device=device,
                log=wlog,
            )
            result_queue.put(summary)
        except Exception as exc:  # noqa: BLE001, isolate failures per cell
            wlog.error("  Error on cell %s: %s", cell.study_name(), exc, exc_info=True)
            result_queue.put(
                {"cell": cell.study_name(), "status": "error", "error": str(exc)},
            )


def _run_optuna(
    condition: str,
    config: dict,
    *,
    workers: int = 0,
    sequential: bool = False,
) -> None:
    """Per-cell Optuna search with median pruning (parallel across cells).

    For each ``(dataset, model, embedding)`` cell we create (or load)
    an Optuna study and run ``hp_search.optuna.n_trials`` trials.
    Cells are independent studies, so they are dispatched to worker
    processes (B7); trials WITHIN a cell remain sequential, keeping the
    TPE sampler conditioned on every prior trial of its study.  Cells
    whose studies already hold ``n_trials`` legitimate outcomes are
    skipped, so a killed run resumes where it stopped (requires a
    persistent ``hp_search.optuna.storage``).
    """
    device = resolve_device(config["device"])
    processed_dir = config["paths"]["data_processed"]
    embeddings_dir = config["paths"]["embeddings"]
    from src.recommenders.hp_budget import resolve_hp_budget

    # The trial count is a per-dataset budget field (hp_budget override).
    n_trials = {ds: resolve_hp_budget(config, ds)["n_trials"] for ds in config.get("datasets", [])}

    cells = _list_cells(condition, config, processed_dir, embeddings_dir)
    logger.info("Optuna cells to process: %d (n_trials per dataset=%s)", len(cells), n_trials)

    # D5: an enabled recommender with zero cells must fail, not vanish.
    counts = {name: 0 for name in _resolve_model_names(config)}
    for cell_key, _n_users, _n_items, _emb_path in cells:
        counts[cell_key.model_name] += 1
    assert_enabled_recommenders_have_cells(counts, condition)

    if not cells:
        return

    resources = resolve_resources(config)
    n_workers = (
        1
        if sequential
        else _resolve_optuna_workers(
            workers, device, len(cells), reserve_bytes=resources.host.reserved_bytes
        )
    )

    if n_workers == 1:
        try:
            for cell, n_users, n_items, emb_path in cells:
                _optimize_one_cell(
                    cell,
                    n_users,
                    n_items,
                    emb_path,
                    config=config,
                    processed_dir=processed_dir,
                    device=device,
                )
        except KeyboardInterrupt:
            logger.warning("Optuna study interrupted by user.")
            raise
        return

    if config["hp_search"]["optuna"].get("storage") is None:
        logger.warning(
            "hp_search.optuna.storage is null: studies live in worker memory "
            "and completed-cell skip will not survive a restart. Set a "
            "sqlite storage for resumable parallel search.",
        )

    storage_url = config["hp_search"]["optuna"].get("storage")
    if storage_url:
        # Create the schema ONCE in the parent before any worker touches
        # the database: N workers racing RDBStorage's create_all on a
        # fresh sqlite file lose cells to "table studies already exists".
        import optuna

        if storage_url.startswith("sqlite:///"):
            Path(storage_url[len("sqlite:///") :]).parent.mkdir(parents=True, exist_ok=True)
        optuna.storages.RDBStorage(url=storage_url)

    import torch.multiprocessing as mp

    logger.info("Optuna inter-cell parallelism: %d workers", n_workers)
    ctx = mp.get_context("spawn")
    cell_queue = ctx.Queue()
    result_queue = ctx.Queue()
    for item in cells:
        cell_queue.put(item)
    for _ in range(n_workers):
        cell_queue.put(None)

    procs = []
    for i in range(n_workers):
        p = ctx.Process(
            target=_optuna_cell_worker,
            args=(i, cell_queue, result_queue, n_workers, config, processed_dir, device),
            daemon=True,
        )
        p.start()
        procs.append(p)

    # One slot per submitted cell: a duplicate delivery cannot end the
    # loop early and leave a real cell unaccounted.
    expected = [cell.study_name() for cell, _n_users, _n_items, _emb_path in cells]
    results: dict[str, dict] = {}
    total = len(cells)
    # Live cell-level battery bar (done/total, %, elapsed<ETA, rate).
    # Parent-side observability only — never touches worker computation.
    # Renders in place under a TTY (compose ``tty: true``); auto-quiet
    # off a TTY via ``disable=None``.
    import time as _time

    t_start = _time.monotonic()
    with tqdm(total=total, desc="Training (Optuna cells)", unit="cell", disable=None) as pbar:
        while len(results) < total:
            try:
                last = result_queue.get(timeout=30)
            except Exception:  # noqa: BLE001, queue.Empty from a spawn context
                if not any(p.is_alive() for p in procs):
                    logger.warning("All Optuna workers exited early.")
                    break
                continue
            if last.get("cell") in results:
                logger.warning("Ignoring duplicate result for cell %s", last.get("cell"))
                continue
            results[str(last.get("cell"))] = last
            pbar.update(1)
            # Plain-log ETA for `docker logs` followers, where the tqdm
            # bar does not render (no TTY): rate = cells done / elapsed.
            done = len(results)
            elapsed = _time.monotonic() - t_start
            eta_s = elapsed / done * (total - done)
            logger.info(
                "cell %d/%d done (%s, %s) — avg %.1f min/cell, ETA ~%dh%02dm",
                done,
                total,
                last.get("cell", "?"),
                last.get("status", "?"),
                elapsed / done / 60,
                int(eta_s // 3600),
                int(eta_s % 3600 // 60),
            )
    for p in procs:
        p.join(timeout=30)

    ok = sum(1 for r in results.values() if r.get("status") == "ok")
    logger.info("Optuna search complete: %d/%d cells succeeded.", ok, len(cells))
    # Studies of the completed cells are already persisted in the
    # storage; the exception denies the step its success marker only.
    _raise_if_work_failed(list(results.values()), expected, id_key="cell", unit="cell")


def _list_cells(
    condition: str,
    config: dict,
    processed_dir: str,
    embeddings_dir: str,
) -> list[tuple[CellKey, int, int, str | None]]:
    """Enumerate every ``(dataset, model, embedding)`` cell to optimise.

    Shares :func:`_iter_cells` with :func:`build_job_list` but stops at
    the cell granularity (no per-HP enumeration).
    """
    model_names = _resolve_model_names(config)
    return [
        (
            CellKey(cell.dataset_name, cell.model_name, cell.embedding_name),
            cell.n_users,
            cell.n_items,
            cell.embedding_path,
        )
        for cell in _iter_cells(condition, config, processed_dir, embeddings_dir, model_names)
    ]


def _train_one_optuna_trial(
    *,
    cell: CellKey,
    hyperparams: dict,
    n_users: int,
    n_items: int,
    embeddings_path: str | None,
    processed_dir: str,
    device: str,
    config: dict,
    trial=None,
    ranking_budget_bytes: int | None = None,
) -> float:
    """Single trial entry point: load data, train one model, return metric."""
    from src.fusions import load_embedding
    from src.recommenders import get_recommender_class
    from src.utils.training import train_single_run

    dataset_name = cell.dataset_name
    train_path = Path(processed_dir) / dataset_name / "train.csv"
    # Model selection (early stopping + the Optuna objective) runs on
    # VALIDATION users, masking each user's TRAIN items. The test set is
    # never read during training/selection — it is touched only by the
    # final evaluate step. Mirrors the grid worker (src/utils/parallel.py).
    val_path = Path(processed_dir) / dataset_name / "val.csv"

    import pandas as pd

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)

    train_interactions: dict = {}
    for u, i in zip(train_df["user_idx"], train_df["item_idx"], strict=False):
        train_interactions.setdefault(int(u), set()).add(int(i))
    val_interactions: dict = {}
    for u, i in zip(val_df["user_idx"], val_df["item_idx"], strict=False):
        val_interactions.setdefault(int(u), set()).add(int(i))

    visual_embeddings = None
    if embeddings_path is not None:
        visual_embeddings = load_embedding(
            embeddings_path, lazy=lazy_features_for(config, embeddings_path)
        )

    model_cls = get_recommender_class(cell.model_name)
    # Resume checkpoints live under the run's ``paths.checkpoints`` so a
    # seed-isolated config (battery replay, --seeds) gets its own
    # namespace instead of colliding on the default root (R01/E03).
    checkpoint_mgr = CheckpointManager(checkpoint_root(config))

    item_categories = None
    if getattr(model_cls, "wants_categories", False):
        from src.data.categories import item_category_array

        item_categories = item_category_array(dataset_name, processed_dir)

    return train_single_run(
        model_cls=model_cls,
        model_name=cell.model_name,
        n_users=n_users,
        n_items=n_items,
        visual_embeddings=visual_embeddings,
        train_interactions=train_interactions,
        selection_interactions=val_interactions,
        hyperparams=hyperparams,
        config=config,
        checkpoint_mgr=checkpoint_mgr,
        dataset_name=cell.dataset_name,
        embedding_name=cell.embedding_name,
        device=device,
        optuna_trial=trial,
        item_categories=item_categories,
        ranking_budget_bytes=ranking_budget_bytes,
        identity_context=identity_context_for(
            processed_dir, cell.dataset_name, cell.embedding_name, embeddings_path
        ),
    )


def train_replay(
    *,
    cell: CellKey,
    hyperparams: dict,
    n_users: int,
    n_items: int,
    embeddings_path: str | None,
    processed_dir: str,
    device: str,
    config: dict,
) -> float:
    """D2 replay: train ONE fixed config (no search), early stopping on val.

    The clean "train with a given config" entry point the battery runner
    (Task I) invokes to replicate a search's best config on the
    non-primary seeds.  Validation early stopping is active (via
    ``train_single_run``); ``config['seed']`` selects the seed.  Returns
    the best validation metric.  Orchestration across seeds is Task I's.
    """
    return _train_one_optuna_trial(
        cell=cell,
        hyperparams=hyperparams,
        n_users=n_users,
        n_items=n_items,
        embeddings_path=embeddings_path,
        processed_dir=processed_dir,
        device=device,
        config=config,
        trial=None,
    )
