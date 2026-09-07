"""Step 06, Final evaluation on the test set.

Loads the best model checkpoint produced during step 05 for every
``(dataset, model, embedding)`` combination and computes
precision/recall/F1/MAP/NDCG at the configured cut-offs.

Per-dataset partial CSVs are written incrementally so an interrupted
run can resume; the final ``{dataset}_evaluation_{condition}.csv`` is
written when every cell finishes.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.evaluation.protocol import Evaluator
from src.recommenders.registry import (
    get_recommender_spec,
    registered_recommender_names,
)
from src.utils.artifact_names import (
    BEST_SUFFIX,
    is_finetuned_artifact,
    parse_checkpoint_stem,
)
from src.utils.atomic_io import atomic_write
from src.utils.checkpoint import load_best_checkpoint
from src.utils.config import load_config
from src.utils.device import cap_process_vram, resolve_device
from src.utils.identity import (
    EVALUATION_SPLITS,
    IdentityError,
    canonical_digest,
    experiment_identity,
    implementation_digest,
    resolve_data_identity,
    stream_digest,
)
from src.utils.logging import get_logger
from src.utils.resources import resolve_resources
from src.utils.splits import assert_holdout_disjoint
from src.utils.timing import note_skipped_cell, time_cell
from src.utils.variant_filters import checkpoint_matches_config

logger = get_logger(__name__)


def _build_interactions(df: pd.DataFrame) -> dict[int, set[int]]:
    """Build ``{user_idx: set(item_idx)}`` from a (user_idx, item_idx) DataFrame."""
    interactions: dict[int, set[int]] = {}
    # zip over the two columns instead of iterrows(): the latter
    # materialises a Series per row, ~10x slower on million-row CSVs.
    for u, i in zip(df["user_idx"], df["item_idx"], strict=True):
        interactions.setdefault(int(u), set()).add(int(i))
    return interactions


def load_data(processed_dir: str, dataset_name: str):
    """Load processed data: train + val + test for final evaluation.

    For final evaluation, ``train`` and ``val`` are merged and used as the
    "seen" set whose items are masked from the candidate ranking.
    Metrics are then computed against the held-out ``test`` set.
    Per-user disjointness ``test ∩ (train ∪ val) = ∅`` is asserted (a
    duplicated held-out would be masked and become unhittable).

    Returns ``(n_users, n_items, seen_interactions, test_interactions,
    train_interactions)``.  The pure-train ``train_interactions`` (without
    val) is returned separately so history-consuming models (ACF) rebuild
    the exact same user profile they trained on, while the merged
    ``seen_interactions`` still drives candidate masking.
    """
    base = Path(processed_dir) / dataset_name
    train_df = pd.read_csv(base / "train.csv")
    val_df = pd.read_csv(base / "val.csv")
    test_df = pd.read_csv(base / "test.csv")

    with open(base / "user2idx.json") as f:
        user2idx = json.load(f)
    with open(base / "item2idx.json") as f:
        item2idx = json.load(f)

    n_users = len(user2idx)
    n_items = len(item2idx)

    train_interactions = _build_interactions(train_df)
    val_interactions = _build_interactions(val_df)

    seen_interactions: dict[int, set[int]] = {
        uid: set(items) for uid, items in train_interactions.items()
    }
    for uid, items in val_interactions.items():
        seen_interactions.setdefault(uid, set()).update(items)

    test_interactions = _build_interactions(test_df)
    # A3 guard: shared with the selection-side val ∩ train check (R6).
    assert_holdout_disjoint(seen_interactions, test_interactions, dataset_name, holdout_name="test")

    return n_users, n_items, seen_interactions, test_interactions, train_interactions


def build_evaluator(
    config: dict,
    seen_interactions: dict[int, set[int]],
    test_interactions: dict[int, set[int]],
    n_items: int,
) -> Evaluator:
    """Build the final-evaluation ``Evaluator`` from the ``evaluation:`` block.

    Single construction path shared by the sequential evaluate step
    (:func:`run`) and the battery executor
    (``src.battery.execute._evaluate_one_cell``), so both honour
    ``evaluation.protocol`` / ``n_negatives`` / ``negative_sampling_seed``
    with identical defaults.

    Tie-break rule (unified): ``tiebreak_seed`` is the run's ACTIVE seed —
    ``config['seed']``.  The battery executor sets ``config['seed']`` to
    the cell's seed (matching its per-seed results directories); the
    sequential step passes the global run config, whose ``seed`` is the
    run's seed.  Either way the permutation is shared by every
    model/trial of a ``(dataset, seed)`` run.
    """
    eval_cfg = config.get("evaluation") or {}
    protocol = eval_cfg.get("protocol", "full_ranking")
    n_negatives = eval_cfg.get("n_negatives", 100)
    if protocol == "sampled":
        logger.warning(
            "evaluate: protocol='sampled' selected (n_negatives=%d).  "
            "Sampled metrics are inconsistent with full-ranking "
            "(Krichene & Rendle 2020); use only for comparability with "
            "prior work, never as the headline benchmark.",
            n_negatives,
        )
    return Evaluator(
        seen_interactions,
        test_interactions,
        n_items,
        k_values=config.get("k_values", [5, 10, 20]),
        protocol=protocol,
        n_negatives=n_negatives,
        negative_sampling_seed=eval_cfg.get("negative_sampling_seed", 42),
        tiebreak_seed=int(config.get("seed", 42)),
    )


def find_best_models(dataset_name: str, results_dir: Path | str = "results") -> list[dict]:
    """List ``*_best.pt`` checkpoints saved during step 05 for a dataset.

    Filenames follow ``{model_name}_{embedding_name}_best.pt``.  We
    resolve the boundary by matching the longest registered recommender
    name as the *prefix*, naively splitting on the first underscore
    breaks for multi-token recommender names like ``uniform_noise``,
    which would otherwise be parsed as model=``uniform`` /
    embedding=``noise_<rest>``.
    """
    models_dir = Path(results_dir) / "models" / dataset_name
    if not models_dir.exists():
        return []

    # Longest first so e.g. ``uniform_noise`` matches before any
    # hypothetical ``uniform`` recommender.
    known_models = sorted(registered_recommender_names(), key=len, reverse=True)

    results: list[dict] = []
    for model_path in sorted(models_dir.glob("*_best.pt")):
        stem = model_path.stem.replace(BEST_SUFFIX, "")
        parsed = parse_checkpoint_stem(stem, known_models)
        if parsed is None:
            logger.warning("  Unrecognised checkpoint filename: %s", model_path.name)
            continue
        model_name, embedding_name = parsed

        results.append(
            {
                "model_name": model_name,
                "embedding_name": embedding_name,
                "path": str(model_path),
            }
        )
    return results


def _route_targets(model_name: str, embedding_name: str) -> list[str]:
    """Return the battery file(s) a cell's results belong to.

    The embedding name encodes the visual-backbone condition
    (``*_finetuned_*`` => fine-tuned backbone). ``bpr`` / the ``none``
    embedding is a non-visual baseline written to both batteries as the
    common reference.
    """
    if model_name == "bpr" or embedding_name == "none":
        return ["frozen", "finetuned"]
    if is_finetuned_artifact(embedding_name):
        return ["finetuned"]
    return ["frozen"]


def _write_mean_table(results_dir: Path, dataset_name: str, target: str) -> None:
    """Write ``{ds}_evaluation_mean_{target}.csv``: one MEAN row per config.

    The per-user ``{ds}_evaluation_{target}.csv`` is the statistical
    step's input and stays untouched; this companion is the
    dissertation-facing view — mean of every metric column per
    (model, embedding) plus ``n_users`` — so nobody has to aggregate
    per-user rows by hand (or, worse, rank them raw).
    """
    src = results_dir / f"{dataset_name}_evaluation_{target}.csv"
    if not src.exists():
        return
    df = pd.read_csv(src)
    if "user_id" not in df.columns or df.empty:
        return
    keys = ["model_name", "embedding_name"]
    metric_cols = [
        c
        for c in df.columns
        if c not in keys
        and c not in ("user_id", "dataset")
        and pd.api.types.is_numeric_dtype(df[c])
    ]
    mean_df = df.groupby(keys, as_index=False).agg(
        **{c: (c, "mean") for c in metric_cols},
        n_users=("user_id", "size"),
    )
    out = results_dir / f"{dataset_name}_evaluation_mean_{target}.csv"
    mean_df.to_csv(out, index=False)
    logger.info("  mean table: %d configs -> %s", len(mean_df), out)


def _done_path(results_dir: Path, dataset_name: str) -> Path:
    """Path of the per-dataset resume sidecar."""
    return results_dir / f"{dataset_name}_evaluation_done.csv"


_DONE_KEY = ["target", "model_name", "embedding_name"]
_DONE_BINDING = ["checkpoint_digest", "identity_digest"]


def _load_done_index(path: Path) -> dict[tuple[str, str, str], dict]:
    """Completed triples with the binding they were recorded under (E04).

    Legacy rows (written before the binding columns existed) map to an
    empty binding and therefore never match a current checkpoint.
    """
    if not path.exists():
        return {}
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    index: dict[tuple[str, str, str], dict] = {}
    for row in df.to_dict("records"):
        key = (str(row["target"]), str(row["model_name"]), str(row["embedding_name"]))
        index[key] = {c: row[c] for c in _DONE_BINDING if row.get(c)}
    return index


def _load_done(path: Path) -> set[tuple[str, str, str]]:
    """Load completed ``(target, model_name, embedding_name)`` triples."""
    return set(_load_done_index(path))


def _record_done(
    path: Path,
    rows: list[tuple[str, str, str]],
    *,
    bindings: dict[tuple[str, str, str], dict] | None = None,
) -> None:
    """Upsert completed triples into the done table, atomically.

    One row per triple (a repeated evaluation replaces its row instead
    of appending a duplicate); ``bindings`` carries the checkpoint and
    identity digests the completion is valid for.
    """
    index = _load_done_index(path)
    for key in rows:
        index[key] = dict((bindings or {}).get(key, {}))
    table = pd.DataFrame(
        [
            {**dict(zip(_DONE_KEY, key, strict=True)), **{c: v.get(c, "") for c in _DONE_BINDING}}
            for key, v in index.items()
        ],
        columns=_DONE_KEY + _DONE_BINDING,
    )
    atomic_write(lambda tmp: table.to_csv(tmp, index=False), path)


def _binding_matches(entry: dict | None, checkpoint_digest: str, identity_digest: str) -> bool:
    """Whether a done entry was recorded for exactly this winner and identity."""
    if not entry:
        return False
    return (
        entry.get("checkpoint_digest") == checkpoint_digest
        and entry.get("identity_digest") == identity_digest
    )


def _lazy_features(config: dict, embeddings_dir: str, dataset_name: str, stem: str) -> bool:
    from src.steps.train import lazy_features_for

    return lazy_features_for(config, _embedding_artifact(embeddings_dir, dataset_name, stem))


def _checkpoint_digest(path: str) -> str:
    """Digest of the selected checkpoint's bytes; ``""`` when the file is absent.

    An absent winner can match no recorded completion, so the cell is
    evaluated and fails loudly in :func:`_evaluate_cell` instead of
    being skipped as done.
    """
    try:
        return stream_digest(path)
    except IdentityError:
        return ""


def _embedding_artifact(embeddings_dir: str, dataset_name: str, stem: str) -> Path | None:
    base = Path(embeddings_dir) / dataset_name
    for candidate in (base / f"{stem}.npy", base / f"{stem}.json"):
        if candidate.exists():
            return candidate
    return None


def evaluation_identity(
    config: dict,
    dataset_name: str,
    model_info: dict,
    evaluator: Evaluator,
    *,
    processed_dir: str,
    embeddings_dir: str,
    checkpoint_digest: str,
) -> dict:
    """C02 identity of one final-evaluation cell (E04).

    Binds the dataset, every split (train/val/test), the feature artifact
    content, the model implementation, the evaluation protocol (candidate
    set, cutoffs, tie-break seed, protocol version) and the exact bytes
    of the selected checkpoint.  The checkpoint carries the effective
    hyperparameters, so they enter through its digest.
    """
    from src.evaluation.persistence import EVAL_PROTOCOL_VERSION

    spec = get_recommender_spec(model_info["model_name"])
    emb_path = None
    if spec.requires_visual and model_info["embedding_name"] != "none":
        emb_path = _embedding_artifact(embeddings_dir, dataset_name, model_info["embedding_name"])
    try:
        data = resolve_data_identity(
            processed_dir, dataset_name, emb_path, splits=EVALUATION_SPLITS
        )
    except IdentityError as exc:
        logger.warning("%s/%s: data identity unresolved (%s).", dataset_name, model_info, exc)
        data = None
    condition = "finetuned" if is_finetuned_artifact(model_info["embedding_name"]) else "frozen"
    identity = experiment_identity(
        data=data,
        model_name=model_info["model_name"],
        implementation=implementation_digest(spec.cls),
        hyperparams={},
        selection_budget=None,
        seed=int(config.get("seed", 42)),
        protocol={
            "candidates": evaluator.protocol,
            "k_values": list(evaluator.k_values),
            "n_negatives": int(evaluator.n_negatives),
            "tiebreak_seed": int(config.get("seed", 42)),
            "eval_protocol_version": EVAL_PROTOCOL_VERSION,
        },
        condition=condition,
    )
    identity["checkpoint_digest"] = checkpoint_digest
    return identity


def _append_cell(df: pd.DataFrame, target_path: Path) -> None:
    """Upsert a cell's per-user rows into a battery CSV (E06).

    The battery CSV is a rebuildable view over the per-cell generations,
    not a completion authority: rows of the same ``(model_name,
    embedding_name)`` are replaced, never appended twice, so a repeated
    evaluation cannot add a second scientific observation.  The whole
    table is rewritten atomically.
    """
    keys = [c for c in ("model_name", "embedding_name") if c in df.columns]
    if target_path.exists():
        existing = pd.read_csv(target_path)
        if keys and all(k in existing.columns for k in keys):
            incoming = set(map(tuple, df[keys].astype(str).drop_duplicates().to_numpy()))
            stale = existing[keys].astype(str).apply(tuple, axis=1).isin(incoming)
            existing = existing[~stale]
        df = pd.concat([existing, df], ignore_index=True)
    atomic_write(lambda tmp: df.to_csv(tmp, index=False), target_path)


def _evaluate_cell(
    model_info: dict,
    dataset_name: str,
    n_users: int,
    n_items: int,
    evaluator: Evaluator,
    embeddings_dir: str,
    device: str,
    train_interactions: dict[int, set[int]] | None = None,
    per_user_out_dir: str | None = None,
    seed: int = 42,
    *,
    identity: dict | None = None,
    lazy_features: bool = False,
) -> pd.DataFrame | None:
    """Load a cell's best checkpoint and return its per-user metrics.

    Returns ``None`` (skip) when the recommender is unknown or its
    embedding cannot be resolved — same semantics as the previous inline
    logic.  ``identity`` (see :func:`evaluation_identity`) is recorded in
    the per-user artifact's ``config_hash`` so the artifact says which
    data, protocol and checkpoint bytes produced it.  ``lazy_features``
    reads the artifact through bounded row access (M05 opt-in via
    ``resources.features.residency``); the default keeps the resident
    matrix.
    """
    try:
        spec = get_recommender_spec(model_info["model_name"])
    except KeyError:
        logger.warning("    Unknown model: %s", model_info["model_name"])
        return None
    model_cls = spec.cls

    if not spec.requires_visual or model_info["embedding_name"] == "none":
        visual_emb = None
    else:
        base = Path(embeddings_dir) / dataset_name
        stem = model_info["embedding_name"]
        npy = base / f"{stem}.npy"
        sidecar = base / f"{stem}.json"
        if npy.exists():
            emb_path = npy
        elif sidecar.exists():
            emb_path = sidecar
        else:
            logger.warning("    Embeddings not found: neither %s nor %s", npy, sidecar)
            return None
        from src.fusions import load_embedding

        visual_emb = load_embedding(emb_path, lazy=lazy_features)

    # I02: a cell is evaluable only from a loadable, complete winner.  An
    # absent / truncated / legacy flat checkpoint raises
    # ``BestCheckpointError`` here instead of being skipped or guessed
    # (the old fallback fabricated ``latent_dim=64``), so the cell fails
    # loudly rather than disappearing from the expected set.
    saved = load_best_checkpoint(model_info["path"], map_location=device)
    state_dict = saved["model_state"]
    model_config = {**saved["hyperparams"]}
    # Same seeded history subsample (ACF) the training run used.
    model_config["history_seed"] = seed

    # History-consuming models (ACF) need the pure-train interactions at
    # construction; category-consuming models (DeepStyle) need the item→
    # category index array; other models keep the 4-argument constructor.
    ctor_kwargs: dict = {}
    if getattr(model_cls, "wants_history", False):
        ctor_kwargs["train_interactions"] = train_interactions
    if getattr(model_cls, "wants_categories", False):
        from src.data.categories import item_category_array

        config_paths = load_config()["paths"]
        ctor_kwargs["item_categories"] = item_category_array(
            dataset_name, config_paths["data_processed"]
        )

    model = model_cls(
        n_users=n_users,
        n_items=n_items,
        visual_embeddings=visual_emb,
        config=model_config,
        **ctor_kwargs,
    ).to(device)
    model.configure_item_block(resolve_resources(load_config()).features.item_block)
    model.load_state_dict(state_dict)

    # Task F: persist the per-user sufficient statistic (held-out rank +
    # top-20) when a destination is given, from the SAME scoring pass as
    # the metrics — no recompute, no second pass. Full-ranking only.
    if per_user_out_dir is not None and evaluator.protocol == "full_ranking":
        from src.evaluation.persistence import CellMetadata, write_cell_artifact

        per_user, records = evaluator.evaluate_with_records(model, device=device)
        metadata = CellMetadata(
            dataset=dataset_name,
            visual_config=model_info["embedding_name"],
            recommender=model_info["model_name"],
            seed=seed,
            d=int(model_config.get("latent_dim", 0)),
            split="test",
            n_users=n_users,
            n_items=n_items,
            config_hash=canonical_digest(identity) if identity is not None else None,
        )
        write_cell_artifact(
            records,
            metadata,
            per_user_out_dir,
            expected_users=evaluator.test_users,
            identity_digest=metadata.config_hash,
        )
    else:
        per_user = evaluator.evaluate_per_user(model, device=device)

    # Provenance columns required by the v2 protocol: every recorded
    # result must say which evaluation protocol produced it, the visual
    # input dimensionality the model consumed, and the trainable-param
    # count (E scales with the backbone's native dim — an expected
    # second-order effect that must be reported, not hidden).
    n_trainable = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    return per_user.assign(
        protocol=evaluator.protocol,
        visual_input_dim=int(getattr(model, "visual_dim_raw", 0)),
        n_trainable_params=n_trainable,
    )


def run(condition: str = "frozen") -> None:
    """Evaluate every best model, writing per-user rows routed by embedding.

    ``condition`` is accepted for backward compatibility with the
    legacy ``--condition`` CLI flag but is no longer used: every cell
    is auto-routed to the frozen and/or finetuned battery file based
    on its embedding name (a ``_finetuned`` suffix marks the
    finetuned battery).  ``main.py`` calls this step once per pipeline
    invocation regardless of the configured condition.
    """
    if condition not in {"frozen", "finetuned", "both"}:
        raise ValueError(f"condition must be 'frozen', 'finetuned' or 'both', got {condition!r}")

    config = load_config()
    device = resolve_device(config["device"])
    # Final evaluation is the heaviest ranking in the pipeline (every
    # test user against the whole catalogue), and it runs in its own
    # process -- the training workers' cap does not reach it.  Without
    # this, ``default_ranking_budget`` sizes the user-batch off the whole
    # card and the desktop freezes for the duration of the step.
    cap_process_vram(vram_share=resolve_resources(config).gpu.vram_share)
    processed_dir = config["paths"]["data_processed"]
    embeddings_dir = config["paths"]["embeddings"]
    datasets = config.get("datasets", [])
    if not datasets:
        logger.info("evaluate step skipped: datasets list is empty in configs/default.yaml.")
        return

    results_root = Path(config.get("paths", {}).get("results", "results"))
    results_dir = results_root / "tables"
    results_dir.mkdir(parents=True, exist_ok=True)

    for dataset_name in datasets:
        logger.info("=== Dataset: %s ===", dataset_name)
        n_users, n_items, seen_inter, test_inter, train_only_inter = load_data(
            processed_dir, dataset_name
        )
        # Shared construction path with the battery executor: same
        # ``evaluation:`` block, same defaults, tiebreak_seed = the run's
        # active seed (here: the global run seed from config['seed']).
        evaluator = build_evaluator(config, seen_inter, test_inter, n_items)

        done_path = _done_path(results_dir, dataset_name)
        done = _load_done_index(done_path)
        best_models = find_best_models(dataset_name, results_dir=results_root)
        # The models directory accumulates checkpoints across runs; only
        # cells the CURRENT config would train are evaluated (same
        # disk-vs-config rule as the train step's cell filters).
        eligible = [m for m in best_models if checkpoint_matches_config(m, config)]
        if len(eligible) != len(best_models):
            logger.info(
                "  %d checkpoint(s) on disk skipped (model/backbone/fusion "
                "not enabled in the current config).",
                len(best_models) - len(eligible),
            )
        best_models = eligible
        logger.info("  Found %d models to evaluate", len(best_models))

        for model_info in best_models:
            mn = model_info["model_name"]
            en = model_info["embedding_name"]
            targets = _route_targets(mn, en)
            # Reuse is bound to the exact checkpoint bytes and the
            # evaluation identity (E04, Q13): a completion recorded for
            # another winner, split, feature content, seed or protocol
            # — or a legacy row without a binding — is not a completion.
            checkpoint_digest = _checkpoint_digest(model_info["path"])
            identity = evaluation_identity(
                config,
                dataset_name,
                model_info,
                evaluator,
                processed_dir=processed_dir,
                embeddings_dir=embeddings_dir,
                checkpoint_digest=checkpoint_digest,
            )
            identity_digest = canonical_digest(identity)
            pending = [
                t
                for t in targets
                if not _binding_matches(done.get((t, mn, en)), checkpoint_digest, identity_digest)
            ]
            stale = [t for t in pending if (t, mn, en) in done]
            if stale:
                logger.warning(
                    "  %s/%s: completion for %s does not match the current "
                    "checkpoint/identity (or is legacy); re-evaluating.",
                    mn,
                    en,
                    stale,
                )
            if not pending:
                logger.info("  %s/%s: already done, skipping.", mn, en)
                note_skipped_cell()
                continue

            logger.info("  Evaluating: %s/%s -> %s", mn, en, pending)
            with time_cell("evaluate", dataset=dataset_name, model_key=f"{mn}_{en}"):
                per_user = _evaluate_cell(
                    model_info,
                    dataset_name,
                    n_users,
                    n_items,
                    evaluator,
                    embeddings_dir,
                    device,
                    train_interactions=train_only_inter,
                    per_user_out_dir=str(results_root),
                    seed=int(config.get("seed", 42)),
                    identity=identity,
                    lazy_features=_lazy_features(config, embeddings_dir, dataset_name, en),
                )
            if per_user is None:
                continue

            per_user = per_user.assign(dataset=dataset_name, model_name=mn, embedding_name=en)
            recorded: list[tuple[str, str, str]] = []
            for target in pending:
                _append_cell(
                    per_user,
                    results_dir / f"{dataset_name}_evaluation_{target}.csv",
                )
                recorded.append((target, mn, en))
            binding = {"checkpoint_digest": checkpoint_digest, "identity_digest": identity_digest}
            _record_done(done_path, recorded, bindings=dict.fromkeys(recorded, binding))
            done.update(dict.fromkeys(recorded, binding))

        for target in ("frozen", "finetuned"):
            _write_mean_table(results_dir, dataset_name, target)
        logger.info("  Dataset %s complete.", dataset_name)

    logger.info("Evaluation complete.")
