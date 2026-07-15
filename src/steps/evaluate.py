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
import torch

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
from src.utils.config import load_config
from src.utils.device import resolve_device
from src.utils.logging import get_logger
from src.utils.timing import time_cell

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

    return n_users, n_items, seen_interactions, test_interactions, train_interactions


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


def _done_path(results_dir: Path, dataset_name: str) -> Path:
    """Path of the per-dataset resume sidecar."""
    return results_dir / f"{dataset_name}_evaluation_done.csv"


def _load_done(path: Path) -> set[tuple[str, str, str]]:
    """Load completed ``(target, model_name, embedding_name)`` triples."""
    if not path.exists():
        return set()
    df = pd.read_csv(path, dtype=str)
    return {(row.target, row.model_name, row.embedding_name) for row in df.itertuples(index=False)}


def _record_done(path: Path, rows: list[tuple[str, str, str]]) -> None:
    """Append completed ``(target, model_name, embedding_name)`` triples."""
    header = not path.exists()
    pd.DataFrame(rows, columns=["target", "model_name", "embedding_name"]).to_csv(
        path, mode="a", header=header, index=False
    )


def _append_cell(df: pd.DataFrame, target_path: Path) -> None:
    """Append a cell's per-user rows to a battery CSV.

    The header is written only when the file is first created so the
    file stays a single valid CSV across many appended cells.
    """
    header = not target_path.exists()
    df.to_csv(target_path, mode="a", header=header, index=False)


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
) -> pd.DataFrame | None:
    """Load a cell's best checkpoint and return its per-user metrics.

    Returns ``None`` (skip) when the recommender is unknown or its
    embedding cannot be resolved — same semantics as the previous inline
    logic.
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

        visual_emb = load_embedding(emb_path)

    saved = torch.load(model_info["path"], map_location=device, weights_only=False)
    if isinstance(saved, dict) and "model_state" in saved:
        state_dict = saved["model_state"]
        model_config = {**saved["hyperparams"]}
    else:
        state_dict = saved
        model_config = {"latent_dim": 64, "l2_reg": 0.0001}
        logger.warning("    Legacy checkpoint (no hyperparams): %s", model_info["path"])

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
    model.load_state_dict(state_dict)
    per_user = evaluator.evaluate_per_user(model, device=device)

    # Task F: persist the per-user sufficient statistic (held-out rank +
    # top-20) when a destination is given. Full-ranking only; a second
    # scoring pass (final eval is a small fraction of the battery).
    if per_user_out_dir is not None and evaluator.protocol == "full_ranking":
        from src.evaluation.persistence import CellMetadata, write_cell_artifact

        records = evaluator.per_user_records(model, device=device)
        metadata = CellMetadata(
            dataset=dataset_name,
            visual_config=model_info["embedding_name"],
            recommender=model_info["model_name"],
            seed=seed,
            d=int(model_config.get("latent_dim", 0)),
            split="test",
            n_users=n_users,
            n_items=n_items,
        )
        write_cell_artifact(records, metadata, per_user_out_dir)

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
    processed_dir = config["paths"]["data_processed"]
    embeddings_dir = config["paths"]["embeddings"]
    k_values = config.get("k_values", [5, 10, 20])
    datasets = config.get("datasets", [])
    if not datasets:
        logger.info("evaluate step skipped: datasets list is empty in configs/default.yaml.")
        return

    eval_cfg = config.get("evaluation") or {}
    protocol = eval_cfg.get("protocol", "full_ranking")
    n_negatives = eval_cfg.get("n_negatives", 100)
    neg_seed = eval_cfg.get("negative_sampling_seed", 42)
    tiebreak_seed = config.get("seed", 42)
    if protocol == "sampled":
        logger.warning(
            "evaluate: protocol='sampled' selected (n_negatives=%d).  "
            "Sampled metrics are inconsistent with full-ranking "
            "(Krichene & Rendle 2020); use only for comparability with "
            "prior work, never as the headline benchmark.",
            n_negatives,
        )

    results_root = Path(config.get("paths", {}).get("results", "results"))
    results_dir = results_root / "tables"
    results_dir.mkdir(parents=True, exist_ok=True)

    for dataset_name in datasets:
        logger.info("=== Dataset: %s ===", dataset_name)
        n_users, n_items, seen_inter, test_inter, train_only_inter = load_data(
            processed_dir, dataset_name
        )
        evaluator = Evaluator(
            seen_inter,
            test_inter,
            n_items,
            k_values=k_values,
            protocol=protocol,
            n_negatives=n_negatives,
            negative_sampling_seed=neg_seed,
            tiebreak_seed=tiebreak_seed,
        )

        done_path = _done_path(results_dir, dataset_name)
        done = _load_done(done_path)
        best_models = find_best_models(dataset_name, results_dir=results_root)
        logger.info("  Found %d models to evaluate", len(best_models))

        for model_info in best_models:
            mn = model_info["model_name"]
            en = model_info["embedding_name"]
            targets = _route_targets(mn, en)
            pending = [t for t in targets if (t, mn, en) not in done]
            if not pending:
                logger.info("  %s/%s: already done, skipping.", mn, en)
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
            _record_done(done_path, recorded)

        logger.info("  Dataset %s complete.", dataset_name)

    logger.info("Evaluation complete.")
