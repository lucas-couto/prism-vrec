"""Production per-cell executor for the battery runner (Task I).

Runs ONE cell end-to-end, reusing the existing, battle-tested per-cell
functions rather than reimplementing training/evaluation:

* search cells (primary seed) → the Optuna search for that cell
  (``_optimize_one_cell``), which saves the best-trial checkpoint;
* replay cells (other seeds) → ``train_replay`` (Task H) with the search's
  best config, under the cell's own seed;
* then a single-cell final evaluation writes the per-user artifact (F).

The battery runner supplies idempotency/resume/manifest around this.
NOTE: designed to be driven by :func:`src.battery.runner.run_battery`;
multi-seed checkpoint/result path isolation follows the existing
``paths.results``/``paths.checkpoints`` config (set per seed by the
caller when running multiple seeds).
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

from src.battery.cells import NO_VISUAL, BatteryCell
from src.utils.logging import get_logger

logger = get_logger(__name__)


def _dims(processed_dir: str, dataset: str) -> tuple[int, int]:
    base = Path(processed_dir) / dataset
    with open(base / "user2idx.json") as fh:
        n_users = len(json.load(fh))
    with open(base / "item2idx.json") as fh:
        n_items = len(json.load(fh))
    return n_users, n_items


def _embedding_path(embeddings_dir: str, dataset: str, visual_config: str) -> str | None:
    if visual_config == NO_VISUAL:
        return None
    base = Path(embeddings_dir) / dataset
    npy = base / f"{visual_config}.npy"
    if npy.exists():
        return str(npy)
    sidecar = base / f"{visual_config}.json"
    return str(sidecar) if sidecar.exists() else None


def execute_cell(cell: BatteryCell, config: dict) -> dict:
    """Run one battery cell: search|replay → final evaluation (F artifact)."""
    from src.recommenders.hp_search import CellKey, create_study
    from src.steps.train import _optimize_one_cell, train_replay
    from src.utils.device import resolve_device

    cfg = copy.deepcopy(config)
    cfg["seed"] = cell.seed
    processed_dir = cfg["paths"]["data_processed"]
    embeddings_dir = cfg["paths"]["embeddings"]
    base_results = cfg["paths"]["results"]
    # Isolate the best-model checkpoint per seed. The Optuna study is shared
    # across seeds (D2: search on the primary seed supplies best_params), but
    # each seed's TRAINED model must be the one evaluated — a shared
    # ``_best.pt`` path would let a replay whose val metric is below the
    # search's silently keep the search seed's checkpoint. The F artifact
    # still lands in the shared, seed-keyed directory (``base_results``).
    cfg["paths"] = {**cfg["paths"], "results": f"{base_results}_seed{cell.seed}"}
    device = resolve_device(cfg["device"])
    n_users, n_items = _dims(processed_dir, cell.dataset)
    emb_path = _embedding_path(embeddings_dir, cell.dataset, cell.visual_config)
    ck = CellKey(cell.dataset, cell.recommender, cell.visual_config)

    if cell.role == "search":
        _optimize_one_cell(
            ck, n_users, n_items, emb_path, config=cfg, processed_dir=processed_dir, device=device
        )
    else:
        # Replay: the study is seed-independent (name has no seed), so the
        # primary seed's search supplies best_params; retrain under this seed.
        study = create_study(ck, cfg)
        if not study.trials:
            raise RuntimeError(
                f"replay cell {cell.key()} has no completed search "
                f"(study {ck.study_name()!r} empty); run the primary seed first."
            )
        best = study.best_params
        train_replay(
            cell=ck,
            hyperparams=best,
            n_users=n_users,
            n_items=n_items,
            embeddings_path=emb_path,
            processed_dir=processed_dir,
            device=device,
            config=cfg,
        )

    _evaluate_one_cell(cell, cfg, n_users, n_items, emb_path, device, f_out_dir=base_results)
    return {"seed": cell.seed, "role": cell.role}


def _evaluate_one_cell(
    cell: BatteryCell,
    cfg: dict,
    n_users: int,
    n_items: int,
    emb_path: str | None,
    device: str,
    *,
    f_out_dir: str,
) -> None:
    """Final full-ranking evaluation for one cell → per-user artifact (F).

    Reads the best checkpoint from the seed-isolated results dir
    (``cfg['paths']['results']``); writes the F artifact to the shared,
    seed-keyed ``f_out_dir`` so all seeds land in one per_user directory.
    """
    from src.evaluation.protocol import Evaluator
    from src.steps.evaluate import _evaluate_cell, find_best_models, load_data

    _, _, seen_inter, test_inter, train_only = load_data(
        cfg["paths"]["data_processed"], cell.dataset
    )
    evaluator = Evaluator(
        seen_inter,
        test_inter,
        n_items,
        k_values=cfg.get("k_values", [5, 10, 20]),
        tiebreak_seed=int(cfg.get("seed", 42)),
    )
    models = [
        m
        for m in find_best_models(cell.dataset, results_dir=cfg["paths"]["results"])
        if m["model_name"] == cell.recommender and m["embedding_name"] == cell.visual_config
    ]
    if not models:
        raise RuntimeError(f"no best checkpoint found for cell {cell.key()} after training.")
    _evaluate_cell(
        models[0],
        cell.dataset,
        n_users,
        n_items,
        evaluator,
        cfg["paths"]["embeddings"],
        device,
        train_interactions=train_only,
        per_user_out_dir=f_out_dir,
        seed=int(cell.seed),
    )
