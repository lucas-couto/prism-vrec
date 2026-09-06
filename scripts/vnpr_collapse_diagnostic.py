"""VNPR collapse diagnostic experiment driver (SDD S04).

Runs the controlled native-versus-fused comparison that the science SPEC
asks for, with the opt-in probe diagnostics of
:mod:`src.utils.diagnostics` enabled, and writes a report whose
hypothesis decision table is filled from the recorded probes.  It does
NOT change VNPR; it only measures.

Two input modes share every downstream step:

* **synthetic** (default; what the test suite and the S04 record run):
  a latent-factor interaction model, leave-one-out validation, and two
  "native" sources that both carry preference signal but differ in scale
  (norm ~10 and ~30, mirroring the ResNet-50 / ViT-B/16 norms recorded in
  ``docs/protocol.md`` §10.3);
* **real** (``--processed-dir`` + ``--sources``; NOT run here): the
  dataset's ``train.csv`` / ``val.csv`` and two native ``.npy`` feature
  files.  Selection stays on validation; the test split is never read.

Conditions (same split, sampler, budget and init policy; only the visual
input differs):

``native_a`` / ``native_b`` raw sources; ``native_a_unit`` the scale-only
control (``native_a`` L2-normalised per row: same directions, norm 1);
``learned_mean`` / ``learned_sum`` learned alignment with the §10.3
normalisation; ``learned_mean_nonorm`` the same alignment without it
(scale control on the fused side); ``stacked_adaptive_gated`` the
non-learned online sidecar recipe (S02: per-source unit rows) — only
when both sources share a width.

Usage (inside the project container, CPU)::

    python scripts/vnpr_collapse_diagnostic.py --out results/diagnostics/vnpr_collapse
    python scripts/vnpr_collapse_diagnostic.py --processed-dir data/processed/amazon_women \\
        --sources data/embeddings/amazon_women/resnet50.npy data/embeddings/amazon_women/vit_b16.npy
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.fusions.online import RaggedSources, StackedSources  # noqa: E402
from src.fusions.strategies import l2_normalize  # noqa: E402
from src.recommenders.vnpr import VNPR  # noqa: E402
from src.utils.checkpoint import CheckpointManager  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402
from src.utils.training import SelectionMetricError, train_single_run  # noqa: E402

logger = get_logger("vnpr_collapse_diagnostic")

DIAG_STEPS = (1, 5, 20, 100, 500)
DECISION_ROWS = (
    (
        "inactive_relu_at_init",
        "Inactive ReLU at initialization",
        "Compare branch preactivations and initialization across seeds",
        "Replace activation without an approved model variant",
    ),
    (
        "gradients_without_movement",
        "Finite gradients but no parameter movement",
        "Verify scaler skips, optimizer groups and step counters",
        "Increase LR without tracing step application",
    ),
    (
        "scale_correlates",
        "Native/fused scale correlates with collapse",
        "Controlled scale-only validation ablation, explicitly versioned",
        "Claim causality from correlation",
    ),
    (
        "constant_only_in_eval",
        "Constant scores only in evaluation",
        "Compare training/inference branches and cache generations",
        "Exclude tied users from metrics",
    ),
    (
        "zero_with_nonconstant_scores",
        "Zero metric with nonconstant scores",
        "Verify full-rank target positions, data identity and expected random baseline",
        "Treat every zero as a numeric failure",
    ),
)


@dataclass(frozen=True)
class Inputs:
    train: dict[int, set[int]]
    val: dict[int, set[int]]
    n_users: int
    n_items: int
    source_a: np.ndarray
    source_b: np.ndarray
    origin: str


# --------------------------------------------------------------------- inputs
def synthetic_inputs(
    *, n_users: int, n_items: int, latent: int, dv_a: int, dv_b: int, seed: int
) -> Inputs:
    """Latent-factor interactions plus two signal-carrying sources of unequal scale."""
    rng = np.random.default_rng(seed)
    z_u = rng.standard_normal((n_users, latent))
    z_i = rng.standard_normal((n_items, latent))
    logits = z_u @ z_i.T
    train: dict[int, set[int]] = {}
    val: dict[int, set[int]] = {}
    for u in range(n_users):
        p = np.exp(logits[u] - logits[u].max())
        chosen = rng.choice(n_items, size=9, replace=False, p=p / p.sum())
        train[u] = {int(i) for i in chosen[:8]}
        val[u] = {int(chosen[8])}
    a = _source(rng, z_i, dv_a, scale=10.0)
    b = _source(rng, z_i, dv_b, scale=30.0)
    return Inputs(train, val, n_users, n_items, a, b, origin=f"synthetic(seed={seed})")


def _source(rng: np.random.Generator, z_i: np.ndarray, dv: int, *, scale: float) -> np.ndarray:
    proj = rng.standard_normal((z_i.shape[1], dv)) / math.sqrt(z_i.shape[1])
    feats = z_i @ proj + 0.5 * rng.standard_normal((z_i.shape[0], dv))
    feats /= np.linalg.norm(feats, axis=1, keepdims=True).mean()
    return (feats * scale).astype(np.float32)


def real_inputs(processed_dir: Path, sources: list[Path]) -> Inputs:
    """``train.csv`` / ``val.csv`` (``user_idx,item_idx``) and two native ``.npy`` files."""
    import pandas as pd

    def _read(name: str) -> dict[int, set[int]]:
        frame = pd.read_csv(processed_dir / name)
        out: dict[int, set[int]] = {}
        for u, i in zip(frame["user_idx"], frame["item_idx"], strict=True):
            out.setdefault(int(u), set()).add(int(i))
        return out

    train, val = _read("train.csv"), _read("val.csv")
    a, b = (np.load(p).astype(np.float32) for p in sources)
    if a.shape[0] != b.shape[0]:
        raise ValueError(f"sources disagree on n_items: {a.shape[0]} vs {b.shape[0]}")
    n_users = max(max(train), max(val)) + 1
    return Inputs(train, val, n_users, int(a.shape[0]), a, b, origin=str(processed_dir))


# ----------------------------------------------------------------- conditions
def build_conditions(inputs: Inputs, aligned_dim: int) -> dict[str, Any]:
    """Visual inputs per condition; each is what ``train_single_run`` receives."""
    a, b = inputs.source_a, inputs.source_b
    dims = [int(a.shape[1]), int(b.shape[1])]
    concat = np.concatenate([a, b], axis=1)

    def ragged(strategy: str, normalize: bool) -> RaggedSources:
        return RaggedSources(
            concat,
            source_dims=dims,
            strategy=strategy,
            aligned_dim=aligned_dim,
            normalize=normalize,
        )

    conditions: dict[str, Any] = {
        "native_a": a,
        "native_b": b,
        "native_a_unit": l2_normalize(a),
        "learned_mean": ragged("mean", True),
        "learned_sum": ragged("sum", True),
        "learned_mean_nonorm": ragged("mean", False),
    }
    if dims[0] == dims[1]:
        stacked = np.stack([l2_normalize(a), l2_normalize(b)], axis=1)
        conditions["stacked_adaptive_gated"] = StackedSources(
            stacked, normalize=True, sidecar_recipe_version=2
        )
    return conditions


# ------------------------------------------------------------------- one run
def _config(out: Path, *, seed: int, epochs: int, eval_every: int, batch_size: int) -> dict:
    return {
        "seed": seed,
        "paths": {"results": str(out / "results")},
        "common": {
            "epochs": epochs,
            "batch_size": batch_size,
            "early_stopping_patience": epochs,
            "early_stopping_metric": "ndcg@10",
            "eval_every_epochs": eval_every,
        },
        "diagnostics": {
            "enabled": True,
            "probe_users": 32,
            "probe_items": 128,
            "probe_pairs": 64,
            "steps": list(DIAG_STEPS),
            "output_dir": str(out / "diagnostics"),
        },
    }


def run_one(
    inputs: Inputs, visual: Any, *, condition: str, seed: int, lr: float, args: argparse.Namespace
) -> dict[str, Any]:
    """Train one VNPR cell with diagnostics on; never raises on a failed run."""
    out = Path(args.out)
    hyperparams = {"learning_rate": lr, "latent_dim": args.latent_dim, "l2_reg": args.l2_reg}
    embedding_name = f"{condition}_lr{lr}"
    result: dict[str, Any] = {
        "condition": condition,
        "seed": seed,
        "learning_rate": lr,
        "run_id": None,
        "status": "failed",
        "best_metric": None,
        "error": None,
    }
    try:
        metric = train_single_run(
            model_cls=VNPR,
            model_name="vnpr",
            n_users=inputs.n_users,
            n_items=inputs.n_items,
            visual_embeddings=visual,
            train_interactions=inputs.train,
            selection_interactions=inputs.val,
            hyperparams=hyperparams,
            config=_config(
                out,
                seed=seed,
                epochs=args.epochs,
                eval_every=args.eval_every,
                batch_size=args.batch_size,
            ),
            checkpoint_mgr=CheckpointManager(str(out / "checkpoints")),
            dataset_name=f"seed{seed}",
            embedding_name=embedding_name,
            device=args.device,
        )
        result.update(status="completed", best_metric=float(metric))
    except (SelectionMetricError, RuntimeError, ValueError) as exc:
        logger.error("%s seed=%d lr=%g failed: %s", condition, seed, lr, exc)
        result["error"] = f"{type(exc).__name__}: {exc}"
    diag_path = _find_diagnostics(out / "diagnostics", inputs, embedding_name, hyperparams, seed)
    result["run_id"] = CheckpointManager.get_run_id(
        f"seed{seed}", embedding_name, "vnpr", hyperparams
    )
    result.update(_summarise(diag_path))
    return result


def _find_diagnostics(
    diag_dir: Path, inputs: Inputs, embedding_name: str, hyperparams: dict, seed: int
) -> Path | None:
    run_id = CheckpointManager.get_run_id(f"seed{seed}", embedding_name, "vnpr", hyperparams)
    path = diag_dir / f"{run_id}.json"
    return path if path.exists() else None


# ------------------------------------------------------------------- summary
def _summarise(diag_path: Path | None) -> dict[str, Any]:
    """Bounded per-run fields read back from the diagnostics JSON."""
    if diag_path is None:
        return {"diagnostics": None}
    payload = json.loads(diag_path.read_text(encoding="utf-8"))
    steps = [r for r in payload["records"] if r["phase"] in ("init", "step")]
    vals = [r for r in payload["records"] if r["phase"] == "validation"]
    first, last = steps[0], steps[-1]
    deltas = [p["delta_from_init"] for p in last["parameters"].values()]
    return {
        "diagnostics": str(diag_path),
        "fused_norm_median_init": first["features"]["fused_norm_quantiles"][2],
        "active_pos_init": first["train_branches"]["pos"]["active_fraction"],
        "active_neg_init": first["train_branches"]["neg"]["active_fraction"],
        "active_pos_last": last["train_branches"]["pos"]["active_fraction"],
        "active_eval_last": last["eval_branches"]["pos"]["active_fraction"],
        "preact_pos_max_init": first["train_branches"]["pos"]["max"],
        "score_unique_init": first["scores"]["unique_fraction"],
        "score_unique_last": last["scores"]["unique_fraction"],
        "score_std_last": last["scores"]["std"],
        "all_tied_users_last": last["scores"]["all_tied_user_fraction"],
        "grad_total_init": first["gradient_norm_total"],
        "grad_total_last": last["gradient_norm_total"],
        "max_param_delta_last": max(deltas) if deltas else 0.0,
        "steps_attempted": last["optimizer"]["steps_attempted"],
        "steps_applied": last["optimizer"]["steps_applied"],
        "last_step_recorded": last["step"],
        "validations": len(vals),
        "zero_validations": sum(1 for v in vals if v["zero_metric"]),
        "tie_frequency_last_val": vals[-1]["tie_frequency"] if vals else None,
        "checkpoint_exists_last_val": vals[-1]["checkpoint_exists"] if vals else None,
        "random_baseline_ndcg10": payload["random_baseline"]["ndcg@10"],
    }


def aggregate(results: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_condition: dict[str, dict[str, Any]] = {}
    for r in results:
        agg = by_condition.setdefault(
            r["condition"], {"expected": 0, "completed": 0, "zero": 0, "failed": 0, "runs": []}
        )
        agg["expected"] += 1
        agg["runs"].append(r)
        if r["status"] != "completed":
            agg["failed"] += 1
            continue
        agg["completed"] += 1
        if r["best_metric"] == 0.0:
            agg["zero"] += 1
    for agg in by_condition.values():
        done = [r for r in agg["runs"] if r.get("diagnostics")]
        agg["mean_active_pos_init"] = _mean(done, "active_pos_init")
        agg["mean_active_pos_last"] = _mean(done, "active_pos_last")
        agg["mean_fused_norm_median_init"] = _mean(done, "fused_norm_median_init")
        agg["mean_score_unique_last"] = _mean(done, "score_unique_last")
        agg["mean_best_metric"] = _mean(
            [r for r in agg["runs"] if r["status"] == "completed"], "best_metric"
        )
        del agg["runs"]
    return by_condition


def _mean(rows: list[dict], key: str) -> float | None:
    vals = [r[key] for r in rows if r.get(key) is not None]
    return float(np.mean(vals)) if vals else None


# -------------------------------------------------------------- decision table
def decide(results: list[dict[str, Any]], by_condition: dict[str, dict[str, Any]]) -> dict:
    done = [r for r in results if r.get("diagnostics")]
    dead_init = [r for r in done if r["active_pos_init"] == 0.0 and r["active_neg_init"] == 0.0]
    frozen = [
        r
        for r in done
        if r["grad_total_init"] > 0
        and math.isfinite(r["grad_total_init"])
        and r["max_param_delta_last"] == 0.0
    ]
    eval_only = [r for r in done if r["active_eval_last"] == 0.0 and r["active_pos_last"] > 0.0]
    zero_nonconst = [
        r
        for r in done
        if r["status"] == "completed" and r["best_metric"] == 0.0 and r["all_tied_users_last"] < 1.0
    ]
    scale = _scale_comparison(by_condition)
    return {
        "inactive_relu_at_init": _verdict(dead_init, done),
        "gradients_without_movement": _verdict(frozen, done),
        "scale_correlates": scale,
        "constant_only_in_eval": _verdict(eval_only, done),
        "zero_with_nonconstant_scores": _verdict(zero_nonconst, done),
    }


def _verdict(hits: list[dict], done: list[dict]) -> dict[str, Any]:
    return {
        "observed": bool(hits),
        "runs": len(hits),
        "of": len(done),
        "examples": [f"{r['condition']}/seed{r['seed']}/lr{r['learning_rate']}" for r in hits[:5]],
    }


def _scale_comparison(by_condition: dict[str, dict[str, Any]]) -> dict[str, Any]:
    pairs = (("native_a", "native_a_unit"), ("learned_mean_nonorm", "learned_mean"))
    rows = []
    for raw, unit in pairs:
        if raw in by_condition and unit in by_condition:
            r, u = by_condition[raw], by_condition[unit]
            rows.append(
                {
                    "raw": raw,
                    "unit": unit,
                    "zero_rate_raw": r["zero"] / max(r["completed"], 1),
                    "zero_rate_unit": u["zero"] / max(u["completed"], 1),
                    "active_last_raw": r["mean_active_pos_last"],
                    "active_last_unit": u["mean_active_pos_last"],
                    "fused_norm_raw": r["mean_fused_norm_median_init"],
                    "fused_norm_unit": u["mean_fused_norm_median_init"],
                }
            )
    observed = any(
        (row["zero_rate_unit"] or 0) > (row["zero_rate_raw"] or 0)
        or (row["active_last_unit"] or 0) < 0.5 * (row["active_last_raw"] or 0)
        for row in rows
    )
    return {"observed": observed, "comparisons": rows}


# ---------------------------------------------------------------------- report
def write_report(
    out: Path, args: argparse.Namespace, inputs: Inputs, results, by_condition, decisions
):
    lines = [
        "# VNPR collapse diagnostic — experiment report",
        "",
        f"- Inputs: `{inputs.origin}` — {inputs.n_users} users, {inputs.n_items} items, "
        f"sources {inputs.source_a.shape[1]}-d (norm median "
        f"{np.median(np.linalg.norm(inputs.source_a, axis=1)):.2f}) and "
        f"{inputs.source_b.shape[1]}-d (norm median "
        f"{np.median(np.linalg.norm(inputs.source_b, axis=1)):.2f})",
        f"- Seeds: {args.seeds}; learning rates: {args.learning_rates}; epochs: {args.epochs}; "
        f"eval every {args.eval_every}; latent_dim {args.latent_dim}; aligned_dim "
        f"{args.aligned_dim}; l2_reg {args.l2_reg}; batch {args.batch_size}; device {args.device}",
        f"- Diagnostics steps: {list(DIAG_STEPS)}; selection on validation only (test never read)",
        "- Model: VNPR as shipped (no activation / regularisation / normalisation change)",
        "",
        "## Seed-level outcomes",
        "",
        "| condition | seed | lr | status | best ndcg@10 | zero | fused ‖f‖ med (init) | "
        "active pos init→last | eval active last | score unique last | all-tied users last | "
        "grad init→last | max Δparam | steps applied/attempted | ckpt |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(_row(r))
    lines += [
        "",
        "## Per-condition counts (expected / completed / zero / failed)",
        "",
        "| condition | expected | completed | zero | failed | mean best | mean fused ‖f‖ | "
        "mean active init→last | mean score unique last |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name, agg in by_condition.items():
        lines.append(
            f"| {name} | {agg['expected']} | {agg['completed']} | {agg['zero']} | {agg['failed']} | "
            f"{_fmt(agg['mean_best_metric'])} | {_fmt(agg['mean_fused_norm_median_init'])} | "
            f"{_fmt(agg['mean_active_pos_init'])}→{_fmt(agg['mean_active_pos_last'])} | "
            f"{_fmt(agg['mean_score_unique_last'])} |"
        )
    lines += [
        "",
        "## Hypothesis decision table",
        "",
        "| Observation | Observed here | Next discriminating experiment | Forbidden shortcut |",
        "|---|---|---|---|",
    ]
    for key, label, nxt, forbidden in DECISION_ROWS:
        lines.append(f"| {label} | {_describe(decisions[key])} | {nxt} | {forbidden} |")
    lines += [
        "",
        "## Conclusion",
        "",
        *_conclusion(inputs, decisions),
        "",
        "## Pointing this driver at real data",
        "",
        "```",
        "python scripts/vnpr_collapse_diagnostic.py --processed-dir data/processed/<dataset> \\",
        "    --sources data/embeddings/<dataset>/resnet50.npy data/embeddings/<dataset>/vit_b16.npy \\",
        "    --out results/diagnostics/vnpr_collapse_<dataset> --seeds 0 1 2",
        "```",
        "",
        "Raw bounded probes: one JSON per run under `diagnostics/`; `summary.json` holds "
        "every field of the tables above.",
    ]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _row(r: dict[str, Any]) -> str:
    if not r.get("diagnostics"):
        return (
            f"| {r['condition']} | {r['seed']} | {r['learning_rate']} | {r['status']} | "
            f"{_fmt(r['best_metric'])} | — | — | — | — | — | — | — | — | — | — |"
        )
    return (
        f"| {r['condition']} | {r['seed']} | {r['learning_rate']} | {r['status']} | "
        f"{_fmt(r['best_metric'])} | {'yes' if r['best_metric'] == 0.0 else 'no'} | "
        f"{_fmt(r['fused_norm_median_init'])} | {_fmt(r['active_pos_init'])}→{_fmt(r['active_pos_last'])} | "
        f"{_fmt(r['active_eval_last'])} | {_fmt(r['score_unique_last'])} | {_fmt(r['all_tied_users_last'])} | "
        f"{_fmt(r['grad_total_init'])}→{_fmt(r['grad_total_last'])} | {_fmt(r['max_param_delta_last'])} | "
        f"{r['steps_applied']}/{r['steps_attempted']} | {r['checkpoint_exists_last_val']} |"
    )


def _fmt(value: Any) -> str:
    if value is None:
        return "—"
    return f"{value:.4g}" if isinstance(value, float) else str(value)


def _describe(verdict: dict[str, Any]) -> str:
    if "comparisons" in verdict:
        parts = [
            f"{c['raw']}→{c['unit']}: zero rate {c['zero_rate_raw']:.2f}→{c['zero_rate_unit']:.2f}, "
            f"active last {_fmt(c['active_last_raw'])}→{_fmt(c['active_last_unit'])}, "
            f"‖f‖ {_fmt(c['fused_norm_raw'])}→{_fmt(c['fused_norm_unit'])}"
            for c in verdict["comparisons"]
        ]
        return ("SUPPORTED — " if verdict["observed"] else "not supported — ") + "; ".join(parts)
    tail = f" ({', '.join(verdict['examples'])})" if verdict["examples"] else ""
    return f"{'OBSERVED' if verdict['observed'] else 'not observed'} in {verdict['runs']}/{verdict['of']} runs{tail}"


def _conclusion(inputs: Inputs, decisions: dict) -> list[str]:
    supported = [label for key, label, _, _ in DECISION_ROWS if decisions[key]["observed"]]
    scope = (
        "synthetic inputs; nothing here establishes the cause on amazon_women / tradesy"
        if inputs.origin.startswith("synthetic")
        else f"inputs from {inputs.origin}"
    )
    lines = [f"Evidence scope: {scope}."]
    if supported:
        lines.append(
            "Hypotheses with supporting observations in this run: " + "; ".join(supported) + "."
        )
    else:
        lines.append("No hypothesis of the decision table was observed in this run.")
    lines.append(
        "Cause of the historical VNPR × hybrid collapse: **still unresolved** until the same probes "
        "are recorded on the real datasets; any model repair requires a separate approved spec."
    )
    return lines


# ------------------------------------------------------------------------ main
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--out", default="results/diagnostics/vnpr_collapse")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--learning-rates", type=float, nargs="+", default=[0.001, 0.01])
    p.add_argument("--conditions", nargs="+", default=None, help="subset of condition names")
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--eval-every", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--latent-dim", type=int, default=16)
    p.add_argument("--aligned-dim", type=int, default=32)
    p.add_argument("--l2-reg", type=float, default=1e-4)
    p.add_argument("--device", default="cpu")
    p.add_argument("--n-users", type=int, default=300)
    p.add_argument("--n-items", type=int, default=600)
    p.add_argument("--dv-a", type=int, default=32)
    p.add_argument("--dv-b", type=int, default=32)
    p.add_argument("--data-seed", type=int, default=0)
    p.add_argument("--processed-dir", type=Path, default=None)
    p.add_argument("--sources", type=Path, nargs=2, default=None)
    return p.parse_args(argv)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    return value


def main(argv: list[str] | None = None) -> Path:
    args = parse_args(argv)
    if len(args.seeds) < 3:
        logger.warning("fewer than 3 seeds (%s): below the SPEC's diagnostic budget.", args.seeds)
    if args.processed_dir is not None:
        if args.sources is None:
            raise SystemExit("--processed-dir requires --sources A.npy B.npy")
        inputs = real_inputs(args.processed_dir, list(args.sources))
    else:
        inputs = synthetic_inputs(
            n_users=args.n_users,
            n_items=args.n_items,
            latent=8,
            dv_a=args.dv_a,
            dv_b=args.dv_b,
            seed=args.data_seed,
        )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    conditions = build_conditions(inputs, args.aligned_dim)
    if args.conditions:
        conditions = {k: v for k, v in conditions.items() if k in set(args.conditions)}
    results = [
        run_one(inputs, visual, condition=name, seed=seed, lr=lr, args=args)
        for name, visual in conditions.items()
        for lr in args.learning_rates
        for seed in args.seeds
    ]
    by_condition = aggregate(results)
    decisions = decide(results, by_condition)
    summary = {
        "args": {k: _jsonable(v) for k, v in vars(args).items()},
        "inputs": {"origin": inputs.origin, "n_users": inputs.n_users, "n_items": inputs.n_items},
        "results": results,
        "by_condition": by_condition,
        "decisions": decisions,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(out, args, inputs, results, by_condition, decisions)
    logger.info("report written to %s", out / "report.md")
    return out


if __name__ == "__main__":
    main()
