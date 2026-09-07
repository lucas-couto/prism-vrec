"""Pipeline entrypoint.

The pipeline is a fixed sequence of named steps living under
``src/steps``.  Each step exposes a ``run(...)`` function; this module
dispatches the right ones depending on the ``pipeline:`` block in
``configs/default.yaml``.

The YAML is the only control surface for run configuration
----------------------------------------------------------
Which steps run, for which condition, with which search strategy,
protocol and seeds is decided by the merged ``configs/*.yaml`` and by
nothing else, so a run reproduces from ``git checkout`` plus the
untracked ``configs/zz_local.yaml`` of the night.  Example::

    # configs/default.yaml (or configs/zz_local.yaml, merged last)
    pipeline:
      run_all: false          # start_from / stop_at are ignored while true
      start_from: train
      stop_at: beyond_accuracy
      condition: finetuned    # frozen | finetuned | both

    hp_search:
      strategy: optuna        # configs/recommenders.yaml
      optuna:
        n_trials: 30
    evaluation:
      protocol: full_ranking  # configs/evaluation.yaml
    seeds: [42, 99, 7]        # multi-seed run

``python main.py`` (and ``docker compose up -d --build``) runs that
plan; ``python main.py --show-plan`` prints it without running.  The
remaining flags are tools around the run (``--battery``, ``--folds``,
``--report``, ``--inspect-pending``, ``--validate-*``, ``--list-*``,
``--config-dir``).  The former step / condition / search / protocol /
seed flags were removed (3.0.0, by the researcher's decision); passing
one fails with the YAML key that replaced it.

The script never re-orders steps: ``start_from`` / ``stop_at`` and the
ordering enforced by :data:`STEP_ORDER` always reflect the natural
pipeline order.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

# Set BEFORE any multiprocessing-using import (incl. torch) so the
# resource_tracker subprocess inherits the same warning filter.  Without
# this it logs a ``leaked semaphore`` UserWarning after we exit via
# ``os._exit(0)``, which lands on the terminal *after* the shell prompt
# returned and leaves the cursor parked on the warning text.
os.environ.setdefault("PYTHONWARNINGS", "ignore::UserWarning")

from src.steps import (  # noqa: E402
    beyond_accuracy,
    download,
    evaluate,
    evaluate_finetuning,
    export_best,
    extract,
    finetune,
    fuse,
    preprocess,
    statistical,
    train,
    validate_features,
)
from src.utils.config import load_config  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402
from src.utils.memory import warn_if_budget_exceeds_cgroup  # noqa: E402
from src.utils.resources import resolve_resources  # noqa: E402

logger = get_logger("main")


# Steps that take a condition (frozen / finetuned).  When
# ``pipeline.condition == "both"``, these run twice, once per condition.
CONDITION_STEPS = {"fuse", "train", "evaluate"}

# Steps whose work is meaningful only for the *frozen* battery.  When
# the user asks for ``condition: finetuned`` and the step list came
# from automatic expansion (run_all / start_from / stop_at), these are
# silently dropped, the FT step does its own re-extraction, so the
# frozen embeddings are unused in a finetuned-only run.
FROZEN_ONLY_STEPS = {"extract"}

# Steps whose work is meaningful only for the *finetuned* battery.  When
# the user asks for ``condition: frozen`` (or never plans to fine-tune),
# these are dropped from the auto-expanded pipeline so the frozen-only
# run does not pay the multi-hour fine-tuning cost it would never use.
FINETUNED_ONLY_STEPS = {"finetune", "evaluate_finetuning"}

STEP_ORDER: list[str] = [
    "download",
    "preprocess",
    "extract",
    "finetune",
    "evaluate_finetuning",
    "fuse",
    "train",
    "evaluate",
    "beyond_accuracy",
    "statistical",
    "export_best",
]

STEP_FUNCTIONS: dict[str, Callable] = {
    "download": download.run,
    "preprocess": preprocess.run,
    "extract": extract.run,
    "finetune": finetune.run,
    "evaluate_finetuning": evaluate_finetuning.run,
    "fuse": fuse.run,
    "train": train.run,
    "evaluate": evaluate.run,
    "beyond_accuracy": beyond_accuracy.run,
    "statistical": statistical.run,
    "export_best": export_best.run,
}


def _slice_steps(start: str | None, stop: str | None) -> list[str]:
    """Return ``STEP_ORDER`` clipped to the [start, stop] inclusive range."""
    start_idx = STEP_ORDER.index(start) if start else 0
    stop_idx = STEP_ORDER.index(stop) if stop else len(STEP_ORDER) - 1
    if start_idx > stop_idx:
        raise ValueError(
            f"start_from ({start}) cannot come after stop_at ({stop}) in the pipeline order"
        )
    return STEP_ORDER[start_idx : stop_idx + 1]


def _release_gpu_memory() -> None:
    """Drop cached CUDA blocks held by this process before the next step.

    Steps such as ``extract`` and ``fuse`` leave the orchestrating process
    holding several GB of cached VRAM; ``train`` then spawns workers that
    have to share what is left.  Freeing the cache here keeps the parent's
    footprint out of the workers' budget.
    """
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # pragma: no cover - torch missing or CUDA broken
        return


def _log_resource_plan(config: dict[str, Any]) -> None:
    """Resolve ``configs/resources.yaml`` once at startup and log what applies.

    Fails loud on an invalid block (before any step runs) and warns when
    an explicit host budget exceeds the cgroup limit in effect.
    """
    resources = resolve_resources(config)
    warn_if_budget_exceeds_cgroup(resources)
    logger.info(
        "Resources: vram_share=%.2f ranking_vram_share=%.3f host_budget=%s headroom=%.1f GB "
        "workers(training=%d fusion=%d dataloader=%s) residency=%s item_block=%d",
        resources.gpu.vram_share,
        resources.gpu.ranking_vram_share,
        "cgroup" if resources.host.budget_bytes is None else resources.host.budget_bytes,
        resources.host.headroom_bytes / 1024**3,
        resources.workers.training,
        resources.workers.fusion,
        "auto" if resources.workers.dataloader is None else resources.workers.dataloader,
        resources.features.residency,
        resources.features.item_block,
    )


def _run_step(name: str, condition: str | None) -> None:
    """Invoke a step, passing ``condition`` only when the step accepts it."""
    from src.utils import telemetry
    from src.utils.timing import cell_counts, now_iso, record_step

    fn = STEP_FUNCTIONS[name]
    label = name if name not in CONDITION_STEPS else f"{name} ({condition})"
    logger.info("===== %s =====", label)
    _release_gpu_memory()
    started_iso = now_iso()
    started = time.time()
    marker = telemetry.mark()
    cells_before = cell_counts()
    try:
        if name in CONDITION_STEPS:
            fn(condition=condition)
        elif name == "statistical":
            fn(condition=condition or "frozen")
        else:
            fn()
    finally:
        # Record in a finally block so a step that raises still leaves its
        # partial throughput / cost window in the manifest — that is exactly
        # the run a reader needs telemetry for.
        duration = time.time() - started
        metrics = telemetry.summarise_since(marker)
        no_new_work = _did_no_new_work(cells_before, cell_counts())
        if no_new_work:
            # Every cell found its output already on disk.  Timing and
            # costing that window would credit this run with an hour of
            # extraction it never performed.
            metrics = None
        record_step(label, started_iso, duration, metrics, skipped=no_new_work)
    logger.info("===== %s done in %.1fs =====", label, duration)
    _log_step_telemetry(label, metrics)


def _did_no_new_work(before: tuple[int, int], after: tuple[int, int]) -> bool:
    """True when a step ran only cells that were already done.

    *before* / *after* are ``(recorded, skipped)`` snapshots from
    :func:`src.utils.timing.cell_counts`.  Requiring at least one
    skipped cell is what keeps steps that emit no cells at all
    (``preprocess``, ``report``) out of the skipped bucket: they are
    timed as usual.  ``download`` does emit cells, but never skips
    them, so it is never marked skipped either.
    """
    recorded = after[0] - before[0]
    skipped = after[1] - before[1]
    return recorded == 0 and skipped > 0


def _log_step_telemetry(label: str, metrics: dict[str, Any] | None) -> None:
    """Echo the headline throughput / cost figures into the run log.

    The manifest holds the full breakdown; this one line exists so a
    researcher watching ``docker logs -f`` sees immediately whether a
    step was network-bound, compute-bound or idle.
    """
    if not metrics:
        return

    parts: list[str] = []
    throughput = metrics.get("throughput") or {}
    net = throughput.get("network_mb_per_s")
    if net:
        parts.append(f"net {net['mean']:.1f} MB/s (min {net.get('min', net['mean']):.1f})")
    flops = throughput.get("flops_per_s")
    if flops:
        parts.append(f"compute {flops['mean'] / 1e12:.2f} TFLOP/s")

    cost = metrics.get("cost") or {}
    gpu = cost.get("gpu_util_percent")
    if gpu:
        parts.append(f"gpu {gpu['mean']:.0f}%")
    power = cost.get("gpu_power_watts")
    if power:
        parts.append(f"{power['mean']:.0f} W")
    energy = cost.get("energy_wh")
    if energy:
        parts.append(f"{energy:.2f} Wh")

    if parts:
        logger.info("      %s telemetry: %s", label, " | ".join(parts))


def _run_steps(names: list[str], condition: str | None, run_both_conditions: bool) -> None:
    """Run a sequence of steps, expanding condition steps when requested."""
    for name in names:
        if name in CONDITION_STEPS and run_both_conditions:
            _run_step(name, "frozen")
            _run_step(name, "finetuned")
        elif name == "statistical" and run_both_conditions:
            _run_step(name, "all")
        else:
            _run_step(name, condition)


#: Flags removed in 3.0.0 (the YAML is the only control surface) and
#: the key that provides each one's behaviour.  Passing one fails with
#: argparse's standard error plus this hint.
REMOVED_FLAGS: dict[str, str] = {
    "--all": "pipeline.run_all: true (configs/default.yaml)",
    "--step": "pipeline.run_all: false with start_from and stop_at set to the step",
    "--from": "pipeline.run_all: false with pipeline.start_from",
    "--to": "pipeline.run_all: false with pipeline.stop_at",
    "--condition": "pipeline.condition (frozen | finetuned | both)",
    "--hp-search": "hp_search.strategy (configs/recommenders.yaml)",
    "--n-trials": "hp_search.optuna.n_trials (configs/recommenders.yaml)",
    "--eval-protocol": "evaluation.protocol (configs/evaluation.yaml)",
    "--seeds": "seeds: [...] (configs/default.yaml)",
}


def _reject_removed_flags(parser: argparse.ArgumentParser, argv: list[str]) -> None:
    """Fail loud on a removed flag, naming the YAML key that replaced it."""
    for token in argv:
        flag = token.split("=", 1)[0]
        if flag in REMOVED_FLAGS:
            parser.error(
                f"{flag} was removed: the YAML is the only control surface; "
                f"set {REMOVED_FLAGS[flag]} instead"
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the prism-vrec pipeline as configured by configs/*.yaml "
            "(pipeline: section).  Flags are tools around the run, never "
            "overrides of the YAML."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--inspect-pending",
        choices=["frozen", "finetuned"],
        metavar="CONDITION",
        help=(
            "Print how many grid-search jobs are still pending for the "
            "given condition (does not run the pipeline)."
        ),
    )
    selection.add_argument(
        "--list-extractors",
        action="store_true",
        help="Print every registered visual extractor and exit.",
    )
    selection.add_argument(
        "--list-fusions",
        action="store_true",
        help="Print every registered fusion strategy and exit.",
    )
    selection.add_argument(
        "--list-recommenders",
        action="store_true",
        help="Print every registered recommender and exit.",
    )
    selection.add_argument(
        "--list-datasets",
        action="store_true",
        help="Print every registered dataset provider and exit.",
    )
    selection.add_argument(
        "--show-plan",
        action="store_true",
        help=(
            "Resolve the pipeline plan from the merged YAML and print which "
            "steps would run (with condition filtering applied), then exit."
        ),
    )
    selection.add_argument(
        "--validate-dataset",
        metavar="NAME",
        help=(
            "Run schema and image-coverage checks on dataset NAME and "
            "exit with non-zero status if problems are found.  Useful "
            "before launching a multi-day grid search."
        ),
    )
    selection.add_argument(
        "--validate-features",
        nargs="*",
        metavar="DATASET BACKBONE",
        help=(
            "Sanity-check extracted feature matrices (shape, native dim, "
            "dtype, NaN/Inf, zero-norm rows) and exit non-zero on any "
            "failure.  No args = every enabled (dataset, backbone); pass "
            "DATASET or DATASET BACKBONE to narrow.  Run before the battery."
        ),
    )
    selection.add_argument(
        "--battery",
        action="store_true",
        help=(
            "Run the full battery via the resumable runner: enumerate cells, "
            "skip completed ones (idempotent), track state in the manifest, "
            "and retry-safe after a spot-instance interruption."
        ),
    )
    selection.add_argument(
        "--folds",
        action="store_true",
        help=(
            "Run the user-level K-fold cross-validation (configs/default.yaml "
            "-> folds:) over every battery cell with frozen hyperparameters: "
            "train on K-1 folds, fold the held-out users in, evaluate them on "
            "their single target, concatenate the K per-user artifacts."
        ),
    )
    selection.add_argument(
        "--battery-status",
        action="store_true",
        help="Print the battery manifest state counts + remaining-cost projection.",
    )
    selection.add_argument(
        "--retry-failed",
        action="store_true",
        help="With --battery, also re-run cells currently marked failed.",
    )
    selection.add_argument(
        "--report",
        action="store_true",
        help=(
            "Aggregate every evaluation CSV under results/tables/ into "
            "results/report.md (top-N by metric, best per recommender, "
            "frozen vs finetuned delta) and exit."
        ),
    )

    parser.add_argument(
        "--report-metric",
        default="ndcg@10",
        metavar="METRIC",
        help="Metric used to rank configurations in --report (default: ndcg@10).",
    )
    parser.add_argument(
        "--report-top",
        type=int,
        default=15,
        metavar="N",
        help="Number of top configurations to list in --report (default: 15).",
    )

    parser.add_argument(
        "--config-dir",
        default=None,
        metavar="PATH",
        help=(
            "Alternative directory of YAML config files (an ablation or "
            "validation profile that overrides configs/).  Defaults to 'configs/'."
        ),
    )

    return parser


def _list_extractors() -> None:
    """Print every registered extractor (name, role, raw dim) and exit."""
    import src.extractors  # noqa: F401
    from src.extractors.registry import registered_extractor_names

    config = load_config()
    catalogue = config.get("extractors", {})

    print(f"{'name':22s} {'role':12s} {'raw_dim':>8s}  source")
    print("-" * 70)
    for name in registered_extractor_names():
        meta = catalogue.get(name, {})
        role = meta.get("role", "-")
        raw_dim = meta.get("raw_dim", "-")
        model_name = meta.get("model_name") or meta.get("pretrained") or "-"
        print(f"  {name:20s} {role:12s} {str(raw_dim):>8s}  {model_name}")


def _list_fusions() -> None:
    """Print every registered fusion strategy and exit."""
    import src.fusions  # noqa: F401
    from src.fusions.registry import iter_specs

    print(f"{'name':22s} {'equal_dim_required':>22s}")
    print("-" * 50)
    for spec in iter_specs():
        print(f"  {spec.name:20s} {str(spec.equal_dim_required):>22s}")


def _list_recommenders() -> None:
    """Print every registered recommender and exit."""
    import src.recommenders  # noqa: F401
    from src.recommenders.registry import iter_specs

    print(
        f"{'name':14s} {'priority':>8s} {'requires_visual':>16s} {'uses_visual_dim':>16s}  hp_keys",
    )
    print("-" * 80)
    for spec in iter_specs():
        hp_keys = ", ".join(spec.extra_hyperparam_keys) or "-"
        print(
            f"  {spec.name:12s} {spec.priority:>8d} "
            f"{str(spec.requires_visual):>16s} {str(spec.uses_visual_dim):>16s}  {hp_keys}",
        )


def _list_datasets() -> None:
    """Print every registered dataset provider and exit."""
    import src.data  # noqa: F401
    from src.data.base import registered_dataset_names

    config = load_config()
    enabled = set(config.get("datasets") or [])

    print(f"{'name':24s} {'enabled':>8s}")
    print("-" * 40)
    for name in registered_dataset_names():
        mark = "✓" if name in enabled else "-"
        print(f"  {name:22s} {mark:>8s}")


def _validate_dataset(name: str) -> int:
    """Validate the on-disk layout for ``name`` and return an exit code."""
    import src.data  # noqa: F401
    from src.data.base import registered_dataset_names, validate_layout

    config = load_config()
    raw_dir = config["paths"]["data_raw"]
    processed_dir = config["paths"]["data_processed"]

    if name not in registered_dataset_names():
        print(
            f"Dataset {name!r} is not registered.  Registered datasets: "
            f"{registered_dataset_names()}",
        )
        return 2

    problems = validate_layout(name, raw_dir=raw_dir, processed_dir=processed_dir)
    if not problems:
        print(f"Dataset {name!r}: layout OK.")
        return 0

    print(f"Dataset {name!r}: {len(problems)} problem(s) detected:")
    for problem in problems:
        print(f"  - {problem}")
    return 1


def _show_plan() -> None:
    """Resolve the plan from the merged YAML and print which steps would run."""
    config = load_config()
    steps, condition, run_both = _resolve_plan(config)

    print(
        f"Plan resolved with condition={'both' if run_both else condition!r} ({len(steps)} steps):",
    )
    for idx, step in enumerate(steps, start=1):
        if step in CONDITION_STEPS and run_both:
            print(f"  {idx:2d}. {step:24s}  (runs twice: frozen + finetuned)")
        elif step == "statistical" and run_both:
            print(f"  {idx:2d}. {step:24s}  (condition='all')")
        else:
            print(f"  {idx:2d}. {step}")


def _inspect_pending(condition: str) -> None:
    """Print pending grid-search jobs for the given condition.

    Mirrors the behaviour of the old ``scripts/list_pending_jobs.py``
    helper without requiring users to remember its path.
    """
    from collections import Counter

    from src.steps.train import build_job_list

    config = load_config()
    jobs = build_job_list(
        condition,
        config,
        config["paths"]["data_processed"],
        config["paths"]["embeddings"],
        config["device"],
    )

    label = "Battery 1" if condition == "frozen" else "Battery 2"
    print(f"Pending {label} jobs: {len(jobs)}")
    print()

    by_ds_model = Counter((j.dataset_name, j.model_name) for j in jobs)
    print(f"{'dataset':18s} {'model':10s} {'pending':>8s}")
    print("-" * 40)
    for (ds, m), n in sorted(by_ds_model.items()):
        print(f"  {ds:16s} {m:10s} {n:>8d}")

    print()
    by_ds = Counter(j.dataset_name for j in jobs)
    print(f"{'dataset':18s} {'total pending':>14s}")
    print("-" * 36)
    for ds, n in sorted(by_ds.items()):
        print(f"  {ds:16s} {n:>14d}")


def _resolve_plan(config: dict[str, Any]) -> tuple[list[str], str | None, bool]:
    """Decide which steps to run, with which condition, from the merged YAML.

    Resolution rules
    ----------------
    1. Selection (``pipeline:``): ``run_all: true`` runs every step;
       ``false`` uses ``start_from`` / ``stop_at`` (both inclusive,
       ``null`` = the ends of :data:`STEP_ORDER`).  The range keys are
       ignored while ``run_all`` is ``true``.
    2. Condition: ``pipeline.condition`` (default ``both``).
    3. Condition filtering: ``frozen`` drops ``finetune`` /
       ``evaluate_finetuning``; ``finetuned`` drops ``extract`` (the FT
       step does its own re-extraction).  A plan that ends up empty is
       an error, not a silent no-op.
    """
    pipeline_cfg = config.get("pipeline", {})

    if pipeline_cfg.get("run_all", True):
        steps = list(STEP_ORDER)
    else:
        steps = _slice_steps(pipeline_cfg.get("start_from"), pipeline_cfg.get("stop_at"))

    cond = pipeline_cfg.get("condition", "both")
    if cond not in {"frozen", "finetuned", "both"}:
        raise ValueError(
            f"pipeline.condition must be 'frozen', 'finetuned' or 'both', got {cond!r}"
        )

    run_both = cond == "both"
    condition = None if run_both else cond

    if not run_both:
        steps = _filter_steps_by_condition(steps, cond)
    if not steps:
        raise ValueError(
            f"pipeline resolved to no steps: start_from={pipeline_cfg.get('start_from')!r} "
            f"stop_at={pipeline_cfg.get('stop_at')!r} are all irrelevant to "
            f"pipeline.condition={cond!r}"
        )

    return steps, condition, run_both


def _plan_record(steps: list[str], condition: str | None, run_both: bool) -> dict[str, Any]:
    """The resolved plan as written to ``manifest['plan']``."""
    return {"steps": list(steps), "condition": "both" if run_both else condition}


def _filter_steps_by_condition(steps: list[str], condition: str) -> list[str]:
    """Drop steps whose work is irrelevant to the chosen battery.

    Logs every dropped step so the user always knows which work was
    skipped and why.
    """
    if condition == "frozen":
        irrelevant = FINETUNED_ONLY_STEPS
    elif condition == "finetuned":
        irrelevant = FROZEN_ONLY_STEPS
    else:
        return steps

    kept: list[str] = []
    for name in steps:
        if name in irrelevant:
            logger.info(
                "Skipping step %r, irrelevant to condition=%s.",
                name,
                condition,
            )
        else:
            kept.append(name)
    return kept


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    _reject_removed_flags(parser, sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(argv)

    if args.config_dir:
        from src.utils.config import set_config_dir

        set_config_dir(args.config_dir)

    if args.list_extractors:
        _list_extractors()
        return
    if args.list_fusions:
        _list_fusions()
        return
    if args.list_recommenders:
        _list_recommenders()
        return
    if args.list_datasets:
        _list_datasets()
        return
    if args.show_plan:
        _show_plan()
        return
    if args.validate_dataset:
        sys.exit(_validate_dataset(args.validate_dataset))
    if args.validate_features is not None:
        ds = args.validate_features[0] if len(args.validate_features) >= 1 else None
        bb = args.validate_features[1] if len(args.validate_features) >= 2 else None
        sys.exit(validate_features.run(dataset=ds, backbone=bb))
    if args.battery_status:
        from src.battery.runner import battery_status

        cfg = load_config()
        battery_status(cfg["paths"]["results"])
        return
    if args.folds:
        from src.folds.runner import run_folds

        cfg = load_config()
        _log_resource_plan(cfg)
        _require_complete(run_folds(cfg, cfg["paths"]["results"]), label="K-fold run")
        return
    if args.battery:
        from src.battery.execute import execute_cell
        from src.battery.runner import run_battery

        cfg = load_config()
        _log_resource_plan(cfg)
        manifest = run_battery(
            cfg, cfg["paths"]["results"], execute_cell, retry_failed=args.retry_failed
        )
        _require_complete(manifest, label="battery")
        return
    if args.report:
        from src.utils.report import write_report

        config = load_config()
        results_dir = Path(config.get("paths", {}).get("results", "results"))
        tables_dir = results_dir / "tables"
        out_path = results_dir / "report.md"
        written = write_report(
            out_path=out_path,
            tables_dir=tables_dir,
            metric=args.report_metric,
            top_n=args.report_top,
        )
        print(f"Report written to {written}")
        return
    if args.inspect_pending:
        _inspect_pending(args.inspect_pending)
        return

    config = load_config()
    steps, condition, run_both = _resolve_plan(config)
    _log_resource_plan(config)

    from src.utils.logging import session_log_path

    session_path = session_log_path()
    if session_path is not None:
        logger.info("Unified session log: %s, `tail -f` it to follow the run.", session_path)

    logger.info(
        "Pipeline plan: steps=%s condition=%s run_both=%s",
        steps,
        condition if condition is not None else "(both)",
        run_both,
    )

    seeds = config.get("seeds")
    if seeds:
        _run_multi_seed(seeds, config, steps, condition, run_both)
    else:
        _run_single(config, steps, condition, run_both)


class IncompleteRunError(RuntimeError):
    """A battery / K-fold manifest still holds cells that did not finish.

    The runners return their manifest even when cells failed; without
    this check ``main.py`` exited zero on a battery with failed cells
    (audit F04).  The manifest itself is untouched, so ``--battery
    --retry-failed`` resumes exactly the cells listed here.
    """


def _require_complete(manifest: Any, *, label: str) -> None:
    """Raise :class:`IncompleteRunError` unless every cell is ``done``."""
    summary = manifest.summary()
    unfinished = {state: n for state, n in summary.items() if state != "done" and n > 0}
    if not unfinished:
        return
    breakdown = ", ".join(f"{n} {state}" for state, n in sorted(unfinished.items()))
    raise IncompleteRunError(
        f"{label} finished with unfinished cells ({breakdown}); "
        f"{summary.get('done', 0)} done. See the manifest for the cell list."
    )


def run_cli(argv: list[str] | None = None) -> int:
    """Run :func:`main` and translate its outcome into a process exit code.

    ``0`` on success, the code carried by ``SystemExit``, ``130`` on
    ``KeyboardInterrupt`` and ``1`` for any other exception -- which is
    logged with its traceback so a step failure that already updated
    the run manifest (see :func:`_run_single`) is never mistaken for a
    clean finish.  This is the boundary tests exercise in-process.
    """
    try:
        main(argv)
    except SystemExit as exc:
        return _exit_code(exc.code)
    except KeyboardInterrupt:
        logger.warning("Interrupted by user.")
        return 130
    except Exception:  # noqa: BLE001 — the boundary must yield a code, not a traceback
        logger.error("Pipeline failed.", exc_info=True)
        return 1
    return 0


def _exit_code(code: object) -> int:
    if code is None:
        return 0
    if isinstance(code, int):
        return code
    # ``sys.exit("message")`` prints the message and exits 1.
    print(code, file=sys.stderr)
    return 1


def _run_single(
    config: dict[str, Any],
    steps: list[str],
    condition: str | None,
    run_both: bool,
) -> Path:
    """Execute the full pipeline once and return the run directory."""
    from src.utils import telemetry
    from src.utils.carbon import tracker as carbon_tracker
    from src.utils.manifest import finish_run, start_run
    from src.utils.timing import bind_run_dir

    results_root = Path(config.get("paths", {}).get("results", "results"))
    run_dir = start_run(
        config_snapshot=config,
        results_root=results_root / "runs",
        plan=_plan_record(steps, condition, run_both),
    )
    bind_run_dir(run_dir)
    # One sampler for the whole invocation; every step and cell slices its
    # own window out of the shared series.
    telemetry.start(config, run_dir)
    exit_status = "ok"
    try:
        with carbon_tracker(run_dir):
            _run_steps(steps, condition, run_both)
        logger.info("Pipeline finished.")
    except KeyboardInterrupt:
        exit_status = "interrupted"
        raise
    except Exception:
        exit_status = "error"
        raise
    finally:
        # Stop before finish_run: the manifest records the probe backends
        # and the sampler flushes its raw series here.
        telemetry.stop()
        finish_run(run_dir, exit_status=exit_status)
        _print_post_run_summary(run_dir, exit_status)
    return run_dir


def _run_multi_seed(
    seeds: list[int],
    base_config: dict[str, Any],
    steps: list[str],
    condition: str | None,
    run_both: bool,
) -> None:
    """Execute the pipeline once per seed under suffixed result/checkpoint paths.

    Inputs (data/raw, data/processed, data/embeddings) are reused
    across seeds since they do not depend on the recommender seed;
    only ``paths.results`` and ``paths.checkpoints`` are suffixed so
    paired statistical analysis across seeds becomes possible.  After
    every seed finishes, a cross-seed aggregation pass writes
    mean/std/median CSVs under the base ``paths.results``.
    """
    from src.utils.config import derive_seed_config, set_config_override

    logger.info("Multi-seed run: seeds=%s", seeds)
    for seed in seeds:
        logger.info(">>> Starting pipeline for seed=%d", seed)
        seed_config = derive_seed_config(base_config, seed)
        set_config_override(seed_config)
        try:
            _run_single(seed_config, steps, condition, run_both)
        finally:
            set_config_override(None)

    set_config_override(None)
    _aggregate_seed_results(base_config, seeds)


def _aggregate_seed_results(base_config: dict[str, Any], seeds: list[int]) -> None:
    """Read each seed's evaluation CSV and emit cross-seed aggregates."""
    try:
        from src.reporting.aggregate_seeds import write_cross_seed_aggregates
    except ImportError as exc:
        logger.warning("Cross-seed aggregation unavailable: %s", exc)
        return

    base_results = Path(base_config.get("paths", {}).get("results", "results"))
    seed_dirs = [Path(f"{base_results}_seed{s}") for s in seeds]
    output_dir = base_results / "aggregated_across_seeds"
    try:
        written = write_cross_seed_aggregates(seed_dirs, output_dir, seeds=seeds)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Cross-seed aggregation failed: %s", exc)
        return
    for label, path in written.items():
        logger.info("Cross-seed %s: %s", label, path)


def _print_post_run_summary(run_dir: Path, exit_status: str) -> None:
    """Render a short human-readable summary of the run.

    Reads the manifest that :func:`finish_run` just wrote and prints
    the total wall-time, the top three most expensive steps, and the
    path to the artefact directory.  Designed to be the last thing a
    researcher sees on the terminal so they do not need to grep the
    log for the duration of the run they just kicked off.
    """
    import json

    manifest_path = Path(run_dir) / "manifest.json"
    if not manifest_path.exists():
        return

    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        logger.debug("skipping post-run summary; could not read %s (%r)", manifest_path, exc)
        return

    total = manifest.get("duration_seconds")
    steps = manifest.get("steps") or []
    top = sorted(steps, key=lambda s: s.get("duration_seconds", 0), reverse=True)[:3]

    print()
    print("=" * 72)
    print(f" Run finished: {manifest.get('run_id', '?')}  [{exit_status}]")
    if total is not None:
        print(f" Total wall-time: {_format_duration(total)}")
    if top:
        print(" Most expensive steps:")
        for s in top:
            print(f"   {s['name']:32s}  {_format_duration(s['duration_seconds'])}")

    totals = (manifest.get("telemetry") or {}).get("totals") or {}
    if totals:
        print(" Measured cost:")
        if "energy_wh" in totals:
            print(f"   GPU energy                        {totals['energy_wh']:.2f} Wh")
        if "total_petaflops" in totals:
            print(f"   Compute                           {totals['total_petaflops']:.4f} PFLOPs")
        if "total_downloaded_gb" in totals:
            print(f"   Downloaded                        {totals['total_downloaded_gb']:.2f} GB")

    print(f" Manifest:        {manifest_path}")
    sidecar = Path(run_dir) / "step_timings.json"
    if sidecar.exists():
        print(f" Per-cell timings: {sidecar}")
    samples = Path(run_dir) / "telemetry_samples.jsonl"
    if samples.exists():
        print(f" Telemetry series: {samples}")
    print("=" * 72)


def _format_duration(seconds: float) -> str:
    """``3725s`` -> ``1h2m5s`` for human-readable totals."""
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m}m{s}s"
    if m:
        return f"{m}m{s}s"
    return f"{s}s"


if __name__ == "__main__":
    _code = run_cli(sys.argv[1:])
    # Workaround: PyTorch / Optuna can leave background threads alive
    # after the pipeline finishes (CPython does not always reap them
    # at shutdown), which leaves the user staring at a frozen prompt
    # for several seconds.  All durable outputs (manifest, CSVs,
    # checkpoints) are fsynced through atomic renames during the run,
    # so an immediate process exit is safe.  The code is the one
    # ``run_cli`` derived: a failed step exits non-zero even though the
    # exception was consumed to reach this cleanup.
    import gc
    import multiprocessing

    gc.collect()
    for _child in multiprocessing.active_children():
        _child.terminate()
    os._exit(_code)
