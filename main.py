"""Pipeline entrypoint.

The pipeline is a fixed sequence of named steps living under
``src/steps``.  Each step exposes a ``run(...)`` function; this module
dispatches the right ones depending on the ``pipeline:`` block in
``configs/default.yaml`` and (optionally) command-line flags.

Defaults, what runs when nothing is specified
----------------------------------------------
``configs/default.yaml`` has a ``pipeline:`` block whose values are the
defaults used when no CLI flags are passed.  ``docker compose up -d
--build`` therefore runs the full pipeline end-to-end without any
arguments.

To change what runs, edit that YAML.  Example::

    # configs/default.yaml
    pipeline:
      run_all: false
      start_from: train
      condition: finetuned

CLI flags always override the YAML when present.

Examples
--------
Run the full pipeline (frozen + finetuned, both batteries), also the
default behaviour::

    python main.py
    python main.py --all

Run only the fine-tuning step::

    python main.py --step finetune

Resume from training onwards (skip download/preprocess/extract/finetune)::

    python main.py --from train

Limit to one condition for the steps that take ``--condition``::

    python main.py --step train --condition finetuned

The script never re-orders steps: ``--from`` / ``--to`` and the step
ordering enforced by :data:`STEP_ORDER` always reflect the natural
pipeline order.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

# Set BEFORE any multiprocessing-using import (incl. torch) so the
# resource_tracker subprocess inherits the same warning filter.  Without
# this it logs a ``leaked semaphore`` UserWarning after we exit via
# ``os._exit(0)``, which lands on the terminal *after* the shell prompt
# returned and leaves the cursor parked on the warning text.
os.environ.setdefault("PYTHONWARNINGS", "ignore::UserWarning")

# Optuna emits a UserWarning every time a categorical distribution receives a
# list (e.g. ``hidden_layers: [[256, 128], [512, 256, 128]]``) because lists
# are not hashable and Optuna recommends tuples for persistent storage.  In
# this project the lists are intentional, they round-trip through YAML and
# are consumed elsewhere as lists.  Functionally the warning is harmless
# (Optuna pickles the choices regardless); silencing it just keeps the log
# readable when training neural recommenders.
warnings.filterwarnings(
    "ignore",
    message=r"Choices for a categorical distribution should be a tuple",
    category=UserWarning,
    module=r"optuna\.distributions",
)

from src.steps import (  # noqa: E402
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
)
from src.utils.config import load_config  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402

logger = get_logger("main")


# Steps that take a ``--condition`` argument (frozen / finetuned).  When
# ``condition == "both"``, these run twice, once per condition.
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


def _run_step(name: str, condition: str | None) -> None:
    """Invoke a step, passing ``condition`` only when the step accepts it."""
    from src.utils.timing import now_iso, record_step

    fn = STEP_FUNCTIONS[name]
    label = name if name not in CONDITION_STEPS else f"{name} ({condition})"
    logger.info("===== %s =====", label)
    started_iso = now_iso()
    started = time.time()
    if name in CONDITION_STEPS:
        fn(condition=condition)
    elif name == "statistical":
        fn(condition=condition or "frozen")
    elif name == "export_best":
        fn()
    else:
        fn()
    duration = time.time() - started
    logger.info("===== %s done in %.1fs =====", label, duration)
    record_step(label, started_iso, duration)


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the prism-vrec pipeline.  "
            "With no flags, reads defaults from configs/default.yaml "
            "(pipeline: section)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--all",
        action="store_true",
        help="Run every step end-to-end (overrides the YAML).",
    )
    selection.add_argument(
        "--step",
        choices=STEP_ORDER,
        help="Run a single step by name (overrides the YAML).",
    )
    selection.add_argument(
        "--from",
        dest="from_",
        choices=STEP_ORDER,
        help="Run from this step to the end of the pipeline (overrides the YAML).",
    )
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
            "Resolve the pipeline plan from CLI + YAML and print which "
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
        "--to",
        choices=STEP_ORDER,
        help="When used with --from, stop at this step (inclusive).",
    )

    parser.add_argument(
        "--condition",
        choices=["frozen", "finetuned", "both"],
        default=None,
        help=(
            "Condition for fuse/train/evaluate steps.  Defaults to the "
            "value set in configs/default.yaml under pipeline.condition."
        ),
    )

    parser.add_argument(
        "--config-dir",
        default=None,
        metavar="PATH",
        help=(
            "Alternative directory of YAML config files (e.g. configs/smoke "
            "for the bundled smoke profile).  Defaults to 'configs/'."
        ),
    )

    parser.add_argument(
        "--hp-search",
        choices=["grid", "optuna"],
        default=None,
        help=(
            "Override the hyperparameter-search strategy "
            "(``grid`` or ``optuna``).  Defaults to the value in "
            "configs/recommenders.yaml under hp_search.strategy."
        ),
    )
    parser.add_argument(
        "--eval-protocol",
        choices=["full_ranking", "sampled"],
        default=None,
        help=(
            "Override the evaluation protocol.  Defaults to the value "
            "in configs/evaluation.yaml under evaluation.protocol "
            "(full_ranking).  See README §10 for the trade-off."
        ),
    )
    parser.add_argument(
        "--seeds",
        default=None,
        metavar="N1,N2,...",
        help=(
            "Comma-separated list of seeds for a multi-seed run.  Each "
            "seed runs the pipeline once under suffixed result/checkpoint "
            "paths (results_seed<N>, checkpoints_seed<N>) and an "
            "aggregation pass writes mean/std/median across seeds.  "
            "Overrides ``seeds`` in configs/default.yaml when present."
        ),
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Override hp_search.optuna.n_trials per cell.  Ignored "
            "when --hp-search is grid (or grid is the YAML default)."
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


def _show_plan(args: argparse.Namespace) -> None:
    """Resolve the plan and print which steps would run, then exit."""
    config = load_config()
    steps, condition, run_both = _resolve_plan(args, config)

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


def _resolve_plan(
    args: argparse.Namespace, config: dict[str, Any]
) -> tuple[list[str], str | None, bool]:
    """Decide which steps to run, with which condition, given CLI + YAML.

    Resolution rules
    ----------------
    1. Selection mode (which steps):
       - ``--all`` / ``--step`` / ``--from`` win over the YAML.
       - Otherwise, the YAML's ``pipeline.run_all`` controls the behaviour:
         * ``true``  → run every step.
         * ``false`` → use ``pipeline.start_from`` / ``pipeline.stop_at``.

    2. Condition (which condition for fuse/train/evaluate):
       - ``--condition`` wins.
       - Otherwise, the YAML's ``pipeline.condition`` is used (default ``both``).

    3. Condition-based filtering (only when the step list was derived
       automatically, never when the user pinned a specific step via
       ``--step``):
       - ``condition: frozen``    → drop ``finetune`` / ``evaluate_finetuning``.
       - ``condition: finetuned`` → drop ``extract`` (FT does its own
         re-extraction).
    """
    pipeline_cfg = config.get("pipeline", {})

    user_pinned_step = bool(args.step)

    if args.all:
        steps = list(STEP_ORDER)
    elif args.step:
        steps = [args.step]
    elif args.from_:
        steps = _slice_steps(args.from_, args.to)
    else:
        if pipeline_cfg.get("run_all", True):
            steps = list(STEP_ORDER)
        else:
            steps = _slice_steps(
                pipeline_cfg.get("start_from"),
                pipeline_cfg.get("stop_at"),
            )

    cond = args.condition or pipeline_cfg.get("condition", "both")
    if cond not in {"frozen", "finetuned", "both"}:
        raise ValueError(
            f"pipeline.condition must be 'frozen', 'finetuned' or 'both', got {cond!r}"
        )

    run_both = cond == "both"
    condition = None if run_both else cond

    if not user_pinned_step and not run_both:
        steps = _filter_steps_by_condition(steps, cond)

    return steps, condition, run_both


def _filter_steps_by_condition(steps: list[str], condition: str) -> list[str]:
    """Drop steps whose work is irrelevant to the chosen battery.

    Logs every dropped step so the user always knows which work was
    skipped and why.  When the user explicitly pinned a step via
    ``--step`` this filter is bypassed (see :func:`_resolve_plan`).
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
        _show_plan(args)
        return
    if args.validate_dataset:
        sys.exit(_validate_dataset(args.validate_dataset))
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
    if args.hp_search is not None:
        config.setdefault("hp_search", {})["strategy"] = args.hp_search
    if args.n_trials is not None:
        hp_cfg = config.setdefault("hp_search", {})
        hp_cfg.setdefault("optuna", {})["n_trials"] = args.n_trials
    if args.eval_protocol is not None:
        config.setdefault("evaluation", {})["protocol"] = args.eval_protocol
    if args.seeds is not None:
        try:
            seeds_override = [int(s) for s in args.seeds.split(",") if s.strip()]
        except ValueError as exc:
            raise SystemExit(
                f"--seeds expects a comma-separated list of integers, got {args.seeds!r}: {exc}"
            ) from exc
        if not seeds_override:
            raise SystemExit("--seeds must contain at least one integer")
        if len(set(seeds_override)) != len(seeds_override):
            raise SystemExit("--seeds entries must be unique")
        config["seeds"] = seeds_override

    steps, condition, run_both = _resolve_plan(args, config)

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


def _run_single(
    config: dict[str, Any],
    steps: list[str],
    condition: str | None,
    run_both: bool,
) -> Path:
    """Execute the full pipeline once and return the run directory."""
    from src.utils.carbon import tracker as carbon_tracker
    from src.utils.manifest import finish_run, start_run
    from src.utils.timing import bind_run_dir

    results_root = Path(config.get("paths", {}).get("results", "results"))
    run_dir = start_run(config_snapshot=config, results_root=results_root / "runs")
    bind_run_dir(run_dir)
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
    print(f" Manifest:        {manifest_path}")
    sidecar = Path(run_dir) / "step_timings.json"
    if sidecar.exists():
        print(f" Per-cell timings: {sidecar}")
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
    main(sys.argv[1:])
    # Workaround: PyTorch / Optuna can leave background threads alive
    # after the pipeline finishes (CPython does not always reap them
    # at shutdown), which leaves the user staring at a frozen prompt
    # for several seconds.  All durable outputs (manifest, CSVs,
    # checkpoints) are fsynced through atomic renames during the run,
    # so an immediate process exit is safe.
    import gc
    import multiprocessing

    gc.collect()
    for _child in multiprocessing.active_children():
        _child.terminate()
    os._exit(0)
