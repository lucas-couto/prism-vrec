"""Aggregated-results report generator.

Reads the per-dataset evaluation CSVs produced by ``src/steps/evaluate.py``
under ``results/tables/`` and emits a single Markdown (and optional
LaTeX) summary covering:

* Top-N rows by the chosen metric across every (dataset, condition).
* Best row per recommender, per dataset.
* Frozen vs finetuned head-to-head when both are present.
* Pointer to the bootstrap-CI / Friedman / Wilcoxon CSVs the
  statistical step emits, so the reader knows where the
  significance numbers live.

The intent is dissertation-friendly: drop the generated Markdown
into a chapter, or inline the LaTeX tables directly.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pandas as pd

DEFAULT_METRIC = "ndcg@10"
DEFAULT_TOP_N = 15


def _list_evaluation_csvs(tables_dir: Path) -> list[Path]:
    """Find every ``<dataset>_evaluation_<condition>.csv`` under *tables_dir*."""
    if not tables_dir.is_dir():
        return []
    return sorted(tables_dir.glob("*_evaluation_*.csv"))


def _parse_dataset_condition(path: Path) -> tuple[str, str] | None:
    """Decode the ``<dataset>_evaluation_<condition>.csv`` filename."""
    stem = path.stem  # e.g. amazon_fashion_evaluation_frozen
    if "_evaluation_" not in stem:
        return None
    dataset, _, condition = stem.partition("_evaluation_")
    return dataset, condition


def _load_all(tables_dir: Path) -> pd.DataFrame:
    """Load every evaluation CSV, tag each row with dataset + condition."""
    frames: list[pd.DataFrame] = []
    for csv_path in _list_evaluation_csvs(tables_dir):
        decoded = _parse_dataset_condition(csv_path)
        if decoded is None:
            continue
        dataset, condition = decoded
        df = pd.read_csv(csv_path)
        df["dataset"] = dataset
        df["condition"] = condition
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _md_table(df: pd.DataFrame, columns: Iterable[str]) -> str:
    """Render a small DataFrame slice as a GitHub-flavoured markdown table."""
    cols = list(columns)
    df = df[cols].copy()
    for col in cols:
        if pd.api.types.is_float_dtype(df[col]):
            df[col] = df[col].map(lambda v: f"{v:.4f}" if pd.notna(v) else "—")
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    rows = ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([header, sep, *rows])


def _section_top_n(df: pd.DataFrame, metric: str, n: int) -> str:
    if metric not in df.columns:
        return f"_Metric `{metric}` not present in any evaluation CSV._\n"
    top = df.nlargest(n, metric)
    columns = ["dataset", "condition", "model_name", "embedding_name", metric]
    return _md_table(top, columns)


def _section_best_per_recommender(df: pd.DataFrame, metric: str) -> str:
    if metric not in df.columns:
        return f"_Metric `{metric}` not present in any evaluation CSV._\n"
    idx = df.groupby(["dataset", "condition", "model_name"])[metric].idxmax()
    best = df.loc[idx].sort_values(["dataset", "condition", metric], ascending=[True, True, False])
    columns = ["dataset", "condition", "model_name", "embedding_name", metric]
    return _md_table(best, columns)


def _section_frozen_vs_finetuned(df: pd.DataFrame, metric: str) -> str:
    if metric not in df.columns:
        return ""
    if df["condition"].nunique() < 2:
        return "_Only one condition present; frozen vs finetuned diff omitted._\n"

    # Pivot on (dataset, model_name) only — embedding_name differs between
    # frozen and finetuned conditions (e.g. "resnet50_D128" vs
    # "resnet50_finetuned_D128"), so including it as a pivot key would
    # prevent any row from having both a frozen and a finetuned value.
    # aggfunc="max" picks the best embedding per (dataset, model, condition).
    pivot_keys = ["dataset", "model_name"]
    pivot = df.pivot_table(
        index=pivot_keys,
        columns="condition",
        values=metric,
        aggfunc="max",
    ).reset_index()
    if "frozen" not in pivot.columns or "finetuned" not in pivot.columns:
        return "_Frozen/finetuned columns missing; diff omitted._\n"
    pivot["delta"] = pivot["finetuned"] - pivot["frozen"]
    pivot = pivot.sort_values("delta", ascending=False)

    columns = pivot_keys + ["frozen", "finetuned", "delta"]
    return _md_table(pivot, columns)


def _section_artefact_links(tables_dir: Path) -> str:
    """List the bootstrap / Friedman / Wilcoxon CSVs without inlining them."""
    if not tables_dir.is_dir():
        return "_No `results/tables/` directory found._\n"

    by_kind: dict[str, list[str]] = {
        "bootstrap (per-(model, dataset, condition) CI)": [],
        "Friedman omnibus": [],
        "pairwise Wilcoxon": [],
        "cross-dataset aggregation": [],
    }
    for path in sorted(tables_dir.iterdir()):
        name = path.name
        if "_summary_" in name:
            by_kind["bootstrap (per-(model, dataset, condition) CI)"].append(name)
        elif "_friedman_" in name:
            by_kind["Friedman omnibus"].append(name)
        elif "_pairwise_" in name:
            by_kind["pairwise Wilcoxon"].append(name)
        elif "_aggregated_" in name:
            by_kind["cross-dataset aggregation"].append(name)

    out: list[str] = []
    for kind, files in by_kind.items():
        if not files:
            continue
        out.append(f"- **{kind}** — {len(files)} files:")
        for fname in files[:10]:
            out.append(f"  - `{fname}`")
        if len(files) > 10:
            out.append(f"  - … {len(files) - 10} more")
    return "\n".join(out) if out else "_No statistical artefacts found._\n"


def generate(
    tables_dir: str | Path = "results/tables",
    metric: str = DEFAULT_METRIC,
    top_n: int = DEFAULT_TOP_N,
) -> str:
    """Build the consolidated Markdown report and return it as a string."""
    tables_dir = Path(tables_dir)
    df = _load_all(tables_dir)

    if df.empty:
        return (
            f"# Aggregated results report\n\n"
            f"_No evaluation CSVs found under `{tables_dir}`. "
            f"Run `python main.py --step evaluate` first._\n"
        )

    parts: list[str] = [
        "# Aggregated results report",
        "",
        f"Source: `{tables_dir}` ({len(df)} rows across {df['dataset'].nunique()} datasets).",
        "",
        f"## Top {top_n} configurations by `{metric}`",
        "",
        _section_top_n(df, metric, top_n),
        "",
        "## Best configuration per recommender",
        "",
        "Within each `(dataset, condition, recommender)` cell, the row "
        f"with the highest `{metric}` is reported.",
        "",
        _section_best_per_recommender(df, metric),
        "",
        "## Frozen vs finetuned diff",
        "",
        f"`delta = finetuned − frozen` on `{metric}` (higher is better for FT).",
        "",
        _section_frozen_vs_finetuned(df, metric),
        "",
        "## Statistical artefacts",
        "",
        "Bootstrap CIs, Friedman tests and pairwise Wilcoxon results live in "
        f"`{tables_dir}` alongside the raw evaluation CSVs:",
        "",
        _section_artefact_links(tables_dir),
        "",
    ]
    return "\n".join(parts)


def write_report(
    out_path: str | Path = "results/report.md",
    tables_dir: str | Path = "results/tables",
    metric: str = DEFAULT_METRIC,
    top_n: int = DEFAULT_TOP_N,
) -> Path:
    """Generate the report and atomically write it to *out_path*."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    body = generate(tables_dir=tables_dir, metric=metric, top_n=top_n)
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(body, encoding="utf-8")
    tmp.rename(out)
    return out


__all__ = ["generate", "write_report"]
