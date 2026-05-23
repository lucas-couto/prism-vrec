"""Long-format consolidation of evaluation and statistical outputs.

Kept separate from ``src.evaluation`` so it has no heavy ML dependencies
(``torch`` etc) — these tables are pure pandas reshaping and can be used
from offline analysis scripts on a laptop without a GPU stack installed.
"""

from src.reporting.aggregate_seeds import (
    aggregate_bootstrap_ci,
    aggregate_evaluation,
    write_cross_seed_aggregates,
)
from src.reporting.consolidate import (
    consolidate_bootstrap,
    consolidate_evaluation,
    consolidate_statistical_tests,
    write_consolidated,
)
from src.reporting.long_format import (
    classify_table_file,
    evaluation_to_long,
    friedman_to_long,
    pairwise_to_long,
    parse_config,
    parse_embedding_name,
    summary_to_long,
)

__all__ = [
    "aggregate_bootstrap_ci",
    "aggregate_evaluation",
    "classify_table_file",
    "consolidate_bootstrap",
    "consolidate_evaluation",
    "consolidate_statistical_tests",
    "evaluation_to_long",
    "friedman_to_long",
    "pairwise_to_long",
    "parse_config",
    "parse_embedding_name",
    "summary_to_long",
    "write_consolidated",
    "write_cross_seed_aggregates",
]
