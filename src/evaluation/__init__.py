"""Evaluation module: metrics, protocol, and statistical tests."""

from src.evaluation.metrics import (
    compute_all_metrics,
    f1_at_k,
    map_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from src.evaluation.protocol import Evaluator
from src.evaluation.statistical import (
    bonferroni_correction,
    pairwise_significance,
    wilcoxon_test,
)

__all__ = [
    "precision_at_k",
    "recall_at_k",
    "f1_at_k",
    "map_at_k",
    "ndcg_at_k",
    "compute_all_metrics",
    "Evaluator",
    "wilcoxon_test",
    "bonferroni_correction",
    "pairwise_significance",
]
