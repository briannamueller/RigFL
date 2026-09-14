"""Evaluation: the correct val/test protocol, metrics, and multi-seed reporting."""

from rigfl.eval.comparison import compare_configurations
from rigfl.eval.metrics import accuracy, balanced_accuracy
from rigfl.eval.protocol import evaluate_split
from rigfl.eval.report import format_table, summarize
from rigfl.eval.transfer import negative_transfer_summary, paired_client_gains

__all__ = [
    "accuracy",
    "balanced_accuracy",
    "compare_configurations",
    "evaluate_split",
    "format_table",
    "negative_transfer_summary",
    "paired_client_gains",
    "summarize",
]
