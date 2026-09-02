"""Evaluation: the correct val/test protocol, metrics, and multi-seed reporting."""

from rigfl.eval.metrics import accuracy, balanced_accuracy
from rigfl.eval.protocol import evaluate_split
from rigfl.eval.report import format_table, summarize, win_rate

__all__ = [
    "accuracy",
    "balanced_accuracy",
    "evaluate_split",
    "summarize",
    "win_rate",
    "format_table",
]
