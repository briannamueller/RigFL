"""The prediction value passed from an algorithm to the evaluator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


#: Absolute tolerance for probability row sums, including float32 ensemble mixtures.
PROB_SUM_ATOL = 1e-3


class PredictionError(ValueError):
    """An algorithm returned something that is not a valid prediction."""


@dataclass
class Predictions:
    """Predicted labels and the optional distributions behind them for a batch.

    ``labels``          ``[N]`` integer class ids -- the prediction.
    ``probabilities``   ``[N, C]`` normalized class probabilities, non-negative
                        and summing to one per row, or ``None``.

    Labels are stored explicitly because an algorithm's decision rule need not be
    the argmax of its reported probabilities.
    """

    labels: torch.Tensor
    probabilities: Optional[torch.Tensor]

    def __post_init__(self):
        if self.labels.ndim != 1:
            raise PredictionError(
                f"labels must be 1-D [N], got shape {tuple(self.labels.shape)}")
        if self.probabilities is not None:
            check_probabilities(self.probabilities, n=self.labels.numel())

    # ── the ways an algorithm builds one ──

    @classmethod
    def from_logits(cls, logits: torch.Tensor) -> "Predictions":
        """Softmax of raw scores. The usual case: any model ending in a linear head.

        A one-logit binary head is normalized to two probability columns,
        ``[1 - p, p]`` for ``p = sigmoid(logit)``.
        """
        if logits.ndim == 1:
            logits = logits.unsqueeze(1)
        if logits.ndim != 2:
            raise PredictionError(
                f"logits must be [N] or [N, C], got shape {tuple(logits.shape)}")
        if logits.shape[1] == 1:
            l = logits.squeeze(1).float()
            p = torch.sigmoid(l)
            probs = torch.stack([1 - p, p], dim=1)
        else:
            probs = torch.softmax(logits.float(), dim=1)
        return cls(labels=probs.argmax(dim=1), probabilities=probs)

    @classmethod
    def from_probabilities(cls, probs: torch.Tensor,
                           labels: Optional[torch.Tensor] = None) -> "Predictions":
        """Already-normalized probabilities -- an ensemble average, a calibrated pool.

        ``labels`` may be given when the algorithm's decision rule is not the argmax
        of these probabilities; it defaults to the argmax.
        """
        # Shape-checked before the argmax, so a 1-D or 3-D input is reported as
        # what it is rather than as a labels problem downstream.
        probs = check_probabilities(probs.float())
        return cls(labels=probs.argmax(dim=1) if labels is None else labels,
                   probabilities=probs)

    @classmethod
    def labels_only(cls, labels: torch.Tensor) -> "Predictions":
        """A prediction that supports label-based but not probability-based metrics."""
        return cls(labels=labels, probabilities=None)


def check_probabilities(probs: torch.Tensor, *, n: Optional[int] = None,
                        num_classes: Optional[int] = None) -> torch.Tensor:
    """Shape, finiteness, non-negativity and row sums, or a message saying which."""
    if probs.ndim != 2:
        raise PredictionError(
            f"probabilities must be [N, C], got shape {tuple(probs.shape)}")
    if n is not None and probs.shape[0] != n:
        raise PredictionError(
            f"probabilities has {probs.shape[0]} rows for {n} label(s)")
    if num_classes is not None and probs.shape[1] != num_classes:
        raise PredictionError(
            f"probabilities has {probs.shape[1]} columns for {num_classes} classes")
    if not torch.isfinite(probs).all():
        raise PredictionError("probabilities contain NaN or inf")
    if (probs < 0).any():
        raise PredictionError(
            f"probabilities contain negative values (min {probs.min().item():.4g}); "
            "these look like logits or log-probabilities rather than probabilities")
    sums = probs.sum(dim=1)
    if not torch.allclose(sums, torch.ones_like(sums), atol=PROB_SUM_ATOL):
        worst = (sums - 1).abs().argmax().item()
        raise PredictionError(
            f"probability rows must sum to 1 (row {worst} sums to "
            f"{sums[worst].item():.6g}, tolerance {PROB_SUM_ATOL}). Normalize "
            "them -- softmax the logits, or divide by the row sum -- rather than "
            "passing unnormalized scores.")
    return probs


def as_predictions(value) -> Predictions:
    """Normalize a :class:`Predictions` value or bare label tensor."""
    if isinstance(value, Predictions):
        return value
    if isinstance(value, torch.Tensor):
        return Predictions.labels_only(value)
    raise PredictionError(
        f"predict() must return Predictions (or, for a label-only algorithm, "
        f"a 1-D tensor of class ids); got {type(value).__name__}")
