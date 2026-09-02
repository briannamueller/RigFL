"""Classification metrics and their optimization directions."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from rigfl.prediction import (Predictions, as_predictions,
                              check_probabilities)

#: Clamp applied to a true-class probability before taking its log. An algorithm may
#: legitimately assign a class zero probability -- a hard-vote ensemble does it
#: routinely -- and ``-log 0`` is infinite, which would make one sample dominate
#: a client's mean and poison every aggregate downstream. The clamp bounds a
#: single sample's contribution at ``-log(1e-12) ~= 27.6``.
LOG_LOSS_EPS = 1e-12

def accuracy(preds: torch.Tensor, labels: torch.Tensor) -> float:
    return (preds == labels).float().mean().item()


def balanced_accuracy(preds: torch.Tensor, labels: torch.Tensor, num_classes: int) -> float:
    """Mean recall over the classes present -- macro recall.

    Useful when classes are imbalanced, because a model that ignores a rare class
    cannot score well by getting the common one right. That makes it a good
    choice for some experiments, not a universally preferable one.
    """
    if labels.numel() and int(labels.max()) >= num_classes:
        raise ValueError(f"balanced_accuracy: label {int(labels.max())} >= num_classes {num_classes}")
    recalls = []
    for c in range(num_classes):
        in_c = labels == c
        if in_c.any():
            recalls.append((preds[in_c] == c).float().mean().item())
    return sum(recalls) / len(recalls) if recalls else 0.0


def macro_f1(preds: torch.Tensor, labels: torch.Tensor, num_classes: int) -> float:
    """Unweighted mean F1 over classes present in labels or predictions."""
    if labels.numel() and int(labels.max()) >= num_classes:
        raise ValueError(f"macro_f1: label {int(labels.max())} >= num_classes {num_classes}")
    scores = []
    for c in range(num_classes):
        in_c = labels == c
        if not in_c.any() and not (preds == c).any():
            continue
        tp = int(((preds == c) & in_c).sum())
        fp = int(((preds == c) & ~in_c).sum())
        fn = int(((preds != c) & in_c).sum())
        denom = 2 * tp + fp + fn
        scores.append(2 * tp / denom if denom else 0.0)
    return sum(scores) / len(scores) if scores else 0.0


def log_loss(probs: torch.Tensor, labels: torch.Tensor, num_classes: int) -> float:
    """Multiclass log loss (cross-entropy) of a predicted distribution.

    .. math:: \\frac{1}{N} \\sum_i -\\log \\hat{p}_{i, y_i}

    the mean over samples of the negative log of the probability the algorithm
    assigned to the *true* class, with that probability clamped to
    ``[LOG_LOSS_EPS, 1]`` before the log. Natural log, so the unit is nats and a
    uniform prediction over ``C`` classes scores ``log C``.

    This is an **evaluation** loss: it is computed from the distribution the
    algorithm predicts at the evaluation boundary, on a held-out split, and it is
    the same function for every algorithm. It is not any algorithm's training
    objective -- those carry regularizers (prototype pull, mutual distillation,
    contrastive margins) that are not comparable across algorithms and are measured
    on the training data the model just fit.
    """
    probs = check_probabilities(probs, n=labels.numel(), num_classes=num_classes)
    if labels.numel() == 0:
        return 0.0
    if int(labels.max()) >= num_classes or int(labels.min()) < 0:
        raise ValueError(f"log_loss: label {int(labels.max())} outside [0, {num_classes})")
    true = probs.gather(1, labels.long().view(-1, 1).to(probs.device)).squeeze(1)
    return (-torch.log(true.clamp(min=LOG_LOSS_EPS, max=1.0))).mean().item()


# ── The registry ────────────────────────────────────────────────────────────

#: What a metric function needs from a :class:`~rigfl.core.interfaces.Predictions`.
#: Stated per metric, because it decides both what is passed to ``fn`` and
#: whether a label-only algorithm can produce the metric at all.
INPUTS = ("labels", "probabilities")


class MetricInputUnavailable(ValueError):
    """The prediction does not carry what this metric needs."""


@dataclass(frozen=True)
class MetricSpec:
    name: str
    direction: str                  # "maximize" | "minimize"
    computed: bool                  # does RigFL's evaluation produce it?
    note: str = ""
    #: ``fn(input, labels, num_classes) -> float`` when RigFL computes the metric.
    fn: object = None
    #: Which part of the prediction ``fn`` receives as its first argument.
    needs: str = "labels"

    @property
    def higher_is_better(self) -> bool:
        return self.direction == "maximize"


#: Canonical names. Aliases are resolved at input boundaries by :func:`canonical`,
#: so one representation is used everywhere internally.
METRICS: dict[str, MetricSpec] = {
    "accuracy": MetricSpec("accuracy", "maximize", computed=True,
                           fn=lambda p, y, c: accuracy(p, y)),
    "balanced_accuracy": MetricSpec("balanced_accuracy", "maximize", computed=True,
                                    fn=balanced_accuracy),
    "macro_f1": MetricSpec("macro_f1", "maximize", computed=True, fn=macro_f1),
    # Computed from the distribution every built-in algorithm returns. Distinct from
    # any algorithm's training loss, which is measured on training data and carries
    # algorithm-specific regularizers.
    "loss": MetricSpec("loss", "minimize", computed=True, fn=log_loss,
                       needs="probabilities",
                       note="multiclass log loss: mean over samples of -log of the "
                            "probability assigned to the true class, clamped at "
                            f"{LOG_LOSS_EPS:g}. Needs normalized class probabilities, "
                            "which a label-only algorithm does not provide."),
}

#: Short spellings accepted at config and CLI boundaries.
ALIASES = {"acc": "accuracy", "bacc": "balanced_accuracy",
           "balanced_acc": "balanced_accuracy", "f1": "macro_f1",
           "log_loss": "loss", "cross_entropy": "loss"}

#: Metrics that RigFL evaluation computes, in a stable order.
COMPUTED_METRICS = [n for n, s in METRICS.items() if s.computed]


def canonical(name: str) -> str:
    """Resolve an alias to its canonical name, or raise with the known set."""
    key = str(name).strip()
    resolved = ALIASES.get(key, key)
    if resolved not in METRICS:
        import difflib
        close = difflib.get_close_matches(key, sorted(METRICS) + sorted(ALIASES), n=1)
        hint = f' Did you mean "{close[0]}"?' if close else ""
        raise ValueError(
            f'Unknown metric "{name}".{hint} '
            f'Known: {", ".join(sorted(METRICS))} (aliases: {", ".join(sorted(ALIASES))}).')
    return resolved


def spec(name: str) -> MetricSpec:
    return METRICS[canonical(name)]


def direction_of(name: str) -> str:
    return spec(name).direction


def require_computable(name: str) -> str:
    """Canonical name of a metric RigFL can actually produce, or a clear error."""
    s = spec(name)
    if not s.computed:
        raise ValueError(f'Metric "{s.name}" is not available in RigFL. {s.note}')
    return s.name


def register(name: str, direction: str, *, fn=None, note: str = "",
             needs: str = "labels") -> None:
    """Add a custom metric.

    ``direction`` is required -- there is no default. ``fn`` makes it computed:
    without one the metric is only a name and a direction, usable for selecting
    over a history that already contains it but never produced by evaluation.

    ``needs`` says what ``fn`` receives first, and defaults to ``"labels"``.
    Use ``"probabilities"`` for confidence-dependent metrics.
    """
    if direction not in ("maximize", "minimize"):
        raise ValueError(f'direction must be "maximize" or "minimize", got {direction!r}')
    if needs not in INPUTS:
        raise ValueError(f'needs must be one of {INPUTS}, got {needs!r}')
    METRICS[name] = MetricSpec(name, direction, computed=fn is not None, note=note,
                               fn=fn, needs=needs)
    if fn is not None and name not in COMPUTED_METRICS:
        COMPUTED_METRICS.append(name)
    elif fn is None and name in COMPUTED_METRICS:
        COMPUTED_METRICS.remove(name)


def unregister(name: str) -> None:
    """Remove a metric (used by tests that register temporary ones)."""
    METRICS.pop(name, None)
    if name in COMPUTED_METRICS:
        COMPUTED_METRICS.remove(name)


def metric_input(spec: MetricSpec, output: Predictions) -> torch.Tensor:
    """The part of a prediction a metric consumes, or why it is not there."""
    if spec.needs == "labels":
        return output.labels
    value = getattr(output, spec.needs)
    if value is None:
        raise MetricInputUnavailable(unavailable_reason(spec.name))
    return value


def unavailable_reason(name: str) -> str:
    """Why a metric could not be computed for a label-only prediction."""
    s = spec(name)
    if s.needs == "labels":
        return ""
    return (f'"{s.name}" needs {s.needs} from the algorithm\'s prediction, and this run '
            f"recorded none. An algorithm whose predict() returns bare labels -- or "
            f"Predictions.labels_only -- can serve accuracy, balanced_accuracy "
            f"and macro_f1, which depend only on which class was predicted, but not "
            f'"{s.name}", which depends on how confidently. Return Predictions '
            f"carrying normalized class probabilities to make it computable; one-hot "
            f"probabilities derived from the labels would report a confidence the "
            f"algorithm never expressed.")


def compute_all(output: "Predictions | torch.Tensor", labels: torch.Tensor,
                num_classes: int) -> dict[str, float | None]:
    """Every computed metric, for one client's predictions.

    Registry entries with functions are computed automatically. Metrics whose
    required prediction input is unavailable are recorded as ``None`` so metric
    vectors remain aligned.
    """
    output = as_predictions(output)
    out: dict[str, float | None] = {}
    for name in COMPUTED_METRICS:
        s = METRICS[name]
        try:
            out[name] = s.fn(metric_input(s, output), labels, num_classes)
        except MetricInputUnavailable:
            out[name] = None
    return out
