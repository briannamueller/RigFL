"""Classification metrics and their optimization directions."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from rigfl.prediction import Predictions, check_predictions, check_probabilities

#: Floor on the true-class probability in log loss; caps one sample at ~27.6 nats.
LOG_LOSS_EPS = 1e-12

def accuracy(preds: torch.Tensor, labels: torch.Tensor) -> float:
    return (preds == labels).float().mean().item()


def balanced_accuracy(preds: torch.Tensor, labels: torch.Tensor, num_classes: int) -> float:
    """Mean recall over the classes present -- macro recall."""
    recalls = []
    for c in range(num_classes):
        in_c = labels == c
        if in_c.any():
            recalls.append((preds[in_c] == c).float().mean().item())
    return sum(recalls) / len(recalls) if recalls else 0.0


def macro_f1(preds: torch.Tensor, labels: torch.Tensor, num_classes: int) -> float:
    """Unweighted mean F1 over classes present in labels or predictions."""
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

    Computed identically for every algorithm on held-out data; unrelated to
    training losses.
    """
    if labels.numel() == 0:
        return 0.0
    true = probs.gather(1, labels.long().view(-1, 1).to(probs.device)).squeeze(1)
    return (-torch.log(true.clamp(min=LOG_LOSS_EPS, max=1.0))).mean().item()


def _binary_ranking_inputs(
    probs: torch.Tensor, labels: torch.Tensor, num_classes: int
) -> tuple[list[float], list[bool]] | None:
    """Class-1 scores and targets, or ``None`` unless both outcomes are present."""
    if num_classes != 2:
        raise ValueError(
            f"AUROC and AUPRC are available only for binary tasks, got "
            f"{num_classes} classes"
        )
    values = [float(value) for value in probs[:, 1].detach().cpu().tolist()]
    positive = [int(label) == 1 for label in labels.cpu().tolist()]
    count = sum(positive)
    if count == 0 or count == len(positive):
        return None
    return values, positive


def auroc(
    probs: torch.Tensor, labels: torch.Tensor, num_classes: int
) -> float | None:
    """Binary area under the ROC curve, with label 1 as the positive class."""
    prepared = _binary_ranking_inputs(probs, labels, num_classes)
    if prepared is None:
        return None
    values, positive = prepared
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        average_rank = ((start + 1) + end) / 2
        for index in order[start:end]:
            ranks[index] = average_rank
        start = end
    positives = sum(positive)
    negatives = len(positive) - positives
    positive_rank_sum = sum(
        rank for rank, is_positive in zip(ranks, positive) if is_positive
    )
    return (
        positive_rank_sum - positives * (positives + 1) / 2
    ) / (positives * negatives)


def average_precision(
    probs: torch.Tensor, labels: torch.Tensor, num_classes: int
) -> float | None:
    """Binary average precision (RigFL's AUPRC), with label 1 as the positive class."""
    prepared = _binary_ranking_inputs(probs, labels, num_classes)
    if prepared is None:
        return None
    values, positive = prepared
    order = sorted(range(len(values)), key=values.__getitem__, reverse=True)
    positives = sum(positive)
    true_positives = 0
    false_positives = 0
    previous_recall = 0.0
    average_precision = 0.0
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        group = order[start:end]
        true_positives += sum(positive[index] for index in group)
        false_positives += sum(not positive[index] for index in group)
        recall = true_positives / positives
        precision = true_positives / (true_positives + false_positives)
        average_precision += (recall - previous_recall) * precision
        previous_recall = recall
        start = end
    return average_precision


class MetricInputUnavailable(ValueError):
    """The prediction does not carry what this metric needs."""


@dataclass(frozen=True)
class MetricSpec:
    name: str
    direction: str                  # "maximize" | "minimize"
    note: str = ""
    #: ``fn(input, labels, num_classes) -> float`` when RigFL computes the metric.
    fn: object = None
    #: Which part of the prediction ``fn`` receives as its first argument.
    needs: str = "labels"
    #: Defined only for binary tasks; recorded as ``None`` otherwise.
    binary_only: bool = False


#: Canonical names. Aliases are resolved at input boundaries by :func:`canonical`,
#: so one representation is used everywhere internally.
METRICS: dict[str, MetricSpec] = {
    "accuracy": MetricSpec("accuracy", "maximize",
                           fn=lambda p, y, c: accuracy(p, y)),
    "balanced_accuracy": MetricSpec("balanced_accuracy", "maximize",
                                    fn=balanced_accuracy),
    "macro_f1": MetricSpec("macro_f1", "maximize", fn=macro_f1),
    "loss": MetricSpec("loss", "minimize", fn=log_loss,
                       needs="probabilities",
                       note="multiclass log loss: mean over samples of -log of the "
                            "probability assigned to the true class, clamped at "
                            f"{LOG_LOSS_EPS:g}. Needs normalized class probabilities, "
                            "which a label-only algorithm does not provide."),
    "auroc": MetricSpec(
        "auroc", "maximize", fn=auroc,
        needs="probabilities", binary_only=True,
        note="Binary tasks only, with label 1 as the positive class. It is "
             "unavailable when a split lacks a positive or a negative example.",
    ),
    "auprc": MetricSpec(
        "auprc", "maximize", fn=average_precision,
        needs="probabilities", binary_only=True,
        note="Average precision; binary tasks only, with label 1 as the "
             "positive class. It is unavailable when a split lacks a positive "
             "or a negative example.",
    ),
}

#: Short spellings accepted at config and CLI boundaries.
ALIASES = {"acc": "accuracy", "bacc": "balanced_accuracy",
           "balanced_acc": "balanced_accuracy", "f1": "macro_f1",
           "log_loss": "loss", "cross_entropy": "loss",
           "roc_auc": "auroc", "average_precision": "auprc"}

#: Metrics that RigFL evaluation computes, in a stable order.
COMPUTED_METRICS = list(METRICS)


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


def require_task_metric(name: str, num_classes: int) -> str:
    """Canonical name of a metric defined for this task; raises ValueError otherwise."""
    s = spec(name)
    if s.binary_only and num_classes != 2:
        raise ValueError(
            f'"{s.name}" is available only for binary tasks; this task has '
            f"{num_classes} classes."
        )
    return s.name


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
    reason = (
        f'"{s.name}" needs class probabilities, but this run recorded only labels. '
        f"Return Predictions with normalized probabilities from predict()."
    )
    return f"{reason} {s.note}" if s.note else reason


def compute_all(
    output: "Predictions | torch.Tensor",
    labels: torch.Tensor,
    num_classes: int,
) -> dict[str, float | None]:
    """Every computed metric, for one client's predictions.

    Registry entries with functions are computed automatically. Metrics whose
    required prediction input is unavailable, and binary-only metrics on a
    multiclass task, are recorded as ``None`` so metric vectors remain aligned.
    """
    output = check_predictions(output)
    if labels.ndim != 1:
        raise ValueError(
            f"evaluation labels must be 1-D [N], got shape {tuple(labels.shape)}"
        )
    if output.labels.numel() != labels.numel():
        raise ValueError(
            f"prediction has {output.labels.numel()} labels for "
            f"{labels.numel()} evaluation sample(s)"
        )
    integer_dtypes = {
        torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64,
    }
    for name, values in (("prediction", output.labels), ("evaluation", labels)):
        if values.dtype not in integer_dtypes:
            raise ValueError(f"{name} labels must contain integer class ids")
        if values.numel() and (
            int(values.min()) < 0 or int(values.max()) >= num_classes
        ):
            raise ValueError(
                f"{name} labels must be in [0, {num_classes}); got range "
                f"[{int(values.min())}, {int(values.max())}]"
            )
    if output.probabilities is not None:
        check_probabilities(
            output.probabilities, n=labels.numel(), num_classes=num_classes
        )
    out: dict[str, float | None] = {}
    for name in COMPUTED_METRICS:
        s = METRICS[name]
        if s.binary_only and num_classes != 2:
            out[name] = None
            continue
        try:
            out[name] = s.fn(metric_input(s, output), labels, num_classes)
        except MetricInputUnavailable:
            out[name] = None
    return out
