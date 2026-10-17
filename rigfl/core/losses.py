"""Shared supervised classification losses and class-weighting setup."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _label_counts(client, num_classes: int) -> torch.Tensor:
    counts = torch.zeros(num_classes, dtype=torch.long)
    dataset = client.train_loader.dataset
    for index in range(len(dataset)):
        _, label = dataset[index]
        value = int(label.item()) if isinstance(label, torch.Tensor) else int(label)
        if value < 0 or value >= num_classes:
            raise ValueError(
                f"training label {value} is outside [0, {num_classes})"
            )
        counts[value] += 1
    return counts


def _balanced_weights(counts: torch.Tensor, context: str) -> torch.Tensor:
    missing = torch.where(counts == 0)[0].tolist()
    if missing:
        raise ValueError(
            f"{context} class weighting requires every class in the training "
            f"split; missing classes: {missing}"
        )
    total = counts.sum().to(torch.float64)
    return (total / (len(counts) * counts.to(torch.float64))).to(torch.float32)


def apply_class_weighting(clients, mode: str, num_classes: int) -> None:
    """Set client loss weights from training labels."""
    if mode == "none":
        return

    counts = [_label_counts(client, num_classes) for client in clients]
    if mode == "global_balanced":
        weights = _balanced_weights(
            torch.stack(counts).sum(dim=0), "global_balanced"
        )
        for client in clients:
            client.class_weights = weights.clone()
        return
    if mode == "local_balanced":
        for client_id, (client, client_counts) in enumerate(zip(clients, counts)):
            client.class_weights = _balanced_weights(
                client_counts, f"local_balanced client {client_id}"
            )
        return
    raise ValueError(f"unknown class-weighting mode {mode!r}")


def classification_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    class_weights: torch.Tensor | None,
) -> torch.Tensor:
    """Cross-entropy with optional training-derived class weights."""
    weight = None
    if class_weights is not None:
        weight = class_weights.to(device=logits.device, dtype=logits.dtype)
    return F.cross_entropy(logits, labels, weight=weight)


def distribution_classification_loss(
    probabilities: torch.Tensor,
    labels: torch.Tensor,
    class_weights: torch.Tensor | None,
) -> torch.Tensor:
    """Cross-entropy for normalized probabilities with identical weighting."""
    weight = None
    if class_weights is not None:
        weight = class_weights.to(
            device=probabilities.device, dtype=probabilities.dtype
        )
    return F.nll_loss(
        probabilities.clamp_min(1e-12).log(), labels, weight=weight
    )
