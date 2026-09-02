"""The per-client model: a private backbone, an alignment adapter, and a head."""

from __future__ import annotations

import torch
import torch.nn as nn

from rigfl.core.adapters import Adapter


class ClientModel(nn.Module):
    """A client's model, split into three explicit pieces.

    * ``backbone`` — a *private*, possibly heterogeneous feature extractor that
      emits native features ``(B, native_dim)``. It is never averaged as raw
      weights across clients with different architectures.
    * ``adapter`` — maps native features into the shared representation width
      ``(B, shared_dim)`` (interface ``d``); see :mod:`rigfl.core.adapters`.
    * ``head`` — maps the shared representation to logits ``(B, num_classes)``
      (interface ``C``).

    An algorithm reads whichever interface it synchronizes on:

    * ``rep(x)`` — the shared-dim representation, for representation-space (``d``)
      algorithms (prototypes, shared representations, an averaged head).
    * ``forward(x)`` — logits, for logit-space (``C``) algorithms (mutual
      distillation).
    """

    def __init__(self, backbone: nn.Module, adapter: Adapter, head: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self.adapter = adapter
        self.head = head

    def rep(self, x: torch.Tensor) -> torch.Tensor:
        """Shared-dim representation (interface ``d``)."""
        return self.adapter(self.backbone(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Logits (interface ``C``)."""
        return self.head(self.rep(x))


def assemble_model(backbone: nn.Module, *, shared_dim: int, num_classes: int,
                   adapter) -> ClientModel:
    """Apply RigFL's one backbone–adapter–head construction contract."""
    return ClientModel(
        backbone,
        adapter(backbone.out_dim, shared_dim),
        nn.Linear(shared_dim, num_classes),
    )
