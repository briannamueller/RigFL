"""The per-client model: a private backbone, an alignment adapter, and a head."""

from __future__ import annotations

import torch
import torch.nn as nn

from rigfl.core.adapters import Adapter, Identity


class ClientModel(nn.Module):
    """A client's model, split into three explicit pieces.

    * ``backbone`` — a *private*, possibly heterogeneous feature extractor that
      emits native features ``(B, native_dim)``. It is never averaged as raw
      weights across clients with different architectures.
    * ``adapter`` — maps native features into the shared representation width
      ``(B, shared_dim)`` (interface ``d``) for algorithms that exchange
      representations, and is ``Identity`` otherwise; see
      :mod:`rigfl.core.adapters`.
    * ``head`` — maps the representation to logits ``(B, num_classes)``
      (interface ``C``).

    An algorithm reads whichever interface it synchronizes on:

    * ``rep(x)`` — the representation, for representation-space (``d``)
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
        """Representation fed to the head (interface ``d``)."""
        return self.adapter(self.backbone(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Logits (interface ``C``)."""
        return self.head(self.rep(x))


def assemble_model(backbone: nn.Module, *, shared_dim: int, num_classes: int,
                   adapter=None) -> ClientModel:
    """Build a ClientModel: backbone, adapter, linear head.

    ``adapter`` is a factory ``(native_dim, shared_dim) -> Adapter``. Without
    one, the head sits directly on the backbone's native output and
    ``shared_dim`` is unused.
    """
    if adapter is None:
        return ClientModel(
            backbone,
            Identity(backbone.out_dim),
            nn.Linear(backbone.out_dim, num_classes),
        )
    return ClientModel(
        backbone,
        adapter(backbone.out_dim, shared_dim),
        nn.Linear(shared_dim, num_classes),
    )
