"""The optimizer every algorithm uses for client-model training."""

from __future__ import annotations

import torch

from rigfl.core.config import ClientOptimizerConfig


def client_optimizer(params, config: ClientOptimizerConfig, lr: float) -> torch.optim.Optimizer:
    """The configured optimizer over ``params`` (parameters or parameter groups).

    Parameter groups may set their own ``lr``; ``lr`` is the default for the rest.
    """
    if config.optimizer == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=config.weight_decay)
    return torch.optim.SGD(params, lr=lr, momentum=config.momentum,
                           weight_decay=config.weight_decay)
