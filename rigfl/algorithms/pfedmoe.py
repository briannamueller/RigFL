"""pFedMoE -- data-level personalization with mixture of experts (TKDE 2026).

Every client mixes representations from its private heterogeneous model and a
shared homogeneous proxy extractor using a private sample-dependent gate.  Only
the proxy extractor is communicated.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as F
from pydantic import Field
from torch import nn

from rigfl.algorithms.fedavg import (
    ModelUpload,
    clone_state_dict,
    weighted_average_states,
)
from rigfl.core.config import AlgorithmConfig
from rigfl.core.interfaces import Algorithm
from rigfl.prediction import Predictions


class PFedMoEConfig(AlgorithmConfig):
    local_epochs: int = Field(1, ge=1, description="Client training epochs per round.")
    lr: float = Field(0.01, gt=0, description="Private-model learning rate.")
    proxy_lr: float = Field(0.01, gt=0, description="Proxy-extractor learning rate.")
    gate_lr: float = Field(0.01, gt=0, description="Private gating-network learning rate.")
    gate_hidden_dim: int = Field(
        32, ge=1, description="Width of the private two-layer gating network."
    )
    proxy_model: str | None = Field(
        None,
        description="Architecture used for the shared homogeneous proxy extractor.",
    )


class ProxyExtractor(nn.Module):
    """A registered backbone plus alignment adapter, deliberately without a head."""

    def __init__(self, backbone: nn.Module, adapter: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.adapter = adapter

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.adapter(self.backbone(x))


class GatingNetwork(nn.Module):
    """Lightweight sample-dependent two-expert gate."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.input = nn.LazyLinear(hidden_dim)
        self.normalization = nn.LayerNorm(hidden_dim)
        self.output = nn.Linear(hidden_dim, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        flattened = x.flatten(1).float()
        hidden = torch.sigmoid(self.normalization(self.input(flattened)))
        return torch.softmax(self.output(hidden), dim=1)


def mixed_representation(
    proxy: ProxyExtractor, private_model, gate: GatingNetwork, x: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """pFedMoE Eqs. 2-4 and the sample-specific expert weights."""
    weights = gate(x)
    proxy_representation = proxy(x)
    private_representation = private_model.rep(x)
    if proxy_representation.shape != private_representation.shape:
        raise ValueError(
            "pFedMoE requires proxy and private representations to have the same "
            f"shape, got {tuple(proxy_representation.shape)} and "
            f"{tuple(private_representation.shape)}."
        )
    mixed = (
        weights[:, :1] * proxy_representation
        + weights[:, 1:] * private_representation
    )
    return mixed, weights


class PFedMoE(Algorithm):
    """A private local expert and gate around one shared proxy extractor."""

    algorithm_name = "pFedMoE"

    def __init__(
        self,
        config: PFedMoEConfig,
        proxy_extractor_factory: Callable[[], ProxyExtractor],
    ):
        super().__init__(config)
        if proxy_extractor_factory is None:
            raise ValueError("pFedMoE requires a proxy extractor factory.")
        self.proxy_extractor_factory = proxy_extractor_factory

    @classmethod
    def from_config(
        cls, config, *, proxy_extractor_factory=None, **resources
    ):
        return cls(config, proxy_extractor_factory)

    def init_globals(self) -> ProxyExtractor:
        return self.proxy_extractor_factory()

    def local_train(self, client, shared: ProxyExtractor) -> ModelUpload:
        private_model = client.model.to(self.device)
        proxy = client.state.get("pfedmoe_proxy")
        if proxy is None:
            proxy = self.proxy_extractor_factory()
            client.state["pfedmoe_proxy"] = proxy
        proxy.to(self.device)
        proxy.load_state_dict(shared.state_dict())

        gate = client.state.get("pfedmoe_gate")
        if gate is None:
            gate = GatingNetwork(self.config.gate_hidden_dim)
            client.state["pfedmoe_gate"] = gate
        gate.to(self.device)

        private_model.train()
        proxy.train()
        gate.train()
        optimizer = torch.optim.SGD(
            [
                {"params": private_model.parameters(), "lr": self.config.lr},
                {"params": proxy.parameters(), "lr": self.config.proxy_lr},
                {"params": gate.parameters(), "lr": self.config.gate_lr},
            ]
        )
        for _ in range(self.config.local_epochs):
            for x, y in client.train_loader:
                x, y = x.to(self.device), y.to(self.device)
                representation, _ = mixed_representation(
                    proxy, private_model, gate, x
                )
                loss = F.cross_entropy(private_model.head(representation), y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        return ModelUpload(
            clone_state_dict(proxy.state_dict()), len(client.train_loader.dataset)
        )

    def aggregate(
        self, uploads: list[ModelUpload], shared: ProxyExtractor
    ) -> ProxyExtractor:
        averaged = weighted_average_states(
            uploads, device=self.device, algorithm=self.algorithm_name
        )
        shared.to(self.device)
        shared.load_state_dict(averaged)
        return shared

    @torch.no_grad()
    def predict(self, client, x, shared: ProxyExtractor) -> Predictions:
        gate = client.state.get("pfedmoe_gate")
        if gate is None:
            raise RuntimeError("pFedMoE cannot predict before the client has trained.")
        shared.to(x.device)
        client.model.to(x.device)
        gate.to(x.device)
        representation, _ = mixed_representation(shared, client.model, gate, x)
        return Predictions.from_logits(client.model.head(representation))
