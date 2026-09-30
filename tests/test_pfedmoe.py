"""pFedMoE proxy-extractor and sample-gating behavior."""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from rigfl.algorithms.pfedmoe import (
    PFedMoE,
    PFedMoEConfig,
    ProxyExtractor,
    mixed_representation,
)
from rigfl.core import Client, ClientModel, Identity


class _Backbone(nn.Module):
    out_dim = 2

    def __init__(self, value: float = 1.0):
        super().__init__()
        self.weight = nn.Parameter(torch.eye(2) * value)

    def forward(self, x):
        return x @ self.weight.T


def _model(value: float = 1.0) -> ClientModel:
    return ClientModel(_Backbone(value), Identity(2), nn.Linear(2, 2))


def _proxy() -> ProxyExtractor:
    return ProxyExtractor(_Backbone(0.5), Identity(2))


def _loader() -> DataLoader:
    x = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 1.0]])
    y = torch.tensor([0, 1, 0, 1])
    return DataLoader(TensorDataset(x, y), batch_size=2, shuffle=False)


def test_mixture_combines_proxy_and_private_representations():
    proxy = _proxy()
    private = _model(1.0)

    class FixedGate(nn.Module):
        def forward(self, x):
            return torch.tensor([[0.25, 0.75]], dtype=x.dtype).repeat(len(x), 1)

    mixed, weights = mixed_representation(
        proxy, private, FixedGate(), torch.tensor([[2.0, 4.0]])
    )

    assert torch.allclose(weights, torch.tensor([[0.25, 0.75]]))
    assert torch.allclose(mixed, torch.tensor([[1.75, 3.5]]))


def test_local_training_uploads_only_proxy_extractor_state():
    algorithm = PFedMoE(PFedMoEConfig(local_epochs=1), _proxy)
    algorithm.device = torch.device("cpu")
    algorithm.total_rounds = 1
    algorithm.round_idx = 0
    client = Client(_model(), _loader(), client_id=0)

    upload = algorithm.local_train(client, algorithm.init_globals())

    assert set(upload.state) == set(_proxy().state_dict())
