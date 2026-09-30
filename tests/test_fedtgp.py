"""FedTGP's paper-specific prototype calculations."""

import pytest
import torch

from rigfl.algorithms.fedtgp import FedTGP, FedTGPConfig


def test_adaptive_margin_is_the_largest_classwise_nearest_class_gap():
    algorithm = FedTGP(
        FedTGPConfig(margin_cap=100.0), num_classes=3, feature_dim=2
    )
    uploads = [
        {
            0: torch.tensor([0.0, 0.0]),
            1: torch.tensor([3.0, 0.0]),
            2: torch.tensor([0.0, 4.0]),
        }
    ]

    # Per-class nearest-other-class distances are 3, 3, and 4. FedTGP uses
    # their maximum (4), not the largest pairwise distance (5).
    assert algorithm._adaptive_margin(uploads, torch.device("cpu")) == pytest.approx(4.0)

    capped = FedTGP(FedTGPConfig(margin_cap=2.5), num_classes=3, feature_dim=2)
    assert capped._adaptive_margin(uploads, torch.device("cpu")) == 2.5


def test_server_epochs_are_minibatch_passes(monkeypatch):
    algorithm = FedTGP(
        FedTGPConfig(server_epochs=3),
        num_classes=3,
        feature_dim=2,
        batch_size=2,
        seed=7,
    )
    algorithm.device = torch.device("cpu")
    shared = algorithm.init_globals()
    uploads = [
        {0: torch.tensor([0.0, 0.0]), 1: torch.tensor([1.0, 0.0])},
        {
            0: torch.tensor([0.0, 1.0]),
            1: torch.tensor([1.0, 1.0]),
            2: torch.tensor([2.0, 1.0]),
        },
    ]
    steps = 0
    original_step = torch.optim.SGD.step

    def counted_step(optimizer, *args, **kwargs):
        nonlocal steps
        steps += 1
        return original_step(optimizer, *args, **kwargs)

    monkeypatch.setattr(torch.optim.SGD, "step", counted_step)

    algorithm.aggregate(uploads, shared)

    assert steps == 3 * 3  # epochs * ceil(5 uploaded prototypes / batch size 2)


def test_server_shuffle_has_its_own_reproducible_random_stream():
    uploads = [
        {0: torch.tensor([0.0, 0.0]), 1: torch.tensor([1.0, 0.0])},
        {
            0: torch.tensor([0.0, 1.0]),
            1: torch.tensor([1.0, 1.0]),
            2: torch.tensor([2.0, 1.0]),
        },
    ]

    def train(*, unrelated_draws: int):
        torch.manual_seed(11)
        algorithm = FedTGP(
            FedTGPConfig(server_epochs=3, server_lr=0.1),
            num_classes=3,
            feature_dim=2,
            batch_size=2,
            seed=19,
        )
        algorithm.device = torch.device("cpu")
        shared = algorithm.init_globals()
        torch.rand(unrelated_draws)
        return algorithm.aggregate(uploads, shared)["tgp"].state_dict()

    first = train(unrelated_draws=1)
    second = train(unrelated_draws=100)

    for name, value in first.items():
        assert torch.equal(value, second[name])
