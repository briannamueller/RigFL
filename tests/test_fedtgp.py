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


def test_adaptive_margin_is_capped_by_tau():
    algorithm = FedTGP(
        FedTGPConfig(margin_cap=2.5), num_classes=2, feature_dim=2
    )
    uploads = [
        {0: torch.tensor([0.0, 0.0]), 1: torch.tensor([3.0, 4.0])}
    ]

    assert algorithm._adaptive_margin(uploads, torch.device("cpu")) == 2.5
