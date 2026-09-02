"""Every core algorithm fulfils the four-function Algorithm contract."""

from __future__ import annotations

import inspect

import pytest

from rigfl.algorithms.fedgh import FedGH
from rigfl.algorithms.fedavg import FedAvg
from rigfl.algorithms.feddes import FedDES
from rigfl.algorithms.fedprox import FedProx
from rigfl.algorithms.fedkd import FedKD
from rigfl.algorithms.fedproto import FedProto
from rigfl.algorithms.fedtgp import FedTGP
from rigfl.algorithms.fml import FML
from rigfl.algorithms.global_ensemble import GlobalEnsemble
from rigfl.algorithms.lgfedavg import LGFedAvg
from rigfl.algorithms.local import Local
from rigfl.core.config import AlgorithmConfig
from rigfl.core.interfaces import Algorithm

BASELINES = [Local, GlobalEnsemble, FedAvg, FedProx, FedProto, FedGH,
             LGFedAvg, FML, FedKD, FedTGP]
CONTRACT = ("init_globals", "local_train", "aggregate", "predict")


def test_default_construction_hook_stores_the_validated_configuration():
    class ExampleConfig(AlgorithmConfig):
        lr: float = 0.1

    class ExampleAlgorithm(Algorithm):
        pass

    config = ExampleConfig(lr=0.2)
    algorithm = ExampleAlgorithm.from_config(config, unused_resource=object())

    assert algorithm.config is config


def test_one_registry_entry_is_enough_to_construct_an_ordinary_algorithm(
    monkeypatch,
):
    from rigfl.experiment.config import ExperimentConfig
    from rigfl.experiment.registry import (AlgorithmSpec, REGISTRY,
                                           build_algorithm)

    class ExampleConfig(AlgorithmConfig):
        lr: float = 0.1

    class ExampleAlgorithm(Algorithm):
        pass

    monkeypatch.setitem(
        REGISTRY, "example", AlgorithmSpec(ExampleAlgorithm, ExampleConfig))
    config = ExampleConfig(lr=0.2)

    algorithm = build_algorithm("example", ExperimentConfig(), config)

    assert isinstance(algorithm, ExampleAlgorithm)
    assert algorithm.config is config


@pytest.mark.parametrize("cls", BASELINES, ids=[c.__name__ for c in BASELINES])
def test_algorithm_exposes_contract(cls):
    for name in CONTRACT:
        attr = getattr(cls, name, None)
        assert attr is not None, f"{cls.__name__} is missing {name}"
        assert callable(attr), f"{cls.__name__}.{name} is not callable"


@pytest.mark.parametrize("cls", BASELINES, ids=[c.__name__ for c in BASELINES])
def test_iterative_operations_share_the_standard_signatures(cls):
    local = list(inspect.signature(cls.local_train).parameters)
    server = list(inspect.signature(cls.aggregate).parameters)
    assert len(local) == 3 and local[:2] == ["self", "client"]
    assert len(server) == 3 and server[:2] == ["self", "uploads"]


@pytest.mark.parametrize("cls", BASELINES + [FedDES],
                         ids=[c.__name__ for c in BASELINES + [FedDES]])
def test_predict_receives_client_inputs_and_shared_state(cls):
    params = list(inspect.signature(cls.predict).parameters)
    assert len(params) == 4 and params[:3] == ["self", "client", "x"]
