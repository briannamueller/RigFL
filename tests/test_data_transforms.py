"""Dataset transform registration and conversion."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import torch
from datasets import Dataset, DatasetDict, Features, Value

from rigfl.data import flower
from rigfl.data.config import FlowerDatasetSettings
from rigfl.data.partitions import partition_fingerprint
from rigfl.data.transforms import DATA_TRANSFORMS, get_data_transform


def _paysim_rows(offset: float):
    return {
        "step": [offset - 1, offset + 1],
        "type": ["PAYMENT", "TRANSFER"],
        "amount": [9.0, 11.0],
        "nameOrig": ["a", "b"],
        "oldbalanceOrg": [19.0, 21.0],
        "newbalanceOrig": [4.0, 6.0],
        "nameDest": ["c", "d"],
        "oldbalanceDest": [2.0, 4.0],
        "newbalanceDest": [6.0, 8.0],
        "isFraud": [0, 1],
        "isFlaggedFraud": [0, 0],
        "BankID": [0, 1],
    }


def test_paysim_uses_training_statistics_and_flower_feature_definition():
    train = Dataset.from_dict(_paysim_rows(2.0))
    test_row = {key: [values[0]] for key, values in _paysim_rows(2.0).items()}
    for column, value in {
        "step": 2.0,
        "amount": 10.0,
        "oldbalanceOrg": 20.0,
        "newbalanceOrig": 5.0,
        "oldbalanceDest": 3.0,
        "newbalanceDest": 7.0,
        "type": "CASH_OUT",
    }.items():
        test_row[column] = [value]
    transformed, parameters = get_data_transform("paysim_fraud").prepare(
        DatasetDict({"train": train, "test": Dataset.from_dict(test_row)}),
        "train",
    )

    features = np.asarray(transformed["test"]["features"])

    assert features.shape == (1, 15)
    assert parameters["input_dim"] == 15
    np.testing.assert_allclose(features[0, :6], np.zeros(6))
    np.testing.assert_allclose(features[0, 6:10], [15.0, 4.0, 5.0, 6.0])
    np.testing.assert_array_equal(features[0, 10:], [0, 1, 0, 0, 0])
    assert transformed["test"].column_names == ["isFraud", "BankID", "features"]
    feature_schema = transformed["test"].features["features"]
    assert feature_schema.feature.dtype == "float32"
    assert feature_schema.length == 15


def test_transform_version_contributes_to_partition_fingerprint(monkeypatch):
    settings = FlowerDatasetSettings(
        source_dataset="organization/data", data_transform="cifar10"
    )
    baseline = partition_fingerprint("images", settings)
    transform = DATA_TRANSFORMS["cifar10"]
    monkeypatch.setitem(
        DATA_TRANSFORMS, "cifar10", replace(transform, version=transform.version + 1)
    )

    assert baseline != partition_fingerprint("images", settings)


def test_paysim_transform_runs_through_partition_generation(monkeypatch, tmp_path):
    features = Features(
        {
            "step": Value("int64"),
            "type": Value("string"),
            "amount": Value("float64"),
            "nameOrig": Value("string"),
            "oldbalanceOrg": Value("float64"),
            "newbalanceOrig": Value("float64"),
            "nameDest": Value("string"),
            "oldbalanceDest": Value("float64"),
            "newbalanceDest": Value("float64"),
            "isFraud": Value("int64"),
            "isFlaggedFraud": Value("int64"),
            "BankID": Value("int64"),
        }
    )

    def rows(size):
        return {
            "step": list(range(size)),
            "type": ["PAYMENT", "TRANSFER"] * (size // 2),
            "amount": [float(index + 1) for index in range(size)],
            "nameOrig": [f"origin-{index}" for index in range(size)],
            "oldbalanceOrg": [float(index + 20) for index in range(size)],
            "newbalanceOrig": [float(index + 10) for index in range(size)],
            "nameDest": [f"destination-{index}" for index in range(size)],
            "oldbalanceDest": [float(index) for index in range(size)],
            "newbalanceDest": [float(index + 3) for index in range(size)],
            "isFraud": [index % 7 == 0 for index in range(size)],
            "isFlaggedFraud": [0] * size,
            "BankID": [index % 2 for index in range(size)],
        }

    source = DatasetDict(
        {
            "train": Dataset.from_dict(rows(40), features=features),
            "test": Dataset.from_dict(rows(20), features=features),
        }
    )
    builder = SimpleNamespace(
        config=SimpleNamespace(name="default"),
        info=SimpleNamespace(features=features, supervised_keys=None),
    )
    monkeypatch.setattr(flower, "load_dataset_builder", lambda *args, **kwargs: builder)
    monkeypatch.setattr(
        flower, "get_dataset_config_names", lambda *args, **kwargs: ["default"]
    )
    monkeypatch.setattr(
        flower, "get_dataset_split_names", lambda *args, **kwargs: ["train", "test"]
    )

    class FakeFederatedDataset:
        def __init__(self, *, preprocessor, partitioners, **_kwargs):
            dataset = preprocessor(source)
            self.partitioners = partitioners
            for split, partitioner in partitioners.items():
                partitioner.dataset = dataset[split]

        def load_partition(self, partition_id, split):
            return self.partitioners[split].load_partition(partition_id)

    monkeypatch.setattr("flwr_datasets.FederatedDataset", FakeFederatedDataset)
    settings = FlowerDatasetSettings(
        source_dataset="organization/paysim",
        data_transform="paysim_fraud",
        partition={
            "scheme": "natural_id",
            "partition_by": "BankID",
            "train_per_client": None,
            "validation_per_client": None,
            "test_per_client": None,
        },
    )

    manifest = flower.generate_flower_partition(settings, tmp_path)
    saved_inputs, _ = torch.load(
        tmp_path / "clients" / "client_0" / "train.pt", weights_only=True
    )

    transform = manifest["source"]["data_transform"]
    assert manifest["num_clients"] == 2
    assert manifest["input_spec"] == {"kind": "numeric", "shape": [15]}
    assert manifest["target_spec"]["class_names"] == ["not_fraud", "fraud"]
    assert transform["fitted_parameters"]["input_dim"] == 15
    assert transform["source"].endswith("fed-fin-fraud/fed_fraud/task.py")
    assert saved_inputs.shape == (16, 15)
