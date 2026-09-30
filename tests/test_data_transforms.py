"""Dataset transform registration and conversion."""

from dataclasses import replace

import numpy as np
import torch
from datasets import Dataset, DatasetDict

from rigfl.data.config import FlowerDatasetSettings
from rigfl.data.partitions import partition_fingerprint
from rigfl.data.transforms import DATA_TRANSFORMS, get_data_transform
from rigfl.data.transforms.phishing import MAX_LENGTH, encode_urls


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
    transformed, _ = get_data_transform("paysim_fraud").prepare(
        DatasetDict({"train": train, "test": Dataset.from_dict(test_row)}),
        "train",
    )

    features = np.asarray(transformed["test"]["features"])

    # test rows at the training mean normalize to zero
    np.testing.assert_allclose(features[0, :6], np.zeros(6))


def test_phishing_urls_are_normalized_and_byte_encoded():
    inputs = encode_urls(["A%20B", "https://example.com"])

    assert inputs.dtype == torch.int16
    assert inputs.shape == (2, MAX_LENGTH)
    assert inputs[0, :3].tolist() == [ord("a") + 2, ord(" ") + 2, ord("b") + 2]
    assert torch.count_nonzero(inputs[0, 3:]) == 0


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
