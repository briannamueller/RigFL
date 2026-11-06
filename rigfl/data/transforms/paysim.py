"""PaySim features adapted from Flower Labs' FedFinFraud application."""

from __future__ import annotations

import numpy as np
from datasets import Features, Sequence, Value

SOURCE_COMMIT = "07b1b6fe3b1d50a252000f1c839497a87f13b144"
SOURCE_PATH = "fed-fin-fraud/fed_fraud/task.py"
SOURCE_URL = (
    "https://github.com/flwrlabs/flower-hub-benchmark/blob/"
    f"{SOURCE_COMMIT}/{SOURCE_PATH}"
)

# PaySim cancels detected fraud, so the balances after a transaction encode the label
BALANCE_COLUMNS = (
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
)
NUMERIC_COLUMNS = ("step", "amount", *BALANCE_COLUMNS)
TYPE_VALUES = ("CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER")
REQUIRED_COLUMNS = (*NUMERIC_COLUMNS, "type", "isFraud", "BankID")


def _float_array(values) -> np.ndarray:
    return np.fromiter(
        (0.0 if value is None else float(value) for value in values),
        dtype=np.float64,
        count=len(values),
    )


def _balance_features(numeric) -> np.ndarray:
    origin_delta = numeric["oldbalanceOrg"] - numeric["newbalanceOrig"]
    destination_delta = numeric["newbalanceDest"] - numeric["oldbalanceDest"]
    return np.column_stack(
        [
            origin_delta,
            destination_delta,
            origin_delta - numeric["amount"],
            numeric["amount"] - destination_delta,
        ]
    )


def prepare_paysim(dataset, training_split: str, include_balances: bool = True):
    numeric_columns = NUMERIC_COLUMNS if include_balances else ("step", "amount")
    training = dataset[training_split]
    means = {}
    stds = {}
    for column in numeric_columns:
        values = _float_array(training[column])
        means[column] = float(values.mean())
        std = float(values.std())
        stds[column] = std if std > 1e-12 else 1.0

    observed_types = {str(value).strip() for value in training.unique("type")}
    type_values = sorted(set(TYPE_VALUES).union(observed_types))
    type_to_index = {value: index for index, value in enumerate(type_values)}
    num_engineered = 4 if include_balances else 0
    input_dim = len(numeric_columns) + num_engineered + len(type_values)

    def encode(batch):
        numeric = {column: _float_array(batch[column]) for column in numeric_columns}
        standardized = np.column_stack(
            [
                (numeric[column] - means[column]) / stds[column]
                for column in numeric_columns
            ]
        )
        if include_balances:
            engineered = _balance_features(numeric)
        else:
            engineered = np.empty((len(standardized), 0))
        encoded_types = np.zeros(
            (len(batch["type"]), len(type_values)), dtype=np.float32
        )
        for row, value in enumerate(batch["type"]):
            index = type_to_index.get(str(value).strip())
            if index is not None:
                encoded_types[row, index] = 1.0
        features = np.concatenate(
            [standardized, engineered, encoded_types], axis=1
        ).astype(np.float32)
        return {"features": features.tolist()}

    removable = [
        column
        for column in next(iter(dataset.values())).column_names
        if column not in {"isFraud", "BankID"}
    ]
    output_features = Features(
        {
            "isFraud": Value("int64"),
            "BankID": Value("int64"),
            "features": Sequence(Value("float32"), length=input_dim),
        }
    )
    transformed = dataset.map(
        encode,
        batched=True,
        batch_size=10_000,
        remove_columns=removable,
        features=output_features,
    )
    parameters = {
        "means": means,
        "stds": stds,
        "type_values": type_values,
        "input_dim": input_dim,
    }
    return transformed, parameters


def prepare_paysim_without_balances(dataset, training_split: str):
    return prepare_paysim(dataset, training_split, include_balances=False)
