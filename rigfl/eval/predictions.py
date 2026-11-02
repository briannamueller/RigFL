"""Test predictions at the rounds a report can select.

Reports pick rounds after a run, so ``SelectedPredictions`` keeps each
client's test predictions at every round that some selection rule currently
picks (the client's own best, or the client-averaged best with uniform or
sample-count weighting) and drops the rest. Rounds are compared the same way
``rigfl.eval.selection`` compares them.

Samples are matched by position; a client whose test labels change order
between evaluations saves nothing.
"""

from __future__ import annotations

import torch

from rigfl.eval.metrics import COMPUTED_METRICS, direction_of
from rigfl.eval.selection import aggregate
from rigfl.prediction import Predictions

SELECTION_METRICS = tuple(m for m in COMPUTED_METRICS if direction_of(m) == "maximize")
AGGREGATIONS = ("uniform", "sample_count")


def sample_record(predictions: Predictions, labels: torch.Tensor) -> dict:
    """Per-sample predicted label and true-class probability for one split."""
    labels = labels.long().cpu()
    record = {"labels": labels.tolist(),
              "predicted": predictions.labels.long().cpu().tolist(),
              "true_class_probability": None}
    if predictions.probabilities is not None:
        probabilities = predictions.probabilities.cpu()
        record["true_class_probability"] = (
            probabilities.gather(1, labels.unsqueeze(1)).squeeze(1).tolist())
    return record


class SelectedPredictions:
    """Test predictions at every round some selection rule currently picks."""

    def __init__(self):
        self._client_best: dict[tuple[str, str], tuple[float, int]] = {}
        self._shared_best: dict[tuple[str, str], tuple[float, int]] = {}
        self._rounds: dict[str, dict[int, dict]] = {}
        self._labels: dict[str, list] = {}
        self._unordered: set[str] = set()

    def observe_client(self, cid, rnd: int, validation: dict | None,
                       test: dict | None) -> None:
        """One client's evaluation: its validation metrics and ``sample_record`` on test."""
        cid = str(cid)
        if self._observe_client(cid, rnd, validation, test):
            self._prune(cid)

    def observe_round(self, rnd: int, validation: dict, tests: dict) -> None:
        """One evaluation of every client.

        ``validation`` is an ``evaluate_split`` block and ``tests`` maps each
        client id to its ``sample_record`` on test.
        """
        changed = {cid for cid, metrics in validation["clients"].items()
                   if self._observe_client(cid, rnd, metrics, tests.get(cid))}
        counts = validation["sample_counts"]
        for how in AGGREGATIONS:
            for name in SELECTION_METRICS:
                values = [m.get(name) if m else None
                          for m in validation["clients"].values()]
                weights = list(counts.values()) if how == "sample_count" else None
                if how == "sample_count" and not any(weights):
                    continue
                value = aggregate(values, weights, how)
                best = self._shared_best.get((how, name))
                if value is None or (best is not None and value <= best[0]):
                    continue
                self._shared_best[(how, name)] = (value, rnd)
                for cid, record in tests.items():
                    if record is not None:
                        self._keep(str(cid), rnd, record)
                changed.update(str(cid) for cid in tests)
        for cid in changed:
            self._prune(cid)

    def to_dict(self) -> dict:
        return {"clients": {
            cid: {"labels": self._labels[cid],
                  "rounds": {str(rnd): record for rnd, record in sorted(rounds.items())}}
            for cid, rounds in sorted(self._rounds.items(), key=lambda kv: int(kv[0]))
            if cid not in self._unordered
        }}

    def _observe_client(self, cid: str, rnd: int, validation, test) -> bool:
        if not validation or test is None:
            return False
        improved = False
        for name in SELECTION_METRICS:
            value = validation.get(name)
            best = self._client_best.get((cid, name))
            if value is None or (best is not None and value <= best[0]):
                continue
            self._client_best[(cid, name)] = (value, rnd)
            improved = True
        if improved:
            self._keep(cid, rnd, test)
        return improved

    def _keep(self, cid: str, rnd: int, record: dict) -> None:
        if self._labels.setdefault(cid, record["labels"]) != record["labels"]:
            self._unordered.add(cid)
        if cid in self._unordered:
            return
        self._rounds.setdefault(cid, {})[rnd] = {
            "predicted": record["predicted"],
            "true_class_probability": record["true_class_probability"],
        }

    def _prune(self, cid: str) -> None:
        used = {rnd for (c, _), (_, rnd) in self._client_best.items() if c == cid}
        used |= {rnd for _, rnd in self._shared_best.values()}
        rounds = self._rounds.get(cid, {})
        for rnd in list(rounds):
            if rnd not in used:
                del rounds[rnd]


def predictions_at(record: dict, cid: str, rnd: int) -> dict | None:
    """Saved test predictions of one client at one round, with its labels."""
    saved = record.get("result", {}).get("test_predictions")
    client = (saved or {}).get("clients", {}).get(str(cid))
    if client is None:
        return None
    at = client["rounds"].get(str(rnd))
    return None if at is None else {"labels": client["labels"], **at}
