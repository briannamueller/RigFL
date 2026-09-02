"""The canonical evaluation history and early stopping."""

from __future__ import annotations

import contextlib
import io

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from rigfl.core import Client, ClientModel, LearnedProjection, iterative
from rigfl.algorithms.local import Local, LocalConfig

NC, DIM = 3, 8
_CENTERS = torch.randn(NC, DIM, generator=torch.Generator().manual_seed(0)) * 2.5


def _loader(n=24):
    xs = [_CENTERS[c] + 2.0 * torch.randn(n, DIM) for c in range(NC)]
    ys = [torch.full((n,), c) for c in range(NC)]
    return DataLoader(TensorDataset(torch.cat(xs), torch.cat(ys)), batch_size=16, shuffle=True)


def _client(hidden=16, *, val=True):
    return Client(model=ClientModel(nn.Sequential(nn.Linear(DIM, hidden), nn.ReLU()),
                                    LearnedProjection(hidden, 6), nn.Linear(6, NC)),
                  train_loader=_loader(),
                  val_loader=_loader(12) if val else None,
                  test_loader=_loader(12))


def _run(**kw):
    torch.manual_seed(0)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        out = iterative(Local(LocalConfig(local_epochs=1, lr=0.05)),
                            kw.pop("clients", [_client(16), _client(24)]),
                            num_rounds=kw.pop("num_rounds", 6),
                            device=torch.device("cpu"), num_classes=NC,
                            eval_gap=kw.pop("eval_gap", 2), verbose=False, **kw)
    return out, buf.getvalue()


# 10 -- the alignment invariant
def test_every_metric_vector_aligns_with_evaluation_rounds():
    out, _ = _run()
    h = out["evaluation_history"]
    n = len(h["evaluation_rounds"])
    assert n > 1
    for cid, splits in h["clients"].items():
        for split, per_metric in splits.items():
            for name, series in per_metric.items():
                assert len(series) == n, f"{cid}/{split}/{name}: {len(series)} != {n}"
    for split, per_client in h["client_sample_counts"].items():
        for cid, series in per_client.items():
            assert len(series) == n


def test_evaluation_rounds_need_not_start_at_one_or_be_contiguous():
    out, _ = _run(eval_gap=3, num_rounds=7)
    rounds = out["evaluation_history"]["evaluation_rounds"]
    assert rounds[0] == 0                       # evaluation starts at round 0
    assert rounds != list(range(len(rounds)))   # and is not every round


def test_a_client_without_a_split_is_null_not_skipped():
    """A skipped client used to shift every later client's position."""
    out, _ = _run(clients=[_client(16), _client(24, val=False)])
    h = out["evaluation_history"]
    assert set(h["clients"]) == {"0", "1"}
    assert all(v is None for v in h["clients"]["1"]["validation"]["accuracy"])
    assert all(v is not None for v in h["clients"]["1"]["test"]["accuracy"])


def test_sample_counts_are_preserved_for_weighted_summaries():
    out, _ = _run()
    counts = out["evaluation_history"]["client_sample_counts"]["validation"]
    assert all(isinstance(v, int) and v > 0 for v in counts["0"])


# 12 -- early stopping is configured separately from result selection
def test_early_stopping_metric_is_independent_of_selection_metric():
    out, _ = _run(early_stopping={"enabled": True, "metric": "balanced_accuracy",
                                  "patience": 2, "min_delta": 0.0})
    es = out["early_stopping"]
    assert es["enabled"] and es["metric"] == "balanced_accuracy"
    assert es["direction"] == "maximize" and es["aggregation"] == "mean"
    assert es["termination_reason"] in ("early_stopping", "completed_all_rounds")
    assert es["best_round"] in out["evaluation_history"]["evaluation_rounds"]

    # the history is complete either way, and can be selected on a different metric
    from rigfl.eval.selection import select_global
    sel = select_global(out["evaluation_history"], "accuracy")
    assert sel["selection_metric"] == "accuracy" != es["metric"]


def test_early_stopping_off_by_default_records_why_it_stopped():
    out, _ = _run()
    es = out["early_stopping"]
    assert es["enabled"] is False
    assert es["termination_reason"] == "completed_all_rounds"
    # Disabled means no control metric governed the run, so none is claimed --
    # recording one would read as though it had.
    assert es["metric"] is None and es["direction"] is None
    assert es["best_round"] is None and es["best_value"] is None
    assert es["patience"] is None and es["min_delta"] is None


def test_early_stopping_can_end_a_run_early():
    """A metric that cannot improve must trigger the configured patience."""
    out, _ = _run(num_rounds=12, eval_gap=1,
                  early_stopping={"enabled": True, "metric": "accuracy",
                                  "patience": 1, "min_delta": 0.9})
    assert out["early_stopping"]["termination_reason"] == "early_stopping"
    assert len(out["evaluation_history"]["evaluation_rounds"]) < 12


def test_new_results_carry_a_schema_version():
    out, _ = _run()
    assert out["schema_version"] == 3
    assert out["selection_views_supported"] == ["global", "per-client"]


class _IterativeWithExtraOperation(Local):
    """An operation outside the iterative contract must not be invoked."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.events = []

    def prepare(self, clients, shared, client_state, device):
        raise AssertionError("iterative must not call one-shot preparation")

    def local_train(self, client, shared):
        self.events.append(f"train:{self.round_idx}")
        return super().local_train(client, shared)


def test_iterative_runner_invokes_only_its_own_lifecycle_operations():
    torch.manual_seed(0)
    algorithm = _IterativeWithExtraOperation(
        LocalConfig(local_epochs=1, lr=0.05))
    out = iterative(algorithm, [_client()], num_rounds=2,
                        device=torch.device("cpu"), num_classes=NC,
                        eval_gap=1, verbose=False)
    assert algorithm.events == ["train:0", "train:1"]
    history = out["evaluation_history"]
    assert history["evaluation_rounds"] == [0, 1]
    assert set(history) == {"evaluation_rounds", "clients", "client_sample_counts"}


def test_early_stopping_rejects_bad_config_passed_as_a_raw_dict():
    """A dict handed straight to iterative bypasses Pydantic entirely."""
    for cfg, match in (({"enabled": True, "metric": "nonsense"}, "Unknown metric"),
                       ({"enabled": True, "metric": "accuracy", "split": "test"},
                        "validation split")):
        with pytest.raises(ValueError, match=match):
            _run(early_stopping=cfg)
