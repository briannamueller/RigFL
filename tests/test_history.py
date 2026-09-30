"""The canonical evaluation history and early stopping."""

from __future__ import annotations

import contextlib
import io

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from rigfl.algorithms.local import Local, LocalConfig
from rigfl.core import Client, ClientModel, LearnedProjection, iterative

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


# Evaluation-history alignment
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


def test_a_client_without_a_split_is_null_not_skipped():
    """A missing split does not change client positions in the history."""
    out, _ = _run(clients=[_client(16), _client(24, val=False)])
    h = out["evaluation_history"]
    assert set(h["clients"]) == {"0", "1"}
    assert all(v is None for v in h["clients"]["1"]["validation"]["accuracy"])
    assert all(v is not None for v in h["clients"]["1"]["test"]["accuracy"])
