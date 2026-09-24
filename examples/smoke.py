"""Run the included algorithms on a small synthetic federation.

    python examples/smoke.py
"""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from rigfl.algorithms.fedgh import FedGH, FedGHConfig
from rigfl.algorithms.fedproto import FedProto, FedProtoConfig
from rigfl.algorithms.fedtgp import FedTGP, FedTGPConfig
from rigfl.algorithms.fml import FML, FMLConfig
from rigfl.algorithms.global_ensemble import GlobalEnsemble, GlobalEnsembleConfig
from rigfl.algorithms.local import Local, LocalConfig
from rigfl.core import Client, ClientModel, LearnedProjection, iterative
from rigfl.eval.report import format_table, summarize

NUM_CLASSES = 4
INPUT_DIM = 32
SHARED_DIM = 16
DEVICE = torch.device("cpu")
SEEDS = [0, 1, 2]

# The synthetic classes are balanced, so reports select on accuracy.
SELECTION_METRIC = "accuracy"
_SEL = {"view": "global", "aggregation": "mean", "tie_break": "earliest"}

# Fixed class centers (independent of the per-seed RNG) so the class structure is
# stable across seeds while the samples and model init vary.
CENTERS = torch.randn(NUM_CLASSES, INPUT_DIM, generator=torch.Generator().manual_seed(12345)) * 3.0


def make_loader(n_per_class: int, spread: float = 2.5, batch: int = 32) -> DataLoader:
    xs, ys = [], []
    for c in range(NUM_CLASSES):
        xs.append(CENTERS[c] + spread * torch.randn(n_per_class, INPUT_DIM))
        ys.append(torch.full((n_per_class,), c))
    x, y = torch.cat(xs), torch.cat(ys)
    return DataLoader(TensorDataset(x, y), batch_size=batch, shuffle=True)


def make_client(hidden: int) -> Client:
    model = ClientModel(
        nn.Sequential(nn.Linear(INPUT_DIM, hidden), nn.ReLU()),
        LearnedProjection(hidden, SHARED_DIM),
        nn.Linear(SHARED_DIM, NUM_CLASSES),
    )
    return Client(model=model, train_loader=make_loader(64),
                  val_loader=make_loader(16), test_loader=make_loader(16))


def build_clients() -> list[Client]:
    return [make_client(h) for h in (24, 48, 24)]   # heterogeneous backbones


def mentee() -> ClientModel:
    return ClientModel(nn.Sequential(nn.Linear(INPUT_DIM, 32), nn.ReLU()),
                       LearnedProjection(32, SHARED_DIM), nn.Linear(SHARED_DIM, NUM_CLASSES))


def run_all_seeds(make_algorithm) -> list[dict]:
    """One record per seed, shaped like the JSON the experiment layer writes.

    The whole evaluation history travels because selection happens afterwards;
    the loop marks no round as chosen.
    """
    records = []
    for seed in SEEDS:
        torch.manual_seed(seed)
        clients = build_clients()
        result = iterative(make_algorithm(), clients, num_rounds=12,
                               device=DEVICE, num_classes=NUM_CLASSES,
                               eval_gap=4, verbose=False)
        records.append({"config": {"experiment": {"seed": seed}}, "result": result})
    return records


ALGORITHMS = {
    "Local": lambda: Local(LocalConfig(local_epochs=2, lr=0.05)),
    "Global": lambda: GlobalEnsemble(
        GlobalEnsembleConfig(local_epochs=2, lr=0.05)),
    "FedProto": lambda: FedProto(
        FedProtoConfig(lamda=1.0, local_epochs=2, lr=0.05)),
    "FedGH": lambda: FedGH(
        FedGHConfig(local_epochs=2, lr=0.05, server_epochs=1, server_lr=0.05),
        SHARED_DIM, NUM_CLASSES),
    "FML": lambda: FML(FMLConfig(local_epochs=2, lr=0.05), mentee),
    "FedTGP": lambda: FedTGP(
        FedTGPConfig(local_epochs=2, lr=0.05, server_epochs=2,
                     server_lr=0.05, margin_cap=100.0),
        NUM_CLASSES, SHARED_DIM),
}


if __name__ == "__main__":
    all_results = {}
    failures = []
    for name, make in ALGORITHMS.items():
        try:
            all_results[name] = run_all_seeds(make)
            print(f"ran {name}")
        except Exception as e:
            failures.append(name)
            print(f"{name} FAILED: {type(e).__name__}: {e}")

    rows = {}
    for name, records in all_results.items():
        rows[name] = summarize(records, SELECTION_METRIC, **_SEL)

    print(f"\nselection: metric={SELECTION_METRIC}, split=validation, "
          f"view={_SEL['view']}, aggregation={_SEL['aggregation']}, "
          f"tie_break={_SEL['tie_break']}")
    print(format_table(rows, SELECTION_METRIC))
    if failures:
        raise SystemExit(f"smoke test failed: {', '.join(failures)}")
