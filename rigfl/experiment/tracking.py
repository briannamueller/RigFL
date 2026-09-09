"""Optional experiment tracking. The loop calls a Tracker; the default does nothing.

Tracking is off unless you ask for it (``--wandb``), and W&B is an optional
dependency -- so the repo runs, and the loop stays clean, with zero W&B setup.
``core.round`` only ever touches the tiny :class:`Tracker` interface below.
"""

from __future__ import annotations

import os


class Tracker:
    """No-op tracker (the default). Subclass to log somewhere."""

    def update_resources(self, resources: dict) -> None:
        pass

    def log_round(self, rnd: int, val: dict, test: dict) -> None:
        pass

    def finish(self, best: dict) -> None:
        pass


class WandbTracker(Tracker):
    """Log config and per-round metrics to Weights & Biases.

    ``pip install rigfl[wandb]``; run with ``--wandb`` (and ``--wandb-project``).
    """

    def __init__(self, project: str, config: dict, name: str | None = None):
        import wandb                          # lazy -- only needed when --wandb is used
        self.run = wandb.init(project=project, config=config, name=name)
        self.resources = None

    def update_resources(self, resources: dict) -> None:
        self.resources = resources

    def log_round(self, rnd: int, val: dict, test: dict) -> None:
        """Validation every round; test only in the final summary.

        Test metrics are computed each evaluation round because the protocol
        reports the test score of the best *validation* round, and that cannot
        be recovered afterwards. Streaming them to a dashboard is a different
        thing: it puts a per-round test curve in front of you while you are
        choosing hyperparameters, which is how validation-based selection gets
        quietly undone by hand. Set RIGFL_LOG_TEST_ROUNDS=1 to see them anyway.
        """
        from rigfl.eval.metrics import COMPUTED_METRICS
        from rigfl.eval.protocol import mean_over_clients
        payload = {"round": rnd}
        for m in COMPUTED_METRICS:
            v = mean_over_clients(val, m)
            if v is not None:
                payload[f"val/{m}"] = v
        if os.environ.get("RIGFL_LOG_TEST_ROUNDS") == "1":
            for m in COMPUTED_METRICS:
                v = mean_over_clients(test, m)
                if v is not None:
                    payload[f"test/{m}"] = v
        resources = getattr(self, "resources", None)
        if resources:
            checkpoint = resources.get("checkpoints", [])[-1:]
            if checkpoint:
                values = checkpoint[0]
                payload["resources/communication_bytes"] = values[
                    "communication_bytes"]
                payload["resources/algorithm_wall_seconds"] = values[
                    "algorithm_wall_seconds"]
                if values.get("algorithm_flops") is not None:
                    payload["resources/algorithm_flops"] = values[
                        "algorithm_flops"]
        self.run.log(payload, step=rnd)

    def finish(self, result: dict) -> None:
        """Finish the run after recording its termination information.

        Round selection is collection-time analysis and is therefore not stored
        in, or reported from, the raw training result.
        """
        summary = {"schema_version": result.get("schema_version")}
        es = result.get("early_stopping", {})
        summary |= {f"early_stopping_{k}": es[k]
                    for k in ("termination_reason", "best_round") if k in es}
        resources = getattr(self, "resources", None)
        if resources:
            observed = resources["observed"]
            summary["resources/communication_bytes"] = observed[
                "communication_bytes"]["total"]
            summary["resources/observed_runner_wall_seconds"] = observed[
                "wall_seconds"]["total"]
            summary["resources/observed_algorithm_wall_seconds"] = observed[
                "wall_seconds"]["algorithm_operations"]
            attributed = resources["attributed_training"]
            if attributed["wall_seconds"] is not None:
                summary["resources/attributed_training_wall_seconds"] = attributed[
                    "wall_seconds"]
            if attributed["flops"] is not None:
                summary["resources/attributed_training_flops"] = attributed["flops"]
        self.run.summary.update(summary)
        self.run.finish()


def make_tracker(name: str, exp, cfg) -> Tracker:
    """Build the tracker implied by the experiment config (W&B if enabled, else no-op)."""
    if not getattr(exp, "wandb", False):
        return Tracker()
    run_name = f"{name}_{exp.dataset}_seed{exp.seed}"
    config = {"algorithm": name, "experiment": exp.model_dump(),
              "algorithm_config": cfg.model_dump()}
    return WandbTracker(project=exp.wandb_project, config=config, name=run_name)
