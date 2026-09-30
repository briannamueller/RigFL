"""Optional run tracking; the default Tracker does nothing."""

from __future__ import annotations


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

    Install the ``wandb`` extra and enable it in the experiment configuration.
    """

    def __init__(self, project: str, config: dict, name: str | None = None):
        import wandb  # lazy -- only needed when tracking is enabled
        self.run = wandb.init(project=project, config=config, name=name)
        self.resources = None

    def update_resources(self, resources: dict) -> None:
        self.resources = resources

    def log_round(self, rnd: int, val: dict, test: dict) -> None:
        """Only validation metrics are logged to W&B."""
        from rigfl.eval.metrics import COMPUTED_METRICS
        from rigfl.eval.protocol import mean_over_clients
        payload = {"round": rnd}
        for m in COMPUTED_METRICS:
            v = mean_over_clients(val, m)
            if v is not None:
                payload[f"val/{m}"] = v
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
        """Record termination info and finish the W&B run.

        Round selection is post-hoc and not stored in the run.
        """
        summary = {}
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
            algorithm_flops = observed["flops"]["algorithm_operations"]
            if algorithm_flops is not None:
                summary["resources/observed_algorithm_flops"] = algorithm_flops
        self.run.summary.update(summary)
        self.run.finish()


def make_tracker(name: str, exp, cfg) -> Tracker:
    """Build the tracker implied by the experiment config (W&B if enabled, else no-op)."""
    if not getattr(exp, "wandb", False):
        return Tracker()
    run_name = f"{name}_{exp.dataset}_seed{exp.training_seed}"
    config = {"algorithm": name, "experiment": exp.model_dump(),
              "algorithm_config": cfg.model_dump()}
    return WandbTracker(project=exp.wandb_project, config=config, name=run_name)
