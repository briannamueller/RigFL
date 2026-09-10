"""The user-facing experiment runner."""

from pathlib import Path

import pytest

from rigfl.experiment import run as run_module
from tests.helpers import resolved_experiment


def _resolved(exp):
    values = exp.model_dump()
    values.update(
        partition_id="partition_12345678",
        model_family=None,
        model="fedavg_cnn",
        resolved_models=["fedavg_cnn"],
    )
    return resolved_experiment(
        **values,
    )


def test_cifar10_example_configuration_loads():
    config = Path(__file__).parents[1] / "experiments" / "cifar10_run.yaml"
    experiment, algorithm = run_module.load_run_config(str(config))

    assert experiment["dataset"] == "cifar10"
    assert experiment["out_dir"] == "results/cifar10_run"
    assert algorithm == {"local_epochs": 1, "lr": 0.01}


def test_run_experiment_loads_the_yaml_configuration(monkeypatch, tmp_path):
    captured = {}
    monkeypatch.setattr(
        run_module, "resolve_experiment_data", lambda exp: (_resolved(exp), None)
    )

    def fake_run(name, exp, cfg, *, data, force=False):
        captured.update(name=name, exp=exp, cfg=cfg, data=data, force=force)
        return Path(exp.out_dir) / "result.json"

    monkeypatch.setattr(run_module, "_run_resolved_experiment", fake_run)

    config = tmp_path / "experiment.yaml"
    config.write_text(
        "experiment:\n"
        "  dataset: cifar10\n"
        "  model: fedavg_cnn\n"
        f"  out_dir: {tmp_path}\n"
        "algorithm:\n"
        "  local_epochs: 2\n"
        "  lr: 0.02\n"
    )

    path = run_module.run_experiment("fedavg", config, force=True)

    assert path == tmp_path / "result.json"
    assert captured["name"] == "fedavg"
    assert captured["exp"].partition_id == "partition_12345678"
    assert captured["cfg"].local_epochs == 2
    assert captured["cfg"].lr == pytest.approx(0.02)
    assert captured["force"] is True


def test_run_experiment_rejects_settings_not_used_by_the_algorithm(monkeypatch,
                                                                tmp_path):
    monkeypatch.setattr(
        run_module, "resolve_experiment_data", lambda exp: (_resolved(exp), None)
    )

    config = tmp_path / "experiment.yaml"
    config.write_text(
        "experiment:\n"
        "  dataset: cifar10\n"
        "  model: fedavg_cnn\n"
        "algorithm:\n"
        "  mu: 0.1\n"
    )

    with pytest.raises(ValueError, match="unknown fedavg algorithm setting.*mu"):
        run_module.run_experiment("fedavg", config)


def test_run_experiment_accepts_nested_feddes_settings(monkeypatch, tmp_path):
    captured = {}
    monkeypatch.setattr(
        run_module, "resolve_experiment_data", lambda exp: (_resolved(exp), None)
    )

    def fake_run(name, exp, cfg, *, data, force=False):
        captured["cfg"] = cfg
        return tmp_path / "result.json"

    monkeypatch.setattr(run_module, "_run_resolved_experiment", fake_run)
    config = tmp_path / "feddes.yaml"
    config.write_text(
        "experiment:\n"
        "  dataset: cifar10\n"
        "  model: fedavg_cnn\n"
        "algorithm:\n"
        "  graphroute:\n"
        "    graph:\n"
        "      node_feature_source: feature_space\n"
        "      edge_feature_source: embedding_mean\n"
        "      k: 7\n"
    )

    run_module.run_experiment("feddes", config)

    graph = captured["cfg"].graphroute.graph
    assert graph.node_feature_source == "feature_space"
    assert graph.edge_feature_source == "embedding_mean"
    assert graph.k == 7
