"""The user-facing experiment runner."""

from pathlib import Path

import pytest

from rigfl.experiment import run as run_module


def _resolved(exp):
    return exp.model_copy(
        update={
            "scheme": "generated",
            "partition": "partition_12345678",
            "model_architecture_family": None,
            "model_architectures": ["fedavg_cnn"],
        }
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

    def fake_run(name, exp, cfg, *, force=False):
        captured.update(name=name, exp=exp, cfg=cfg, force=force)
        return Path(exp.out_dir) / "result.json"

    monkeypatch.setattr(run_module, "_run_resolved_experiment", fake_run)

    config = tmp_path / "experiment.yaml"
    config.write_text(
        "experiment:\n"
        "  dataset: cifar10\n"
        "  model_architectures: [fedavg_cnn]\n"
        f"  out_dir: {tmp_path}\n"
        "algorithm:\n"
        "  local_epochs: 2\n"
        "  lr: 0.02\n"
    )

    path = run_module.run_experiment("fedavg", config, force=True)

    assert path == tmp_path / "result.json"
    assert captured["name"] == "fedavg"
    assert captured["exp"].partition == "partition_12345678"
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
        "  model_architectures: [fedavg_cnn]\n"
        "algorithm:\n"
        "  mu: 0.1\n"
    )

    with pytest.raises(ValueError, match="unknown fedavg algorithm setting.*mu"):
        run_module.run_experiment("fedavg", config)
