"""The user-facing experiment runner."""

from pathlib import Path

import pytest

from rigfl.experiment import run as run_module


def test_cifar10_example_configuration_loads():
    config = (
        Path(__file__).parents[1]
        / "configs"
        / "experiments"
        / "cifar10_run.yaml"
    )
    experiment, algorithm = run_module.load_run_config(str(config))

    assert experiment["dataset"] == "cifar10"
    assert experiment["out_dir"] == "results"
    assert algorithm == {"local_epochs": 1, "lr": 0.01}


def test_run_experiment_loads_the_yaml_configuration(monkeypatch, tmp_path):
    captured = {}
    from rigfl.experiment import launch as launch_module

    def fake_run(task, out_dir, **options):
        captured.update(task=task, out_dir=out_dir, options=options)
        return {"_source_file": "result.json"}

    monkeypatch.setattr(launch_module, "run_config", fake_run)

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

    assert path == tmp_path / "runs/result.json"
    assert captured["task"]["algorithm"] == "fedavg"
    assert captured["task"]["experiment"]["dataset"] == "cifar10"
    assert captured["task"]["algorithm_config"]["local_epochs"] == 2
    assert captured["task"]["algorithm_config"]["lr"] == pytest.approx(0.02)
    assert captured["out_dir"] == tmp_path / "runs"
    assert captured["options"] == {"force": True}


def test_run_experiment_rejects_settings_not_used_by_the_algorithm(monkeypatch,
                                                                tmp_path):
    config = tmp_path / "experiment.yaml"
    config.write_text(
        "experiment:\n"
        "  dataset: cifar10\n"
        "  model: fedavg_cnn\n"
        "algorithm:\n"
        "  mu: 0.1\n"
    )

    with pytest.raises(SystemExit, match="unknown fedavg algorithm setting.*mu"):
        run_module.run_experiment("fedavg", config)
