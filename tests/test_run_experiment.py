"""The user-facing experiment runner."""


import pytest

from rigfl.experiment import run as run_module
from tests.helpers import STARTER_CONFIGS


def test_mnist_example_configuration_loads():
    config = (
        STARTER_CONFIGS / "experiments" / "mnist_run.yaml"
    )
    name, experiment, algorithm = run_module._split_run_config(
        run_module._read_run_config(config), source=str(config)
    )

    assert name == "fedavg"
    assert experiment["dataset"] == "mnist"
    assert experiment["results_root"] == "results"
    assert algorithm == {"local_epochs": 1, "lr": 0.05}


def test_run_experiment_loads_the_yaml_configuration(monkeypatch, tmp_path):
    captured = {}
    from rigfl.experiment import launch as launch_module

    def fake_run(task, out_dir, **options):
        captured.update(task=task, out_dir=out_dir, options=options)
        return {"_source_file": "result.json"}

    monkeypatch.setattr(launch_module, "run_config", fake_run)

    config = tmp_path / "experiment.yaml"
    config.write_text(
        "algorithm: fedavg\n"
        "dataset: cifar10\n"
        "model_arch: fedavg_cnn\n"
        f"results_root: {tmp_path}\n"
        "local_epochs: 2\n"
        "lr: 0.02\n"
    )

    path = run_module.run_experiment(config, force=True)

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
        "algorithm: fedavg\n"
        "dataset: cifar10\n"
        "model_arch: fedavg_cnn\n"
        "mu: 0.1\n"
    )

    with pytest.raises(SystemExit, match="unknown configuration setting 'mu'"):
        run_module.run_experiment(config)
