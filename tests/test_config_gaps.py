"""Validation of configuration files, overrides, and launch settings."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from rigfl.experiment.launch import expand
from rigfl.experiment.run import _read_run_config, _split_run_config, build_configs


def _args(**over):
    base = dict(config=None, set=[])
    base.update(over)
    return SimpleNamespace(**base)


def _yaml(tmp_path, text):
    path = tmp_path / "config.yaml"
    path.write_text(text)
    return str(path)


def test_a_misspelt_setting_is_refused(tmp_path):
    with pytest.raises(SystemExit, match="btach"):
        path = _yaml(tmp_path, "algorithm: local\nbtach: 64\n")
        _split_run_config(_read_run_config(path), source=path)


def test_set_overrides_yaml_for_one_run(tmp_path):
    config = _yaml(tmp_path, "algorithm: local\nrounds: 100\nbatch: 64\n")
    name, exp, algorithm = build_configs(
        _args(config=config, set=["rounds=50", "lr=0.1"])
    )

    assert name == "local"
    assert exp.rounds == 50 and exp.batch == 64
    assert algorithm == {"lr": 0.1}


def test_a_malformed_fixed_setting_fails_at_launch_not_on_a_worker():
    with pytest.raises(SystemExit, match="does not validate"):
        expand({
            "base": {"lr": -5},
            "sweep": {"algorithm": ["local"], "training_seed": [0]},
        })
