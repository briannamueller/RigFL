"""Validation of configuration files, overrides, and launch settings."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from rigfl.experiment.launch import build_grid
from rigfl.experiment.run import build_configs, load_run_config


def _args(**over):
    base = dict(config=None, set=[])
    base.update(over)
    return SimpleNamespace(**base)


def _yaml(tmp_path, text):
    path = tmp_path / "config.yaml"
    path.write_text(text)
    return str(path)


# ── the single-run YAML ──────────────────────────────────────────────────────

def test_a_yaml_that_is_not_a_mapping_is_refused(tmp_path):
    with pytest.raises(SystemExit, match="must be a mapping"):
        load_run_config(_yaml(tmp_path, "- batch\n- 64\n"))


def test_a_misspelt_section_is_refused(tmp_path):
    with pytest.raises(SystemExit) as e:
        load_run_config(_yaml(tmp_path, "experimnt:\n  batch: 64\n"))
    assert "experimnt" in str(e.value)
    assert 'Did you mean "experiment"?' in str(e.value)


def test_a_sweep_file_handed_to_the_single_run_path_is_refused(tmp_path):
    """Sweep sections are accepted only by the sweep launcher."""
    with pytest.raises(SystemExit, match="unknown top-level section"):
        load_run_config(_yaml(tmp_path, "base:\n  experiment:\n    batch: 64\n"
                                        "sweep:\n  experiment.seed: [0, 1]\n"))


def test_a_section_that_is_not_a_mapping_is_refused(tmp_path):
    with pytest.raises(SystemExit, match="'experiment' must be a mapping"):
        load_run_config(_yaml(tmp_path, "experiment: 5\n"))


def test_a_valid_config_still_loads(tmp_path):
    exp, algorithm = load_run_config(
        _yaml(tmp_path, "experiment:\n  batch: 64\nalgorithm:\n  lr: 0.1\n"))
    assert exp == {"batch": 64} and algorithm == {"lr": 0.1}


# ── --set ────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("bad, match", [
    ("experimnt.batch=64", 'Did you mean "experiment"?'),
    ("methd.lr=0.1", 'Use algorithm'),
    ("batch=64", "expected <section>.<field>=<value>"),
])
def test_an_unknown_set_prefix_is_refused(bad, match):
    """Command-line overrides require a recognized configuration section."""
    with pytest.raises(SystemExit, match=match):
        build_configs(_args(set=[bad]))


def test_the_documented_set_prefixes_still_work():
    exp, algorithm = build_configs(
        _args(set=["experiment.batch=64", "algorithm.lr=0.1"])
    )
    assert exp.batch == 64 and algorithm == {"lr": "0.1"}


def test_set_overrides_yaml_for_one_run(tmp_path):
    config = _yaml(tmp_path, "experiment:\n  rounds: 100\n")
    exp, _ = build_configs(
        _args(config=config, set=["experiment.rounds=50"])
    )

    assert exp.rounds == 50


# ── the sweep file ───────────────────────────────────────────────────────────

def test_a_misspelt_sweep_key_is_refused():
    with pytest.raises(SystemExit) as e:
        build_grid({"algorithms": ["local"], "swep": {"seed": [0, 1]}})
    assert "swep" in str(e.value)
    assert 'Did you mean "sweep"?' in str(e.value)


def test_a_sweep_section_that_is_not_a_mapping_is_refused():
    with pytest.raises(SystemExit, match="'sweep' must be a mapping"):
        build_grid({"algorithms": ["local"], "sweep": ["seed"]})


def test_sweep_paths_require_a_configuration_section():
    with pytest.raises(SystemExit, match="must start with 'experiment.' or 'algorithm.'"):
        build_grid({"algorithms": ["local"], "sweep": {"seed": [0, 1]}})


@pytest.mark.parametrize("base", [
    {"experiment": {"alpha": -1}},
    {"algorithm": {"lr": -5}},
])
def test_a_malformed_fixed_setting_fails_at_launch_not_on_a_worker(base):
    with pytest.raises(SystemExit, match="does not validate"):
        build_grid({
            "algorithms": ["local"],
            "base": base,
            "sweep": {"experiment.seed": [0]},
        })


def test_algorithm_specific_scoping_is_unchanged():
    grid = build_grid({"algorithms": ["feddes", "local"],
                       "sweep": {"experiment.seed": [0, 1],
                                 "algorithm.graphroute.graph.k": [3, 5]}})
    counts = {m: sum(t["algorithm"] == m for t in grid) for m in ("feddes", "local")}
    assert counts == {"feddes": 4, "local": 2}
