"""Configuration mistakes that used to be absorbed instead of reported.

Each of these ran a different experiment than the file or command line described,
without saying so: a misspelt YAML section, a misspelt ``--set`` prefix, a
misspelt sweep key, a fixed setting that only a compute node would reject, and a
``--quiet`` flag that was not passed overwriting ``quiet: true`` from the config.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from rigfl.experiment.launch import build_grid
from rigfl.experiment.run import build_configs, load_run_config


def _args(**over):
    base = dict(config=None, set=[], quiet=False, wandb=False, wandb_project=None,
                rounds=None, seed=None, shared_dim=None, eval_gap=None, device=None,
                out_dir=None, lr=None, local_epochs=None)
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
    """base/sweep belong to rigfl.experiment.launch, and used to be ignored here."""
    with pytest.raises(SystemExit, match="unknown top-level section"):
        load_run_config(_yaml(tmp_path, "base:\n  experiment:\n    batch: 64\n"
                                        "sweep:\n  seed: [0, 1]\n"))


def test_a_section_that_is_not_a_mapping_is_refused(tmp_path):
    with pytest.raises(SystemExit, match="'experiment' must be a mapping"):
        load_run_config(_yaml(tmp_path, "experiment: 5\n"))


def test_a_valid_config_still_loads(tmp_path):
    exp, algorithm = load_run_config(
        _yaml(tmp_path, "experiment:\n  batch: 64\nalgorithm:\n  lr: 0.1\n"))
    assert exp == {"batch": 64} and algorithm == {"lr": 0.1}


# ── --set ────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("bad, match", [
    ("expp.batch=64", 'Did you mean "exp"?'),
    ("methd.lr=0.1", 'Use algorithm'),
    ("batch=64", "expected <section>.<field>=<value>"),
])
def test_an_unknown_set_prefix_is_refused(bad, match):
    """Every non-``exp`` prefix used to be routed into the algorithm config."""
    with pytest.raises(SystemExit, match=match):
        build_configs(_args(set=[bad]))


def test_the_documented_set_prefixes_still_work():
    exp, algorithm = build_configs(_args(set=["exp.batch=64", "algorithm.lr=0.1"]))
    assert exp.batch == 64 and algorithm == {"lr": "0.1"}


# ── --quiet ──────────────────────────────────────────────────────────────────

def test_an_omitted_quiet_flag_does_not_overwrite_the_config(tmp_path):
    config = _yaml(tmp_path, "experiment:\n  quiet: true\n")
    assert build_configs(_args(config=config))[0].quiet is True
    assert build_configs(_args(config=config, quiet=True))[0].quiet is True
    assert build_configs(_args())[0].quiet is False        # run.py's own default


# ── the sweep file ───────────────────────────────────────────────────────────

def test_a_misspelt_sweep_key_is_refused():
    with pytest.raises(SystemExit) as e:
        build_grid({"algorithms": ["local"], "swep": {"seed": [0, 1]}})
    assert "swep" in str(e.value)
    assert 'Did you mean "sweep"?' in str(e.value)


def test_a_sweep_section_that_is_not_a_mapping_is_refused():
    with pytest.raises(SystemExit, match="'sweep' must be a mapping"):
        build_grid({"algorithms": ["local"], "sweep": ["seed"]})


@pytest.mark.parametrize("base", [
    {"experiment": {"alpha": -1}},
    {"algorithm": {"lr": -5}},
])
def test_a_malformed_fixed_setting_fails_at_launch_not_on_a_worker(base):
    with pytest.raises(SystemExit, match="does not validate"):
        build_grid({"algorithms": ["local"], "base": base, "sweep": {"seed": [0]}})


def test_algorithm_specific_scoping_is_unchanged():
    grid = build_grid({"algorithms": ["feddes", "local"],
                       "sweep": {"seed": [0, 1], "algorithm.graph_k": [3, 5]}})
    counts = {m: sum(t["algorithm"] == m for t in grid) for m in ("feddes", "local")}
    assert counts == {"feddes": 4, "local": 2}
