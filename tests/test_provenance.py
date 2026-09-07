"""Provenance metadata recorded with experiment results."""

from __future__ import annotations

import subprocess

from rigfl.experiment import env as env_mod
from rigfl.experiment.config import result_filename, run_fingerprint
from rigfl.experiment.env import _git_dirty, capture_env
from rigfl.experiment.registry import config_class
from tests.helpers import resolved_experiment


def _git(*args, cwd):
    subprocess.run(["git", *args], cwd=str(cwd), check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _repo(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    _git("init", cwd=tmp_path)
    _git("config", "user.email", "t@example.com", cwd=tmp_path)
    _git("config", "user.name", "t", cwd=tmp_path)
    (tmp_path / "a.txt").write_text("one\n")
    _git("add", "a.txt", cwd=tmp_path)
    _git("commit", "-m", "first", cwd=tmp_path)
    return tmp_path


# ── git_dirty ────────────────────────────────────────────────────────────────

def test_a_clean_checkout_reports_false(tmp_path):
    assert _git_dirty(_repo(tmp_path)) is False


def test_an_untracked_file_makes_it_dirty(tmp_path):
    repo = _repo(tmp_path)
    (repo / "scratch.py").write_text("x = 1\n")
    assert _git_dirty(repo) is True


def test_a_modified_file_makes_it_dirty(tmp_path):
    repo = _repo(tmp_path)
    (repo / "a.txt").write_text("two\n")
    assert _git_dirty(repo) is True


def test_a_staged_file_makes_it_dirty(tmp_path):
    repo = _repo(tmp_path)
    (repo / "b.txt").write_text("new\n")
    _git("add", "b.txt", cwd=repo)
    assert _git_dirty(repo) is True


def test_outside_a_git_checkout_it_is_none(tmp_path):
    """Not the same fact as clean, so not the same value."""
    plain = tmp_path / "no_git"
    plain.mkdir()
    assert _git_dirty(plain) is None


def test_capture_env_records_it(monkeypatch):
    monkeypatch.setattr(env_mod, "_git_dirty", lambda cwd=None: False)
    env = capture_env()
    assert env["git_dirty"] is False
    assert "git_commit" in env


def test_the_dirty_warning_is_printed_once_per_process(monkeypatch, capsys):
    monkeypatch.setattr(env_mod, "_git_dirty", lambda cwd=None: True)
    monkeypatch.setattr(env_mod, "_warned_dirty", False)
    for _ in range(3):                       # a multi-algorithm run captures env per algorithm
        assert capture_env()["git_dirty"] is True
    assert capsys.readouterr().out.count("[rigfl][warn]") == 1


def test_a_clean_checkout_warns_about_nothing(monkeypatch, capsys):
    monkeypatch.setattr(env_mod, "_git_dirty", lambda cwd=None: False)
    monkeypatch.setattr(env_mod, "_warned_dirty", False)
    capture_env()
    assert "[rigfl][warn]" not in capsys.readouterr().out


# ── it decides nothing ───────────────────────────────────────────────────────

def test_git_state_is_not_part_of_run_identity():
    """A dirty tree must not rerun a configuration that is already done."""
    exp = resolved_experiment(rounds=2, num_clients=2)
    algorithm = config_class("local")().model_dump()
    fingerprint = run_fingerprint(exp, algorithm)
    name = result_filename(exp, "local", fingerprint)

    for dirty in (True, False, None):
        assert run_fingerprint(exp, algorithm) == fingerprint
        assert result_filename(exp, "local", fingerprint) == name

    # ...and the fields it could have leaked through carry no trace of it
    dumped = exp.model_dump()
    assert not any("git" in key or "dirty" in key for key in dumped)


def test_the_environment_block_is_not_in_the_fingerprint():
    from rigfl.experiment.artifacts import make_run_record

    exp = resolved_experiment(rounds=2, num_clients=2)
    cfg = config_class("local")()
    fingerprint = run_fingerprint(exp, cfg.model_dump())
    record = make_run_record(
        algorithm="local", experiment=exp.model_dump(), algorithm_config=cfg.model_dump(),
        run_fingerprint=fingerprint, result={}, env={"git_dirty": True})
    assert record["env"]["git_dirty"] is True
    assert record["run_fingerprint"] == fingerprint


def test_provenance_describes_rigfl_not_the_working_directory(tmp_path, monkeypatch):
    """Launching from inside another checkout must not borrow its commit.

    ``git`` answers relative to the process's working directory, so a RigFL run
    started from a sibling repository used to record that repository's commit and
    dirty state as its own.
    """
    other = tmp_path / "other_repo"
    other.mkdir()
    _git("init", cwd=other)
    _git("config", "user.email", "t@example.com", cwd=other)
    _git("config", "user.name", "t", cwd=other)
    (other / "a.txt").write_text("one\n")
    _git("add", "a.txt", cwd=other)
    _git("commit", "-m", "first", cwd=other)
    other_commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=str(other),
                                  capture_output=True, text=True).stdout.strip()

    monkeypatch.chdir(other)
    env = capture_env()
    from rigfl.experiment.env import _git_commit, _rigfl_path

    assert env["git_commit"] == _git_commit(_rigfl_path())
    assert env["git_commit"] != other_commit          # not the directory we ran from


def test_an_installed_wheel_gets_no_commit_from_an_enclosing_repo(tmp_path):
    """git walks upward; a package under site-packages must not inherit a commit."""
    from rigfl.experiment.env import _git_commit, _repo_root

    repo = _repo(tmp_path / "repo")
    tracked = repo / "pkg"
    tracked.mkdir()
    (tracked / "__init__.py").write_text("")
    _git("add", "pkg/__init__.py", cwd=repo)
    _git("commit", "-m", "add pkg", cwd=repo)
    assert _repo_root(str(tracked)) is not None       # its own checkout -> a commit
    assert _git_commit(str(tracked)) is not None

    # a wheel unpacked into a virtualenv inside that same repo: git finds the
    # enclosing checkout, but it tracks nothing here
    installed = repo / ".venv" / "lib" / "site-packages" / "pkg"
    installed.mkdir(parents=True)
    (installed / "__init__.py").write_text("")
    assert _repo_root(str(installed)) is None
    assert _git_commit(str(installed)) is None


def test_package_versions_come_from_installed_metadata():
    """Not from a ``__version__`` attribute a package may not define."""
    from rigfl.experiment.env import _package

    for name in ("rigfl", "graphroute"):
        info = _package(name)
        if info is None:                              # not installed here
            continue
        assert info["version"] == "0.1.0"
