"""Provenance metadata recorded with experiment results."""

from __future__ import annotations

import json
import subprocess
from types import SimpleNamespace

from rigfl.experiment import env as env_mod


def _git(*args, cwd):
    subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _repo(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    _git("init", cwd=tmp_path)
    _git("config", "user.email", "t@example.com", cwd=tmp_path)
    _git("config", "user.name", "t", cwd=tmp_path)
    (tmp_path / "a.txt").write_text("one\n")
    _git("add", "a.txt", cwd=tmp_path)
    _git("commit", "-m", "first", cwd=tmp_path)
    return tmp_path


def test_provenance_records_source_state_without_its_path(tmp_path, monkeypatch):
    repo = _repo(tmp_path / "checkout")
    package = repo / "example"
    package.mkdir()
    module_path = package / "__init__.py"
    module_path.write_text("")
    _git("add", "example/__init__.py", cwd=repo)
    _git("commit", "-m", "add package", cwd=repo)
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()
    monkeypatch.setattr(
        env_mod.importlib,
        "import_module",
        lambda _name: SimpleNamespace(__file__=str(module_path), __version__="1.2.3"),
    )

    info = env_mod._package("rigfl_example_not_installed")

    assert info == {"version": "1.2.3", "source_commit": head, "source_dirty": False}
    assert str(repo) not in json.dumps(info)


def test_an_installed_wheel_gets_no_commit_from_an_enclosing_repo(tmp_path):
    """Git must not attribute an untracked site-packages file to its parent repo."""
    repo = _repo(tmp_path / "repo")
    tracked = repo / "pkg"
    tracked.mkdir()
    (tracked / "__init__.py").write_text("")
    _git("add", "pkg/__init__.py", cwd=repo)
    _git("commit", "-m", "add pkg", cwd=repo)
    assert env_mod._repo_root(tracked) is not None

    installed = repo / ".venv" / "lib" / "site-packages" / "pkg"
    installed.mkdir(parents=True)
    (installed / "__init__.py").write_text("")
    assert env_mod._repo_root(installed) is None


def test_provenance_finds_the_source_of_a_distribution_with_another_import_name(
    tmp_path, monkeypatch
):
    repo = _repo(tmp_path / "checkout")
    (repo / "feddes_paper").mkdir()
    (repo / "feddes_paper" / "__init__.py").write_text("")
    (repo / ".gitignore").write_text("__pycache__/\n")
    _git("add", ".gitignore", "feddes_paper/__init__.py", cwd=repo)
    _git("commit", "-m", "add package", cwd=repo)
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()
    # an editable install: metadata in site-packages, code in the checkout
    info = tmp_path / "site" / "feddes_paper-0.1.dist-info"
    info.mkdir(parents=True)
    (info / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: feddes-paper\nVersion: 0.1\n"
    )
    (info / "top_level.txt").write_text("feddes_paper\n")
    monkeypatch.syspath_prepend(str(tmp_path / "site"))
    monkeypatch.syspath_prepend(str(repo))

    packages = env_mod.capture_env(("feddes-paper",))["packages"]

    assert packages["feddes-paper"] == {
        "version": "0.1", "source_commit": head, "source_dirty": False,
    }
