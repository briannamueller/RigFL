"""Provenance metadata recorded with experiment results."""

from __future__ import annotations

import json
import subprocess
from importlib import metadata
from pathlib import Path
from types import SimpleNamespace

from rigfl.experiment import env as env_mod
from rigfl.experiment.config import result_filename, run_fingerprint
from rigfl.experiment.env import _git_dirty, capture_env
from rigfl.experiment.registry import config_class
from tests.helpers import resolved_experiment


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


def _distribution(root: Path, name: str, version: str, files: list[str]):
    marker = root / f"{name}-{version}.dist-info"
    marker.mkdir(parents=True)
    (marker / "METADATA").write_text(
        f"Metadata-Version: 2.4\nName: {name}\nVersion: {version}\n"
    )
    (marker / "WHEEL").write_text("Wheel-Version: 1.0\n")
    (marker / "RECORD").write_text(
        "".join(f"{path},,\n" for path in [*files, f"{marker.name}/METADATA"])
    )
    return metadata.PathDistribution(marker)


# ── Checkout state ──────────────────────────────────────────────────────────


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
    plain = tmp_path / "no_git"
    plain.mkdir()
    assert _git_dirty(plain) is None


# ── Installation mode ───────────────────────────────────────────────────────


def test_wheel_provenance_uses_matching_installed_metadata(tmp_path, monkeypatch):
    site = tmp_path / "site-packages"
    package = site / "example"
    package.mkdir(parents=True)
    module_path = package / "__init__.py"
    module_path.write_text("")
    distribution = _distribution(site, "example", "1.2.3", ["example/__init__.py"])
    monkeypatch.setattr(
        env_mod.importlib,
        "import_module",
        lambda _name: SimpleNamespace(__file__=str(module_path)),
    )
    monkeypatch.setattr(env_mod, "_distribution_candidates", lambda _name: [distribution])

    assert env_mod._package("example") == {
        "version": "1.2.3",
        "installation_mode": "wheel",
    }


def test_editable_provenance_records_source_state_without_its_path(
    tmp_path, monkeypatch
):
    repo = _repo(tmp_path / "checkout")
    package = repo / "example"
    package.mkdir()
    module_path = package / "__init__.py"
    module_path.write_text("")
    _git("add", "example/__init__.py", cwd=repo)
    _git("commit", "-m", "add package", cwd=repo)

    site = tmp_path / "site-packages"
    distribution = _distribution(site, "example", "1.2.3", [])
    (distribution._path / "direct_url.json").write_text(
        json.dumps({"url": repo.as_uri(), "dir_info": {"editable": True}})
    )
    monkeypatch.setattr(
        env_mod.importlib,
        "import_module",
        lambda _name: SimpleNamespace(__file__=str(module_path)),
    )
    monkeypatch.setattr(env_mod, "_distribution_candidates", lambda _name: [distribution])

    info = env_mod._package("example")

    assert info == {
        "version": "1.2.3",
        "installation_mode": "editable",
        "source_commit": env_mod._git_commit(module_path),
        "source_dirty": False,
    }
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
    assert env_mod._git_commit(tracked) is not None

    installed = repo / ".venv" / "lib" / "site-packages" / "pkg"
    installed.mkdir(parents=True)
    (installed / "__init__.py").write_text("")
    assert env_mod._repo_root(installed) is None
    assert env_mod._git_commit(installed) is None


# ── Captured record and warnings ────────────────────────────────────────────


def test_capture_env_records_structured_package_provenance(monkeypatch):
    monkeypatch.setattr(
        env_mod,
        "_package",
        lambda name: {
            "version": "0.3.0",
            "installation_mode": "editable",
            "source_commit": "abc123",
            "source_dirty": False,
        }
        if name == "rigfl"
        else None,
    )

    env = capture_env()

    assert env["packages"] == {
        "rigfl": {
            "version": "0.3.0",
            "installation_mode": "editable",
            "source_commit": "abc123",
            "source_dirty": False,
        }
    }


def test_the_dirty_warning_is_printed_once_per_package(monkeypatch, capsys):
    monkeypatch.setattr(
        env_mod,
        "_package",
        lambda name: {
            "version": "0.3.0",
            "installation_mode": "editable",
            "source_commit": "abc123",
            "source_dirty": True,
        }
        if name == "rigfl"
        else None,
    )
    monkeypatch.setattr(env_mod, "_warned_dirty", set())

    for _ in range(3):
        assert capture_env()["packages"]["rigfl"]["source_dirty"] is True

    assert capsys.readouterr().out.count("[rigfl][warn]") == 1


def test_a_clean_checkout_warns_about_nothing(monkeypatch, capsys):
    monkeypatch.setattr(
        env_mod,
        "_package",
        lambda name: {
            "version": "0.3.0",
            "installation_mode": "editable",
            "source_dirty": False,
        }
        if name == "rigfl"
        else None,
    )
    monkeypatch.setattr(env_mod, "_warned_dirty", set())

    capture_env()

    assert "[rigfl][warn]" not in capsys.readouterr().out


def test_provenance_describes_rigfl_not_the_working_directory(tmp_path, monkeypatch):
    other = _repo(tmp_path / "other_repo")
    other_commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=str(other),
        capture_output=True,
        text=True,
    ).stdout.strip()
    monkeypatch.chdir(other)

    env = capture_env()
    rigfl = env["packages"]["rigfl"]

    assert rigfl["source_commit"] == env_mod._git_commit(Path(env_mod.__file__))
    assert rigfl["source_commit"] != other_commit


# ── It decides nothing ──────────────────────────────────────────────────────


def test_git_state_is_not_part_of_run_identity():
    """A dirty tree must not rerun a configuration that is already done."""
    exp = resolved_experiment(rounds=2, num_clients=2)
    algorithm = config_class("local")().model_dump()
    fingerprint = run_fingerprint(exp, algorithm)
    name = result_filename(exp, "local", fingerprint)

    for _dirty in (True, False, None):
        assert run_fingerprint(exp, algorithm) == fingerprint
        assert result_filename(exp, "local", fingerprint) == name

    dumped = exp.model_dump()
    assert not any("git" in key or "dirty" in key for key in dumped)


def test_the_environment_block_is_not_in_the_fingerprint():
    from rigfl.experiment.artifacts import make_run_record

    exp = resolved_experiment(rounds=2, num_clients=2)
    cfg = config_class("local")()
    fingerprint = run_fingerprint(exp, cfg.model_dump())
    provenance = {
        "packages": {"rigfl": {"source_commit": "abc123", "source_dirty": True}}
    }
    record = make_run_record(
        algorithm="local",
        experiment=exp.model_dump(),
        algorithm_config=cfg.model_dump(),
        run_fingerprint=fingerprint,
        result={},
        env=provenance,
    )

    assert record["env"] == provenance
    assert record["run_fingerprint"] == fingerprint
