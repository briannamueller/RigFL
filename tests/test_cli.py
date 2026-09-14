"""Tests for the top-level RigFL command."""

from pathlib import Path

import pytest

from rigfl.cli import init_project, main

ROOT = Path(__file__).parents[1]


def test_init_project_creates_the_packaged_starter_files(tmp_path):
    project = tmp_path / "project"

    created = init_project(project)

    assert created == [
        project / "configs/datasets.yaml",
        project / "experiments/cifar10_run.yaml",
        project / "experiments/cifar10_sweep.yaml",
        project / ".gitignore",
    ]
    assert (project / "configs/datasets.yaml").read_bytes() == (
        ROOT / "configs/datasets.yaml"
    ).read_bytes()
    assert (project / "experiments/cifar10_run.yaml").read_bytes() == (
        ROOT / "experiments/cifar10_run.yaml"
    ).read_bytes()
    assert (project / "experiments/cifar10_sweep.yaml").read_bytes() == (
        ROOT / "experiments/cifar10_sweep.yaml"
    ).read_bytes()
    assert "data/" in (project / ".gitignore").read_text()
    assert not (project / "data").exists()
    assert not (project / "results").exists()


def test_init_project_refuses_all_writes_when_a_target_exists(tmp_path):
    project = tmp_path / "project"
    existing = project / "configs/datasets.yaml"
    existing.parent.mkdir(parents=True)
    existing.write_text("keep me\n")

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        init_project(project)

    assert existing.read_text() == "keep me\n"
    assert not (project / "experiments").exists()
    assert not (project / ".gitignore").exists()


def test_init_command_reports_next_steps(tmp_path, capsys):
    project = tmp_path / "project"

    main(["init", str(project)])

    output = capsys.readouterr().out
    assert f"Created RigFL project in {project}" in output
    assert "python -m rigfl.data.generate --dataset cifar10" in output
