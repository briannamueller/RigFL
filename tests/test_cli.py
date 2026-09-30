"""Tests for the top-level RigFL command."""


import pytest

from rigfl.cli import init_project
from tests.helpers import STARTER_CONFIGS


def test_init_project_creates_the_packaged_starter_files(tmp_path):
    project = tmp_path / "project"

    init_project(project)

    for template in STARTER_CONFIGS.rglob("*.yaml"):
        copied = project / "configs" / template.relative_to(STARTER_CONFIGS)
        assert copied.read_bytes() == template.read_bytes()


def test_init_project_refuses_all_writes_when_a_target_exists(tmp_path):
    project = tmp_path / "project"
    existing = project / "configs/datasets.yaml"
    existing.parent.mkdir(parents=True)
    existing.write_text("keep me\n")

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        init_project(project)

    assert existing.read_text() == "keep me\n"
    assert not (project / "configs/experiments").exists()
    assert not (project / ".gitignore").exists()
