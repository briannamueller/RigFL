"""Tests for the top-level RigFL command."""

from pathlib import Path

import pytest

from rigfl import __version__
from rigfl.cli import init_project, legacy_config_argv, main

ROOT = Path(__file__).parents[1]


def test_legacy_config_option_is_translated_without_becoming_public_help():
    assert legacy_config_argv(
        ["--algorithm", "fedavg", "--config", "run.yaml"]
    ) == ["run.yaml", "--algorithm", "fedavg"]


def test_init_project_creates_the_packaged_starter_files(tmp_path):
    project = tmp_path / "project"

    created = init_project(project)

    assert created == [
        project / "configs/datasets.yaml",
        project / "configs/reporting.yaml",
        project / "configs/experiments/cifar10_run.yaml",
        project / "configs/experiments/cifar10_sweep.yaml",
        project / ".gitignore",
        project / "requirements.txt",
    ]
    assert (project / "configs/datasets.yaml").read_bytes() == (
        ROOT / "configs/datasets.yaml"
    ).read_bytes()
    assert (project / "configs/reporting.yaml").read_bytes() == (
        ROOT / "configs/reporting.yaml"
    ).read_bytes()
    assert (project / "configs/experiments/cifar10_run.yaml").read_bytes() == (
        ROOT / "configs/experiments/cifar10_run.yaml"
    ).read_bytes()
    assert (project / "configs/experiments/cifar10_sweep.yaml").read_bytes() == (
        ROOT / "configs/experiments/cifar10_sweep.yaml"
    ).read_bytes()
    assert "data/" in (project / ".gitignore").read_text()
    assert (project / "requirements.txt").read_text() == f"rigfl=={__version__}\n"
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
    assert not (project / "configs/experiments").exists()
    assert not (project / ".gitignore").exists()


def test_init_command_reports_next_steps(tmp_path, capsys):
    project = tmp_path / "project"

    main(["init", str(project)])

    output = capsys.readouterr().out
    assert f"Created RigFL project in {project}" in output
    assert "rigfl data generate --dataset cifar10" in output
    assert (
        "rigfl run configs/experiments/cifar10_run.yaml --algorithm fedavg"
        in output
    )


def test_task_command_runs_one_indexed_grid_task(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(
        "rigfl.experiment.launch.run_task",
        lambda grid, index, out_dir, **options: calls.append(
            (grid, index, out_dir, options)
        ),
    )

    main([
        "task", "snapshot/grid.jsonl", "3",
        "--results-root", str(tmp_path), "--dry-run",
    ])

    assert calls == [(
        "snapshot/grid.jsonl",
        3,
        tmp_path / "runs",
        {"dry_run": True, "force": False},
    )]


def test_run_command_delegates_to_the_single_run_interface(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "rigfl.experiment.run.main",
        lambda argv, *, prog: calls.append((argv, prog)),
    )

    main(["run", "configs/experiments/run.yaml", "--algorithm", "fedavg"])

    assert calls == [(
        ["configs/experiments/run.yaml", "--algorithm", "fedavg"],
        "rigfl run",
    )]


def test_run_command_requires_an_algorithm(tmp_path):
    config = tmp_path / "run.yaml"
    config.write_text("experiment: {}\nalgorithm: {}\n")

    with pytest.raises(SystemExit, match="2"):
        main(["run", str(config)])


def test_sweep_command_delegates_to_the_yaml_sweep_interface(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "rigfl.experiment.launch.main",
        lambda argv, *, prog: calls.append((argv, prog)),
    )

    main(["sweep", "configs/experiments/sweep.yaml"])

    assert calls == [(["configs/experiments/sweep.yaml"], "rigfl sweep")]


def test_data_generate_command_delegates_to_the_partition_interface(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "rigfl.data.generate.main",
        lambda argv, *, prog: calls.append((argv, prog)),
    )

    main(["data", "generate", "--dataset", "cifar10"])

    assert calls == [(["--dataset", "cifar10"], "rigfl data generate")]


def test_snapshot_command_reports_scheduler_neutral_task_command(
    tmp_path, monkeypatch, capsys
):
    snapshot = tmp_path / "snapshots" / "fixed" / "grid.jsonl"
    monkeypatch.setattr(
        "rigfl.experiment.launch.stage_task_snapshot",
        lambda grid, results_root: snapshot,
    )

    main(["snapshot", "working/grid.jsonl", "--results-root", str(tmp_path)])

    output = capsys.readouterr().out
    assert f"Wrote immutable task snapshot: {snapshot}" in output
    assert f"rigfl task {snapshot} 1" in output
