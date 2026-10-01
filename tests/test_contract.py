"""The public construction and runner contract for registered algorithms."""

from __future__ import annotations

import os
import subprocess
import sys
from importlib import metadata

import pytest

from rigfl.core.config import AlgorithmConfig
from rigfl.core.interfaces import Algorithm
from rigfl.experiment.registry import (
    ALL_ALGORITHMS,
    REGISTRY,
    AlgorithmSpec,
    build_algorithm,
    register_algorithm,
    register_entry_points,
)
from tests.helpers import resolved_experiment


def test_external_algorithm_registration_is_explicit_and_refuses_replacement():
    class ExternalConfig(AlgorithmConfig):
        pass

    class ExternalAlgorithm(Algorithm):
        pass

    spec = AlgorithmSpec(ExternalAlgorithm, ExternalConfig)
    config = ExternalConfig()
    try:
        register_algorithm("external_example", spec)

        algorithm = build_algorithm("external_example", resolved_experiment(), config)
        assert isinstance(algorithm, ExternalAlgorithm)
        assert algorithm.config is config
        with pytest.raises(ValueError, match="already registered"):
            register_algorithm("external_example", spec)
    finally:
        REGISTRY.pop("external_example", None)
        if "external_example" in ALL_ALGORITHMS:
            ALL_ALGORITHMS.remove("external_example")


def _entry_points(directory, entry: str):
    info = directory / "ext_example-0.1.dist-info"
    info.mkdir(parents=True)
    (info / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: ext-example\nVersion: 0.1\n"
    )
    (info / "entry_points.txt").write_text(
        f"[rigfl.algorithms]\n{entry} = ext_example:SPEC\n"
    )
    (dist,) = metadata.distributions(path=[str(directory)])
    return dist.entry_points.select(group="rigfl.algorithms")


def test_installed_entry_points_register_with_their_distribution(
    tmp_path, monkeypatch
):
    (tmp_path / "ext_example.py").write_text(
        "from rigfl.core.config import AlgorithmConfig\n"
        "from rigfl.core.interfaces import Algorithm\n"
        "from rigfl.experiment.registry import AlgorithmSpec\n"
        "class ExtConfig(AlgorithmConfig):\n"
        "    pass\n"
        "SPEC = AlgorithmSpec(Algorithm, ExtConfig, "
        "provenance_packages=('graphroute',))\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    try:
        register_entry_points(_entry_points(tmp_path / "ok", "ext_example"))

        assert ALL_ALGORITHMS[-1] == "ext_example"
        assert REGISTRY["ext_example"].provenance_packages == (
            "ext-example", "graphroute",
        )
        with pytest.raises(ValueError, match="already registered"):
            register_entry_points(_entry_points(tmp_path / "clash", "local"))
    finally:
        REGISTRY.pop("ext_example", None)
        if "ext_example" in ALL_ALGORITHMS:
            ALL_ALGORITHMS.remove("ext_example")


def test_a_plugin_imported_before_rigfl_still_registers(tmp_path):
    (tmp_path / "ext_first.py").write_text(
        "from rigfl.core.config import AlgorithmConfig\n"
        "from rigfl.core.interfaces import Algorithm\n"
        "from rigfl.experiment.registry import AlgorithmSpec\n"
        "class ExtConfig(AlgorithmConfig):\n"
        "    pass\n"
        "SPEC = AlgorithmSpec(Algorithm, ExtConfig)\n"
    )
    info = tmp_path / "ext_first-0.1.dist-info"
    info.mkdir()
    (info / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: ext-first\nVersion: 0.1\n"
    )
    (info / "entry_points.txt").write_text(
        "[rigfl.algorithms]\next_first = ext_first:SPEC\n"
    )
    # a fresh interpreter, so the plugin really is imported first
    script = (
        "import ext_first\n"
        "from rigfl.experiment.registry import algorithm_spec\n"
        "assert algorithm_spec('ext_first').config is ext_first.ExtConfig\n"
    )
    subprocess.run(
        [sys.executable, "-c", script], check=True,
        env={**os.environ, "PYTHONPATH": str(tmp_path)},
    )
