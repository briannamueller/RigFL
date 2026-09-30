"""Capture the software provenance of an experiment before training starts."""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
from importlib import metadata
from pathlib import Path


def _repo_root(path: str | Path) -> str | None:
    """The Git checkout that tracks ``path``, or ``None``."""
    try:
        resolved = os.path.realpath(path)
        out = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=resolved if os.path.isdir(resolved) else os.path.dirname(resolved),
            stderr=subprocess.DEVNULL,
            text=True,
        )
        root = os.path.realpath(out.strip())
        tracked = subprocess.check_output(
            ["git", "ls-files", "--", resolved],
            cwd=root,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except Exception:
        return None
    return root if tracked.strip() else None


def _git_commit(root: str) -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except Exception:
        return None


def _git_dirty(root: str) -> bool | None:
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=root,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except Exception:
        return None
    return bool(out.strip())


def _import_name(distribution: str) -> str:
    """The top-level module a distribution installs, e.g. feddes-paper -> feddes_paper."""
    try:
        top_level = metadata.distribution(distribution).read_text("top_level.txt")
    except metadata.PackageNotFoundError:
        top_level = None
    names = (top_level or "").split()
    return names[0] if names else distribution.replace("-", "_")


def _package(name: str) -> dict | None:
    """Describe the installed distribution whose code this process actually imported."""
    try:
        module = importlib.import_module(_import_name(name))
    except Exception:
        return None
    source = getattr(module, "__file__", None)
    if not source:
        return None
    try:
        version = metadata.version(name)
    except metadata.PackageNotFoundError:
        version = getattr(module, "__version__", None)

    info = {"version": version}
    root = _repo_root(source)
    if root is not None:
        info["source_commit"] = _git_commit(root)
        info["source_dirty"] = _git_dirty(root)
    return {key: value for key, value in info.items() if value is not None}


_warned_dirty: set[str] = set()


def _warn_dirty_once(name: str) -> None:
    if name in _warned_dirty:
        return
    _warned_dirty.add(name)
    print(
        f"[rigfl][warn] {name} has uncommitted changes; source_commit does not "
        "fully identify this code (source_dirty is recorded as true)."
    )


def capture_env(extra_packages: tuple[str, ...] = ()) -> dict:
    """Return portable execution provenance, excluding local source paths.

    ``extra_packages`` are distribution names recorded alongside RigFL and BioSilo.
    """
    import numpy
    import torch

    packages = {}
    for name in dict.fromkeys(("rigfl", "biosilo", *extra_packages)):
        info = _package(name)
        if info is None:
            continue
        packages[name] = info
        if info.get("source_dirty"):
            _warn_dirty_once(name)

    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "numpy": numpy.__version__,
        "packages": packages,
    }
