"""Capture repository state and key software versions for a run."""

from __future__ import annotations

import os
import subprocess
import sys


def _repo_root(path: str) -> str | None:
    """The Git checkout that tracks ``path``, or ``None``."""
    try:
        out = subprocess.check_output(["git", "rev-parse", "--show-toplevel"],
                                      cwd=path, stderr=subprocess.DEVNULL, text=True)
        root = os.path.realpath(out.strip())
        tracked = subprocess.check_output(["git", "ls-files", "--", path],
                                          cwd=root, stderr=subprocess.DEVNULL, text=True)
    except Exception:
        return None
    return root if tracked.strip() else None


def _git_commit(path: str | None = None) -> str | None:
    """The commit of the checkout containing ``path`` (default: RigFL's own)."""
    root = _repo_root(path if path is not None else _rigfl_path())
    if root is None:
        return None                              # not a git checkout -- fine
    try:
        out = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                      cwd=root, stderr=subprocess.DEVNULL, text=True)
        return out.strip()
    except Exception:
        return None


def _rigfl_path() -> str:
    """Where RigFL itself is installed -- the only tree its provenance describes."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _git_dirty(path: str | None = None) -> bool | None:
    """Whether tracked, staged, or untracked changes exist; ``None`` if unknown."""
    root = _repo_root(path if path is not None else _rigfl_path())
    if root is None:
        return None
    try:
        out = subprocess.check_output(["git", "status", "--porcelain"],
                                      cwd=root, stderr=subprocess.DEVNULL, text=True)
    except Exception:
        return None
    return bool(out.strip())


#: Prevent repeated warnings in a multi-algorithm process.
_warned_dirty = False


def _warn_dirty_once() -> None:
    global _warned_dirty
    if _warned_dirty:
        return
    _warned_dirty = True
    print("[rigfl][warn] the checkout has uncommitted changes, so env.git_commit "
          "does not fully describe the code that produced this result "
          "(env.git_dirty is recorded as true).")


def _package(name: str) -> dict | None:
    """Installed version and, for a checkout, its commit."""
    import importlib
    try:
        mod = importlib.import_module(name)
    except Exception:
        return None
    info: dict = {}
    try:
        from importlib.metadata import version
        info["version"] = version(name)
    except Exception:                            # not installed as a distribution
        info["version"] = getattr(mod, "__version__", None)
    path = getattr(mod, "__file__", None)
    if path:
        info["commit"] = _git_commit(os.path.dirname(os.path.dirname(
            os.path.abspath(path))))
    return {k: v for k, v in info.items() if v is not None} or None


def capture_env() -> dict:
    import numpy
    import torch
    dirty = _git_dirty()
    if dirty:
        _warn_dirty_once()
    env = {
        "git_commit": _git_commit(),
        "git_dirty": dirty,
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "numpy": numpy.__version__,
    }
    rigfl = _package("rigfl")
    if rigfl and rigfl.get("version"):
        env["rigfl"] = rigfl["version"]
    # Record optional packages when they are installed.
    for name in ("graphroute", "biosilo"):
        info = _package(name)
        if info:
            env[name] = info
    return env
