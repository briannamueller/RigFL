"""Capture the software provenance of an experiment before training starts."""

from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from urllib.parse import unquote, urlparse
from urllib.request import url2pathname


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


def _git_commit(path: str | Path) -> str | None:
    """The commit of the checkout containing ``path``, when available."""
    root = _repo_root(path)
    if root is None:
        return None
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


def _git_dirty(path: str | Path) -> bool | None:
    """Whether the checkout containing ``path`` has any changes."""
    root = _repo_root(path)
    if root is None:
        return None
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


def _normalise_distribution_name(name: str) -> str:
    return name.lower().replace("-", "_").replace(".", "_")


def _distribution_candidates(name: str) -> list[metadata.Distribution]:
    wanted = _normalise_distribution_name(name)
    return [
        distribution
        for distribution in metadata.distributions()
        if _normalise_distribution_name(distribution.metadata.get("Name", ""))
        == wanted
    ]


def _editable_root(distribution: metadata.Distribution) -> Path | None:
    """Return the PEP 610 editable source root without persisting it."""
    try:
        raw = distribution.read_text("direct_url.json")
        direct = json.loads(raw) if raw else {}
        parsed = urlparse(direct.get("url", ""))
        if not direct.get("dir_info", {}).get("editable") or parsed.scheme != "file":
            return None
        return Path(url2pathname(unquote(parsed.path))).resolve()
    except Exception:
        return None


def _contains(root: Path, path: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _distribution_for(
    name: str, module_path: Path
) -> tuple[metadata.Distribution | None, bool]:
    """Find the installed metadata belonging to the code that was imported."""
    candidates = _distribution_candidates(name)
    for distribution in candidates:
        root = _editable_root(distribution)
        if root is not None and _contains(root, module_path):
            return distribution, True
    for distribution in candidates:
        for installed in distribution.files or ():
            try:
                if Path(distribution.locate_file(installed)).resolve() == module_path:
                    return distribution, False
            except Exception:
                continue
    return None, False


def _package(name: str) -> dict | None:
    """Describe the installed package whose code this process actually imported."""
    try:
        module = importlib.import_module(name)
    except Exception:
        return None
    source = getattr(module, "__file__", None)
    if not source:
        return None
    module_path = Path(source).resolve()
    distribution, editable = _distribution_for(name, module_path)
    checkout = _repo_root(module_path)

    if editable:
        installation_mode = "editable"
    elif checkout is not None:
        installation_mode = "source_checkout"
    elif distribution is not None:
        installation_mode = "wheel"
    else:
        installation_mode = "unknown"

    info = {
        "version": (
            distribution.version
            if distribution is not None
            else getattr(module, "__version__", None)
        ),
        "installation_mode": installation_mode,
    }
    if checkout is not None:
        info["source_commit"] = _git_commit(module_path)
        info["source_dirty"] = _git_dirty(module_path)
    return {key: value for key, value in info.items() if value is not None}


# Prevent repeated warnings in a multi-algorithm process.
_warned_dirty: set[str] = set()


def _warn_dirty_once(name: str) -> None:
    if name in _warned_dirty:
        return
    _warned_dirty.add(name)
    print(
        f"[rigfl][warn] {name} is running from a checkout with uncommitted "
        "changes; its source_commit alone does not fully identify the code "
        "that produced this result (source_dirty is recorded as true)."
    )


def capture_env() -> dict:
    """Return portable execution provenance, excluding local source paths."""
    import numpy
    import torch

    packages = {}
    for name in ("rigfl", "biosilo"):
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
