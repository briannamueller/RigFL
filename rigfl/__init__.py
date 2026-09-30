"""RigFL — a modular framework for rigorous federated learning experimentation."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("rigfl")
except PackageNotFoundError:
    __version__ = "0+unknown"
