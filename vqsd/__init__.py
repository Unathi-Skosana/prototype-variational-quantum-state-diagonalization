"""Variational Quantum State Diagonilzation."""

from importlib_metadata import version as metadata_version, PackageNotFoundError

try:
    __version__ = (metadata_version("vqsd"),)
except PackageNotFoundError:  # pragma: no cover
    # package is not installed
    pass
