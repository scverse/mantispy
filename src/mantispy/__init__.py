"""Image-based profiling on AnnData."""

from importlib.metadata import version

from mantispy import ds, get, io, metrics, pl, pp, tl
from mantispy._settings import settings

__version__ = version("mantispy")

__all__ = ["__version__", "ds", "get", "io", "metrics", "pl", "pp", "settings", "tl"]
