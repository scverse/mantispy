"""Image-based profiling on AnnData."""

from importlib.metadata import version

from mantispy import ds, get, io, metrics, pp, tl
from mantispy._settings import settings

__version__ = version("mantispy")

__all__ = ["__version__", "ds", "get", "io", "metrics", "pp", "settings", "tl"]
