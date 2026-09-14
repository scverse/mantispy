from typing import TYPE_CHECKING, Any

from mantispy.io._profiles import CHANNEL_ALIASES, METADATA_PREFIXES, read_profiles

if TYPE_CHECKING:
    from mantispy.io._plate import read_plate

__all__ = ["CHANNEL_ALIASES", "METADATA_PREFIXES", "read_plate", "read_profiles"]

# read_plate pulls in spatialdata and the image stack; importing mantispy must not pay for them.
_LAZY = {"read_plate": "mantispy.io._plate"}


def __getattr__(name: str) -> Any:
    """Import the spatial readers only when they are first asked for."""
    if name in _LAZY:
        from importlib import import_module

        return getattr(import_module(_LAZY[name]), name)
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
