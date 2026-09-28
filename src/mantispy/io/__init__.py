"""Readers and writers."""

from typing import TYPE_CHECKING, Any

from mantispy._core.schema import validate

from ._jump import read_jump
from ._profiles import METADATA_PREFIXES, read, read_profiles, stamp, write

if TYPE_CHECKING:
    from ._plate import read_plate

__all__ = ["METADATA_PREFIXES", "read", "read_jump", "read_plate", "read_profiles", "stamp", "validate", "write"]

# read_plate needs the spatial extra, which importing mantispy must not.
_LAZY = {"read_plate": "mantispy.io._plate"}


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        from importlib import import_module

        return getattr(import_module(_LAZY[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
