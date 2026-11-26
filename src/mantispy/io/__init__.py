"""Readers and writers."""

from typing import TYPE_CHECKING, Any

from mantispy._core.schema import validate

from ._jump import read_jump, unexpressed_genes
from ._profiles import METADATA_PREFIXES, from_dataframe, from_embedding, read, read_profiles, stamp, write

if TYPE_CHECKING:
    from ._export import read_cellprofiler_zarr
    from ._gallery import read_cellprofiler_gallery

__all__ = [
    "METADATA_PREFIXES",
    "from_dataframe",
    "from_embedding",
    "read",
    "read_cellprofiler_gallery",
    "read_cellprofiler_zarr",
    "read_jump",
    "read_profiles",
    "stamp",
    "unexpressed_genes",
    "validate",
    "write",
]

# These readers need the images extra, which importing mantispy must not.
_LAZY = {
    "read_cellprofiler_gallery": "mantispy.io._gallery",
    "read_cellprofiler_zarr": "mantispy.io._export",
}


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        from importlib import import_module

        return getattr(import_module(_LAZY[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
