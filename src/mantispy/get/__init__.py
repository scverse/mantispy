"""Accessors that turn a mantispy AnnData into plain Python objects."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from mantispy.get._accessors import controls, features, to_dataframe

if TYPE_CHECKING:
    from scanpy.get import obs_df, var_df

#: Names re-exported from :mod:`scanpy.get`, fetched on first access rather than at import.
#: Importing them eagerly pulls in scanpy, and with it sklearn and matplotlib, so the first access pays that cost instead of every ``import mantispy``.
_SCANPY_NAMES = ("obs_df", "var_df")

__all__ = ["controls", "features", "obs_df", "to_dataframe", "var_df"]


def __getattr__(name: str) -> Any:
    """Fetch a name re-exported from :mod:`scanpy.get` on first access, so that importing mantispy does not import scanpy.

    Args:
        name: Attribute looked up on this module.

    Returns:
        The scanpy function, also bound in this module so that later lookups do not come back here.

    Raises:
        AttributeError: `name` is not one of the names re-exported from scanpy.
    """
    if name not in _SCANPY_NAMES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import scanpy.get

    value = getattr(scanpy.get, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """List the module's public names, including the scanpy ones no one has touched yet."""
    return sorted(__all__)
