"""Accessors that turn a mantispy AnnData into plain Python objects."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from mantispy.get._accessors import controls, features, to_dataframe

if TYPE_CHECKING:
    from scanpy.get import obs_df, var_df

#: Names re-exported from :mod:`scanpy.get`, fetched on first access rather than at import.
#: Importing them eagerly reached scanpy, and through it sklearn and matplotlib: ``scanpy <- scanpy.get <- mantispy.get`` was the parent chain in every ``python -X importtime`` trace, and sklearn arrived under ``scanpy.get._aggregated`` rather than from any import of ours.
#: That chain is a structural fact and holds regardless of load, which is why it is recorded here.
#: Measured on an idle machine, minimum of seven interleaved rounds: ``import mantispy`` takes 1.85 s against 3.43 s before this deferral, where ``import anndata`` alone costs 1.59 s, so the overhead this package adds on top of anndata falls from 1.84 s to 0.27 s.
#: The cost is deferred rather than removed: the first call that draws pays 0.64 s for matplotlib, and the first ``get.obs_df`` pays scanpy in full, once per process.
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
    from scanpy.get import obs_df, var_df

    globals().update(obs_df=obs_df, var_df=var_df)
    return globals()[name]


def __dir__() -> list[str]:
    """List the module's public names, including the scanpy ones no one has touched yet."""
    return sorted(__all__)
