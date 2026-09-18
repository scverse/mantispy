"""Aggregate single cells into well- or perturbation-level profiles."""

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy

import anndata as ad
import numpy as np
import pandas as pd
from anndata import AnnData

from mantispy._core._numba import MEAN, MEDIAN
from mantispy._core._reduce import group_codes, reduce_grouped
from mantispy._core.frames import as_frame, categorize_metadata
from mantispy._core.logging import get_logger
from mantispy._core.provenance import record_params
from mantispy._core.schema import stamp

#: Aggregation functions, mapped to the kernel selector that computes them.
FUNCTIONS = {"median": MEDIAN, "mean": MEAN}

_WELL_KEYS = {"Metadata_Plate", "Metadata_Well"}

#: uns["mantispy"] keys that survive aggregation because they describe the features or the experiment.
#: Result tables are keyed on the input rows and are dropped.
_INHERITED = frozenset({"channels", "dataset", "truth", "feature_select", "blocklist"})


def aggregate(
    adata: AnnData,
    by: Sequence[str] = ("Metadata_Plate", "Metadata_Well"),
    func: str = "median",
    min_cells: int = 10,
    layer: str | None = None,
) -> AnnData:
    """Aggregate ``adata`` to one profile per group.

    Args:
        adata: Single-cell object to aggregate.
        by: Columns defining a profile. The default is one profile per well.
        func: ``"median"`` (the pycytominer default) or ``"mean"``.
        min_cells: Groups with fewer cells than this are dropped.
        layer: Aggregate this layer instead of ``X``.

    Returns:
        A new :class:`~anndata.AnnData` with one row per group.
        ``var`` is carried over unchanged; ``obs`` holds the grouping columns, ``Metadata_CellCount``, and every other ``Metadata_`` column that is constant within every group.
        The resolution recorded is ``"well"`` when ``by`` holds both ``Metadata_Plate`` and ``Metadata_Well``, since a finer grouping such as one row per site is still per-well or finer, and ``"perturbation"`` otherwise.

    Raises:
        ValueError: ``func`` is not one of ``FUNCTIONS``.

    Notes:
        This uses mantispy's own NaN-skipping kernel rather than :func:`scanpy.get.aggregate`, which propagates NaN and is measurably slower on both mean and median.
    """
    if func not in FUNCTIONS:
        raise ValueError(f"func must be one of {tuple(FUNCTIONS)}, got {func!r}")
    columns = [by] if isinstance(by, str) else list(by)

    values, keys, counts = reduce_grouped(adata, columns, FUNCTIONS[func], layer=layer)
    codes, _ = group_codes(adata, columns)

    obs = _group_obs(adata, columns, keys, codes, counts)
    keep = counts >= min_cells
    if not keep.any():
        get_logger().warning("aggregate dropped every group; min_cells=%d exceeds every group size", min_cells)

    obs = obs.loc[keep].reset_index(drop=True)
    obs.index = pd.Index([str(index) for index in range(len(obs))])

    result = ad.AnnData(X=values[keep].astype(np.float32), obs=obs, var=as_frame(adata.var).copy())
    # A grouping that splits the well further, by site or by anything else, is still per-well or finer,
    # and it carries the plate and well columns that validate requires of a well-resolution object.
    stamp(result, resolution="well" if _WELL_KEYS <= set(columns) else "perturbation")
    store = adata.uns.get("mantispy", {})
    # Deep copies, so the aggregate and its source do not share mutable frames.
    result.uns["mantispy"].update({key: deepcopy(value) for key, value in store.items() if key in _INHERITED})
    dropped = sorted(set(store) - _INHERITED - {"resolution", "schema_version", "params"})
    if dropped:
        get_logger().debug("aggregate dropped %s, which describe the input rows", dropped)
    result.uns["mantispy"]["aggregated_from"] = {
        "by": columns,
        "func": func,
        "n_obs": int(adata.n_obs),
        "min_cells": int(min_cells),
    }
    record_params(result, "aggregate", {"by": columns, "func": func, "min_cells": min_cells, "layer": layer})
    return result


def _group_obs(
    adata: AnnData, columns: list[str], keys: pd.Index, codes: np.ndarray, counts: np.ndarray
) -> pd.DataFrame:
    """Build the aggregated ``obs``: grouping keys, cell count, constant metadata."""
    if len(columns) == 1:
        obs = pd.DataFrame({columns[0]: np.asarray(keys)})
    else:
        obs = pd.DataFrame({name: keys.get_level_values(position) for position, name in enumerate(columns)})
    obs = obs.reset_index(drop=True)
    obs["Metadata_CellCount"] = counts

    frame = as_frame(adata.obs)
    carried = [
        column
        for column in frame.columns
        if column.startswith("Metadata_") and column not in columns and column != "Metadata_CellCount"
    ]
    if carried:
        grouped = frame[carried].groupby(codes, observed=True)
        constant = grouped.nunique(dropna=False).le(1).all()
        for column in carried:
            if constant[column]:
                obs[column] = grouped[column].first().to_numpy()
            else:
                get_logger().debug("aggregate dropped non-constant metadata column %s", column)
    return categorize_metadata(obs)
