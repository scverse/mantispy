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
from mantispy._core.schema import get_resolution, resolution_for, stamp

#: Aggregation functions, mapped to the kernel selector that computes them.
FUNCTIONS = {"median": MEDIAN, "mean": MEAN}

#: uns["mantispy"] keys that survive aggregation because they describe the features or the experiment.
#: Result tables are keyed on the input rows and are dropped.
_INHERITED = frozenset({"channels", "dataset", "truth", "feature_select", "blocklist"})

#: Columns that add up over a group, so they are recomputed for it rather than carried as constant metadata.
_TALLIES = ("Metadata_CellCount", "Metadata_SiteCount", "Metadata_ReplicateCount")


def aggregate(
    adata: AnnData,
    by: Sequence[str] = ("Metadata_Plate", "Metadata_Well"),
    func: str = "median",
    min_cells: int = 10,
    layer: str | None = None,
    count_key: str = "Metadata_CellCount",
    site_key: str = "Metadata_SiteCount",
) -> AnnData:
    """Aggregate ``adata`` to one profile per group.

    Args:
        adata: Single cells, or profiles to aggregate further, as its recorded resolution says.
        by: Columns defining a profile. The default is one profile per well.
        func: ``"median"`` (the pycytominer default) or ``"mean"``.
        min_cells: Groups with fewer cells than this are dropped.
        layer: Aggregate this layer instead of ``X``.
        count_key: ``obs`` column the cell count is written to, and read from when ``adata`` holds profiles.
        site_key: ``obs`` column the number of fields of view is written to, and read from when ``adata`` holds profiles.

    Returns:
        A new :class:`~anndata.AnnData` with one row per group.
        ``var`` is carried over unchanged; ``obs`` holds the grouping columns, `count_key`, `site_key` when the fields of view are known, and every other ``Metadata_`` column that is constant within every group.
        `count_key` is the number of cells behind a row, so its scope follows ``by``: grouping by site counts the cells of one field of view, grouping by well those of every field. Profiles contribute the cells they carry rather than one each.
        `site_key` is the number of fields that contributed cells, summed where the rows carry it and counted from ``Metadata_Site`` otherwise.
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
    frame = as_frame(adata.obs)
    tallies = {count_key: counts}
    # The recorded resolution decides, not the column: a cell may carry its well's count as a covariate.
    if get_resolution(adata) != "cell":
        # Profiles stand for the cells and fields they summarize, not one cell each.
        for column in (count_key, site_key):
            if column in frame:
                tallies[column] = np.bincount(codes, weights=frame[column].to_numpy(dtype=float), minlength=len(keys))
    elif "Metadata_Site" in frame:
        tallies[site_key] = _site_counts(frame, codes, len(keys))

    obs = _group_obs(adata, columns, keys, codes, tallies)
    # A group whose count is unknown is kept rather than dropped as too small.
    keep = ~(tallies[count_key] < min_cells)
    if not keep.any():
        get_logger().warning("aggregate dropped every group; min_cells=%d exceeds every group size", min_cells)

    obs = obs.loc[keep].reset_index(drop=True)
    obs.index = pd.Index([str(index) for index in range(len(obs))])

    result = ad.AnnData(X=values[keep].astype(np.float32), obs=obs, var=as_frame(adata.var).copy())
    stamp(result, resolution=resolution_for(columns))
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
    record_params(
        result,
        "aggregate",
        {
            "by": columns,
            "func": func,
            "min_cells": min_cells,
            "layer": layer,
            "count_key": count_key,
            "site_key": site_key,
        },
    )
    return result


def _site_counts(frame: pd.DataFrame, codes: np.ndarray, n_groups: int) -> np.ndarray:
    """Distinct fields of view among each group's cells, where site 1 of one well and of the next are different fields."""
    # One integer per field, built from per-column codes rather than a MultiIndex of tuples, which is 7x slower.
    field = np.zeros(len(frame), dtype=np.int64)
    for column in ("Metadata_Plate", "Metadata_Well", "Metadata_Site"):
        if column in frame:
            level, uniques = pd.factorize(frame[column])
            field = field * (len(uniques) + 1) + level + 1
    width = int(field.max()) + 1
    return np.bincount(np.unique(codes.astype(np.int64) * width + field) // width, minlength=n_groups)


def _group_obs(
    adata: AnnData, columns: list[str], keys: pd.Index, codes: np.ndarray, tallies: dict[str, np.ndarray]
) -> pd.DataFrame:
    """Build the aggregated ``obs``: grouping keys, the per-group tallies, constant metadata."""
    if len(columns) == 1:
        obs = pd.DataFrame({columns[0]: np.asarray(keys)})
    else:
        obs = pd.DataFrame({name: keys.get_level_values(position) for position, name in enumerate(columns)})
    obs = obs.reset_index(drop=True)
    for name, values in tallies.items():
        obs[name] = values

    frame = as_frame(adata.obs)
    carried = [
        column
        for column in frame.columns
        if column.startswith("Metadata_") and column not in {*columns, *_TALLIES, *tallies}
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
