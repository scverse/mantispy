"""Aggregate single cells into well- or perturbation-level profiles."""

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy

import anndata as ad
import numpy as np
import pandas as pd
from anndata import AnnData

from mantispy._core._numba import MEAN, MEDIAN
from mantispy._core._reduce import group_codes, reduce_grouped, reduced_var
from mantispy._core.frames import as_frame, categorize_metadata
from mantispy._core.logging import get_logger
from mantispy._core.provenance import level_change_source, record_history, record_params
from mantispy._core.schema import REQUIRED_OBS, REQUIRED_UNS, get_resolution, resolution_for, stamp

FUNCTIONS = {"median": MEDIAN, "mean": MEAN}

#: Keys describing the features or the experiment; result tables are keyed on the input rows, so they are dropped.
_INHERITED = frozenset({"channels", "dataset", "truth", "feature_select", "blocklist"})

#: Columns that add up over a group, so they are recomputed for it rather than carried as constant metadata.
_TALLIES = ("Metadata_CellCount", "Metadata_SiteCount", "Metadata_ReplicateCount")


def aggregate(
    adata: AnnData,
    by: Sequence[str] = ("Metadata_Plate", "Metadata_Well"),
    func: str = "median",
    min_cells: int = 10,
    layer: str | None = None,
    use_rep: str | None = None,
    count_key: str = "Metadata_CellCount",
    site_key: str = "Metadata_SiteCount",
    store_membership: bool = False,
) -> AnnData:
    """Aggregate ``adata`` to one profile per group.

    Args:
        adata: Single cells, or profiles to aggregate further, as its recorded resolution says.
        by: Columns defining a profile.
            The default is one profile per well.
        func: ``"median"`` (the pycytominer default) or ``"mean"``.
        min_cells: Groups with fewer cells than this are dropped.
        layer: Aggregate this layer instead of ``X``.
        use_rep: Aggregate this ``obsm`` representation (e.g. an embedding from ``pp.tvn``/``pp.harmony``) instead of ``X``; the result's ``X`` holds the reduced representation and ``var`` is a plain range index, since the axes are not named features.
            Mutually exclusive with ``layer``.
        count_key: ``obs`` column the cell count is written to, and read from when ``adata`` holds profiles.
        site_key: ``obs`` column the number of fields of view is written to, and read from when ``adata`` holds profiles.
        store_membership: Record which source rows went into each profile under ``uns["mantispy"]["membership"]``, by their stable identity columns (e.g. ``Metadata_ImageID`` + ``Metadata_ObjectNumber`` for objects) rather than row positions. Off by default because object-level membership can be large.

    Returns:
        A new :class:`~anndata.AnnData` with one row per group.
        ``var`` is carried over unchanged; ``obs`` holds the grouping columns, `count_key`, `site_key` when the fields of view are known, and every other ``Metadata_`` column that is constant within every group.
        `count_key` is the number of cells behind a row, so its scope follows ``by``: grouping by site counts the cells of one field of view, grouping by well those of every field.
        Profiles contribute the cells they carry rather than one each, and profiles that carry no count give an unknown one.
        `site_key` is the number of fields that contributed cells, summed where the rows carry it and counted from ``Metadata_Site`` otherwise.
        The resolution recorded is ``"well"`` when ``by`` holds both ``Metadata_Plate`` and ``Metadata_Well``, since a finer grouping such as one row per site is still per-well or finer, and ``"aggregate"`` otherwise.

    Raises:
        ValueError: ``func`` is not one of ``FUNCTIONS``, or ``use_rep`` and ``layer`` are both given, or ``use_rep`` is not a 2-D representation in ``obsm``.

    Notes:
        This uses mantispy's own NaN-skipping kernel rather than :func:`scanpy.get.aggregate`, which propagates NaN and is slower on both mean and median.
    """
    if func not in FUNCTIONS:
        raise ValueError(f"func must be one of {tuple(FUNCTIONS)}, got {func!r}")
    # use_rep/layer mutual exclusion and obsm validation are enforced once, in reduce_grouped's _obsm_source.
    columns = [by] if isinstance(by, str) else list(by)

    values, keys, counts = reduce_grouped(adata, columns, FUNCTIONS[func], layer=layer, use_rep=use_rep)
    codes, _ = group_codes(adata, columns)
    frame = as_frame(adata.obs)
    tallies = {count_key: counts}
    # The recorded resolution, not the column, decides count handling: an object may carry its well's
    # count as a covariate. get_resolution raises on an unstamped object rather than assume object-level.
    if get_resolution(adata) != "object":
        tallies[count_key] = np.full(len(keys), np.nan)
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

    var = reduced_var(adata, use_rep, values.shape[1])
    result = ad.AnnData(X=values[keep].astype(np.float32), obs=obs, var=var)
    store = adata.uns.get("mantispy", {})
    resolution = resolution_for(columns)
    # Inherit the source history so the aggregate carries its lineage (spec §14.3).
    stamp(result, resolution=resolution, grouped_by=columns, history=store.get("history"))
    result.uns["mantispy"].update({key: deepcopy(value) for key, value in store.items() if key in _INHERITED})
    dropped = sorted(set(store) - _INHERITED - set(REQUIRED_UNS) - {"params"})
    if dropped:
        get_logger().debug("aggregate dropped %s, which describe the input rows", dropped)
    result.uns["mantispy"]["aggregated_from"] = {
        "by": columns,
        "func": func,
        "n_obs": int(adata.n_obs),
        "min_cells": int(min_cells),
    }
    if store_membership:
        _record_membership(result, adata, columns, codes, keep)
    call_params = {
        "by": columns,
        "func": func,
        "min_cells": min_cells,
        "layer": layer,
        "use_rep": use_rep,
        "count_key": count_key,
        "site_key": site_key,
    }
    record_params(result, "aggregate", call_params)
    record_history(result, "aggregate", params=call_params, source=level_change_source(adata, resolution, columns))
    return result


def _site_counts(frame: pd.DataFrame, codes: np.ndarray, n_groups: int) -> np.ndarray:
    """Distinct fields of view among each group's cells, where site 1 of one well and of the next are different fields."""
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
    """Build the aggregated ``obs``: grouping keys, the per-group tallies, every column constant within a group.

    The ``Metadata_`` prefix is not a survival filter (spec §13.1): any column constant within every group
    is carried, whatever its name; a column that varies within a group is dropped rather than copied.
    """
    if len(columns) == 1:
        obs = pd.DataFrame({columns[0]: np.asarray(keys)})
    else:
        obs = pd.DataFrame({name: keys.get_level_values(position) for position, name in enumerate(columns)})
    obs = obs.reset_index(drop=True)
    for name, values in tallies.items():
        obs[name] = values

    frame = as_frame(adata.obs)
    carried = [column for column in frame.columns if column not in {*columns, *_TALLIES, *tallies}]
    if carried:
        grouped = frame[carried].groupby(codes, observed=True)
        constant = grouped.nunique(dropna=False).le(1).all()
        for column in carried:
            if constant[column]:
                obs[column] = grouped[column].first().to_numpy()
            else:
                get_logger().debug("aggregate dropped non-constant column %s", column)
    return categorize_metadata(obs)


def _record_membership(
    result: AnnData, adata: AnnData, columns: list[str], codes: np.ndarray, keep: np.ndarray
) -> None:
    """Store which source rows went into each kept profile, by stable identity (spec §13.4).

    ``members`` is long-form: one row per contributing source object, its identity columns plus the
    target profile's ``obs`` index. Source rows are referenced by identity, never by position.
    """
    source_resolution = get_resolution(adata, default="object")
    frame = as_frame(adata.obs)
    identity = [column for column in REQUIRED_OBS.get(source_resolution, ()) if column in frame.columns]
    if not identity:  # an aggregate source has no required identity columns; fall back to its grouping.
        identity = [
            column for column in (adata.uns.get("mantispy", {}).get("grouped_by") or []) if column in frame.columns
        ]
    # Map each source row to its profile's row index in the kept output, or -1 when its group was dropped.
    target_row = np.full(len(keep), -1, dtype=np.int64)
    target_row[keep] = np.arange(int(keep.sum()))
    rows = target_row[codes]
    contributing = rows >= 0
    members = frame.loc[contributing, identity].reset_index(drop=True)
    members["Metadata_AggregateRow"] = rows[contributing].astype(str)
    result.uns["mantispy"]["membership"] = {
        "source_resolution": source_resolution,
        "identity_columns": identity,
        "target_column": "Metadata_AggregateRow",
        "members": members,
    }
