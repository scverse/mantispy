"""Basic quality control metrics and filters."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from anndata import AnnData

from mantispy._core._corr import CHUNK_BYTES
from mantispy._core._numba import MAD, grouped_median_spread
from mantispy._core._reduce import get_matrix, group_codes
from mantispy._core._stats import MAD_TO_SIGMA, nanvar
from mantispy._core.features import blocklist_hits
from mantispy._core.frames import as_frame
from mantispy._core.logging import get_logger, report_drop
from mantispy._core.mutation import inplace_or_copy

#: Robust z above which a cell's area is called an outlier.
AREA_Z_CUTOFF = 5.0


def _n_unique(X: np.ndarray, missing: np.ndarray) -> np.ndarray:
    """Distinct finite values per feature, a column block at a time.

    Sorting a block of columns in one call avoids a Python-level ``np.unique`` per feature, which took nine seconds on 50 640 JUMP wells by 3634 features.
    Missing values sort to the end, so each column's distinct count is the number of value changes in its finite prefix.
    """
    n_obs, n_vars = X.shape
    out = np.empty(n_vars, dtype=np.int32)
    width = max(int(CHUNK_BYTES / max(n_obs, 1) / X.itemsize), 1)
    position = np.arange(1, n_obs)[:, None]
    for start in range(0, n_vars, width):
        columns = slice(start, start + width)
        block = np.sort(X[:, columns], axis=0)
        finite = n_obs - missing[:, columns].sum(axis=0)
        changed = (block[1:] != block[:-1]) & (position < finite)
        out[columns] = changed.sum(axis=0) + (finite > 0)
    return out


@inplace_or_copy()
def calculate_qc_metrics(
    adata: AnnData,
    image_shape: tuple[int, int] | None = None,
    border_margin: int = 10,
    max_nan_fraction: float = 0.5,
    copy: bool = False,
) -> AnnData | None:
    """Compute per-cell and per-feature QC metrics.

    Args:
        adata: Object to annotate.
        image_shape: ``(height, width)`` of a field of view. Without it, and without ``Metadata_Center_X``/``_Y`` in ``obs``, the border flag stays ``False``.
        border_margin: Distance from the image edge, in pixels, inside which a cell is a border cell.
        max_nan_fraction: Largest fraction of missing features a cell may have and still pass. Partial NaN is routine in CellProfiler output (Zernike and RadialDistribution features are undefined for small objects), so requiring no missing values would fail almost every cell.
        copy: Return a modified copy instead of mutating in place.

    Returns:
        ``None``, or the modified copy. Writes the ``obs`` columns ``qc_n_nan_features``, ``qc_nan_fraction``, ``qc_is_border``, ``qc_area_outlier`` and ``qc_pass``, and the ``var`` columns ``qc_n_nan``, ``qc_variance`` and ``qc_n_unique``.

    Raises:
        KeyError: If no column in ``var`` names an area, so ``qc_area_outlier`` cannot be scored and ``qc_pass`` would be an ``and`` over one check fewer than it claims.
    """
    # First, before get_matrix densifies and before @inplace_or_copy's duplicate is touched:
    # a call that is going to be rejected should not read the matrix or copy the object.
    _area_features(adata)

    X = get_matrix(adata)
    missing = np.isnan(X)
    nan_fraction = missing.mean(axis=1)
    border = _border_flag(adata, image_shape, border_margin)
    area_outlier = _area_outlier_flag(adata, X)

    adata.obs["qc_n_nan_features"] = missing.sum(axis=1).astype(np.int32)
    adata.obs["qc_nan_fraction"] = nan_fraction
    adata.var["qc_n_nan"] = missing.sum(axis=0).astype(np.int32)
    adata.var["qc_variance"] = nanvar(X)
    adata.var["qc_n_unique"] = _n_unique(X, missing)

    adata.obs["qc_is_border"] = border
    adata.obs["qc_area_outlier"] = area_outlier
    adata.obs["qc_pass"] = ~border & ~area_outlier & (nan_fraction <= max_nan_fraction)
    return None


def _border_flag(adata: AnnData, image_shape: tuple[int, int] | None, margin: int) -> np.ndarray:
    """Cells whose centroid sits within ``margin`` pixels of the field edge."""
    coordinates = {"Metadata_Center_X", "Metadata_Center_Y"} <= set(adata.obs.columns)
    if image_shape is None or not coordinates:
        if image_shape is not None:
            get_logger().warning(
                "image_shape was given but obs has no Metadata_Center_X/Y, so no cell can be flagged as a border cell"
            )
        return np.zeros(adata.n_obs, dtype=bool)
    height, width = image_shape
    x = as_frame(adata.obs)["Metadata_Center_X"].to_numpy(dtype=float)
    y = as_frame(adata.obs)["Metadata_Center_Y"].to_numpy(dtype=float)
    return (x < margin) | (y < margin) | (x > width - margin) | (y > height - margin)


def _area_features(adata: AnnData) -> pd.Index:
    """The ``var`` names that measure an area, refusing an object where none do.

    ``qc_pass`` is an ``and`` over its checks, so one that could not run would weaken it silently.

    Raises:
        KeyError: No column in ``var`` names an area.
    """
    named = adata.var["feature"].astype(str).eq("Area") if "feature" in adata.var else []
    area = adata.var_names[named]
    if not len(area):
        raise KeyError(
            "no column in var names an area, and qc_area_outlier has nothing to score. var's 'feature' "
            "column is written by mt.io.read_profiles and built by mantispy._core.features."
            "parse_feature_names for a var table made by hand; mt.io.stamp supplies it empty, which "
            "names no area either. An object with no area measurement — an embedding, an "
            "Intensity-only export, or anything from tl.feature_signature — has no cell-level QC to run."
        )
    return area


def _area_outlier_flag(adata: AnnData, X: np.ndarray) -> np.ndarray:
    """Cells whose area is more than :data:`AREA_Z_CUTOFF` robust SDs from the plate median.

    Every compartment that measured an area is scored within its own plate and the flags are OR-ed, so a cell is an outlier when any of its areas is.
    Scoring only the first matching column made the flag, and so ``qc_pass``, depend on the order of ``var``.

    The area columns come from :func:`_area_features`, which :func:`calculate_qc_metrics` has already
    called, so reaching here means at least one column names an area.
    """
    area = _area_features(adata)
    if "Metadata_Plate" not in adata.obs:
        return np.zeros(adata.n_obs, dtype=bool)
    if len(area) > 1:
        get_logger().info("qc_area_outlier flags a cell outlying in any of %s", list(area))

    columns = X[:, np.array([adata.var_names.get_loc(name) for name in area])]
    codes, keys = group_codes(adata, "Metadata_Plate")
    median, mad = grouped_median_spread(columns, codes, len(keys), MAD)

    with np.errstate(invalid="ignore", divide="ignore"):
        z = np.abs(columns - median[codes]) / (MAD_TO_SIGMA * mad[codes])
    # A quantized Area column can have zero MAD, which makes z infinite; posinf=0.0 stops
    # nan_to_num from turning that into 1.8e308 and flagging the whole plate.
    return (np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0) > AREA_Z_CUTOFF).any(axis=1)


@inplace_or_copy()
def filter_cells(
    adata: AnnData,
    min_cells_per_well: int = 50,
    qc_pass: bool = True,
    copy: bool = False,
) -> AnnData | None:
    """Drop cells that fail QC or sit in under-populated wells.

    Args:
        adata: Object to filter.
        min_cells_per_well: Wells with fewer cells than this are dropped entirely, counted over the cells that survive the other checks in this call so that the cells being dropped cannot hold a well above the floor. ``0`` disables the check.
        qc_pass: Also require ``obs["qc_pass"]``, which :func:`calculate_qc_metrics` writes.
        copy: Return a filtered copy instead of filtering in place.

    Returns:
        ``None``, or the filtered copy. Subsets ``obs`` to the surviving cells, and warns when that leaves none.

    Raises:
        KeyError: If ``qc_pass`` is requested but ``obs`` has no such column.
    """
    keep = np.ones(adata.n_obs, dtype=bool)
    if qc_pass:
        if "qc_pass" not in adata.obs:
            raise KeyError("obs has no 'qc_pass'; run mt.pp.calculate_qc_metrics first, or pass qc_pass=False")
        keep &= as_frame(adata.obs)["qc_pass"].to_numpy(dtype=bool)
    if min_cells_per_well > 0:
        codes, keys = group_codes(adata, ["Metadata_Plate", "Metadata_Well"])
        # Count the survivors, not every cell in the well: counting the cells this call is about
        # to drop left a well of 60 cells with 12 passing above a floor of 50, and tl.aggregate
        # then built its profile from 12 cells. Running the two checks as separate calls dropped
        # that well, so the two paths disagreed.
        keep &= np.bincount(codes[keep], minlength=len(keys))[codes] >= min_cells_per_well

    dropped = int((~keep).sum())
    if dropped:
        get_logger().info(
            "filter_cells dropped %d of %d cells (%.1f%%)", dropped, adata.n_obs, 100 * dropped / adata.n_obs
        )
    if not keep.any():
        get_logger().warning("filter_cells removed every cell; check qc_pass and min_cells_per_well")
    adata._inplace_subset_obs(keep)
    return None


@inplace_or_copy()
def filter_features(
    adata: AnnData,
    drop_nan: bool = True,
    min_variance: float = 0.0,
    blocklist: str | Sequence[str] | None = "default",
    copy: bool = False,
) -> AnnData | None:
    """Drop all-NaN, low-variance and blocklisted features.

    Args:
        adata: Object to filter.
        drop_nan: Drop features that are missing everywhere.
        min_variance: Drop features whose variance is at or below this, as :func:`~mantispy.pp.feature_select` and sklearn's ``VarianceThreshold`` do. ``0`` disables the check.
        blocklist: ``"default"`` for the bundled CellProfiler blocklist, an explicit list of names, or ``None`` to skip. Matched against the current names and against ``var["original_name"]``, so it works either side of :func:`~mantispy.pp.standardize_feature_names`.
        copy: Return a filtered copy instead of filtering in place.

    Returns:
        ``None``, or the filtered copy. Subsets ``var`` to the surviving features, and reports how many were dropped.
    """
    X = get_matrix(adata)
    keep = np.ones(adata.n_vars, dtype=bool)
    if drop_nan:
        keep &= ~np.isnan(X).all(axis=0)
    if min_variance > 0:
        # `>` matches pp.feature_select's variance_threshold and sklearn's VarianceThreshold.
        keep &= np.nan_to_num(nanvar(X), nan=0.0, posinf=0.0) > min_variance
    if blocklist is not None:
        # Also check var["original_name"], because pp.standardize_feature_names rewrites names
        # into a grammar no blocklist entry matches.
        names = [adata.var_names.to_numpy()]
        if "original_name" in adata.var:
            names.append(adata.var["original_name"].astype(str).to_numpy())
        keep &= ~blocklist_hits(names, blocklist)

    dropped = int((~keep).sum())
    report_drop("features", dropped, adata.n_vars, remedy="check drop_nan, min_variance and blocklist")
    adata._inplace_subset_var(keep)
    return None
