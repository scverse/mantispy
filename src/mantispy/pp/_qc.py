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
from mantispy._core.features import var_blocklist_hits
from mantispy._core.frames import as_frame
from mantispy._core.logging import get_logger, report_drop
from mantispy._core.mutation import inplace_or_copy

AREA_Z_CUTOFF = 5.0
QC_METRICS = ("nan", "border", "area_outlier", "variance", "n_unique")


def _n_unique(X: np.ndarray, missing: np.ndarray) -> np.ndarray:
    """Distinct finite values per feature, a column block at a time.

    Missing values sort to the end, so each column's distinct count is the number of value changes in its finite prefix.
    """
    n_obs, n_vars = X.shape
    out = np.empty(n_vars, dtype=np.int32)
    width = max(int(CHUNK_BYTES / max(n_obs, 1) / X.itemsize), 1)
    position = None
    for start in range(0, n_vars, width):
        columns = slice(start, start + width)
        block = np.sort(X[:, columns], axis=0)
        finite = n_obs - missing[:, columns].sum(axis=0)
        changed = block[1:] != block[:-1]
        if (finite != n_obs).any():
            # A column with missing values keeps its finite prefix only; the rest sorted to the end and is not
            # counted. Columns with no gaps need no mask, which is the common case and avoids the position array.
            if position is None:
                position = np.arange(1, n_obs)[:, None]
            changed &= position < finite
        out[columns] = changed.sum(axis=0) + (finite > 0)
    return out


@inplace_or_copy()
def calculate_qc_metrics(
    adata: AnnData,
    image_shape: tuple[int, int] | None = None,
    border_margin: int = 10,
    max_nan_fraction: float = 0.5,
    area_feature: str | None = None,
    qc_metrics: Sequence[str] = ("nan", "border", "area_outlier"),
    copy: bool = False,
) -> AnnData | None:
    """Compute per-cell and per-feature QC metrics.

    Args:
        adata: Object to annotate.
        image_shape: ``(height, width)`` of a field of view.
            Without it, and without ``Metadata_Center_X``/``_Y`` in ``obs``, the border flag stays ``False``.
        border_margin: Distance from the image edge, in pixels, inside which a cell is a border cell.
        max_nan_fraction: Largest fraction of missing features a cell may have before it is flagged.
            Partial NaN is routine in CellProfiler output (Zernike and RadialDistribution features are undefined for small objects), so requiring no missing values would flag almost every cell.
        area_feature: The ``var`` name of the area measurement to score for outliers.
            By default the columns whose parsed ``feature`` is ``Area`` are used; pass this for a table whose names do not parse (``mt.io.stamp`` could not fill ``var["feature"]``).
        qc_metrics: Metrics to compute.
            ``"nan"`` writes ``obs["qc_n_nan_features"]``, ``obs["qc_nan_fraction"]`` and ``var["qc_n_nan"]``; ``"border"`` writes ``obs["qc_is_border"]``; ``"area_outlier"`` writes ``obs["qc_area_outlier"]``; ``"variance"`` and ``"n_unique"`` write ``var["qc_variance"]`` and ``var["qc_n_unique"]``.
            The per-feature ``"variance"`` and ``"n_unique"`` are left out by default: :func:`filter_cells` reads neither, and ``"n_unique"`` sorts every feature, which dominates the runtime on large single-cell objects.
        copy: Return a modified copy instead of mutating in place.

    Returns:
        ``None``, or the modified copy.
        Writes the columns of each metric in `qc_metrics`, and ``obs["qc_fail"]``.
        Every ``qc_*`` boolean is ``True`` for a problem, matching the ecosystem convention; ``qc_fail`` is the union of the computed flags and of a missing fraction above `max_nan_fraction`, and is the flag :func:`filter_cells` reads.
        When no area column is found and `area_feature` is not given, ``qc_area_outlier`` is skipped (left all ``False``) with a warning and the other metrics are still computed.

    Raises:
        KeyError: `area_feature` is given but is not a ``var`` name.
        ValueError: `qc_metrics` names an unknown metric.
    """
    unknown = set(qc_metrics) - set(QC_METRICS)
    if unknown:
        raise ValueError(f"qc_metrics: unknown {sorted(unknown)}, choose from {list(QC_METRICS)}")

    X = get_matrix(adata)
    fail = np.zeros(adata.n_obs, dtype=bool)
    if "nan" in qc_metrics:
        missing = np.isnan(X)
        nan_fraction = missing.mean(axis=1)
        adata.obs["qc_n_nan_features"] = missing.sum(axis=1).astype(np.int32)
        adata.obs["qc_nan_fraction"] = nan_fraction
        adata.var["qc_n_nan"] = missing.sum(axis=0).astype(np.int32)
        fail |= nan_fraction > max_nan_fraction
    if "border" in qc_metrics:
        adata.obs["qc_is_border"] = border = _border_flag(adata, image_shape, border_margin)
        fail |= border
    if "area_outlier" in qc_metrics:
        area = _area_features(adata, area_feature)
        if len(area):
            area_outlier = _area_outlier_flag(adata, X, area)
        else:
            get_logger().info(
                "calculate_qc_metrics: no area feature in var, so qc_area_outlier is all False",
                deep="pass area_feature= to name the area column",
            )
            area_outlier = np.zeros(adata.n_obs, dtype=bool)
        adata.obs["qc_area_outlier"] = area_outlier
        fail |= area_outlier
    if "variance" in qc_metrics:
        adata.var["qc_variance"] = nanvar(X)
    if "n_unique" in qc_metrics:
        adata.var["qc_n_unique"] = _n_unique(X, missing if "nan" in qc_metrics else np.isnan(X))

    adata.obs["qc_fail"] = fail
    return None


def _border_flag(adata: AnnData, image_shape: tuple[int, int] | None, margin: int) -> np.ndarray:
    """Cells whose centroid sits within ``margin`` pixels of the field edge."""
    coordinates = {"Metadata_Center_X", "Metadata_Center_Y"} <= set(adata.obs.columns)
    if image_shape is None or not coordinates:
        if image_shape is not None:
            get_logger().warning(
                "calculate_qc_metrics: obs has no Metadata_Center_X/Y, so no cell is flagged as a border cell"
            )
        return np.zeros(adata.n_obs, dtype=bool)
    height, width = image_shape
    x = as_frame(adata.obs)["Metadata_Center_X"].to_numpy(dtype=float)
    y = as_frame(adata.obs)["Metadata_Center_Y"].to_numpy(dtype=float)
    return (x < margin) | (y < margin) | (x > width - margin) | (y > height - margin)


def _area_features(adata: AnnData, area_feature: str | None = None) -> pd.Index:
    """The ``var`` names that measure an area: `area_feature` if given, else those whose ``feature`` is ``Area``.

    Returns an empty index when neither is available, leaving the caller to skip the area metric.

    Raises:
        KeyError: `area_feature` is given but is not a ``var`` name.
    """
    if area_feature is not None:
        if area_feature not in adata.var_names:
            raise KeyError(f"area_feature={area_feature!r} is not a var name")
        return pd.Index([area_feature])
    named = adata.var["feature"].astype(str).eq("Area") if "feature" in adata.var else []
    return adata.var_names[named]


def _area_outlier_flag(adata: AnnData, X: np.ndarray, area: pd.Index) -> np.ndarray:
    """Cells whose area is more than :data:`AREA_Z_CUTOFF` robust SDs from the plate median.

    Every compartment that measured an area is scored within its own plate and the flags are OR-ed, so a cell is an outlier when any of its areas is.
    """
    if "Metadata_Plate" not in adata.obs:
        return np.zeros(adata.n_obs, dtype=bool)

    columns = X[:, np.array([adata.var_names.get_loc(name) for name in area])]
    codes, keys = group_codes(adata, "Metadata_Plate")
    median, mad = grouped_median_spread(columns, codes, len(keys), MAD)

    with np.errstate(invalid="ignore", divide="ignore"):
        z = np.abs(columns - median[codes]) / (MAD_TO_SIGMA * mad[codes])
    # posinf=0.0: a quantized Area column can have zero MAD, and an infinite z would flag the whole plate.
    return (np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0) > AREA_Z_CUTOFF).any(axis=1)


@inplace_or_copy()
def filter_cells(
    adata: AnnData,
    min_cells_per_well: int = 50,
    require_qc: bool = True,
    copy: bool = False,
) -> AnnData | None:
    """Drop cells that fail QC or sit in under-populated wells.

    Args:
        adata: Object to filter.
        min_cells_per_well: Wells with fewer cells than this are dropped entirely, counted over the cells that survive the other checks in this call so that the cells being dropped cannot hold a well above the floor.
            ``0`` disables the check.
        require_qc: Also drop cells flagged by ``obs["qc_fail"]``, which :func:`calculate_qc_metrics` writes.
        copy: Return a filtered copy instead of filtering in place.

    Returns:
        ``None``, or the filtered copy.
        Subsets ``obs`` to the surviving cells, and warns when that leaves none.

    Raises:
        KeyError: If ``require_qc`` is set but ``obs`` has no ``qc_fail`` column.
    """
    keep = np.ones(adata.n_obs, dtype=bool)
    if require_qc:
        if "qc_fail" not in adata.obs:
            raise KeyError("obs has no 'qc_fail'; run mt.pp.calculate_qc_metrics first, or pass require_qc=False")
        keep &= ~as_frame(adata.obs)["qc_fail"].to_numpy(dtype=bool)
    if min_cells_per_well > 0:
        codes, keys = group_codes(adata, ["Metadata_Plate", "Metadata_Well"])
        # Count the survivors, so one call agrees with running the two checks as separate calls.
        keep &= np.bincount(codes[keep], minlength=len(keys))[codes] >= min_cells_per_well

    dropped = int((~keep).sum())
    if not dropped:
        # Nothing fails the checks, so the subset would copy the whole matrix only to return the same rows.
        return None
    get_logger().info("filter_cells dropped %d of %d cells (%.1f%%)", dropped, adata.n_obs, 100 * dropped / adata.n_obs)
    if not keep.any():
        get_logger().warning("filter_cells: removed every cell", deep="check qc_fail and min_cells_per_well")
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
        min_variance: Drop features whose variance is at or below this, as :func:`~mantispy.pp.feature_select` and sklearn's ``VarianceThreshold`` do.
            ``0`` disables the check.
        blocklist: ``"default"`` for the bundled CellProfiler blocklist, an explicit list of names, or ``None`` to skip.
            Matched against the current names and against ``var["original_name"]``, so it works either side of :func:`~mantispy.pp.standardize_feature_names`.
        copy: Return a filtered copy instead of filtering in place.

    Returns:
        ``None``, or the filtered copy.
        Subsets ``var`` to the surviving features, and reports how many were dropped.
    """
    X = get_matrix(adata)
    keep = np.ones(adata.n_vars, dtype=bool)
    if drop_nan:
        keep &= ~np.isnan(X).all(axis=0)
    if min_variance > 0:
        keep &= np.nan_to_num(nanvar(X), nan=0.0, posinf=0.0) > min_variance
    if blocklist is not None:
        keep &= ~var_blocklist_hits(adata, blocklist)

    dropped = int((~keep).sum())
    report_drop("features", dropped, adata.n_vars, remedy="check drop_nan, min_variance and blocklist")
    adata._inplace_subset_var(keep)
    return None
