"""Image-level quality control, from CellProfiler's MeasureImageQuality columns.

No pixels are read. Everything here works off ``uns["mantispy"]["image_table"]``, which
the reader fills from ``Image.csv``.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence

import numpy as np
import pandas as pd
from anndata import AnnData

from mantispy._core._utils import as_frame, get_logger, inplace_or_copy, report_drop

#: Metrics MeasureImageQuality writes that say something about usable image quality.
DEFAULT_METRICS = ("FocusScore", "PowerLogLogSlope", "PercentMaximal", "PercentMinimal", "Saturation")

METHODS = ("mad", "knn")

#: Default robust-z cutoffs. The "mad" cutoff is higher because its score is the maximum
#: |z| over several metrics, while "knn" scores a single one-sided dissimilarity.
DEFAULT_CUTOFF = {"mad": 5.0, "knn": 3.5}


def _metric_columns(table: pd.DataFrame, metrics: Sequence[str], channel: str | None) -> list[str]:
    columns = [
        column
        for column in table.columns
        if any(f"ImageQuality_{metric}" in column for metric in metrics)
        and (channel is None or column.endswith(f"_{channel}"))
    ]
    if not columns:
        available = sorted({c for c in table.columns if "ImageQuality" in c})
        raise KeyError(
            f"the image table has no ImageQuality columns for metrics={list(metrics)} "
            f"channel={channel!r}. Available: {available[:8]}"
        )
    return columns


def _robust_z(values: np.ndarray) -> np.ndarray:
    median = np.nanmedian(values, axis=0)
    mad = 1.4826 * np.nanmedian(np.abs(values - median), axis=0)
    mad = np.where((mad == 0) | ~np.isfinite(mad), np.nan, mad)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.abs(values - median) / mad


def _lower_half_z(scores: np.ndarray) -> np.ndarray:
    """Robust z for a one-sided, right-skewed score.

    The spread is taken from the values at or below the median, because outliers sit in
    the upper half and would inflate it. A few badly out-of-focus images can otherwise
    raise the threshold enough to hide most of them.
    """
    median = np.median(scores)
    lower = scores[scores <= median]
    spread = 1.4826 * np.median(np.abs(lower - median))
    if spread == 0 or not np.isfinite(spread):
        return np.zeros_like(scores)
    return (scores - median) / spread


def _knn_dissimilarity(values: np.ndarray, k: int) -> np.ndarray:
    """Mean distance to the k nearest images in standardized metric space."""
    from sklearn.neighbors import NearestNeighbors

    spread = np.nanstd(values, axis=0)
    spread = np.where((spread == 0) | ~np.isfinite(spread), 1.0, spread)
    centered = np.nan_to_num((values - np.nanmean(values, axis=0)) / spread, nan=0.0, posinf=0.0, neginf=0.0)
    n_neighbors = min(k + 1, len(centered))
    distances, _ = NearestNeighbors(n_neighbors=n_neighbors).fit(centered).kneighbors(centered)
    return distances[:, 1:].mean(axis=1)


@inplace_or_copy()
def image_qc(
    adata: AnnData,
    metrics: Sequence[str] = DEFAULT_METRICS,
    channel: str | None = None,
    method: str = "mad",
    threshold: float | str = "auto",
    by: str | None = "Metadata_Plate",
    k: int = 15,
    copy: bool = False,
) -> AnnData | None:
    """Flag low-quality images and broadcast the verdict onto their cells.

    Args:
        adata: Object carrying ``uns["mantispy"]["image_table"]`` and
            ``obs["Metadata_ImageNumber"]``.
        metrics: Which MeasureImageQuality metrics to use.
        channel: Restrict to one channel's metrics. ``None`` uses every channel present.
        method: ``"mad"`` flags an image when any metric is an outlier within its ``by`` group.
            ``"knn"`` flags images that sit far from their neighbors in the standardized
            metric space, which catches unusual combinations of metrics that a per-metric
            rule misses.
        threshold: ``"auto"`` flags a score above the method's default robust-z cutoff
            (``DEFAULT_CUTOFF``). A float thresholds the raw score instead.
        by: Compute thresholds within each group of this column, normally the plate.
            ``None`` pools every image, which flags every image on a dim plate and misses a
            blurred image on a bright one.
        k: Neighbors for ``method="knn"``.
        copy: Return a modified copy instead of mutating in place.

    Returns:
        ``None``, or the modified copy. Writes ``uns["mantispy"]["image_qc"]`` (the image
        table plus ``qc_image_score`` and ``qc_image_pass``) and broadcasts
        ``obs["qc_image_pass"]``.

    Raises:
        KeyError: If the image table is missing, holds none of the requested metrics, or lacks
            the ``by`` column.
        ValueError: If ``method`` is unknown, or some images have no value in ``by``.
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")

    store = adata.uns.setdefault("mantispy", {})
    if "image_table" not in store:
        raise KeyError(
            "uns['mantispy']['image_table'] is missing; read the data with "
            "mt.io.read_profiles on an ExportToSpreadsheet directory, which fills it from Image.csv"
        )
    table = pd.DataFrame(store["image_table"]).copy()

    columns = _metric_columns(table, metrics, channel)
    values = table[columns].to_numpy(dtype=np.float64)

    if by is not None and by not in table.columns:
        raise KeyError(
            f"the image table has no {by!r} column to compute thresholds within. Pass by=None to "
            "pool every image, which flags all images on a plate that is dimmer than the rest."
        )
    groups = table[by].to_numpy() if by is not None else np.zeros(len(table), dtype=int)
    if by is not None and pd.isna(groups).any():
        # `groups == group` is False for NaN, so those images would never be scored and would
        # pass. Images where segmentation found nothing are the ones likely to lack a plate.
        raise ValueError(
            f"{int(pd.isna(groups).sum())} of {len(table)} images have no {by!r}, so they cannot be "
            "thresholded within their group. Fill the column, drop those images, or pass by=None "
            "to pool every image."
        )

    score = np.zeros(len(table))
    passed = np.ones(len(table), dtype=bool)
    for group in pd.unique(groups):
        rows = np.flatnonzero(groups == group)
        block = values[rows]
        if method == "mad":
            block_score = np.nanmax(np.nan_to_num(_robust_z(block), nan=0.0, posinf=0.0), axis=1)
            cutoff = DEFAULT_CUTOFF["mad"] if threshold == "auto" else float(threshold)
        else:
            block_score = _knn_dissimilarity(block, k)
            if threshold == "auto":
                block_score = _lower_half_z(block_score)
                cutoff = DEFAULT_CUTOFF["knn"]
            else:
                cutoff = float(threshold)
        score[rows] = block_score
        passed[rows] = block_score <= cutoff

    table["qc_image_score"] = score
    table["qc_image_pass"] = passed
    store["image_qc"] = table

    if "Metadata_ImageNumber" not in adata.obs:
        raise KeyError("obs has no 'Metadata_ImageNumber' column to broadcast image QC onto")
    broadcast = adata.obs["Metadata_ImageNumber"].map(table["qc_image_pass"])
    n_missing = int(broadcast.isna().sum())
    if n_missing:
        warnings.warn(
            f"{n_missing} cells have an ImageNumber not present in the image table; treating them as passing",
            UserWarning,
            stacklevel=3,
        )
    adata.obs["qc_image_pass"] = broadcast.fillna(True).to_numpy(dtype=bool)
    get_logger().info("image_qc(%s) flagged %d of %d images", method, int((~passed).sum()), len(table))
    return None


@inplace_or_copy()
def filter_images(adata: AnnData, copy: bool = False) -> AnnData | None:
    """Drop every cell belonging to an image that failed :func:`~mantispy.pp.image_qc`."""
    if "qc_image_pass" not in adata.obs:
        raise KeyError("obs has no 'qc_image_pass'; run mt.pp.image_qc first")
    keep = as_frame(adata.obs)["qc_image_pass"].to_numpy(dtype=bool)
    report_drop("cells", int((~keep).sum()), adata.n_obs, remedy="loosen the image QC thresholds")
    adata._inplace_subset_obs(keep)
    return None
