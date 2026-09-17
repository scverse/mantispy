"""Plate-layout plots."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mantispy._core._reduce import get_matrix
from mantispy._core.frames import as_frame
from mantispy._core.plate import PLATE_FORMATS, detect_plate_format, row_label, well_col, well_row

if TYPE_CHECKING:
    from anndata import AnnData

#: How to combine several cells or sites falling in the same well. pandas skips NaN.
AGGREGATIONS = ("median", "mean", "max", "min")


def _values(adata: AnnData, color: str) -> np.ndarray:
    """The per-row quantity to draw, from either a feature or an ``obs`` column."""
    if color in adata.var_names:
        return get_matrix(adata)[:, adata.var_names.get_loc(color)].astype(float)
    if color in adata.obs:
        return as_frame(adata.obs)[color].to_numpy(dtype=float)
    raise KeyError(f"{color!r} is neither a feature name nor an obs column")


def plate(
    adata: AnnData,
    color: str,
    plate: str | None = None,
    agg: str = "median",
    ax: plt.Axes | None = None,
    cmap: str = "viridis",
    **kwargs,
):
    """Well-grid heatmap of ``color``, one panel per plate.

    Args:
        adata: Object to draw. Works at cell or well resolution; several rows landing in the
            same well are combined with ``agg``.
        color: A feature name or an ``obs`` column.
        plate: Draw only this plate. By default every plate gets a panel.
        agg: How to combine rows sharing a well: median, mean, max or min.
        ax: Axes to draw into. Only valid together with ``plate``.
        cmap: Matplotlib colormap.
        kwargs: Passed to :meth:`~matplotlib.axes.Axes.imshow`.

    Returns:
        A single :class:`~matplotlib.axes.Axes`, or an array of them for several plates.
    """
    if agg not in AGGREGATIONS:
        raise ValueError(f"agg must be one of {AGGREGATIONS}, got {agg!r}")
    values = _values(adata, color)

    plates = [plate] if plate is not None else sorted(adata.obs["Metadata_Plate"].unique())
    if ax is not None and len(plates) > 1:
        raise ValueError("pass plate= when supplying a single ax, or leave ax=None")
    if ax is None:
        _, axes = plt.subplots(1, len(plates), figsize=(5 * len(plates), 4), squeeze=False)
        axes = axes.ravel()
    else:
        axes = np.array([ax])

    size = detect_plate_format(adata.obs["Metadata_Well"].unique())
    n_rows, n_cols = PLATE_FORMATS[size]

    for axis, name in zip(axes, plates, strict=True):
        mask = (adata.obs["Metadata_Plate"] == name).to_numpy()
        frame = pd.DataFrame(
            {
                "row": [well_row(well) for well in adata.obs["Metadata_Well"][mask]],
                "col": [well_col(well) for well in adata.obs["Metadata_Well"][mask]],
                "value": values[mask],
            }
        )
        grid = np.full((n_rows, n_cols), np.nan)
        combined = frame.groupby(["row", "col"])["value"].agg(agg)
        grid[
            combined.index.get_level_values("row").to_numpy(dtype=int),
            combined.index.get_level_values("col").to_numpy(dtype=int),
        ] = combined.to_numpy()

        image = axis.imshow(grid, cmap=cmap, aspect="equal", **kwargs)
        axis.set_title(f"{name}\n{color}", fontsize=9)
        step = max(1, n_cols // 12)
        axis.set_xticks(range(0, n_cols, step))
        axis.set_xticklabels([str(col + 1) for col in range(0, n_cols, step)], fontsize=7)
        axis.set_yticks(range(n_rows))
        axis.set_yticklabels([row_label(row) for row in range(n_rows)], fontsize=7)
        axis.figure.colorbar(image, ax=axis, fraction=0.04)

    return axes[0] if len(axes) == 1 else axes
