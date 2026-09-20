"""Plate-layout plots."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from mantispy._core._reduce import get_matrix, group_codes, group_offsets
from mantispy._core.frames import as_frame
from mantispy._core.plate import plate_grid, row_label, well_col, well_row

if TYPE_CHECKING:
    from anndata import AnnData
    from matplotlib.axes import Axes

#: How to combine several cells or sites falling in the same well. pandas skips NaN.
AGGREGATIONS = ("median", "mean", "max", "min")


def _values(adata: AnnData, color: str) -> np.ndarray:
    """The per-row quantity to draw, from either a feature or an ``obs`` column."""
    if color in adata.var_names:
        return get_matrix(adata)[:, adata.var_names.get_loc(color)].astype(float)
    if color in adata.obs:
        return as_frame(adata.obs)[color].to_numpy(dtype=float)
    raise KeyError(f"{color!r} is neither a feature name nor an obs column")


def _natural(text: str) -> list[int | str]:
    """Sort key reading digit runs as numbers: source_2 before source_10."""
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", text)]


def plate(
    adata: AnnData,
    color: str,
    plate: str | None = None,
    groupby: str | None = None,
    agg: str = "median",
    ncols: int = 4,
    share_colorbar: bool = True,
    ax: Axes | None = None,
    cmap: str = "viridis",
    **kwargs: Any,
) -> Axes | np.ndarray:
    """Well-grid heatmap of ``color``, one panel per plate.

    Args:
        adata: Object to draw. Works at cell or well resolution; several rows landing in the same well are combined with ``agg``.
        color: A feature name or an ``obs`` column.
        plate: Draw only this plate. By default every plate gets a panel.
        groupby: ``obs`` column, constant per plate, to order and title the panels by, such as ``"Metadata_Batch"``.
        agg: How to combine rows sharing a well: median, mean, max or min.
        ncols: Panels per row.
        share_colorbar: One color scale and colorbar for all panels, instead of one per panel.
        ax: Axes to draw into. Only valid together with ``plate``.
        cmap: Matplotlib colormap.
        kwargs: Passed to :meth:`~matplotlib.axes.Axes.imshow`. ``vmin``, ``vmax`` or ``norm`` set the scale.

    Returns:
        A single :class:`~matplotlib.axes.Axes`, or an array of them for several plates in panel order, each panel labeled with the plate's own well grid.

    Raises:
        ValueError: ``agg`` is not one of ``AGGREGATIONS``, ``ax`` was passed for more than one plate, ``groupby`` varies within a drawn plate, or it or ``Metadata_Plate`` has missing values.
        KeyError: ``color`` is neither a feature name nor an ``obs`` column, ``groupby`` is not an ``obs`` column, or ``plate`` is not a plate of ``adata``.
    """
    import matplotlib.pyplot as plt

    if agg not in AGGREGATIONS:
        raise ValueError(f"agg must be one of {AGGREGATIONS}, got {agg!r}")
    values = _values(adata, color)
    codes, keys = group_codes(adata, [groupby, "Metadata_Plate"] if groupby else "Metadata_Plate")
    panels = [key if isinstance(key, tuple) else (key,) for key in keys]
    drawn = [index for index, panel in enumerate(panels) if plate is None or str(panel[-1]) == str(plate)]
    if not drawn:
        raise KeyError(f"obs has no plate {plate!r}")
    if len({panels[index][-1] for index in drawn}) < len(drawn):
        raise ValueError(f"{groupby!r} varies within a plate, so it cannot label one")
    if ax is not None and len(drawn) > 1:
        raise ValueError("pass plate= when supplying a single ax, or leave ax=None")
    drawn.sort(key=lambda index: [_natural(str(part)) for part in panels[index]])

    # Each distinct well is parsed once, and each plate reads only its own rows.
    wells = pd.Categorical(as_frame(adata.obs)["Metadata_Well"])
    positions = np.array([(well_row(well), well_col(well)) for well in wells.categories])[wells.codes]
    order, offsets = group_offsets(codes, len(keys))
    grids = []
    for index in drawn:
        rows = order[offsets[index] : offsets[index + 1]]
        grid = np.full(plate_grid(wells.categories[np.unique(wells.codes[rows])]), np.nan)
        combined = pd.Series(values[rows]).groupby([positions[rows, 0], positions[rows, 1]]).agg(agg)
        grid[combined.index.get_level_values(0), combined.index.get_level_values(1)] = combined.to_numpy()
        grids.append(grid)
    if share_colorbar and "norm" not in kwargs:
        everything = np.concatenate([grid.ravel() for grid in grids])
        kwargs.setdefault("vmin", np.nanmin(everything))
        kwargs.setdefault("vmax", np.nanmax(everything))

    if ax is None:
        width = min(ncols, len(grids))
        height = -(-len(grids) // width)
        figure = plt.figure(figsize=(4 * width, 3.2 * height))
        axes = np.array([figure.add_subplot(height, width, number) for number in range(1, len(grids) + 1)])
    else:
        axes = np.array([ax])

    for axis, grid, index in zip(axes, grids, drawn, strict=True):
        image = axis.imshow(grid, cmap=cmap, aspect="equal", **kwargs)
        axis.set_title(" · ".join(map(str, panels[index])), fontsize=9)
        n_rows, n_cols = grid.shape
        xticks, yticks = range(0, n_cols, max(1, n_cols // 12)), range(0, n_rows, max(1, n_rows // 16))
        axis.set_xticks(xticks, [str(col + 1) for col in xticks], fontsize=7)
        axis.set_yticks(yticks, [row_label(row) for row in yticks], fontsize=7)
        if not share_colorbar:
            axis.figure.colorbar(image, ax=axis, fraction=0.04, label=color)
    if share_colorbar:
        axes[0].figure.colorbar(image, ax=axes if len(axes) > 1 else axes[0], fraction=0.04, label=color)

    return axes[0] if len(axes) == 1 else axes
