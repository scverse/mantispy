"""Plate-layout plots."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from mantispy._core._reduce import get_matrix
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
    """``text`` with its digits read as numbers, so that source_2 sorts before source_10."""
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
        groupby: ``obs`` column the panels are ordered and titled by, such as ``"Metadata_Batch"``, so that the plates of one group sit together. Each panel is still one plate.
        agg: How to combine rows sharing a well: median, mean, max or min.
        ncols: Panels per row.
        share_colorbar: Draw every panel on one color scale with one colorbar, so plates can be compared. Otherwise each panel gets its own.
        ax: Axes to draw into. Only valid together with ``plate``.
        cmap: Matplotlib colormap.
        kwargs: Passed to :meth:`~matplotlib.axes.Axes.imshow`. ``vmin`` and ``vmax`` fix the scale.

    Returns:
        A single :class:`~matplotlib.axes.Axes`, or an array of them for several plates in panel order, each panel labeled with the plate's own well grid.

    Raises:
        ValueError: ``agg`` is not one of ``AGGREGATIONS``, ``ax`` was passed for more than one plate, or ``groupby`` varies within a plate.
        KeyError: ``color`` is neither a feature name nor an ``obs`` column, ``groupby`` is not an ``obs`` column, or ``plate`` is not a plate of ``adata``.
    """
    import matplotlib.pyplot as plt

    if agg not in AGGREGATIONS:
        raise ValueError(f"agg must be one of {AGGREGATIONS}, got {agg!r}")
    values = _values(adata, color)
    obs = as_frame(adata.obs)
    if groupby is not None and groupby not in obs:
        raise KeyError(f"obs has no column {groupby!r}")

    # One (group, plate) per panel, ordered by group and then plate.
    keys = [*([groupby] if groupby else []), "Metadata_Plate"]
    panels = sorted(
        set(obs[keys].astype(str).itertuples(index=False, name=None)), key=lambda row: list(map(_natural, row))
    )
    if len({row[-1] for row in panels}) < len(panels):
        raise ValueError(f"{groupby!r} varies within a plate, so it cannot label one")
    if plate is not None:
        panels = [row for row in panels if row[-1] == plate]
        if not panels:
            raise KeyError(f"obs has no plate {plate!r}")
    if ax is not None and len(panels) > 1:
        raise ValueError("pass plate= when supplying a single ax, or leave ax=None")

    grids = []
    for *_, name in panels:
        mask = (obs["Metadata_Plate"].astype(str) == name).to_numpy()
        wells = obs["Metadata_Well"][mask]
        grid = np.full(plate_grid(wells.unique()), np.nan)
        combined = (
            pd.DataFrame(
                {"row": [well_row(w) for w in wells], "col": [well_col(w) for w in wells], "value": values[mask]}
            )
            .groupby(["row", "col"])["value"]
            .agg(agg)
        )
        grid[combined.index.get_level_values("row"), combined.index.get_level_values("col")] = combined.to_numpy()
        grids.append(grid)
    if share_colorbar:
        kwargs.setdefault("vmin", min(np.nanmin(grid) for grid in grids))
        kwargs.setdefault("vmax", max(np.nanmax(grid) for grid in grids))

    if ax is None:
        width = min(ncols, len(grids))
        height = -(-len(grids) // width)
        _, layout = plt.subplots(height, width, figsize=(4 * width, 3.2 * height), squeeze=False)
        for unused in layout.ravel()[len(grids) :]:
            unused.remove()
        axes = layout.ravel()[: len(grids)]
    else:
        axes = np.array([ax])

    for axis, grid, row in zip(axes, grids, panels, strict=True):
        image = axis.imshow(grid, cmap=cmap, aspect="equal", **kwargs)
        axis.set_title(" · ".join(row), fontsize=9)
        n_rows, n_cols = grid.shape
        axis.set_xticks(columns := range(0, n_cols, max(1, n_cols // 12)))
        axis.set_xticklabels([str(col + 1) for col in columns], fontsize=7)
        axis.set_yticks(rows := range(0, n_rows, max(1, n_rows // 16)))
        axis.set_yticklabels([row_label(row) for row in rows], fontsize=7)
        if not share_colorbar:
            axis.figure.colorbar(image, ax=axis, fraction=0.04, label=color)
    if share_colorbar:
        axes[0].figure.colorbar(image, ax=list(axes), fraction=0.04, label=color)

    return axes[0] if len(axes) == 1 else axes
