"""The feature-family signature as a picture."""

from __future__ import annotations

import numpy as np
import pandas as pd
from anndata import AnnData
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from scipy.cluster import hierarchy
from scipy.spatial import distance

from mantispy._core._reduce import get_matrix
from mantispy._core.frames import as_frame


def feature_signature(
    adata: AnnData,
    groupby: str | None = None,
    top: int | None = 40,
    cluster: bool = True,
    cmap: str = "RdBu_r",
    figsize: tuple[float, float] | None = None,
    ax: Axes | None = None,
) -> Axes:
    """Heatmap of perturbations by feature families.

    Args:
        adata: The output of :func:`~mantispy.tl.feature_signature`.
        groupby: ``obs`` column to average rows within, instead of showing one row per perturbation. ``"Metadata_MOA"``, for example, gives one row per mechanism.
        top: Show only this many rows, those with the largest absolute value. ``None`` shows all of them. Ignored when ``groupby`` is given.
        cluster: Order rows and columns by hierarchical clustering, so families that move together are adjacent. Otherwise the object's order is kept.
        cmap: Diverging colormap, centered on zero so decreases and increases read equally.
        figsize: Size of the figure, in inches, or ``None`` for one that grows with the number of rows and columns. Ignored when ``ax`` is given.
        ax: Axes to draw on, or ``None`` for a new figure.

    Returns:
        The axes drawn on, holding perturbations against feature families on a scale centered on zero, with a colorbar beside them.

    Raises:
        KeyError: ``groupby`` was given and ``obs`` has no such column.
    """
    values = get_matrix(adata).astype(np.float64)
    obs = as_frame(adata.obs)
    rows = pd.Index(adata.obs_names.astype(str))

    if groupby is not None:
        if groupby not in obs.columns:
            raise KeyError(f"obs has no column {groupby!r}")
        frame = pd.DataFrame(values, index=obs[groupby].astype(str).to_numpy())
        frame = frame[frame.index != "nan"]
        grouped = frame.groupby(level=0, observed=True).mean()
        values, rows = grouped.to_numpy(), pd.Index(grouped.index.astype(str))
    elif top is not None and values.shape[0] > top:
        order = np.argsort(-np.abs(values).max(axis=1))[:top]
        values, rows = values[order], rows[order]

    columns = pd.Index(adata.var_names.astype(str))
    if cluster and values.shape[0] > 2 and values.shape[1] > 2:
        for axis in (0, 1):
            block = values if axis == 0 else values.T
            finite = np.nan_to_num(block, nan=0.0)
            order = np.asarray(
                hierarchy.leaves_list(
                    hierarchy.linkage(distance.pdist(finite, metric="correlation"), method="average")
                ),
                dtype=np.intp,
            )
            if axis == 0:
                values, rows = values[order], rows[order]
            else:
                values, columns = values[:, order], columns[order]

    if ax is None:
        height = max(2.4, 0.22 * len(rows) + 1.4)
        width = max(4.0, 0.34 * len(columns) + 2.2)
        _, ax = plt.subplots(figsize=figsize or (width, height))

    limit = float(np.nanmax(np.abs(values))) or 1.0
    image = ax.imshow(values, aspect="auto", cmap=cmap, vmin=-limit, vmax=limit)
    ax.set_xticks(range(len(columns)))
    ax.set_xticklabels(columns, rotation=90, fontsize=7)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows, fontsize=7)
    ax.set_xlabel("feature family")
    colorbar = plt.colorbar(image, ax=ax, shrink=0.6)
    colorbar.set_label("mean t", fontsize=8)
    plt.tight_layout()
    return ax
