from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from mantispy.pl._common import axes as _axes
from mantispy.pl._common import returned as _returned

if TYPE_CHECKING:
    from anndata import AnnData
    from matplotlib.axes import Axes


def dendrogram(
    adata: AnnData, key: str = "cluster", color_threshold: float | None = None, ax: Axes | None = None
) -> Axes | None:
    """Draw the hierarchical clustering tree stored by :func:`~mantispy.tl.cluster`.

    Args:
        adata: Object holding the linkage matrix :func:`~mantispy.tl.cluster` wrote with ``method="hierarchical"``.
        key: The ``key_added`` that run used, whose summary (leaf labels and the chosen cut) is read from ``uns["mantispy"][key]``.
        color_threshold: Height below which branches share a colour, marking the clusters.
            Defaults to the cut :func:`~mantispy.tl.cluster` chose, or scipy's own default when that run set the count directly.
        ax: Axes to draw on, or ``None`` for a new figure.

    Returns:
        The axes when the caller passed ``ax``, else ``None`` because the plot then owns the figure it created.
        When returned they hold the tree, with the profiles as leaves and merge height on the y axis.

    Raises:
        KeyError: ``uns["mantispy"]`` holds no ``key + "_linkage"`` (the run used ``method="leiden"``, or none ran).
    """
    from scipy.cluster.hierarchy import dendrogram as scipy_dendrogram

    store = adata.uns.get("mantispy", {})
    linkage_key = f"{key}_linkage"
    if linkage_key not in store:
        raise KeyError(
            f"uns['mantispy'][{linkage_key!r}] is missing; run mt.tl.cluster with method='hierarchical' first"
        )
    linkage_matrix = np.asarray(store[linkage_key], dtype=float)
    summary = store.get(key, {})
    labels = list(summary.get("labels", [str(name) for name in adata.obs_names]))

    if color_threshold is None:
        cut = summary.get("distance_cut", float("nan"))
        color_threshold = float(cut) if np.isfinite(cut) else None

    ax = _axes(ax, (0.18 * len(labels) + 2, 4.0))
    scipy_dendrogram(
        linkage_matrix, labels=labels, color_threshold=color_threshold, ax=ax, leaf_rotation=90, leaf_font_size=6
    )
    ax.set_ylabel("distance")
    return _returned(ax)
