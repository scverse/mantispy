"""Plots for single-cell heterogeneity."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from mantispy._core._reduce import get_matrix, group_codes
from mantispy._core.frames import as_frame
from mantispy.pl._common import axes as _axes
from mantispy.pl._common import table as _table

if TYPE_CHECKING:
    from anndata import AnnData
    from matplotlib.axes import Axes

#: Fixed phase colors, so each phase has the same color in every figure.
PHASE_COLOURS = {"G1": "tab:blue", "S": "tab:grey", "G2M": "tab:red"}


def cluster_composition(composition: AnnData, groupby: str = "Metadata_Perturbation", ax: Axes | None = None) -> Axes:
    """Stacked bars of cell-state fractions, averaged within each group.

    Takes the object :func:`~mantispy.tl.cluster_composition` returns.

    Args:
        composition: The well-level object :func:`~mantispy.tl.cluster_composition` returns, holding one cluster per feature.
        groupby: ``obs`` column whose groups become the bars.
        ax: Axes to draw on, or ``None`` for a new figure.

    Returns:
        The axes drawn on, with one bar per group of ``groupby`` in the order the groups first appear and one stacked segment per cluster.

    Raises:
        KeyError: ``obs`` has no ``groupby`` column.
    """
    import matplotlib.pyplot as plt

    if groupby not in composition.obs:
        raise KeyError(f"obs has no column {groupby!r}")

    fractions = get_matrix(composition).astype(float)
    groups = as_frame(composition.obs)[groupby].astype(str).to_numpy()
    labels = list(dict.fromkeys(groups))
    means = np.stack([np.nanmean(fractions[groups == group], axis=0) for group in labels])

    ax = _axes(ax, (0.4 * len(labels) + 3, 4))
    bottom = np.zeros(len(labels))
    colours = plt.get_cmap("tab20")
    for index, cluster in enumerate(composition.var_names):
        ax.bar(np.arange(len(labels)), means[:, index], bottom=bottom, color=colours(index % 20), label=str(cluster))
        bottom += means[:, index]

    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_ylabel("fraction of cells")
    ax.legend(fontsize=5, ncol=2, bbox_to_anchor=(1.01, 1), loc="upper left", title="cluster")
    return ax


def cell_cycle(
    adata: AnnData,
    dna_feature: str,
    by: str | None = "Metadata_Plate",
    key: str = "Metadata_CellCyclePhase",
    layer: str | None = None,
) -> np.ndarray:
    """Log DNA intensity per group, colored by assigned phase.

    Two separated peaks with the phases split between them indicate a working assignment; a single broad distribution indicates a failed one.

    Args:
        adata: Single-cell object :func:`~mantispy.tl.cell_cycle_phase` has run on.
        dna_feature: The DNA intensity feature the phases were assigned from.
        by: ``obs`` column whose groups become panels, or ``None`` for a single panel over every row.
        key: ``obs`` column holding the assigned phase, whose values are matched against ``PHASE_COLOURS``, so a phase named anything else is not drawn.
        layer: Layer to read the intensity from, or ``None`` for ``X``. Only positive values are drawn, since the plot takes their logarithm.

    Returns:
        A ``(1, n_groups)`` array of axes sharing an x axis, one panel per group of ``by``, each holding one filled histogram per phase.

    Raises:
        KeyError: ``obs`` has no ``key`` column, or ``dna_feature`` is not one of ``var_names``.
        ValueError: The object has no layer named ``layer``.
    """
    import matplotlib.pyplot as plt

    if key not in adata.obs:
        raise KeyError(f"obs has no column {key!r}; run mt.tl.cell_cycle_phase first")

    values = get_matrix(adata, layer)[:, adata.var_names.get_loc(dna_feature)].astype(float)
    positive = np.isfinite(values) & (values > 0)
    phases = as_frame(adata.obs)[key].astype(str).to_numpy()
    codes, keys = group_codes(adata, by)

    figure, axes = plt.subplots(1, len(keys), figsize=(3.2 * len(keys), 3), squeeze=False, sharex=True)
    for index, name in enumerate(keys):
        axis = axes[0, index]
        rows = np.flatnonzero((codes == index) & positive)
        for phase, colour in PHASE_COLOURS.items():
            selected = values[rows[phases[rows] == phase]]
            if selected.size:
                axis.hist(np.log(selected), bins=40, histtype="stepfilled", alpha=0.6, color=colour, label=phase)
        axis.set_title(str(name), fontsize=8)
        axis.set_xlabel("log DNA intensity")
    axes[0, 0].legend(fontsize=6)
    figure.tight_layout()
    return axes


def subpopulation_hits(adata: AnnData, key: str = "subpopulation_hits", top: int = 30, ax: Axes | None = None) -> Axes:
    """Cluster by group heatmap of significance, so an effect in one cell state stands out.

    Args:
        adata: Object holding the table :func:`~mantispy.tl.subpopulation_hits` wrote.
        key: Name of that table in ``uns["mantispy"]``.
        top: How many groups to draw, taken by their most significant cluster.
        ax: Axes to draw on, or ``None`` for a new figure.

    Returns:
        The axes drawn on, holding ``-log10`` q per cluster and group, the drawn groups sorted by name, with a colorbar beside them.

    Raises:
        KeyError: ``uns["mantispy"]`` holds no table under ``key``.
        ValueError: That table is empty, which is what happens when no cluster held both controls and another group.
    """
    table = _table(adata, key, "mt.tl.subpopulation_hits", "no cluster held both controls and another group")
    table = table.assign(significance=-np.log10(np.clip(table["qvalue"].to_numpy(dtype=float), 1e-12, None)))
    grid = table.pivot_table(index="cluster", columns="group", values="significance", aggfunc="max")
    keep = grid.max(axis=0).nlargest(min(top, grid.shape[1])).index
    grid = grid[sorted(keep)]

    ax = _axes(ax, (0.3 * grid.shape[1] + 3, 0.3 * grid.shape[0] + 2))
    image = ax.imshow(grid.to_numpy(dtype=float), aspect="auto", cmap="magma")
    ax.set_xticks(np.arange(grid.shape[1]))
    ax.set_xticklabels(grid.columns, rotation=90, fontsize=6)
    ax.set_yticks(np.arange(grid.shape[0]))
    ax.set_yticklabels(grid.index, fontsize=6)
    ax.set_ylabel("cluster")
    ax.figure.colorbar(image, ax=ax, label="-log10 q")
    return ax


def density(
    adata: AnnData,
    feature: str,
    groupby: str = "Metadata_Perturbation",
    key: str = "Metadata_LocalDensity",
    max_groups: int = 6,
    ax: Axes | None = None,
) -> Axes:
    """Local cell density against a feature, per group.

    Crowding alone changes morphology.
    Use this plot to check whether density explains a phenotype before regressing it out with :func:`~mantispy.pp.regress_out`.
    A group whose points fall on the same line as the controls differs from them only in density.

    Args:
        adata: Single-cell object carrying the density column and the feature.
        feature: Feature to plot against density.
        groupby: Column defining the groups drawn.
        key: ``obs`` column holding the local density.
        max_groups: Number of groups drawn, largest first.
        ax: Axes to draw on.

    Returns:
        The axes drawn on, with a scatter and a fitted line per group and each group's Pearson correlation between density and the feature in the legend. A group with fewer than three usable points is left out.

    Raises:
        KeyError: ``obs`` has no ``key`` column, or ``feature`` is not one of ``var_names``.
    """
    if key not in adata.obs:
        raise KeyError(f"obs has no column {key!r}; run mt.tl.neighbors_local_density first")

    obs = as_frame(adata.obs)
    values = get_matrix(adata)[:, adata.var_names.get_loc(feature)].astype(float)
    crowding = obs[key].to_numpy(dtype=float)
    groups = obs[groupby].astype(str).to_numpy()

    ax = _axes(ax, (5.5, 4.5))
    for name in pd.Series(groups).value_counts().index[:max_groups]:
        rows = np.flatnonzero((groups == name) & np.isfinite(crowding) & np.isfinite(values))
        if rows.size < 3:
            continue
        correlation = float(np.corrcoef(crowding[rows], values[rows])[0, 1])
        points = ax.scatter(crowding[rows], values[rows], s=6, alpha=0.5, label=f"{name} (r = {correlation:+.2f})")
        slope, intercept = np.polyfit(crowding[rows], values[rows], 1)
        grid = np.linspace(crowding[rows].min(), crowding[rows].max(), 2)
        ax.plot(grid, slope * grid + intercept, lw=1.2, color=points.get_facecolor()[0])

    ax.set_xlabel("mean distance to the k nearest cells in the field")
    ax.set_ylabel(feature)
    ax.legend(fontsize=6)
    return ax
