"""Plots for single-cell heterogeneity."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from mantispy._core._reduce import get_matrix, group_codes
from mantispy._core.frames import as_frame
from mantispy.pl._common import axes as _axes
from mantispy.pl._common import maybe_interactive as _maybe_interactive
from mantispy.pl._common import returned as _returned
from mantispy.pl._common import table as _table

if TYPE_CHECKING:
    from anndata import AnnData
    from matplotlib.axes import Axes

PHASE_COLOURS = {"G1": "tab:blue", "S": "tab:grey", "G2M": "tab:red", "off-cycle": "black"}


def cluster_composition(
    composition: AnnData, groupby: str = "Metadata_Perturbation", ax: Axes | None = None
) -> Axes | None:
    """Stacked bars of cell-state fractions, averaged within each group.

    Args:
        composition: The well-level object :func:`~mantispy.tl.cluster_composition` returns, holding one cluster per feature.
        groupby: ``obs`` column whose groups become the bars.
        ax: Axes to draw on, or ``None`` for a new figure.

    Returns:
        The axes when the caller passed ``ax``, else ``None`` because the plot then owns the figure it created.
        When returned they hold one bar per group of ``groupby`` in the order the groups first appear and one stacked segment per cluster.

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

    tidy = pd.DataFrame(means, index=pd.Index(labels, name="group"), columns=[str(c) for c in composition.var_names])
    tidy = tidy.reset_index().melt(id_vars="group", var_name="cluster", value_name="fraction of cells")
    _maybe_interactive(
        "barh", ax=ax, data=tidy, x="fraction of cells", y="group", color="cluster", title="cluster composition"
    )
    return _returned(ax)


def cell_cycle(
    adata: AnnData,
    dna_feature: str,
    by: str | None = "Metadata_Plate",
    key: str = "Metadata_CellCyclePhase",
    layer: str | None = None,
    ncols: int | None = None,
) -> np.ndarray | None:
    """DNA per group colored by phase.

    One panel per group of ``by`` shows the DNA distribution colored by phase. Two separated peaks with the phases
    split between them indicate a working assignment; a single broad distribution indicates a failed one.

    Args:
        adata: Single-cell object :func:`~mantispy.tl.cell_cycle_phase` or :func:`~mantispy.tl.project_cell_cycle`
            has run on.
        dna_feature: The DNA intensity feature the phases were assigned from, drawn on a log scale, or an ``obs``
            column such as the :func:`~mantispy.tl.dna_content` output, drawn as it is.
        by: ``obs`` column whose groups become panels, or ``None`` for a single panel over every row.
        key: ``obs`` column holding the assigned phase, whose values are matched against ``PHASE_COLOURS``, so a phase
            named anything else is not drawn. Pass ``"cell_cycle_phase"`` for the phases of
            :func:`~mantispy.tl.project_cell_cycle`.
        layer: Layer to read the intensity from, or ``None`` for ``X``.
            Only positive values are drawn, since the plot takes their logarithm.
        ncols: Panels per row, or ``None`` for one row.

    Returns:
        ``None``; this plot always creates its own figure, a grid of panels sharing an x axis, one per group of
        ``by``, each holding one filled histogram per phase.

    Raises:
        KeyError: ``obs`` has no ``key`` column, or ``dna_feature`` is neither a feature nor an ``obs`` column.
        ValueError: ``dna_feature`` names both a feature and an ``obs`` column, or the object has no layer named
            ``layer``.
    """
    import matplotlib.pyplot as plt

    if key not in adata.obs:
        raise KeyError(f"obs has no column {key!r}; run mt.tl.cell_cycle_phase or mt.tl.project_cell_cycle first")
    if dna_feature in adata.obs and dna_feature in adata.var_names:
        raise ValueError(f"{dna_feature!r} is both an obs column and a feature; rename one of them")

    if dna_feature in adata.obs:
        values = as_frame(adata.obs)[dna_feature].to_numpy(dtype=float)
        shown, label = np.isfinite(values), str(dna_feature)
    else:
        intensity = get_matrix(adata, layer)[:, adata.var_names.get_loc(dna_feature)].astype(float)
        shown = np.isfinite(intensity) & (intensity > 0)
        values, label = np.log(np.where(shown, intensity, 1.0)), "log DNA intensity"
    phases = as_frame(adata.obs)[key].astype(str).to_numpy()
    codes, keys = group_codes(adata, by)

    columns = min(ncols or len(keys), len(keys))
    rows = -(-len(keys) // columns)
    figure, axes = plt.subplots(rows, columns, figsize=(3.2 * columns, 3 * rows), squeeze=False, sharex=True)
    for index, name in enumerate(keys):
        axis = axes.flat[index]
        selected_rows = np.flatnonzero((codes == index) & shown)
        for phase, colour in PHASE_COLOURS.items():
            selected = values[selected_rows[phases[selected_rows] == phase]]
            if selected.size:
                axis.hist(selected, bins=40, histtype="stepfilled", alpha=0.6, color=colour, label=phase)
        axis.set_title(str(name), fontsize=8)
        axis.set_xlabel(label)
    for axis in axes.flat[len(keys) :]:
        axis.set_visible(False)
    axes[0, 0].legend(fontsize=6)
    figure.tight_layout()
    return _returned(axes, owned=True)


def cell_cycle_occupancy(
    adata: AnnData,
    key: str = "cell_cycle",
    by: str | None = "Metadata_Perturbation",
    highlight: str | None = None,
    seed: int = 0,
) -> None:
    """How the cells of each group occupy the cycle placed by :func:`~mantispy.tl.project_cell_cycle`.

    Each cell contributes one position drawn from its own posterior, so a cell in a plateau spreads over that plateau
    instead of piling onto its median, and an arrest shows as a group's curve rising above the others at one
    position. Off-cycle cells have no position and are left out. When the model is in ``uns["mantispy"][key]``, dotted
    lines mark where its S phase begins and ends.

    Args:
        adata: Single-cell object :func:`~mantispy.tl.project_cell_cycle` has run on.
        key: The ``key_added`` prefix :func:`~mantispy.tl.project_cell_cycle` wrote.
        by: ``obs`` column whose groups become curves, or ``None`` for one curve over every row.
        highlight: A group of ``by``, such as the controls, drawn thicker and in black.
        seed: Seed for the positions drawn.

    Returns:
        ``None``; this plot always creates its own figure, one panel holding one density curve per group of ``by``.

    Raises:
        KeyError: ``obsm`` has no ``key + "_quantiles"``.
    """
    import matplotlib.pyplot as plt

    from mantispy.tl._cell_cycle import _age, _draw_positions

    name = f"{key}_quantiles"
    if name not in adata.obsm:
        raise KeyError(f"obsm has no {name!r}; run mt.tl.project_cell_cycle first")
    quantiles = np.asarray(adata.obsm[name], dtype=float)
    placed = np.isfinite(quantiles).all(axis=1)
    drawn = _draw_positions(np.nan_to_num(quantiles), seed)

    codes, keys = group_codes(adata, by)
    figure, axis = plt.subplots(figsize=(5, 3))
    bins = list(np.linspace(0, 1, 41))
    for index, group in enumerate(keys):
        selected = drawn[(codes == index) & placed]
        if selected.size:
            style: dict[str, Any] = {"color": "black", "linewidth": 2} if str(group) == highlight else {}
            axis.hist(selected, bins=bins, density=True, histtype="step", label=str(group), **style)
    model = adata.uns.get("mantispy", {}).get(key)
    if model is not None and "fractions" in model:
        for edge in _age(np.cumsum(model["fractions"])[:2]):
            axis.axvline(float(edge), color="grey", linestyle=":", linewidth=1)
    axis.set_xlabel("cycle position (0 = start of G1, 1 = division)")
    axis.set_ylabel("density")
    axis.legend(fontsize=6, bbox_to_anchor=(1.01, 1), loc="upper left")
    figure.tight_layout()


def subpopulation_hits(
    adata: AnnData, key: str = "subpopulation_hits", top: int = 30, ax: Axes | None = None
) -> Axes | None:
    """Cluster by group heatmap of significance, so an effect in one cell state stands out.

    Args:
        adata: Object holding the table :func:`~mantispy.tl.subpopulation_hits` wrote.
        key: Name of that table in ``uns["mantispy"]``.
        top: How many groups to draw, taken by their most significant cluster.
        ax: Axes to draw on, or ``None`` for a new figure.

    Returns:
        The axes when the caller passed ``ax``, else ``None`` because the plot then owns the figure it created.
        When returned they hold ``-log10`` q per cluster and group, the drawn groups sorted by name, with a colorbar beside them.

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

    _maybe_interactive(
        "heatmap",
        ax=ax,
        matrix=grid.to_numpy(dtype=float),
        rows=[str(index) for index in grid.index],
        columns=[str(column) for column in grid.columns],
        value_label="-log10 q",
        title="subpopulation hits",
    )
    return _returned(ax)


def density(
    adata: AnnData,
    feature: str,
    groupby: str = "Metadata_Perturbation",
    key: str = "Metadata_LocalDensity",
    max_groups: int = 6,
    ax: Axes | None = None,
) -> Axes | None:
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
        The axes when the caller passed ``ax``, else ``None`` because the plot then owns the figure it created.
        When returned they hold a scatter and a fitted line per group and each group's Pearson correlation between density and the feature in the legend.
        A group with fewer than three usable points is left out.

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
    records = []
    for name in pd.Series(groups).value_counts().index[:max_groups]:
        rows = np.flatnonzero((groups == name) & np.isfinite(crowding) & np.isfinite(values))
        if rows.size < 3:
            continue
        correlation = float(np.corrcoef(crowding[rows], values[rows])[0, 1])
        points = ax.scatter(crowding[rows], values[rows], s=6, alpha=0.5, label=f"{name} (r = {correlation:+.2f})")
        slope, intercept = np.polyfit(crowding[rows], values[rows], 1)
        grid = np.linspace(crowding[rows].min(), crowding[rows].max(), 2)
        ax.plot(grid, slope * grid + intercept, lw=1.2, color=points.get_facecolor()[0])
        records.append(pd.DataFrame({"density": crowding[rows], feature: values[rows], groupby: str(name)}))

    ax.set_xlabel("mean distance to the k nearest cells in the field")
    ax.set_ylabel(feature)
    ax.legend(fontsize=6)

    if records:
        _maybe_interactive(
            "scatter",
            ax=ax,
            data=pd.concat(records, ignore_index=True),
            x="density",
            y=feature,
            color=groupby,
            title=f"{feature} against density",
        )
    return _returned(ax)
