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


def _clock_colour(adata: AnnData, color: str, key: str) -> tuple[str, np.ndarray, str]:
    """Resolve ``color`` to a mode (``"categorical"``, ``"continuous"`` or ``"cyclic"``), its values and a label."""
    obs = as_frame(adata.obs)
    if color in (f"{key}_position", "position"):
        return "cyclic", obs[f"{key}_position"].to_numpy(dtype=float), "cycle position"
    in_obs, in_var = color in obs.columns, color in adata.var_names
    if in_obs and in_var:
        raise ValueError(f"{color!r} is both an obs column and a feature; rename one of them")
    if in_obs:
        series = obs[color]
        if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
            return "continuous", series.to_numpy(dtype=float), color
        return "categorical", series.astype(object).to_numpy(), color
    if in_var:
        return "continuous", get_matrix(adata)[:, adata.var_names.get_loc(color)].astype(float), color
    raise KeyError(f"{color!r} is neither an obs column nor a feature")


def _clock_palette(categories: list[str]) -> dict[str, Any]:
    """A colour per category: the phase colours where they match, else the tab10 cycle."""
    import matplotlib.pyplot as plt

    tab, extra = plt.get_cmap("tab10"), 0
    out: dict[str, Any] = {}
    for category in categories:
        if category in PHASE_COLOURS:
            out[category] = PHASE_COLOURS[category]
        else:
            out[category], extra = tab(extra % 10), extra + 1
    return out


def cell_cycle_clock(
    adata: AnnData,
    key: str = "cell_cycle",
    color: str | None = None,
    groups: list[str] | str | None = None,
    radius: str | None = None,
    groupby: str | None = None,
    ncols: int | None = None,
    ax: Axes | None = None,
    size: float = 6.0,
    cmap: str | None = None,
    rim: bool = True,
    seed: int = 0,
    **kwargs: Any,
) -> Axes | np.ndarray | None:
    """Cells on a ring, placed by angle at a cycle position drawn from their posterior.

    The angle is a cycle position drawn from each cell's posterior from :func:`~mantispy.tl.project_cell_cycle`,
    seeded for a stable figure, so the ring fills rather than stacking on the medians. Position 0 is at the top and
    the angle runs clockwise, so G1 starts at twelve o'clock and a cell comes back to the top as it divides. Cells sit
    on a ring of a fixed radius with a little jitter, or, with ``radius``, on an annulus whose radius follows an
    ``obs`` column such as the DNA content.

    DNA content is flat at 2N through G1 and at 4N through G2M, and the width of either peak is measurement noise, so
    DNA content orders cells only through S. The S arc therefore places each cell from its own DNA, while in the G1
    and G2M arcs the angular spread follows the control age distribution, where a cell is more likely to sit early.
    Those two arcs are shaded and labelled as the population prior, the S arc as per-cell. When
    ``obs[key + "_g2m_order"]`` holds the experimental order from ``refine_g2m``, G2M cells sit on their wedge by that
    order instead, and the wedge is labelled so. Read the clock by where a colour concentrates, with ``color`` set to
    the phase or a gene, not as a fine order within a prior wedge. Off-cycle cells have no position and are left out.

    Args:
        adata: Single-cell object :func:`~mantispy.tl.project_cell_cycle` has run on.
        key: The ``key_added`` prefix :func:`~mantispy.tl.project_cell_cycle` wrote.
        color: What colours the cells: ``None`` for the phase; ``"position"`` or ``key + "_position"`` for the cycle
            position on a cyclic colour map; another ``obs`` column, categorical or numeric; or a feature.
        groups: With a categorical ``color``, draw every other cell as a faint grey background and colour only the
            named categories, each with its own colour and a legend. ``None`` colours every category.
        radius: An ``obs`` column mapped to the radius, so the ring becomes an annulus, its inner edge the column's
            2nd percentile and its outer edge the 98th. ``None`` keeps the jittered ring.
        groupby: ``obs`` column whose groups each get their own clock on a shared layout, or ``None`` for one clock.
        ncols: Clocks per row when ``groupby`` is given, or ``None`` for one row.
        ax: Axes to draw a single clock on, or ``None`` for a new figure. Ignored when ``groupby`` is given.
        size: Marker area.
        cmap: Colour map for a numeric or feature ``color``; defaults to twilight for the cyclic position and viridis
            otherwise.
        rim: Draw the phase arcs outside the ring.
        seed: Seed for the ring jitter and the draw order.
        kwargs: Passed to the underlying scatter.

    Returns:
        The axes when the caller passed ``ax`` and no ``groupby``, else ``None`` because the plot then owns the
        figure it created.

    Raises:
        KeyError: ``obs`` has no ``key + "_position"`` column, or ``color`` or ``radius`` names nothing.
        ValueError: ``color`` is both an ``obs`` column and a feature, ``groups`` is given without a categorical
            ``color``, or a name in ``groups`` is not a category of ``color``.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgba_array
    from matplotlib.lines import Line2D
    from matplotlib.patches import Wedge

    from mantispy.tl._cell_cycle import _age, _draw_positions

    obs = as_frame(adata.obs)
    if f"{key}_quantiles" not in adata.obsm:
        raise KeyError(f"obsm has no {key}_quantiles; run mt.tl.project_cell_cycle first")
    quantiles = np.asarray(adata.obsm[f"{key}_quantiles"], dtype=float)
    placed = np.isfinite(quantiles).all(axis=1)
    position = _draw_positions(np.nan_to_num(quantiles), seed)
    model = adata.uns.get("mantispy", {}).get(key)
    edges = (
        [0.0, *(_age(np.cumsum(model["fractions"])[:2]).tolist()), 1.0]
        if isinstance(model, dict) and "fractions" in model
        else None
    )
    refined = np.zeros(adata.n_obs, dtype=bool)
    if edges is not None and f"{key}_g2m_order" in obs.columns:
        order = obs[f"{key}_g2m_order"].to_numpy(dtype=float)
        refined = placed & np.isfinite(order)
        position = np.where(refined, edges[2] + np.nan_to_num(order) * (edges[3] - edges[2]), position)
    mode, values, label = _clock_colour(adata, color or f"{key}_phase", key)
    text = values.astype(str)

    groups_list = None
    if groups is not None:
        if mode != "categorical":
            raise ValueError("groups= needs a categorical color=")
        groups_list = [groups] if isinstance(groups, str) else list(groups)
        present = set(text[placed])
        unknown = [name for name in groups_list if name not in present]
        if unknown:
            raise ValueError(f"groups {unknown} are not categories of {color!r}; have {sorted(present)[:10]}")

    # A stream apart from the position draws, so the ring jitter does not follow the angle.
    rng = np.random.default_rng([seed, 1])
    angle = 2 * np.pi * position
    if radius is None:
        reach, inner = 1.0 + rng.uniform(-0.06, 0.06, adata.n_obs), 1.0
    else:
        if radius not in obs.columns:
            raise KeyError(f"obs has no column {radius!r}")
        raw = obs[radius].to_numpy(dtype=float)
        low, high = np.nanpercentile(raw[placed], [2, 98])
        inner = 0.55
        reach = inner + 0.6 * np.clip((raw - low) / ((high - low) or 1.0), 0, 1)
    x, y = reach * np.sin(angle), reach * np.cos(angle)

    present_cats = list(dict.fromkeys(text[placed]))
    categories = (
        [phase for phase in PHASE_COLOURS if phase in present_cats]
        if all(name in PHASE_COLOURS for name in present_cats)
        else sorted(present_cats)
    )
    palette = _clock_palette(categories)
    rim_radius = (0.55 + 0.6 if radius is not None else 1.0) + 0.18
    limit = rim_radius + 0.12 if rim and "mantispy" in adata.uns else rim_radius - 0.05

    def draw(axis: Axes, panel: np.ndarray) -> Any:
        axis.add_patch(plt.Circle((0, 0), inner if radius is None else 0.55, fill=False, color="0.85", lw=0.6))
        if radius is not None:
            axis.add_patch(plt.Circle((0, 0), 1.15, fill=False, color="0.9", lw=0.5))
        rows = rng.permutation(np.flatnonzero(placed & panel))
        drawn, handles = None, []
        if mode == "categorical":
            # One scatter in shuffled order, so no category hides another under it; the legend uses proxies.
            shown = groups_list if groups_list is not None else categories
            front = rows
            if groups_list is not None:
                back = rows[~np.isin(text[rows], groups_list)]
                axis.scatter(x[back], y[back], s=size * 0.5, c="0.85", alpha=0.5, linewidths=0, rasterized=True)
                front = rows[np.isin(text[rows], groups_list)]
            if front.size:
                axis.scatter(
                    x[front],
                    y[front],
                    s=size,
                    c=to_rgba_array([palette[name] for name in text[front]]),
                    alpha=0.6 if groups_list is not None else 0.7,
                    linewidths=0,
                    rasterized=True,
                    **kwargs,
                )
            handles = [
                Line2D([], [], marker="o", linestyle="", color=palette[name], label=name)
                for name in shown
                if (text[rows] == name).any()
            ]
        else:
            span = (0.0, 1.0) if mode == "cyclic" else tuple(np.nanpercentile(values[rows], [2, 98]))
            drawn = axis.scatter(
                x[rows],
                y[rows],
                s=size,
                c=values[rows],
                cmap=cmap or ("twilight" if mode == "cyclic" else "viridis"),
                vmin=span[0],
                vmax=span[1],
                alpha=0.8,
                linewidths=0,
                rasterized=True,
                **kwargs,
            )
        if edges is not None:
            # Where the angle follows the control age prior rather than the cell's own data, shade and label it so.
            g2m_note = "experimental order" if refined.any() else "population prior"
            arcs = [((edges[0], edges[1]), "G1", "population prior"), ((edges[2], edges[3]), "G2M", g2m_note)]
            for (start, end), name, note in arcs:
                if note == "population prior":
                    axis.add_patch(
                        Wedge(
                            (0, 0),
                            rim_radius,
                            90 - 360 * end,
                            90 - 360 * start,
                            color="0.6",
                            alpha=0.12,
                            lw=0,
                            zorder=0,
                        )
                    )
                middle = (start + end) / 2
                axis.text(
                    0.58 * rim_radius * np.sin(2 * np.pi * middle),
                    0.58 * rim_radius * np.cos(2 * np.pi * middle),
                    f"{name}\n{note}",
                    ha="center",
                    va="center",
                    fontsize=5,
                    color="0.5",
                    zorder=0,
                )
            middle = (edges[1] + edges[2]) / 2
            axis.text(
                0.58 * rim_radius * np.sin(2 * np.pi * middle),
                0.58 * rim_radius * np.cos(2 * np.pi * middle),
                "S\nper-cell",
                ha="center",
                va="center",
                fontsize=5,
                color="0.4",
                zorder=0,
            )
            if rim:
                for (start, end), colour in zip(
                    list(zip(edges[:-1], edges[1:], strict=True)), ("tab:blue", "tab:grey", "tab:red"), strict=True
                ):
                    turn = np.linspace(start, end, 40)
                    axis.plot(
                        rim_radius * np.sin(2 * np.pi * turn),
                        rim_radius * np.cos(2 * np.pi * turn),
                        color=colour,
                        lw=3,
                        solid_capstyle="round",
                    )
        axis.set_xlim(-limit, limit)
        axis.set_ylim(-limit, limit)
        axis.set_aspect("equal")
        axis.axis("off")
        return drawn, handles

    if groupby is None:
        axis = _axes(ax, (4.5, 4.5))
        drawn, handles = draw(axis, np.ones(adata.n_obs, dtype=bool))
        if mode == "categorical":
            axis.legend(handles=handles, fontsize=6, bbox_to_anchor=(1.0, 1.0), loc="upper left", title=label)
        elif drawn is not None:
            axis.figure.colorbar(drawn, ax=axis, label=label, fraction=0.046, pad=0.04)
        if radius is not None:
            axis.set_title(f"radius: {radius}", fontsize=8)
        return _returned(axis)

    codes, groups_keys = group_codes(adata, groupby)
    columns = min(ncols or len(groups_keys), len(groups_keys))
    rows_count = -(-len(groups_keys) // columns)
    figure, axes = plt.subplots(rows_count, columns, figsize=(3.4 * columns, 3.4 * rows_count), squeeze=False)
    drawn, handles = None, []
    for index, group in enumerate(groups_keys):
        panel_drawn, panel_handles = draw(axes.flat[index], codes == index)
        drawn, handles = panel_drawn or drawn, handles or panel_handles
        axes.flat[index].set_title(str(group), fontsize=8)
    for axis in axes.flat[len(groups_keys) :]:
        axis.set_visible(False)
    if mode == "categorical":
        axes.flat[0].legend(handles=handles, fontsize=6, bbox_to_anchor=(1.0, 1.0), loc="upper left", title=label)
    elif drawn is not None:
        figure.colorbar(drawn, ax=list(figure.axes), label=label, fraction=0.046)
    figure.tight_layout()
    return _returned(axes, owned=True)


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
