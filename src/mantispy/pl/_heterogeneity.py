"""Plots for single-cell heterogeneity."""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from itertools import cycle, pairwise
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from mantispy._core._reduce import get_matrix, group_codes
from mantispy._core.frames import as_frame
from mantispy.pl._common import axes as _axes
from mantispy.pl._common import maybe_interactive as _maybe_interactive
from mantispy.pl._common import obs_or_feature as _obs_or_feature
from mantispy.pl._common import returned as _returned
from mantispy.pl._common import table as _table

if TYPE_CHECKING:
    from collections.abc import Sequence

    from anndata import AnnData
    from matplotlib.axes import Axes

PHASE_COLOURS = {"G1": "tab:blue", "S": "tab:grey", "G2M": "tab:red", "sub-2N": "black", "above-4N": "tab:purple"}


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
    key: str = "cell_cycle_phase",
    layer: str | None = None,
    ncols: int | None = None,
) -> np.ndarray | None:
    """DNA per group colored by phase.

    One panel per group of ``by`` shows the DNA distribution colored by phase. Two separated peaks with the phases
    split between them indicate a working assignment; a single broad distribution indicates a failed one.

    Args:
        adata: Single-cell object :func:`~mantispy.tl.project_cell_cycle` has run on.
        dna_feature: The DNA intensity feature the phases were assigned from, drawn on a log scale, or an ``obs``
            column such as the :func:`~mantispy.tl.dna_content` output, drawn as it is.
        by: ``obs`` column whose groups become panels, or ``None`` for a single panel over every row.
        key: ``obs`` column holding the assigned phase, whose values are matched against ``PHASE_COLOURS``, so a phase
            named anything else is not drawn. The default is the phase :func:`~mantispy.tl.project_cell_cycle` writes.
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
    from matplotlib.patches import Patch

    if key not in adata.obs:
        raise KeyError(f"obs has no column {key!r}; run mt.tl.project_cell_cycle first")
    raw, from_obs = _obs_or_feature(adata, dna_feature, layer)
    if from_obs:
        values = raw.astype(float)
        shown, label = np.isfinite(values), str(dna_feature)
    else:
        shown = np.isfinite(raw) & (raw > 0)
        values, label = np.log(np.where(shown, raw, 1.0)), "log DNA intensity"
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
                axis.hist(selected, bins=40, histtype="stepfilled", alpha=0.6, color=colour)
        axis.set_title(str(name), fontsize=8)
        axis.set_xlabel(label)
        axis.tick_params(labelbottom=True)  # a shared axis hides them above the gaps of a partial last row
    for axis in axes.flat[len(keys) :]:
        axis.set_visible(False)
    present = set(phases[shown])
    handles = [
        Patch(color=colour, alpha=0.6, label=phase) for phase, colour in PHASE_COLOURS.items() if phase in present
    ]
    axes[0, 0].legend(handles=handles, fontsize=6)
    figure.tight_layout()
    return _returned(axes, owned=True)


def _draw_positions(quantiles: np.ndarray, levels: np.ndarray, seed: int = 0) -> np.ndarray:
    """One cycle position per cell drawn from its posterior, given as its quantiles at ``levels``.

    The quantile function is interpolated linearly between the given quantiles and extended along its outer segments
    to the ends of the cycle, so the outer draws spread over the tails instead of collapsing onto the outer quantiles.
    """
    levels = np.r_[0, levels, 1]
    below = quantiles[:, 0] - (quantiles[:, 1] - quantiles[:, 0]) * levels[1] / (levels[2] - levels[1])
    above = quantiles[:, -1] + (quantiles[:, -1] - quantiles[:, -2]) * (1 - levels[-2]) / (levels[-2] - levels[-3])
    knots = np.column_stack([np.clip(below, 0, 1), quantiles, np.clip(above, 0, 1)])
    draw = np.random.default_rng(seed).random(quantiles.shape[0])
    upper = np.clip(np.searchsorted(levels, draw), 1, levels.size - 1)
    fraction = (draw - levels[upper - 1]) / (levels[upper] - levels[upper - 1])
    rows = np.arange(quantiles.shape[0])
    return knots[rows, upper - 1] + fraction * (knots[rows, upper] - knots[rows, upper - 1])


def _placed(adata: AnnData, key: str, seed: int) -> tuple[np.ndarray, np.ndarray, list[float] | None]:
    """Which cells have a position, one position per cell drawn from its posterior, and the model's phase edges."""
    name = f"{key}_quantiles"
    if name not in adata.obsm:
        raise KeyError(f"obsm has no {name!r}; run mt.tl.project_cell_cycle first")
    frame = adata.obsm[name]
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"obsm[{name!r}] must be the frame mt.tl.project_cell_cycle writes, with a column per quantile")
    quantiles = np.asarray(frame, dtype=float)
    levels = np.array([float(str(column).rstrip("%")) / 100 for column in frame.columns])
    model = adata.uns.get("mantispy", {}).get(key)
    edges = [float(edge) for edge in model["edges"]] if isinstance(model, Mapping) and "edges" in model else None
    return np.isfinite(quantiles).all(axis=1), _draw_positions(np.nan_to_num(quantiles), levels, seed), edges


def cell_cycle_occupancy(
    adata: AnnData,
    key: str = "cell_cycle",
    by: str | None = "Metadata_Perturbation",
    highlight: str | None = None,
    palette: Mapping[str, Any] | Sequence[Any] | str | None = None,
    legend_loc: str | None = "right margin",
    legend_fontsize: float = 6,
    seed: int = 0,
    ax: Axes | None = None,
) -> Axes | None:
    """How the cells of each group occupy the cycle placed by :func:`~mantispy.tl.project_cell_cycle`.

    Each cell contributes one position drawn from its own posterior, so a cell in a plateau spreads over that plateau
    instead of piling onto its median, and an arrest shows as a group's curve rising above the others at one
    position. Cells off the cycle have no position and are left out. Dotted lines mark where S begins and ends.

    Args:
        adata: Single-cell object :func:`~mantispy.tl.project_cell_cycle` has run on.
        key: The ``key_added`` prefix :func:`~mantispy.tl.project_cell_cycle` wrote.
        by: ``obs`` column whose groups become curves, or ``None`` for one curve over every row.
        highlight: A group of ``by``, such as the controls, drawn thicker and in black.
        palette: Colours for the groups, read as in :func:`~mantispy.pl.cell_cycle_clock`; ``None`` for the
            matplotlib colour cycle.
        legend_loc: ``"right margin"`` for a legend beside the plot, or ``None`` for no legend.
        legend_fontsize: Font size of the legend.
        seed: Seed for the positions drawn.
        ax: Axes to draw on, or ``None`` for a new figure.

    Returns:
        The axes when the caller passed ``ax``, else ``None`` because the plot then owns the figure it created.
        They hold one density curve per group of ``by``.

    Raises:
        KeyError: ``obsm`` has no ``key + "_quantiles"``.
        ValueError: ``legend_loc`` is not ``"right margin"`` or ``None``.
    """
    if legend_loc not in ("right margin", None):
        raise ValueError(f"legend_loc must be 'right margin' or None, not {legend_loc!r}")
    placed, drawn, edges = _placed(adata, key, seed)
    codes, keys = group_codes(adata, by)
    colours = _palette([str(group) for group in keys], palette) if palette is not None else {}
    axis = _axes(ax, (5, 3))
    bins = list(np.linspace(0, 1, 41))
    for index, group in enumerate(keys):
        selected = drawn[(codes == index) & placed]
        if selected.size:
            style: dict[str, Any] = {"color": colours[str(group)]} if colours else {}
            if str(group) == highlight:
                style = {"color": "black", "linewidth": 2}
            axis.hist(selected, bins=bins, density=True, histtype="step", label=str(group), **style)
    for edge in [] if edges is None else edges[1:3]:
        axis.axvline(edge, color="grey", linestyle=":", linewidth=1)
    axis.set_xlabel("cycle position (0 = start of G1, 1 = division)")
    axis.set_ylabel("density")
    if legend_loc == "right margin":
        axis.legend(fontsize=legend_fontsize, bbox_to_anchor=(1.01, 1), loc="upper left")
    if ax is None:
        import matplotlib.pyplot as plt

        plt.tight_layout()
    return _returned(axis)


def _clock_colour(adata: AnnData, color: str, key: str) -> tuple[str, np.ndarray, str]:
    """Resolve ``color`` to a mode (``"categorical"``, ``"continuous"`` or ``"cyclic"``), its values and a label."""
    if color in (f"{key}_position", "position"):
        if f"{key}_position" not in adata.obs:
            raise KeyError(f"obs has no column {key}_position; run mt.tl.project_cell_cycle first")
        return "cyclic", as_frame(adata.obs)[f"{key}_position"].to_numpy(dtype=float), "cycle position"
    values, from_obs = _obs_or_feature(adata, color)
    if from_obs and values.dtype.kind not in "iuf":
        return "categorical", values.astype(object), color
    return "continuous", values.astype(float), color


def _bound(value: float | str | None, values: np.ndarray, default: float) -> float:
    """A colour-scale end, as scanpy reads it: a number, ``"pN"`` for the N-th percentile, or ``None`` for ``default``."""
    if isinstance(value, str):
        if not value.startswith("p"):
            raise ValueError(f"vmin/vmax {value!r} must be a number or a percentile such as 'p99'")
        return float(np.percentile(values, float(value[1:])))
    return float(np.percentile(values, default)) if value is None else float(value)


def _palette(shown: list[str], palette: Mapping[str, Any] | Sequence[Any] | str | None) -> dict[str, Any]:
    """One colour per shown category: ``palette`` first, then the phase colours, then colours the phases do not use.

    Past the colours the phases do not use, the others follow scanpy's default palettes by count, as spatialdata-plot
    does, and grey past the largest.
    """
    import matplotlib.pyplot as plt
    from scanpy.plotting.legacy.palettes import default_20, default_28, default_102

    if isinstance(palette, str):
        cmap = plt.get_cmap(palette)
        palette = list(getattr(cmap, "colors", cmap(np.linspace(0, 1, len(shown)))))
    if palette is not None and not isinstance(palette, Mapping):
        palette = dict(zip(shown, cycle(palette), strict=False))
    given = palette or {}
    unnamed = sum(name not in given and name not in PHASE_COLOURS for name in shown)
    others: Any = cycle(plt.get_cmap("tab10")([1, 2, 5, 6, 8, 9]))
    for scheme in (default_20, default_28, default_102):
        if 6 < unnamed <= len(scheme):
            others = iter(scheme)
            break
    else:
        if unnamed > len(default_102):
            others = cycle(["grey"])
    return {name: given.get(name) or PHASE_COLOURS.get(name) or next(others) for name in shown}


def cell_cycle_clock(
    adata: AnnData,
    key: str = "cell_cycle",
    color: str | None = None,
    groups: list[str] | str | None = None,
    radius: str | None = None,
    groupby: str | None = None,
    ncols: int | None = None,
    ax: Axes | None = None,
    size: float | None = None,
    alpha: float | None = None,
    width: float | None = None,
    palette: Mapping[str, Any] | Sequence[Any] | str | None = None,
    cmap: str | None = None,
    vmin: float | str | None = None,
    vmax: float | str | None = None,
    na_color: Any = "lightgray",
    legend_loc: str | None = "right margin",
    legend_fontsize: float = 6,
    frameon: bool = False,
    rim: bool = True,
    seed: int = 0,
    kind: str = "cells",
    bandwidth: float = 0.02,
    reference: str | None = None,
    **kwargs: Any,
) -> Axes | None:
    """Cells on a ring, placed by angle at a cycle position drawn from their posterior.

    The angle is a cycle position drawn from each cell's posterior from :func:`~mantispy.tl.project_cell_cycle`,
    seeded for a stable figure, so the ring fills rather than stacking on the medians. Position 0 is at the top and
    the angle runs clockwise, so G1 starts at twelve o'clock and a cell comes back to the top as it divides. Cells sit
    in a band around a ring, spread radially at random within ``width``, or, with ``radius``, on an annulus whose
    radius follows an ``obs`` column such as the DNA content.

    DNA content orders cells only through S, since it is flat at 2N through G1 and at 4N through G2M. The S arc
    therefore places each cell from its own DNA, while in the G1 and G2M arcs the angle follows the reference's age
    distribution, where a cell is more likely to sit early. Those two arcs are shaded and labelled as the population
    prior. When ``obs[key + "_" + phase + "_order"]`` holds an order from :func:`~mantispy.tl.phase_order`, the
    cells of that phase sit on their arc by that order instead, and the arc is labelled as the experimental order.
    Read the clock by where a colour concentrates, not as a fine order within a prior arc. Cells off the cycle have
    no position and are left out.

    With ``kind="rings"`` the same positions are smoothed into one density per category of ``color``, each on its own
    ring in the order of ``groups``, the first outermost, so where in the cycle the cells of each group concentrate
    reads from the ring's height. ``kind="ridge"`` draws the same densities as rows over a straight cycle axis, which
    stays readable for many groups. All densities share one scale, and a dashed line marks the height of cells spread
    evenly over the cycle. Within a prior arc the shape of a density follows the prior, so compare how much of each
    group sits in an arc rather than its shape there.

    The colour and legend parameters follow :func:`scanpy.pl.embedding`.

    Args:
        adata: Single-cell object :func:`~mantispy.tl.project_cell_cycle` has run on.
        key: The ``key_added`` prefix :func:`~mantispy.tl.project_cell_cycle` wrote.
        color: What colours the cells: ``None`` for the phase; ``"position"`` or ``key + "_position"`` for the cycle
            position on a cyclic colour map; another ``obs`` column, categorical or numeric; or a feature. Cells
            with a position but no value are drawn in ``na_color``.
        groups: With a categorical ``color``, colour only the named categories, each with its own colour and a
            legend, and draw every other cell in ``na_color`` behind them. ``None`` colours every category.
        radius: An ``obs`` column mapped to the radius, so the ring becomes an annulus, its inner edge the column's
            2nd percentile and its outer edge the 98th. ``None`` keeps the ring.
        groupby: ``obs`` column whose groups each get their own clock on a shared layout, colour scale and legend, or
            ``None`` for one clock.
        ncols: Clocks per row when ``groupby`` is given, or ``None`` for up to five per row.
        ax: Axes to draw a single clock on, or ``None`` for a new figure.
        size: Marker area; ``None`` scales it with the number of cells, as scanpy does.
        alpha: Marker opacity; ``None`` for a default that depends on ``color``.
        width: Thickness of the band the cells sit in, as a fraction of the ring's radius. A wider band spreads a
            crowded arc out. ``None`` gives 0.3 for the ring and 0.6 for the ``radius`` annulus.
        palette: Colours for a categorical ``color``: a mapping from category to colour, a list of colours taken in
            order, or a colour map name. Categories it leaves out keep the phase colours or a default.
        cmap: Colour map for a numeric or feature ``color``; defaults to twilight for the cyclic position and viridis
            otherwise.
        vmin: Lower end of a numeric colour scale: a number, or ``"pN"`` for the N-th percentile of the drawn values.
            ``None`` gives the 2nd percentile.
        vmax: Upper end, read the same way; ``None`` gives the 98th percentile.
        na_color: Colour of cells with a position but no value, and of the cells outside ``groups``.
        legend_loc: ``"right margin"`` for a legend beside the clock, ``"on data"`` to write each category at the
            angle where its cells concentrate, or ``None`` for no legend.
        legend_fontsize: Font size of the legend or the labels on the clock.
        frameon: Draw a frame around each clock.
        rim: Draw the phase arcs outside the ring.
        seed: Seed for the drawn positions, the radial spread and the draw order.
        kind: ``"cells"`` for one marker per cell, ``"rings"`` for one density ring per category of ``color``, or
            ``"ridge"`` for one density row per category. With ``color=None`` the densities draw one curve over every
            cell, filled by phase. ``radius``, ``size``, ``alpha``, ``width``, the colour scale and ``kwargs`` apply
            to ``"cells"`` only.
        bandwidth: Standard deviation of the smoothing kernel of ``"rings"`` and ``"ridge"``, as a fraction of the
            cycle.
        reference: A category, such as the controls, whose density becomes the dashed line of every ring and row
            instead of the even spread, so a group's arrest shows where it rises above the reference. It is a
            category of ``color``, drawn first, or of ``groupby`` when ``color`` is ``None``. A panel without
            reference cells compares to the reference cells of every panel.
        kwargs: Passed to the scatter, overriding its defaults; ``norm`` sets a numeric scale.

    Returns:
        The axes when the caller passed ``ax``, else ``None`` because the plot then owns the figure it created.

    Raises:
        KeyError: ``obsm`` has no ``key + "_quantiles"``, or ``color`` or ``radius`` names nothing.
        ValueError: ``color`` is both an ``obs`` column and a feature, ``groups`` is given without a categorical
            ``color``, a name in ``groups`` is not a category of ``color``, ``ax`` is given with ``groupby``,
            ``legend_loc`` is not one of its options, ``vmin`` or ``vmax`` is a string that is not a percentile,
            ``kind`` is not one of its options, ``kind`` draws densities of a numeric ``color``, or ``reference`` is
            not a category of ``color`` or ``groupby``.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgba_array
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch, Wedge

    if groupby is not None and ax is not None:
        raise ValueError("ax= draws a single clock; drop it to draw one clock per group of groupby=")
    if legend_loc not in ("right margin", "on data", None):
        raise ValueError(f"legend_loc must be 'right margin', 'on data' or None, not {legend_loc!r}")
    if kind not in ("cells", "rings", "ridge"):
        raise ValueError(f"kind must be 'cells', 'rings' or 'ridge', not {kind!r}")
    obs = as_frame(adata.obs)
    placed, position, edges = _placed(adata, key, seed)
    notes = {"G1": "population prior", "S": "per-cell", "G2M": "population prior"}
    phase_column = obs[f"{key}_phase"].astype(str).to_numpy() if f"{key}_phase" in obs else None
    for index, phase in enumerate(notes):
        if edges is None or phase_column is None or f"{key}_{phase}_order" not in obs:
            continue
        order = obs[f"{key}_{phase}_order"].to_numpy(dtype=float)
        ordered = placed & (phase_column == phase) & np.isfinite(order)
        position[ordered] = edges[index] + order[ordered] * (edges[index + 1] - edges[index])
        if ordered.any():
            notes[phase] = "experimental order"

    mode, values, label = _clock_colour(adata, color or f"{key}_phase", key)
    size = 120_000 / max(adata.n_obs, 1) if size is None else size
    style: dict[str, Any] = {"s": size, "linewidths": 0, "rasterized": True}
    if mode == "categorical":
        valued = pd.notna(values)
        text = values.astype(str)
        present = set(text[placed & valued])
        if groups is None:
            shown = [p for p in PHASE_COLOURS if p in present] if present <= PHASE_COLOURS.keys() else sorted(present)
        else:
            shown = [groups] if isinstance(groups, str) else list(groups)
        if unknown := [name for name in shown if name not in present]:
            raise ValueError(f"groups {unknown} are not categories of {color!r}; have {sorted(present)[:10]}")
        colours_of = _palette(shown, palette)
        valued &= np.isin(text, shown)
        style["alpha"] = (0.7 if groups is None else 0.6) if alpha is None else alpha
    elif groups is not None:
        raise ValueError("groups= needs a categorical color=")
    else:
        valued = np.isfinite(values)
        drawn_values = values[placed & valued]
        if mode == "cyclic":
            low, high = (
                (0.0 if vmin is None else _bound(vmin, drawn_values, 0)),
                (1.0 if vmax is None else _bound(vmax, drawn_values, 100)),
            )
        elif drawn_values.size:
            low, high = _bound(vmin, drawn_values, 2), _bound(vmax, drawn_values, 98)
        else:
            low, high = 0.0, 1.0
        style |= {
            "cmap": cmap or ("twilight" if mode == "cyclic" else "viridis"),
            "alpha": 0.8 if alpha is None else alpha,
        }
        if "norm" not in kwargs:
            style |= {"vmin": low, "vmax": high}
    if kind != "cells":
        if mode != "categorical":
            raise ValueError(f"kind={kind!r} draws one density per category, so it needs a categorical color=")
        codes, keys = group_codes(adata, groupby)
        compared = None
        if reference is not None:
            column = color or groupby
            labels = text if color is not None else np.asarray(keys, dtype=str)[codes] if groupby is not None else None
            if labels is None or reference not in set(labels[placed]):
                raise ValueError(f"reference {reference!r} is not a category of color= or groupby= {column!r}")
            compared = placed & (labels == reference)
            if color is not None:
                shown = [reference, *(name for name in shown if name != reference)]
                given = palette.get(reference) if isinstance(palette, Mapping) else None
                colours_of[reference] = given or "0.55"  # the reference in grey, so the groups keep the colours
        if color is None:
            members, colours_of = {"all cells": placed}, {"all cells": "0.3"}
        else:
            members = {name: placed & (text == name) for name in shown}
        # Each panel draws only the categories it has cells of, so its rings or rows share the space between them.
        curves = [
            {
                name: _circular_density(position[panel], bandwidth)
                for name, rows in members.items()
                if (panel := rows & (codes == index)).any()
            }
            for index in range(len(keys))
        ]
        references = [
            None
            if compared is None
            else _circular_density(
                position[compared & (codes == index) if (compared & (codes == index)).any() else compared], bandwidth
            )
            for index in range(len(keys))
        ]
        top = max(
            (curve.max() for panel in curves for curve in [*panel.values(), *references] if curve is not None),
            default=1.0,
        )
        if kind == "rings" and (most := max(map(len, curves))) > 8:
            warnings.warn(
                f"cell_cycle_clock: {most} rings are hard to read; use kind='ridge', or groupby= for one small clock per group",
                stacklevel=2,
            )
        axes = _clock_panels(ax, len(keys), groupby is not None, ncols, kind, max(map(len, curves)))
        by_phase = edges if color is None else None
        on_data = legend_fontsize if kind == "rings" and color is not None and legend_loc == "on data" else None
        # The reference's own panel comes first, as its ring does.
        panels = sorted(range(len(keys)), key=lambda index: str(keys[index]) != reference)
        for slot, index in enumerate(panels):
            group = keys[index]
            _draw_densities(
                axes[slot],
                curves[index],
                references[index],
                top,
                colours_of,
                kind,
                edges,
                notes,
                by_phase,
                rim,
                frameon,
                on_data,
            )
            if groupby is not None:
                axes[slot].set_title(str(group), fontsize=8, pad=10 if kind == "rings" else 4)
        if kind == "rings" and color is not None and legend_loc == "right margin":
            patches = [Patch(color=colours_of[name], alpha=0.6, label=name) for name in members]
            _margin_legend(axes[0], groupby is not None, patches, f"{label}, outer to inner", legend_fontsize)
        return _returned(axes[0]) if groupby is None else None
    marker = {name: value for name, value in kwargs.items() if name not in ("c", "cmap", "norm", "vmin", "vmax")}
    style |= kwargs

    # A stream apart from the position draws, so the radial spread does not follow the angle.
    rng = np.random.default_rng([seed, 1])
    if radius is None:
        band = 0.3 if width is None else width
        inner, outer = 1.0 - band / 2, 1.0 + band / 2
        reach = rng.uniform(inner, outer, adata.n_obs)
    elif radius not in obs.columns:
        raise KeyError(f"obs has no column {radius!r}")
    else:
        raw = obs[radius].to_numpy(dtype=float)
        low_r, high_r = np.nanpercentile(raw[placed], [2, 98]) if np.isfinite(raw[placed]).any() else (0.0, 1.0)
        outer = 1.15
        inner = outer - (0.6 if width is None else width)
        reach = inner + (outer - inner) * np.clip((raw - low_r) / ((high_r - low_r) or 1.0), 0, 1)
    x, y = reach * np.sin(2 * np.pi * position), reach * np.cos(2 * np.pi * position)
    rim_radius = outer + 0.18
    limit = rim_radius + (0.12 if rim and edges is not None else -0.05)

    def draw(axis: Axes, panel: np.ndarray) -> Any:
        guides = (1.0,) if radius is None else (inner, outer)
        for ring in guides:
            axis.add_patch(plt.Circle((0, 0), ring, fill=False, color="0.85", lw=0.6))
        rows = rng.permutation(np.flatnonzero(placed & panel))
        back = rows[~valued[rows]]
        if back.size:
            background = {"s": size * 0.5, "linewidths": 0, "rasterized": True, "alpha": 0.5, **marker}
            axis.scatter(x[back], y[back], c=[na_color], **background)
        # One scatter in shuffled order, so no category hides another under it; the legend uses proxies.
        rows = rows[valued[rows]]
        colours = to_rgba_array([colours_of[name] for name in text[rows]]) if mode == "categorical" else values[rows]
        drawn = axis.scatter(x[rows], y[rows], c=colours, **style)
        for (start, end), (name, note) in zip(pairwise(edges or ()), notes.items(), strict=False):
            middle = np.pi * (start + end)
            if note == "population prior":
                axis.add_patch(
                    Wedge((0, 0), rim_radius, 90 - 360 * end, 90 - 360 * start, color="0.6", alpha=0.12, lw=0, zorder=0)
                )
            axis.text(
                0.68 * inner * np.sin(middle),
                0.68 * inner * np.cos(middle),
                f"{name}\n{note}",
                ha="center",
                va="center",
                fontsize=5,
                color="0.4" if note == "per-cell" else "0.5",
                zorder=0,
            )
            if rim:
                turn = 2 * np.pi * np.linspace(start, end, 40)
                axis.plot(rim_radius * np.sin(turn), rim_radius * np.cos(turn), color=PHASE_COLOURS[name], lw=3)
        if mode == "categorical" and legend_loc == "on data":
            for name in shown:
                angle = 2 * np.pi * position[rows[text[rows] == name]]
                if angle.size:
                    middle = np.arctan2(np.sin(angle).mean(), np.cos(angle).mean())
                    axis.text(
                        (outer + 0.05) * np.sin(middle),
                        (outer + 0.05) * np.cos(middle),
                        name,
                        ha="center",
                        va="center",
                        fontsize=legend_fontsize,
                        fontweight="bold",
                        color=colours_of[name],
                    )
        axis.set(xlim=(-limit, limit), ylim=(-limit, limit), aspect="equal")
        if frameon:
            axis.set(xticks=[], yticks=[])
        else:
            axis.axis("off")
        return drawn

    codes, keys = group_codes(adata, groupby)
    axes = _clock_panels(ax, len(keys), groupby is not None, ncols, kind, 0)
    for index, group in enumerate(keys):
        drawn = draw(axes[index], codes == index)
        if groupby is not None or radius is not None:
            axes[index].set_title(str(group) if groupby is not None else f"radius: {radius}", fontsize=8)
    if mode == "categorical":
        if legend_loc == "right margin":
            handles = [Line2D([], [], marker="o", linestyle="", color=colours_of[name], label=name) for name in shown]
            _margin_legend(axes[0], groupby is not None, handles, label, legend_fontsize)
    else:
        axes[0].figure.colorbar(drawn, ax=axes[: len(keys)], label=label, fraction=0.046, pad=0.04)
    return _returned(axes[0]) if groupby is None else None


def _clock_panels(ax: Axes | None, n_panels: int, grouped: bool, ncols: int | None, kind: str, n_rows: int) -> list:
    """The axes of a clock, or of one clock per group on a grid, sized for the ``kind`` drawn."""
    import matplotlib.pyplot as plt

    # Densities get a fixed thickness per ring or row, so one ring makes a small clock and many make a large one;
    # past eight rings the clock stops growing and the rings get thinner.
    if kind == "ridge":
        width, height = (3.5 if grouped else 4.5), 0.28 * n_rows + 0.9
    elif kind == "rings":
        width = height = 1.3 + 0.3 * min(n_rows, 8)
    else:
        width = height = 3.4 if grouped else 4.5
    if not grouped:
        return [_axes(ax, (width, height))]
    columns = min(ncols or 5, n_panels)
    rows = -(-n_panels // columns)
    figsize = (width * columns + 1.5, height * rows)  # room for the legend or colour bar
    axes = list(plt.subplots(rows, columns, figsize=figsize, squeeze=False, layout="constrained")[1].ravel())
    for axis in axes[n_panels:]:
        axis.set_visible(False)
    return axes


def _circular_density(positions: np.ndarray, bandwidth: float, bins: int = 360) -> np.ndarray:
    """Density of cycle positions at ``bins`` evenly spaced points, smoothed across division, 1 where cells are even."""
    counts = np.bincount((positions * bins).astype(int) % bins, minlength=bins).astype(float)
    offset = np.minimum(np.arange(bins), bins - np.arange(bins)) / bins
    kernel = np.exp(-0.5 * (offset / bandwidth) ** 2)
    smooth = np.fft.irfft(np.fft.rfft(counts) * np.fft.rfft(kernel), bins)
    return smooth / smooth.mean()


def _draw_densities(
    axis: Axes,
    curves: Mapping[str, np.ndarray],
    compared: np.ndarray | None,
    top: float,
    colours: Mapping[str, Any],
    kind: str,
    edges: list[float] | None,
    notes: Mapping[str, str],
    by_phase: list[float] | None,
    rim: bool,
    frameon: bool,
    on_data: float | None,
) -> None:
    """One density per curve on concentric rings, the first outermost, or on rows, the first on top.

    The dashed line of each density is ``compared``, the reference density, or the even spread when it is ``None``.
    ``on_data`` is the font size of each ring's name written at its top, or ``None`` for no names.
    """
    from matplotlib.patches import Wedge

    grid = (np.arange(360) + 0.5) / 360
    band = 0.75 / max(len(curves), 1)
    arcs = list(zip(pairwise(edges or ()), notes, strict=False))

    def ring(turn: Any, radius: Any) -> tuple[Any, Any]:
        return radius * np.sin(2 * np.pi * turn), radius * np.cos(2 * np.pi * turn)

    for row, (name, curve) in enumerate(curves.items()):
        base = 1.0 - (row + 1) * band if kind == "rings" else len(curves) - 1 - row
        scale = 0.85 * (band if kind == "rings" else 1.0) / top
        pieces = [((start, end), PHASE_COLOURS[phase]) for (start, end), phase in arcs] if by_phase else []
        for (start, end), colour in pieces or [((0.0, 1.0), colours[name])]:
            turn = np.linspace(start, end, 2 + int(360 * (end - start)))
            height = base + scale * np.interp(turn, grid, curve, period=1)
            if kind == "rings":
                x, y = ring(np.r_[turn, turn[::-1]], np.r_[height, np.full(turn.size, base)])
                axis.fill(x, y, color=colour, alpha=0.6, lw=0)
                axis.plot(*ring(turn, height), color=colour, lw=0.8)
            else:
                axis.fill_between(turn, base, height, color=colour, alpha=0.6, lw=0)
                axis.plot(turn, height, color=colour, lw=0.8)
        dashed: dict[str, Any] = {"color": "0.25", "lw": 0.7, "linestyle": "--"}
        line = base + scale * (np.ones_like(grid) if compared is None else compared)
        if kind == "rings":
            axis.plot(*ring(np.r_[grid, grid[0]], np.r_[line, line[0]]), **dashed)
            if on_data is not None:
                axis.text(
                    -0.03, base + 0.4 * band, name, ha="right", va="center", fontsize=on_data, color=colours[name]
                )
        else:
            axis.plot(grid, line, **dashed)

    outer = 1.0 if kind == "rings" else len(curves)
    for (start, end), phase in arcs:
        if notes[phase] == "population prior":
            if kind == "rings":
                wedge = Wedge((0, 0), 1.0, 90 - 360 * end, 90 - 360 * start, width=0.75, color="0.6", alpha=0.12, lw=0)
                axis.add_patch(wedge)
            else:
                axis.axvspan(start, end, color="0.6", alpha=0.12, lw=0)
        if rim:
            turn = np.linspace(start, end, 40)
            middle = (start + end) / 2
            if kind == "rings":
                axis.plot(*ring(turn, np.full(turn.size, 1.06)), color=PHASE_COLOURS[phase], lw=3)
                across, up = ring(middle, 1.0)
                axis.annotate(
                    phase,
                    ring(middle, 1.06),
                    xytext=(5 * across, 5 * up),
                    textcoords="offset points",
                    ha="left" if across > 0.3 else "right" if across < -0.3 else "center",
                    va="bottom" if up > 0.3 else "top" if up < -0.3 else "center",
                    fontsize=6,
                )
            else:
                axis.plot(turn, [outer] * turn.size, color=PHASE_COLOURS[phase], lw=3)
                axis.annotate(
                    phase,
                    (middle, outer),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=6,
                )
    for edge in (edges or [])[1:3]:
        if kind == "rings":
            axis.plot(*ring(np.array([edge, edge]), np.array([0.25, 1.0])), color="grey", lw=0.6, linestyle=":")
        else:
            axis.axvline(edge, color="grey", lw=0.6, linestyle=":")

    if kind == "rings":
        axis.set(xlim=(-1.1, 1.1), ylim=(-1.1, 1.1), aspect="equal")
        if frameon:
            axis.set(xticks=[], yticks=[])
        else:
            axis.axis("off")
    else:
        axis.set(xlim=(0, 1), ylim=(-0.1, outer + 0.1))
        axis.set_yticks(np.arange(len(curves))[::-1], list(curves), fontsize=6)
        axis.set_xlabel("cycle position (0 = start of G1, 1 = division)")
        axis.spines[["top", "right", "left"]].set_visible(False)
        axis.tick_params(axis="y", length=0)


def _margin_legend(axis: Axes, grouped: bool, handles: list, title: str, fontsize: float) -> None:
    """A legend beside a single clock, or beside the whole grid when there is one clock per group."""
    style = {"handles": handles, "fontsize": fontsize, "title": title, "title_fontsize": fontsize}
    if grouped:
        axis.figure.legend(loc="outside right upper", **style)
    else:
        axis.legend(bbox_to_anchor=(1.0, 1.0), loc="upper left", **style)


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

    ax.set_xlabel("local cell density (cells per unit area)")
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
