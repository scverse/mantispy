"""Plots for hit calling, effect sizes and dose response."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from mantispy._core.frames import as_frame
from mantispy.pl._common import axes as _axes
from mantispy.pl._common import table as _table

if TYPE_CHECKING:
    import pandas as pd
    from anndata import AnnData
    from matplotlib.axes import Axes

#: Values below this are clipped so they stay on the plot.
_FLOOR = 1e-12


def _significance(values: pd.Series | np.ndarray) -> np.ndarray:
    """Q-values on a ``-log10`` scale, floored so that an exact zero stays on the plot."""
    return -np.log10(np.clip(np.asarray(values, dtype=float), _FLOOR, None))


def _threshold(adata: AnnData, function: str, default: float = 0.05) -> float:
    """The threshold the run used, so the drawn line matches the table."""
    recorded = adata.uns.get("mantispy", {}).get("params", {}).get(function, {})
    return float(recorded.get("threshold", default))


def hits(adata: AnnData, key: str = "hits", label_top: int = 10, ax: Axes | None = None) -> Axes:
    """Distance from the controls against significance, with the most distant groups labeled.

    A point in the upper right moved far from the controls and is significant under the permutation null.
    The dashed line is the q-value threshold the run used, so every point colored as a hit sits on or above it.

    Args:
        adata: Object holding the table :func:`~mantispy.tl.hit_calling` wrote.
        key: Name of that table in ``uns["mantispy"]``.
        label_top: How many of the most distant groups to label.
        ax: Axes to draw on, or ``None`` for a new figure.

    Returns:
        The axes drawn on.

    Raises:
        KeyError: ``uns["mantispy"]`` holds no table under ``key``.
    """
    table = _table(adata, key, "mt.tl.hit_calling")
    ax = _axes(ax, (5.5, 4.5))

    significance = _significance(table["qvalue"])
    called = table["is_hit"].to_numpy(dtype=bool)
    ax.scatter(table["distance"][~called], significance[~called], s=14, color="lightgrey", label="not called")
    ax.scatter(table["distance"][called], significance[called], s=14, color="crimson", label="hit")

    # The threshold is recorded under the name of the function, not under the name of the table
    # it wrote, so reading it under `key` drew every run against the default of 0.05.
    threshold = _threshold(adata, "hit_calling")
    ax.axhline(-np.log10(threshold), color="grey", ls="--", lw=1, label=f"q = {threshold}")
    for _, row in table.nlargest(label_top, "distance").iterrows():
        ax.annotate(str(row["group"]), (row["distance"], -np.log10(max(float(row["qvalue"]), _FLOOR))), fontsize=6)

    ax.set_xlabel("distance from the controls")
    ax.set_ylabel("-log10 q")
    ax.legend(fontsize=7)
    return ax


def _effects(adata: AnnData, group: str, key: str) -> tuple[pd.DataFrame, pd.Series | None]:
    table = _table(adata, key, "mt.tl.effect_size")
    selected = table[table["group"].astype(str) == str(group)]
    if selected.empty:
        raise KeyError(f"no group {group!r} in uns['mantispy'][{key!r}]; it holds {sorted(set(table['group']))[:5]}")
    var = as_frame(adata.var)
    families = var["feature_group"].astype(str) if "feature_group" in var else None
    return selected, families


def _family_colours(families: pd.Series | None, names: pd.Series) -> tuple[list[Any], dict[str, Any]]:
    """One color per feature family, assigned in sorted order of the families present."""
    import matplotlib.pyplot as plt

    if families is None:
        return ["tab:blue"] * len(names), {}
    labels = [str(families.get(name, "unknown")) for name in names]
    palette = {label: plt.get_cmap("tab20")(index % 20) for index, label in enumerate(sorted(set(labels)))}
    return [palette[label] for label in labels], palette


def effect_sizes(adata: AnnData, group: str, key: str = "effect", top: int = 30, ax: Axes | None = None) -> Axes:
    """The largest effects for one group, colored by feature family.

    Args:
        adata: Object holding the table :func:`~mantispy.tl.effect_size` wrote.
        group: Which group of that table to draw.
        key: Name of that table in ``uns["mantispy"]``.
        top: How many features to draw, taken by absolute effect.
        ax: Axes to draw on, or ``None`` for a new figure.

    Returns:
        The axes drawn on.

    Raises:
        KeyError: There is no such table, or it holds no such group.
    """
    import matplotlib.pyplot as plt

    selected, families = _effects(adata, group, key)
    strongest = selected.reindex(selected["effect"].abs().sort_values(ascending=False).index).head(top)[::-1]
    ax = _axes(ax, (6, 0.22 * len(strongest) + 1.5))

    colours, palette = _family_colours(families, strongest["feature"])
    ax.barh(np.arange(len(strongest)), strongest["effect"], color=colours)
    ax.set_yticks(np.arange(len(strongest)))
    ax.set_yticklabels(strongest["feature"], fontsize=6)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("effect size")
    ax.set_title(str(group), fontsize=9)
    if palette:
        ax.legend(
            handles=[plt.Line2D([], [], color=colour, lw=6, label=label) for label, colour in palette.items()],
            fontsize=6,
            loc="lower right",
        )
    return ax


def feature_volcano(
    adata: AnnData, group: str, key: str = "effect", label_top: int = 8, ax: Axes | None = None
) -> Axes:
    """Effect against significance, per feature, for one group.

    Args:
        adata: Object holding the table :func:`~mantispy.tl.effect_size` wrote.
        group: Which group of that table to draw.
        key: Name of that table in ``uns["mantispy"]``.
        label_top: How many features to label, taken by absolute effect.
        ax: Axes to draw on, or ``None`` for a new figure.

    Returns:
        The axes drawn on.

    Raises:
        KeyError: There is no such table, or it holds no such group.
    """
    import matplotlib.pyplot as plt

    selected, families = _effects(adata, group, key)
    ax = _axes(ax, (5.5, 4.5))

    colours, palette = _family_colours(families, selected["feature"])
    significance = _significance(selected["qvalue"])
    ax.scatter(selected["effect"], significance, s=12, color=colours)

    for _, row in (
        selected.reindex(selected["effect"].abs().sort_values(ascending=False).index).head(label_top).iterrows()
    ):
        ax.annotate(str(row["feature"]), (row["effect"], -np.log10(max(float(row["qvalue"]), _FLOOR))), fontsize=5)

    ax.axhline(-np.log10(0.05), color="grey", ls="--", lw=1)
    ax.axvline(0, color="black", lw=0.6)
    ax.set_xlabel("effect size")
    ax.set_ylabel("-log10 q")
    ax.set_title(str(group), fontsize=9)
    if palette:
        ax.legend(
            handles=[plt.Line2D([], [], color=colour, lw=6, label=label) for label, colour in palette.items()],
            fontsize=5,
            loc="upper left",
        )
    return ax


def dose_response(
    adata: AnnData,
    compound: str,
    key: str = "dose_response",
    compound_key: str = "Metadata_Compound",
    dose_key: str = "Metadata_Concentration",
    response: str = "hits_distance",
    ax: Axes | None = None,
) -> Axes:
    """One compound's response against dose, with the fitted curve when there is one.

    Args:
        adata: Object holding the table :func:`~mantispy.tl.dose_response` wrote.
        compound: Which compound of that table to draw.
        key: Name of that table in ``uns["mantispy"]``.
        compound_key: ``obs`` column naming the compound of each well.
        dose_key: ``obs`` column holding the dose of each well.
        response: ``obs`` column drawn against the dose.
        ax: Axes to draw on, or ``None`` for a new figure.

    Returns:
        The axes drawn on.

    Raises:
        KeyError: There is no such table, or it holds no such compound.
    """
    from mantispy.tl._dose import four_parameter_logistic

    table = _table(adata, key, "mt.tl.dose_response")
    row = table[table["compound"].astype(str) == str(compound)]
    if row.empty:
        raise KeyError(f"no compound {compound!r} in uns['mantispy'][{key!r}]")

    obs = as_frame(adata.obs)
    selected = obs[obs[compound_key].astype(str) == str(compound)]
    doses = selected[dose_key].to_numpy(dtype=float)
    values = selected[response].to_numpy(dtype=float)
    usable = np.isfinite(doses) & np.isfinite(values) & (doses > 0)

    ax = _axes(ax, (5, 4))
    ax.scatter(doses[usable], values[usable], s=18, label="wells")
    ax.set_xscale("log")

    fitted = row.iloc[0]
    if bool(fitted["fit_ok"]):
        grid = np.log10(np.geomspace(doses[usable].min(), doses[usable].max(), 100))
        curve = four_parameter_logistic(
            grid,
            float(fitted["bottom"]),
            float(fitted["top"]),
            np.log10(float(fitted["ec50"])),
            float(fitted["hill_slope"]),
        )
        ax.plot(10.0**grid, curve, color="crimson", lw=1.5, label=f"EC50 = {float(fitted['ec50']):.3g}")

    ax.set_xlabel(dose_key.replace("Metadata_", ""))
    ax.set_ylabel(response)
    ax.set_title(f"{compound}  (spearman {float(fitted['spearman']):.2f})", fontsize=9)
    ax.legend(fontsize=7)
    return ax
