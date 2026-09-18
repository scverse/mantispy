"""Two views of the same effect vectors: what reproduces, and which settings agree."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from mantispy._core.frames import as_frame
from mantispy.pl._common import axes as _axes
from mantispy.pl._common import table as _table
from mantispy.pl._moa import _heatmap

if TYPE_CHECKING:
    from anndata import AnnData
    from matplotlib.axes import Axes


def transport(
    adata: AnnData, key: str = "transport", level: str | None = None, top: int = 25, ax: Axes | None = None
) -> Axes:
    """Agreement per perturbation, ranked, with the ones that reproduce colored.

    With more than ``top`` perturbations, the highest and lowest ranked are shown.
    The perturbations at the bottom are the ones whose effects did not reproduce across settings.

    Args:
        adata: Object holding the table :func:`~mantispy.tl.transport` wrote.
        key: Name of that table in ``uns["mantispy"]``.
        level: Which level of that table to draw, or ``None`` for the last one it holds.
        top: How many perturbations to draw, taken half from each end of the ranking when there are more.
        ax: Axes to draw on, or ``None`` for a new figure.

    Returns:
        The axes drawn on, with one bar per perturbation, the ones that reproduce in crimson, and how many of them do in the title.

    Raises:
        KeyError: There is no such table, or it holds no such level.
    """
    table = _table(adata, key, "mt.tl.transport")
    levels = list(dict.fromkeys(table["level"]))
    chosen = level if level is not None else levels[-1]
    if chosen not in levels:
        raise KeyError(f"no level {chosen!r} in uns['mantispy'][{key!r}]; it holds {levels}")
    block = table[table["level"] == chosen].sort_values("agreement", ascending=False)
    shown = pd.concat([block.head(top // 2), block.tail(top - top // 2)]) if len(block) > top else block

    ax = _axes(ax, (5.5, 0.22 * len(shown) + 1.5))
    positions = np.arange(len(shown))[::-1]
    colours = ["crimson" if flag else "lightgrey" for flag in shown["transports"]]
    ax.barh(positions, shown["agreement"].to_numpy(), color=colours, height=0.7)
    ax.set_yticks(positions)
    ax.set_yticklabels(shown["group"].astype(str), fontsize=6)
    ax.axvline(0.0, color="grey", lw=0.8)
    ax.set_xlabel(f"effect agreement across {chosen.replace('Metadata_', '').lower()}")
    ax.set_title(f"{int(block['transports'].sum())} of {len(block)} reproduce", fontsize=9)
    return ax


def setting_agreement(
    adata: AnnData, key: str = "transport", by: str | None = None, cluster: bool = True, ax: Axes | None = None
) -> Axes:
    """Settings against settings: which plates, batches or laboratories agree with each other.

    Uses the same effect vectors as :func:`transport`, compared between settings instead of between perturbations.
    Two settings agree when the perturbations they share moved the same way in both, weighted by effect size, so agreement on inactive compounds counts for little.

    Args:
        adata: Object :func:`~mantispy.tl.transport` has run on.
        key: The key it was stored under.
        by: ``obs`` column to annotate the settings with, e.g. ``"Metadata_Source"`` when the units are plates. Settings are ordered by it with a line at each boundary, so a laboratory whose plates disagree shows as a broken block.
        cluster: Order the settings by similarity instead. Ignored when ``by`` is given.
        ax: Axes to draw into.

    Returns:
        The axes drawn on, holding the settings-by-settings agreement matrix with a white line at each ``by`` boundary.

    Raises:
        KeyError: ``uns["mantispy"]`` holds no ``key + "_units"`` matrix, or ``by`` was given and no ``obs`` column names the settings it holds.
    """
    matrix = _table(adata, f"{key}_units", "mt.tl.transport").astype(float)
    labels = [str(name) for name in matrix.columns]
    matrix.index = pd.Index(labels)
    matrix.columns = pd.Index(labels)

    annotation = None
    obs = as_frame(adata.obs)
    if by is not None and by in obs:
        naming = next((column for column in obs.columns if set(obs[column].astype(str)) >= set(labels)), None)
        if naming is None:
            raise KeyError(
                f"no obs column holds all the settings {labels[:3]}...; the column mt.tl.transport read them from "
                "is missing"
            )
        lookup = obs.groupby(obs[naming].astype(str), observed=True)[by].first().astype(str)
        annotation = [str(lookup.get(label, "")) for label in labels]
        order = np.argsort(annotation, kind="stable")
    elif cluster and len(labels) > 2:
        from scipy.cluster.hierarchy import leaves_list, linkage
        from scipy.spatial.distance import squareform

        values = matrix.to_numpy(dtype=float)
        # A missing pair shared too few perturbations to compare. The median fill keeps the
        # linkage defined without favoring any pair.
        filled = np.nan_to_num(values, nan=float(np.nanmedian(values)))
        distance = np.clip(1.0 - (filled + filled.T) / 2.0, 0.0, None)
        np.fill_diagonal(distance, 0.0)
        order = leaves_list(linkage(squareform(distance, checks=False), method="average")).astype(np.intp)
    else:
        order = np.arange(len(labels))

    labels = [labels[index] for index in order]
    matrix = matrix.loc[labels, labels]
    if annotation is not None:
        annotation = [annotation[index] for index in order]

    ax = _axes(ax, (0.32 * len(labels) + 3, 0.28 * len(labels) + 2.5))
    _heatmap(ax, matrix.to_numpy(dtype=float), labels, labels, "viridis", "effect agreement")
    if annotation:
        for position in np.flatnonzero(np.asarray(annotation[1:]) != np.asarray(annotation[:-1])) + 1:
            ax.axhline(position - 0.5, color="white", lw=1.2)
            ax.axvline(position - 0.5, color="white", lw=1.2)
    return ax
