"""Helpers shared by the plot modules.

They create axes when the caller passes none, and fetch a result table with an error that names the function that writes it.
"""

from __future__ import annotations

import os
from functools import cache
from importlib.util import find_spec
from typing import TYPE_CHECKING, cast

import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np
    from anndata import AnnData
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

# Marks a figure that :func:`axes` created, so :func:`maybe_interactive` only closes
# figures this plot owns and never a subplot grid the caller passed in.
_OWNED = "_mantispy_owned"


def axes(ax: Axes | None, figsize: tuple[float, float]) -> Axes:
    """The caller's axes, or a new figure sized for this plot.

    Args:
        ax: Axes to draw on, or ``None`` for a new figure.
        figsize: Size of that new figure, in inches.

    Returns:
        The axes to draw on.
    """
    if ax is not None:
        return ax
    import matplotlib.pyplot as plt

    ax = plt.subplots(figsize=figsize)[1]
    setattr(ax.figure, _OWNED, True)
    return ax


def table(adata: AnnData, key: str, produced_by: str, when_empty: str | None = None) -> pd.DataFrame:
    """A result table from ``uns["mantispy"]``, or an error naming what writes it.

    Args:
        adata: Object holding the table.
        key: Name of the table in ``uns["mantispy"]``.
        produced_by: The call to name in the error, for example ``"mt.tl.hit_calling"``.
        when_empty: Why a table this plot can read holds no rows, for the plots where that is a plausible result rather than a missing step. With ``None`` an empty table is handed back for the caller to draw.

    Returns:
        The table as a frame.

    Raises:
        KeyError: ``uns["mantispy"]`` holds no table under ``key``.
        ValueError: The table is empty and ``when_empty`` says what that means.
    """
    store = adata.uns.get("mantispy", {})
    if key not in store:
        raise KeyError(f"uns['mantispy'][{key!r}] is missing; run {produced_by} first")
    frame = pd.DataFrame(store[key])
    if when_empty is not None and frame.empty:
        raise ValueError(f"uns['mantispy'][{key!r}] is empty; {when_empty}")
    return frame


@cache
def interactive_available() -> bool:
    """Whether an interactive plotly twin can be shown in the current runtime.

    True only when plotly is installed and we are in a Jupyter frontend
    (notebook, lab, qtconsole, or a notebook executed at docs build). Plain
    scripts, the terminal REPL, and headless pytest fall back to static plots.

    Returns:
        Whether to draw the interactive twin instead of only the static plot.
    """
    if find_spec("plotly") is None:
        return False
    try:
        from IPython import get_ipython
    except ImportError:
        return False
    if os.environ.get("MANTISPY_PLOTTING_BACKEND") == "static":
        return False
    return type(get_ipython()).__name__ == "ZMQInteractiveShell"


def maybe_interactive(
    kind: str,
    *,
    ax: Axes | None = None,
    data: pd.DataFrame | None = None,
    x: str | None = None,
    y: str | None = None,
    color: str | None = None,
    hover: Sequence[str] | None = None,
    barmode: str | None = None,
    matrix: np.ndarray | None = None,
    rows: Sequence[str] | None = None,
    columns: Sequence[str] | None = None,
    value_label: str | None = None,
    title: str | None = None,
) -> bool:
    """Show a plotly twin of a static plot, only when :func:`interactive_available`.

    The tidy kinds (``"scatter"``, ``"barh"``, ``"line"``, ``"histogram"``, ``"box"``)
    read a long ``data`` frame and its column names; ``"heatmap"`` reads a ``matrix``
    with its ``rows`` and ``columns`` labels. Every twin carries hover tooltips, so the
    identities the static plot can only label for its top few are readable on every mark.

    When a twin fires it closes ``ax``'s figure so the notebook does not also show the
    static PNG, but only when :func:`axes` created that figure; a caller-supplied subplot
    grid is left intact.

    Args:
        kind: One of ``"scatter"``, ``"heatmap"``, ``"barh"``, ``"line"``, ``"histogram"``, ``"box"``.
        ax: The static plot's axes, closed when its figure is owned and a twin fires.
        data: Tidy frame for every kind but ``"heatmap"``.
        x: Column drawn on the x axis, or the value column for ``"barh"``.
        y: Column drawn on the y axis, or the category column for ``"barh"``.
        color: Column that colors the marks, or ``None`` for one color.
        hover: Extra columns to add to the tooltip.
        barmode: ``"group"`` for side-by-side ``"barh"`` bars, or ``None`` for plotly's stacked default.
        matrix: The 2-D array for ``"heatmap"``.
        rows: Row labels of ``matrix``.
        columns: Column labels of ``matrix``.
        value_label: Name of the colored quantity, for the ``"heatmap"`` colorbar and tooltip.
        title: Figure title, or ``None`` for none.

    Returns:
        Whether a twin was displayed.
    """
    if not interactive_available():
        return False
    import plotly.express as px

    hover_list = list(hover) if hover is not None else None
    if kind == "scatter":
        figure = px.scatter(data, x=x, y=y, color=color, hover_data=hover_list, title=title)
    elif kind == "line":
        figure = px.line(data, x=x, y=y, color=color, hover_data=hover_list, markers=True, title=title)
    elif kind == "barh":
        figure = px.bar(
            data, x=x, y=y, color=color, hover_data=hover_list, orientation="h", barmode=barmode, title=title
        )
    elif kind == "histogram":
        figure = px.histogram(data, x=x, color=color, hover_data=hover_list, title=title)
    elif kind == "box":
        figure = px.box(data, x=x, y=y, color=color, hover_data=hover_list, title=title)
    elif kind == "heatmap":
        figure = px.imshow(
            matrix,
            x=None if columns is None else list(columns),
            y=None if rows is None else list(rows),
            labels=None if value_label is None else {"color": value_label},
            aspect="auto",
            title=title,
        )
    else:
        raise ValueError(f"unknown interactive kind {kind!r}")
    # Render with notebook_connected per call (loads plotly.js from CDN) so the twin
    # embeds in exported HTML docs, without mutating the process-global renderer.
    figure.show(renderer="notebook_connected")
    if ax is not None and getattr(ax.figure, _OWNED, False):
        import matplotlib.pyplot as plt

        # _OWNED is only set on a top-level Figure that axes() made, never a SubFigure.
        plt.close(cast("Figure", ax.figure))
    return True
