"""Helpers shared by the plot modules.

They create axes when the caller passes none, and fetch a result table with an error that names the function that writes it.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from functools import cache
from importlib.util import find_spec
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    import numpy as np
    from anndata import AnnData
    from matplotlib.axes import Axes


def axes(ax: Axes | None, figsize: tuple[float, float]) -> Axes:
    """The caller's axes, or a new figure sized for this plot.

    Args:
        ax: Axes to draw on, or ``None`` for a new figure.
        figsize: Size of that new figure, in inches.

    Returns:
        The axes to draw on.
    """
    import matplotlib.pyplot as plt

    return ax if ax is not None else plt.subplots(figsize=figsize)[1]


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
    data: pd.DataFrame | None = None,
    x: str | None = None,
    y: str | None = None,
    color: str | None = None,
    hover: Sequence[str] | None = None,
    text: str | None = None,
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

    Args:
        kind: One of ``"scatter"``, ``"heatmap"``, ``"barh"``, ``"line"``, ``"histogram"``, ``"box"``.
        data: Tidy frame for every kind but ``"heatmap"``.
        x: Column drawn on the x axis, or the value column for ``"barh"``.
        y: Column drawn on the y axis, or the category column for ``"barh"``.
        color: Column that colors the marks, or ``None`` for one color.
        hover: Extra columns to add to the tooltip.
        text: Column whose values are drawn beside the marks.
        matrix: The 2-D array for ``"heatmap"``.
        rows: Row labels of ``matrix``.
        columns: Column labels of ``matrix``.
        value_label: Name of the colored quantity, for the ``"heatmap"`` colorbar and tooltip.
        title: Figure title, or ``None`` for none.

    Returns:
        Whether a twin was displayed; the caller then closes its matplotlib figure so
        the notebook does not also show the static PNG.
    """
    if not interactive_available():
        return False
    import plotly.express as px
    import plotly.io as pio
    from IPython.display import display

    pio.renderers.default = "plotly_mimetype+notebook_connected"
    figure = _build_interactive(
        px,
        kind,
        data=data,
        x=x,
        y=y,
        color=color,
        hover=hover,
        text=text,
        matrix=matrix,
        rows=rows,
        columns=columns,
        value_label=value_label,
        title=title,
    )
    display(figure)
    return True


def _build_interactive(
    px: object,
    kind: str,
    *,
    data: pd.DataFrame | None,
    x: str | None,
    y: str | None,
    color: str | None,
    hover: Sequence[str] | None,
    text: str | None,
    matrix: np.ndarray | None,
    rows: Sequence[str] | None,
    columns: Sequence[str] | None,
    value_label: str | None,
    title: str | None,
) -> object:
    """Build one plotly figure for ``kind``; the dispatch behind :func:`maybe_interactive`."""
    hover_list = list(hover) if hover is not None else None
    if kind == "scatter":
        return px.scatter(data, x=x, y=y, color=color, hover_data=hover_list, text=text, title=title)
    if kind == "line":
        return px.line(data, x=x, y=y, color=color, hover_data=hover_list, markers=True, title=title)
    if kind == "barh":
        return px.bar(data, x=x, y=y, color=color, hover_data=hover_list, orientation="h", title=title)
    if kind == "histogram":
        return px.histogram(data, x=x, color=color, hover_data=hover_list, title=title)
    if kind == "box":
        return px.box(data, x=x, y=y, color=color, hover_data=hover_list, title=title)
    if kind == "heatmap":
        return px.imshow(
            matrix,
            x=None if columns is None else list(columns),
            y=None if rows is None else list(rows),
            labels=None if value_label is None else {"color": value_label},
            aspect="auto",
            title=title,
        )
    raise ValueError(f"unknown interactive kind {kind!r}")


@contextmanager
def needs_plotly(feature: str) -> Iterator[None]:
    """Turn a missing-plotly ImportError into one that names the extra to install.

    Args:
        feature: The call to name in the message, for example ``"mt.pl.plate interactive"``.

    Yields:
        Nothing; the body runs inside the guard.

    Raises:
        ImportError: Plotly is missing, re-raised with the install instructions.
    """
    try:
        yield
    except ImportError as e:
        if (e.name or "").split(".")[0] != "plotly":
            raise
        raise ImportError(
            f"{feature} needs mantispy's interactive extra. Install it with: pip install 'mantispy[interactive]'"
        ) from e
