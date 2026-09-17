"""Helpers shared by the plot modules.

They create axes when the caller passes none, and fetch a result table with an error that names the function that writes it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
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


def table(adata: AnnData, key: str, produced_by: str) -> pd.DataFrame:
    """A result table from ``uns["mantispy"]``, or an error naming what writes it.

    Args:
        adata: Object holding the table.
        key: Name of the table in ``uns["mantispy"]``.
        produced_by: The call to name in the error, for example ``"mt.tl.hit_calling"``.

    Returns:
        The table as a frame.

    Raises:
        KeyError: ``uns["mantispy"]`` holds no table under ``key``.
    """
    store = adata.uns.get("mantispy", {})
    if key not in store:
        raise KeyError(f"uns['mantispy'][{key!r}] is missing; run {produced_by} first")
    return pd.DataFrame(store[key])
