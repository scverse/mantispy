"""Helpers shared by the plot modules.

They create axes when the caller passes none, and fetch a result table with an error that
names the function that writes it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import pandas as pd

if TYPE_CHECKING:
    from anndata import AnnData


def axes(ax: plt.Axes | None, figsize: tuple[float, float]) -> plt.Axes:
    """The caller's axes, or a new figure sized for this plot."""
    return ax if ax is not None else plt.subplots(figsize=figsize)[1]


def table(adata: AnnData, key: str, produced_by: str) -> pd.DataFrame:
    """A result table from ``uns["mantispy"]``, or an error naming what writes it."""
    store = adata.uns.get("mantispy", {})
    if key not in store:
        raise KeyError(f"uns['mantispy'][{key!r}] is missing; run {produced_by} first")
    return pd.DataFrame(store[key])
