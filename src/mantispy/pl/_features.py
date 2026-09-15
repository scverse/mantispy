"""Plots about the feature space itself."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mantispy._core._corr import corr_matrix
from mantispy._core._reduce import get_matrix
from mantispy._core._utils import as_frame, feature_mask

if TYPE_CHECKING:
    from anndata import AnnData


def feature_correlation(
    adata: AnnData,
    key: str | None = "selected",
    groupby: str = "feature_group",
    max_features: int = 300,
    ax: plt.Axes | None = None,
):
    """Correlation heatmap with features ordered by their annotation.

    Features are sorted by ``groupby`` and then by channel, with a line at each group
    boundary. Ordering by annotation instead of clustering shows directly whether
    correlated features fall within the same measurement family.

    Args:
        adata: Object to draw. Usually well-level profiles.
        key: Restrict to features flagged by this boolean ``var`` column. ``None`` uses all.
        groupby: ``var`` column to order and delimit by.
        max_features: Draw at most this many features, taken in the sorted order.
        ax: Axes to draw into.
    """
    mask = feature_mask(adata, key)
    annotation = as_frame(adata.var).loc[mask, [groupby, "channel"]].astype(str)
    order = annotation.sort_values([groupby, "channel"]).index[:max_features]

    positions = adata.var_names.get_indexer(order)
    correlation = corr_matrix(get_matrix(adata)[:, positions])

    ax = ax or plt.subplots(figsize=(7, 6))[1]
    image = ax.imshow(np.nan_to_num(correlation, nan=0.0), cmap="RdBu_r", vmin=-1, vmax=1)

    labels = as_frame(adata.var).loc[order, groupby].astype(str).to_numpy()
    boundaries = np.flatnonzero(labels[1:] != labels[:-1]) + 0.5
    for boundary in boundaries:
        ax.axhline(boundary, color="black", lw=0.5)
        ax.axvline(boundary, color="black", lw=0.5)

    centres = np.concatenate([[0], boundaries, [len(labels)]])
    ticks = (centres[:-1] + centres[1:]) / 2
    ax.set_xticks(ticks)
    ax.set_xticklabels(pd.unique(labels), rotation=90, fontsize=7)
    ax.set_yticks(ticks)
    ax.set_yticklabels(pd.unique(labels), fontsize=7)
    ax.set_title(f"feature correlation ({len(order)} features)", fontsize=9)
    ax.figure.colorbar(image, ax=ax, fraction=0.04, label="correlation")
    return ax


def feature_groups(adata: AnnData, key: str | None = None, ax: plt.Axes | None = None):
    """How many features each group contributes, split by channel.

    Pass ``key="selected"`` after feature selection to see which families survived.
    """
    mask = feature_mask(adata, key)
    annotation = as_frame(adata.var).loc[mask, ["feature_group", "channel"]].astype(str)
    counts = annotation.value_counts().unstack(fill_value=0)

    ax = ax or plt.subplots(figsize=(7, 4))[1]
    counts.plot.bar(stacked=True, ax=ax)
    ax.set_ylabel("features")
    ax.set_xlabel("feature group")
    ax.legend(title="channel", fontsize=6, title_fontsize=7)
    ax.tick_params(axis="x", rotation=45)
    return ax
