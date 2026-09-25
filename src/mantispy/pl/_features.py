"""Plots about the feature space itself."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from mantispy._core._corr import corr_matrix
from mantispy._core._reduce import get_matrix
from mantispy._core.frames import as_frame
from mantispy._core.masks import feature_mask
from mantispy.pl._common import axes as _axes

if TYPE_CHECKING:
    from anndata import AnnData
    from matplotlib.axes import Axes


def feature_correlation(
    adata: AnnData,
    key: str | None = "selected",
    groupby: str = "feature_group",
    max_features: int = 300,
    ax: Axes | None = None,
) -> Axes:
    """Correlation heatmap with features ordered by their annotation.

    Features are sorted by ``groupby`` and then by channel, with a line at each group boundary.
    Ordering by annotation instead of clustering shows directly whether correlated features fall within the same measurement family.

    Args:
        adata: Object to draw. Usually well-level profiles.
        key: Restrict to features flagged by this boolean ``var`` column. ``None``, or a column the object does not hold, uses every feature.
        groupby: ``var`` column to order and delimit by.
        max_features: Draw at most this many features, taken in the sorted order.
        ax: Axes to draw into.

    Returns:
        The axes drawn on, holding the correlation matrix on a diverging scale fixed to ``[-1, 1]``, with a line at each group boundary and one tick per group.

    Raises:
        KeyError: ``var`` has no ``groupby`` column, or no ``channel`` column.
    """
    mask = feature_mask(adata, key)
    annotation = as_frame(adata.var).loc[mask, [groupby, "channel"]].astype(str)
    order = annotation.sort_values([groupby, "channel"]).index[:max_features]

    positions = adata.var_names.get_indexer(order)
    correlation = corr_matrix(get_matrix(adata)[:, positions])

    ax = _axes(ax, (7, 6))
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


def feature_groups(adata: AnnData, key: str | None = None, ax: Axes | None = None) -> Axes:
    """How many features each group contributes, split by channel.

    Pass ``key="selected"`` after feature selection to see which families survived.

    Args:
        adata: Object to draw.
        key: Restrict to features flagged by this boolean ``var`` column. ``None``, or a column the object does not hold, uses every feature.
        ax: Axes to draw into.

    Returns:
        The axes drawn on, with one stacked bar per feature group and one segment per channel.

    Raises:
        KeyError: ``var`` has no ``feature_group`` column, or no ``channel`` column.
    """
    mask = feature_mask(adata, key)
    # A geometry feature has no channel; count it under "none" rather than let value_counts drop the NaN.
    annotation = as_frame(adata.var).loc[mask, ["feature_group", "channel"]].astype(object).fillna("none").astype(str)
    counts = annotation.value_counts().unstack(fill_value=0)
    # Colocalization features carry a pipe-joined channel pair; on their own each pair is a separate
    # legend entry, dozens in all. Collapse them into one "multiple" category.
    combined = [channel for channel in counts.columns if "|" in channel]
    if combined:
        counts = counts.drop(columns=combined).assign(multiple=counts[combined].sum(axis=1))
    order = [c for c in ["none"] if c in counts.columns]
    order += sorted(c for c in counts.columns if c not in ("none", "multiple"))
    order += [c for c in ["multiple"] if c in counts.columns]
    counts = counts[order]

    ax = _axes(ax, (7, 4))
    counts.plot.bar(stacked=True, ax=ax)
    ax.set_ylabel("features")
    ax.set_xlabel("feature group")
    ax.legend(title="channel", fontsize=7, title_fontsize=8, loc="upper left", bbox_to_anchor=(1.0, 1.0), frameon=False)
    ax.tick_params(axis="x", rotation=45)
    return ax
