"""Plots for judging profile strength and correction quality."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mantispy._core.frames import as_frame
from mantispy.metrics._common import embedding, r_squared
from mantispy.pl._common import table

if TYPE_CHECKING:
    from anndata import AnnData


_table = table


def map(adata: AnnData, key: str = "map", label_top: int = 10, ax: plt.Axes | None = None):
    """Mean average precision against significance, with the strongest groups labeled.

    The dashed line is the significance threshold the run used, so a point above it and
    to the right is a perturbation that is both strong and reproducible.
    """
    table = _table(adata, key, "mt.tl.map")
    ax = ax or plt.subplots(figsize=(5.5, 4.5))[1]

    significance = -np.log10(np.clip(table["corrected_p_value"].to_numpy(dtype=float), 1e-12, None))
    ax.scatter(table["mean_average_precision"], significance, s=14)

    threshold = adata.uns.get("mantispy", {}).get("params", {}).get("map", {}).get("threshold", 0.05)
    ax.axhline(-np.log10(threshold), color="grey", ls="--", lw=1, label=f"q = {threshold}")

    group_column = table.columns[0]
    for _, row in table.nlargest(label_top, "mean_average_precision").iterrows():
        ax.annotate(
            str(row[group_column]),
            (row["mean_average_precision"], -np.log10(max(float(row["corrected_p_value"]), 1e-12))),
            fontsize=6,
        )
    ax.set_xlabel("mean average precision")
    ax.set_ylabel("-log10 corrected p")
    ax.legend(fontsize=7)
    return ax


def replicate_correlation(adata: AnnData, key: str = "percent_replicating", ax: plt.Axes | None = None):
    """Observed replicate correlation against each group's permutation threshold.

    Points above the diagonal replicate; the distance from it is the margin.
    """
    table = _table(adata, key, "mt.tl.percent_replicating")
    ax = ax or plt.subplots(figsize=(5, 4.5))[1]

    replicating = table["is_replicating"].to_numpy(dtype=bool)
    ax.scatter(
        table["null_threshold"][~replicating], table["median_replicate_correlation"][~replicating], s=14, label="no"
    )
    ax.scatter(
        table["null_threshold"][replicating],
        table["median_replicate_correlation"][replicating],
        s=14,
        color="seagreen",
        label="yes",
    )
    limits = [
        float(np.nanmin([table["null_threshold"].min(), table["median_replicate_correlation"].min()])),
        float(np.nanmax([table["null_threshold"].max(), table["median_replicate_correlation"].max()])),
    ]
    ax.plot(limits, limits, color="grey", ls="--", lw=1)
    ax.set_xlabel("null threshold")
    ax.set_ylabel("median replicate correlation")
    ax.legend(title="replicating", fontsize=7, title_fontsize=7)
    return ax


def batch_variance(
    adata: AnnData,
    keys: Sequence[str],
    use_rep: str = "X_pca",
    n_comps: int | None = None,
    ax: plt.Axes | None = None,
):
    """R^2 of each principal component on each covariate.

    A covariate with high R^2 in the leading components accounts for much of the
    embedding's structure.
    """
    values = embedding(adata, use_rep)
    if n_comps is not None:
        values = values[:, :n_comps]

    ax = ax or plt.subplots(figsize=(6, 4))[1]
    components = np.arange(1, values.shape[1] + 1)
    for key in keys:
        covariate = as_frame(adata.obs)[key]
        explained = [r_squared(values[:, index], covariate) for index in range(values.shape[1])]
        ax.plot(components, explained, marker="o", ms=3, label=key)
    ax.set_xlabel("principal component")
    ax.set_ylabel("variance explained (R²)")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7)
    return ax


def metrics(table: pd.DataFrame, ax: plt.Axes | None = None):
    """Grouped bars of an :func:`~mantispy.metrics.evaluate_correction` table.

    Takes the table instead of an AnnData because the table already holds every
    representation side by side. When the table has a ``better`` column, each tick label
    says which direction is better for that metric.
    """
    pivot = table.pivot(index="metric", columns="representation", values="value")
    ax = ax or plt.subplots(figsize=(7, 4))[1]
    pivot.plot.bar(ax=ax)

    if "better" in table.columns:
        direction = table.drop_duplicates("metric").set_index("metric")["better"]
        ax.set_xticklabels([f"{name}\n({direction.get(name, '?')} is better)" for name in pivot.index], fontsize=7)
    ax.set_ylabel("value")
    ax.set_xlabel("")
    ax.legend(fontsize=7)
    return ax


def similarity(
    adata: AnnData,
    key: str = "similarity",
    groupby: str | None = "Metadata_Perturbation",
    max_obs: int = 500,
    ax: plt.Axes | None = None,
):
    """Profile-by-profile similarity, ordered by ``groupby`` so blocks are visible.

    Subsamples to ``max_obs`` rows with a fixed seed when the object is larger, because
    the matrix is quadratic.
    """
    if key not in adata.obsp:
        raise KeyError(f"obsp has no {key!r}; run mt.tl.similarity first")
    matrix = np.asarray(adata.obsp[key])

    order = np.arange(adata.n_obs)
    if groupby is not None:
        order = np.argsort(adata.obs[groupby].astype(str).to_numpy(), kind="stable")
    if order.size > max_obs:
        # Sample positions within the ordering, not row indices, so sorting the sample keeps
        # the groupby blocks intact.
        picked = np.sort(np.random.default_rng(0).choice(order.size, size=max_obs, replace=False))
        order = order[picked]

    ax = ax or plt.subplots(figsize=(6, 5))[1]
    image = ax.imshow(matrix[np.ix_(order, order)], cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_title(f"{key} ({order.size} profiles)", fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.figure.colorbar(image, ax=ax, fraction=0.045, label="similarity")
    return ax
