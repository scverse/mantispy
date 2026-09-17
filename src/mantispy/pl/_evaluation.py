"""Plots for judging profile strength and correction quality."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

from mantispy._core.frames import as_frame
from mantispy.metrics._common import embedding, r_squared
from mantispy.pl._common import axes as _axes
from mantispy.pl._common import table as _table

if TYPE_CHECKING:
    import pandas as pd
    from anndata import AnnData
    from matplotlib.axes import Axes


def map(adata: AnnData, key: str = "map", label_top: int = 10, ax: Axes | None = None) -> Axes:
    """Mean average precision against significance, with the strongest groups labeled.

    The dashed line is the significance threshold the run used, so a point above it and to the right is a perturbation that is both strong and reproducible.

    Args:
        adata: Object holding the table :func:`~mantispy.tl.map` wrote.
        key: Name of that table in ``uns["mantispy"]``.
        label_top: How many of the strongest groups to label, taken by mean average precision.
        ax: Axes to draw on, or ``None`` for a new figure.

    Returns:
        The axes drawn on, with one point per group and the threshold drawn as a labeled reference line.

    Raises:
        KeyError: ``uns["mantispy"]`` holds no table under ``key``.
    """
    table = _table(adata, key, "mt.tl.map")
    ax = _axes(ax, (5.5, 4.5))

    significance = -np.log10(np.clip(table["corrected_p_value"].to_numpy(dtype=float), 1e-12, None))
    ax.scatter(table["mean_average_precision"], significance, s=14)

    # The threshold is recorded under the name of the function, not under the name of the table
    # it wrote, so it stays readable whatever `key` is.
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


def replicate_correlation(adata: AnnData, key: str = "percent_replicating", ax: Axes | None = None) -> Axes:
    """Observed replicate correlation against each group's permutation threshold.

    Points above the diagonal replicate; the distance from it is the margin.

    Args:
        adata: Object holding the table :func:`~mantispy.tl.percent_replicating` wrote.
        key: Name of that table in ``uns["mantispy"]``.
        ax: Axes to draw on, or ``None`` for a new figure.

    Returns:
        The axes drawn on, with the groups that replicate colored apart from those that do not and the diagonal drawn as a reference line.

    Raises:
        KeyError: ``uns["mantispy"]`` holds no table under ``key``.
    """
    table = _table(adata, key, "mt.tl.percent_replicating")
    ax = _axes(ax, (5, 4.5))

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
    ax: Axes | None = None,
) -> Axes:
    """R^2 of each principal component on each covariate.

    A covariate with high R^2 in the leading components accounts for much of the embedding's structure.

    Args:
        adata: Object with the embedding to measure in.
        keys: ``obs`` columns to score, one line each.
        use_rep: ``obsm`` key of the embedding.
        n_comps: Draw only the leading components, or ``None`` for every component the embedding holds.
        ax: Axes to draw on, or ``None`` for a new figure.

    Returns:
        The axes drawn on, with one line per entry of ``keys`` and the y axis fixed to ``[0, 1]``.

    Raises:
        KeyError: ``obsm`` holds nothing under ``use_rep``, or ``obs`` has no column for one of ``keys``.
    """
    values = embedding(adata, use_rep)
    if n_comps is not None:
        values = values[:, :n_comps]

    ax = _axes(ax, (6, 4))
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


def metrics(table: pd.DataFrame, ax: Axes | None = None) -> Axes:
    """Grouped bars of an :func:`~mantispy.metrics.evaluate_correction` table.

    Takes the table instead of an AnnData because the table already holds every representation side by side.

    Args:
        table: A tidy frame with ``metric``, ``representation`` and ``value``, as :func:`~mantispy.metrics.evaluate_correction` returns. Its ``better`` column, when present, adds the direction that is an improvement to each tick label.
        ax: Axes to draw on, or ``None`` for a new figure.

    Returns:
        The axes drawn on, with one group of bars per metric and one bar per representation.

    Raises:
        ValueError: The table holds more than one value for some metric and representation, which cannot be pivoted into a grid.
    """
    pivot = table.pivot(index="metric", columns="representation", values="value")
    ax = _axes(ax, (7, 4))
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
    ax: Axes | None = None,
) -> Axes:
    """Profile-by-profile similarity, ordered by ``groupby`` so blocks are visible.

    Subsamples to ``max_obs`` rows with a fixed seed when the object is larger, because the matrix is quadratic.

    Args:
        adata: Object holding the matrix :func:`~mantispy.tl.similarity` wrote in ``obsp``.
        key: Name of that matrix in ``obsp``.
        groupby: ``obs`` column to order the rows and columns by, or ``None`` to keep the object's own order.
        max_obs: Draw at most this many profiles, sampled from the ordering rather than from the rows.
        ax: Axes to draw on, or ``None`` for a new figure.

    Returns:
        The axes drawn on, holding the ordered matrix on a diverging scale fixed to ``[-1, 1]`` with a colorbar beside it.

    Raises:
        KeyError: ``obsp`` holds nothing under ``key``.
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

    ax = _axes(ax, (6, 5))
    image = ax.imshow(matrix[np.ix_(order, order)], cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_title(f"{key} ({order.size} profiles)", fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.figure.colorbar(image, ax=ax, fraction=0.045, label="similarity")
    return ax
