"""Plots for mechanism retrieval and feature-set enrichment."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mantispy._core.frames import as_frame
from mantispy.pl._common import axes as _axes
from mantispy.pl._common import table as _table

if TYPE_CHECKING:
    from collections.abc import Sequence

    from anndata import AnnData
    from matplotlib.axes import Axes


def _heatmap(
    ax: Axes,
    values: np.ndarray,
    rows: Sequence[str],
    columns: Sequence[str],
    cmap: str,
    label: str,
    fmt: str | None = None,
) -> Axes:
    image = ax.imshow(values, aspect="auto", cmap=cmap)
    ax.set_xticks(np.arange(len(columns)))
    ax.set_xticklabels(columns, rotation=90, fontsize=6)
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels(rows, fontsize=6)
    if fmt is not None and values.size <= 400:
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                ax.text(j, i, format(values[i, j], fmt), ha="center", va="center", fontsize=5)
    ax.figure.colorbar(image, ax=ax, label=label)
    return ax


def moa_confusion(adata: AnnData, key: str = "moa", normalize: bool = True, ax: Axes | None = None) -> Axes:
    """The confusion matrix of :func:`~mantispy.tl.nn_moa_classify`, as a heatmap.

    With row normalization the diagonal is per-mechanism recall, and an off-diagonal block marks a pair of mechanisms the morphology does not separate.
    Such pairs usually have similar phenotypes.

    Args:
        adata: Object holding the confusion table :func:`~mantispy.tl.nn_moa_classify` wrote.
        key: Name that run's outputs were stored under, whose confusion table is ``key + "_confusion"``.
        normalize: Divide each row by its total, which turns the counts into per-mechanism recall.
        ax: Axes to draw on, or ``None`` for a new figure.

    Returns:
        The axes drawn on, holding true against predicted mechanisms with the values printed when there are at most 400 cells, and the run's scheme and accuracy in the title.

    Raises:
        KeyError: ``uns["mantispy"]`` holds no ``key + "_confusion"`` table.
    """
    table = _table(adata, f"{key}_confusion", "mt.tl.nn_moa_classify")
    matrix = table.pivot_table(index="true", columns="predicted", values="count", aggfunc="sum", fill_value=0)
    labels = sorted(set(matrix.index) | set(matrix.columns))
    counts = matrix.reindex(index=labels, columns=labels, fill_value=0).to_numpy(dtype=float)

    if normalize:
        totals = counts.sum(axis=1, keepdims=True)
        counts = np.divide(counts, totals, out=np.zeros_like(counts), where=totals > 0)

    ax = _axes(ax, (0.45 * len(labels) + 3, 0.4 * len(labels) + 2.5))
    _heatmap(ax, counts, labels, labels, "Blues", "fraction" if normalize else "count", ".2f" if normalize else ".0f")
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    summary = adata.uns.get("mantispy", {}).get(key, {})
    ax.set_title(f"{summary.get('scheme', '')} accuracy {float(summary.get('accuracy', float('nan'))):.1%}", fontsize=9)
    return ax


def moa_enrichment(
    adata: AnnData, group: str, key: str = "moa_enrichment", top: int = 10, ax: Axes | None = None
) -> Axes:
    """Which mechanisms one profile's neighborhood is enriched for.

    Args:
        adata: Object holding the table :func:`~mantispy.tl.moa_enrichment` wrote.
        group: Which group of that table to draw.
        key: Name of that table in ``uns["mantispy"]``.
        top: How many mechanisms to draw, taken by p-value.
        ax: Axes to draw on, or ``None`` for a new figure.

    Returns:
        The axes drawn on, with one bar of ``-log10`` q per mechanism, labeled by how many of the neighbors carried it, and a reference line at ``q = 0.05``.

    Raises:
        KeyError: There is no such table, or it holds no such group.
    """
    table = _table(adata, key, "mt.tl.moa_enrichment")
    selected = table[table["group"].astype(str) == str(group)]
    if selected.empty:
        raise KeyError(f"no group {group!r} in uns['mantispy'][{key!r}]")

    best = selected.nsmallest(top, "pvalue")[::-1]
    ax = _axes(ax, (5.5, 0.3 * len(best) + 1.5))
    ax.barh(np.arange(len(best)), -np.log10(np.clip(best["qvalue"].to_numpy(dtype=float), 1e-12, None)))
    ax.set_yticks(np.arange(len(best)))
    ax.set_yticklabels([f"{moa}  ({n})" for moa, n in zip(best["moa"], best["n_neighbours"], strict=True)], fontsize=6)
    ax.axvline(-np.log10(0.05), color="grey", ls="--", lw=1)
    ax.set_xlabel("-log10 q")
    ax.set_title(str(group), fontsize=9)
    return ax


def distance_heatmap(
    adata: AnnData, key: str = "edistance", groupby: str | None = "Metadata_MOA", ax: Axes | None = None
) -> Axes:
    """The group-by-group distance matrix, ordered so related groups sit together.

    Args:
        adata: Object holding the pairwise matrix :func:`~mantispy.tl.edistance` wrote with ``reference=None``.
        key: Name that run's outputs were stored under, whose matrix is ``key + "_pairwise"``.
        groupby: ``obs`` column to order the groups by, or ``None`` to keep the matrix's own order. Ordering also needs an ``obs`` column naming the groups of the matrix, and without one the matrix is drawn unordered rather than refused.
        ax: Axes to draw on, or ``None`` for a new figure.

    Returns:
        The axes drawn on, holding the distance matrix with a white line at each ``groupby`` boundary.

    Raises:
        KeyError: ``uns["mantispy"]`` holds no ``key + "_pairwise"`` matrix.
    """
    matrix = _table(adata, f"{key}_pairwise", "mt.tl.edistance(reference=None)")
    labels = list(matrix.columns)
    matrix.index = pd.Index(labels)

    obs = as_frame(adata.obs)
    annotation = None
    if groupby and groupby in obs:
        naming = next((column for column in obs.columns if set(obs[column].astype(str)) >= set(labels)), None)
        if naming is not None:
            lookup = obs.groupby(obs[naming].astype(str), observed=True)[groupby].first().astype(str)
            annotation = [str(lookup.get(label, "")) for label in labels]
            order = np.argsort(annotation, kind="stable")
            labels = [labels[index] for index in order]
            annotation = [annotation[index] for index in order]
            matrix = matrix.loc[labels, labels]

    ax = _axes(ax, (0.25 * len(labels) + 3, 0.22 * len(labels) + 2.5))
    _heatmap(ax, matrix.to_numpy(dtype=float), labels, labels, "magma", "energy distance")
    if annotation:
        for position in np.flatnonzero(np.asarray(annotation[1:]) != np.asarray(annotation[:-1])) + 1:
            ax.axhline(position - 0.5, color="white", lw=0.8)
            ax.axvline(position - 0.5, color="white", lw=0.8)
    return ax


def sets_heatmap(
    adata: AnnData, groupby: str, score_key: str = "score_ulm", top: int = 30, ax: Axes | None = None
) -> Axes:
    """Mean enrichment score per group per feature set.

    Args:
        adata: Object :func:`~mantispy.tl.enrich` has scored, holding the per-row scores in ``obsm``.
        groupby: ``obs`` column whose groups become the rows.
        score_key: ``obsm`` key holding those scores, named after the method that wrote them.
        top: How many feature sets to draw, taken by their largest absolute mean score.
        ax: Axes to draw on, or ``None`` for a new figure.

    Returns:
        The axes drawn on, holding groups against feature sets, the sets it kept sorted by name, on a diverging scale.

    Raises:
        KeyError: ``obsm`` holds nothing under ``score_key``.
    """
    if score_key not in adata.obsm:
        raise KeyError(f"obsm has no {score_key!r}; run mt.tl.enrich first")

    stored = adata.obsm[score_key]
    names = list(stored.columns) if hasattr(stored, "columns") else [str(i) for i in range(np.shape(stored)[1])]
    scores = np.asarray(stored, dtype=float)
    groups = as_frame(adata.obs)[groupby].astype(str).to_numpy()
    labels = list(dict.fromkeys(groups))

    means = np.stack([np.nanmean(scores[groups == group], axis=0) for group in labels])
    keep = np.argsort(-np.nanmax(np.abs(means), axis=0))[:top]
    keep = keep[np.argsort([names[index] for index in keep])]

    ax = _axes(ax, (0.3 * len(keep) + 3, 0.28 * len(labels) + 2))
    _heatmap(ax, means[:, keep], labels, [names[index] for index in keep], "coolwarm", "mean score")
    return ax


def pathway_coherence(adata: AnnData, key: str = "pathway_coherence", top: int = 15, ax: Axes | None = None) -> Axes:
    """Coherence per gene set, the significant ones marked.

    Sets are ordered by coherence, as in the table.
    Under a permutation null every coherent set ties at the p-value floor, so the q-value marks significance and coherence ranks the sets.

    Args:
        adata: Object holding the table :func:`~mantispy.tl.pathway_coherence` wrote.
        key: Name of that table in ``uns["mantispy"]``.
        top: How many sets to draw, taken by coherence.
        ax: Axes to draw on, or ``None`` for a new figure.

    Returns:
        The axes drawn on, with one bar per set labeled by how many of its genes were in the screen, colored by whether its q-value is below 0.05.

    Raises:
        KeyError: ``uns["mantispy"]`` holds no table under ``key``.
        ValueError: That table is empty, which is what happens when no set had enough of its genes in the screen.
    """
    table = _table(adata, key, "mt.tl.pathway_coherence")
    if table.empty:
        raise ValueError(f"uns['mantispy'][{key!r}] is empty; no set had enough of its genes in the screen")

    best = table.nlargest(min(top, len(table)), "coherence")[::-1]
    significant = best["qvalue"].to_numpy(dtype=float) < 0.05
    ax = _axes(ax, (6, 0.3 * len(best) + 1.5))
    ax.barh(np.arange(len(best)), best["coherence"], color=np.where(significant, "crimson", "lightgrey"))
    ax.set_yticks(np.arange(len(best)))
    ax.set_yticklabels([f"{name}  ({n})" for name, n in zip(best["set"], best["n_genes"], strict=True)], fontsize=6)
    ax.set_xlabel("mean similarity among the set's genes")
    ax.legend(
        handles=[
            plt.Line2D([], [], color="crimson", lw=6, label="q < 0.05"),
            plt.Line2D([], [], color="lightgrey", lw=6, label="not significant"),
        ],
        fontsize=6,
        loc="lower right",
    )
    return ax
