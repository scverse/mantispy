"""Compare representations on one table."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import pandas as pd

from mantispy.metrics._common import tidy
from mantispy.metrics._lisi import lisi
from mantispy.metrics._silhouette import silhouette_batch, silhouette_label
from mantispy.metrics._variance import pc_regression

if TYPE_CHECKING:
    from anndata import AnnData

#: Which direction is better, per metric. Batch mixing and biological separation trade off
#: against each other, so read them together.
BETTER = {
    "silhouette_label": "higher",
    "silhouette_batch": "higher",
    "ilisi": "higher",
    "clisi": "lower",
    "pc_regression": "lower",
    "mean_average_precision": "higher",
    "known_relationships": "higher",
}


def _map_row(adata: AnnData, map_key: str, label_key: str) -> pd.DataFrame:
    """The mean mAP of a table :func:`~mantispy.tl.map` wrote, under the representation that run scored.

    The table is read rather than recomputed, so this is one row per call and not one row per representation: the same number repeated under every representation would read as a measured comparison.

    Args:
        adata: Object holding the table and the provenance of the run that wrote it.
        map_key: Name of that table in ``uns["mantispy"]``.
        label_key: ``obs`` column to name in the row's ``key``.

    Returns:
        A one-row tidy frame holding ``mean_average_precision``.

    Raises:
        KeyError: There is no such table.
        KeyError: Nothing recorded which representation :func:`~mantispy.tl.map` scored.
    """
    store = adata.uns.get("mantispy", {})
    table = store.get(map_key)
    if table is None:
        raise KeyError(f"uns['mantispy'] has no {map_key!r}; run mt.tl.map first")

    # Provenance is keyed by function name, and mt.tl.map records use_rep=None when it scores X.
    recorded = store.get("params", {}).get("map")
    if recorded is None:
        raise KeyError(
            f"uns['mantispy']['params'] holds no record of mt.tl.map, so the representation behind "
            f"{map_key!r} is unknown; rerun mt.tl.map to record it"
        )
    rep = recorded.get("use_rep") or "X"
    value = float(pd.DataFrame(table)["mean_average_precision"].mean())
    return tidy("mean_average_precision", rep, label_key, value)


def evaluate_correction(
    adata: AnnData,
    *,
    reps: Sequence[str] = ("X_pca",),
    label_key: str = "Metadata_Perturbation",
    batch_key: str = "Metadata_Batch",
    map_key: str | None = None,
    perplexity: float = 30,
) -> pd.DataFrame:
    """Run every metric for every representation and stack the results.

    Every argument after ``adata`` is keyword-only, so a later metric parameter can be added without changing what an existing argument means.

    Args:
        adata: Object holding the representations in ``obsm``.
        reps: Representations to compare, e.g. ``("X_pca", "X_pca_harmony")``.
        label_key: ``obs`` column with the biological grouping.
        batch_key: ``obs`` column with the nuisance grouping.
        map_key: Name of a table written by :func:`~mantispy.tl.map`, to add its mean mAP as one more row. That table is read rather than recomputed, so the row appears once, under the representation that run scored, and not once per entry of ``reps``.
        perplexity: Perplexity for both :func:`~mantispy.metrics.lisi` rows. The default needs more than 90 rows, and on a smaller object those two rows are NaN unless a smaller value is passed.

    Returns:
        A tidy frame with ``metric``, ``representation``, ``key``, ``value`` and ``better``, the last saying which direction is an improvement for that metric.
        A metric that is undefined for this object, such as a LISI whose perplexity the row count cannot support or a silhouette over one row per label, is NaN in that frame rather than an error, so one undefined metric still leaves the others readable.

    Raises:
        KeyError: ``obsm`` holds nothing under one of ``reps``.
        KeyError: ``map_key`` names no table, or nothing recorded the representation behind it.
    """
    frames = []
    for rep in reps:
        frames += [
            silhouette_label(adata, label_key=label_key, use_rep=rep),
            silhouette_batch(adata, label_key=label_key, batch_key=batch_key, use_rep=rep),
            # kind is explicit because a batch_key such as "Metadata_Site" would otherwise be
            # named clisi and share a metric name with the label row.
            lisi(adata, key=batch_key, use_rep=rep, perplexity=perplexity, kind="batch"),
            lisi(adata, key=label_key, use_rep=rep, perplexity=perplexity, kind="label"),
            pc_regression(adata, key=batch_key, use_rep=rep),
        ]
    if map_key is not None:
        frames.append(_map_row(adata, map_key, label_key))

    result = pd.concat(frames, ignore_index=True)
    result["better"] = result["metric"].map(BETTER)
    return result
