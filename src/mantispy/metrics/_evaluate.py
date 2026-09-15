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
}


def evaluate_correction(
    adata: AnnData,
    reps: Sequence[str] = ("X_pca",),
    label_key: str = "Metadata_Perturbation",
    batch_key: str = "Metadata_Batch",
    map_key: str | None = None,
) -> pd.DataFrame:
    """Run every metric for every representation and stack the results.

    Args:
        adata: Object holding the representations in ``obsm``.
        reps: Representations to compare, e.g. ``("X_pca", "X_pca_harmony")``.
        label_key: ``obs`` column with the biological grouping.
        batch_key: ``obs`` column with the nuisance grouping.
        map_key: Name of a table written by :func:`~mantispy.tl.map`, to add its mean mAP as one
            more row per representation.

    Returns:
        A tidy frame with ``metric``, ``representation``, ``key``, ``value`` and ``better``,
        the last saying which direction is an improvement for that metric.
    """
    frames = []
    for rep in reps:
        frames += [
            silhouette_label(adata, label_key=label_key, use_rep=rep),
            silhouette_batch(adata, label_key=label_key, batch_key=batch_key, use_rep=rep),
            # kind is explicit because a batch_key such as "Metadata_Site" would otherwise be
            # named clisi and share a metric name with the label row.
            lisi(adata, key=batch_key, use_rep=rep, kind="batch"),
            lisi(adata, key=label_key, use_rep=rep, kind="label"),
            pc_regression(adata, key=batch_key, use_rep=rep),
        ]
        if map_key is not None:
            table = adata.uns.get("mantispy", {}).get(map_key)
            if table is None:
                raise KeyError(f"uns['mantispy'] has no {map_key!r}; run mt.tl.map first")
            frames.append(tidy("mean_average_precision", rep, label_key, float(table["mean_average_precision"].mean())))

    result = pd.concat(frames, ignore_index=True)
    result["better"] = result["metric"].map(BETTER)
    return result
