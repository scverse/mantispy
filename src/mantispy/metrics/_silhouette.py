"""Silhouette-based integration metrics, delegated to scib-metrics."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from mantispy.metrics._common import embedding, require_scib_metrics, tidy

if TYPE_CHECKING:
    from anndata import AnnData


def _silhouette_label(adata: AnnData, label_key: str, use_rep: str = "X_pca") -> pd.DataFrame:
    """How well separated the biological labels are, rescaled to ``[0, 1]``.

    Higher means tighter, better separated groups. Computed by scib-metrics.

    Args:
        adata: Object with the embedding to measure in.
        label_key: ``obs`` column with the biological grouping.
        use_rep: ``obsm`` key of the embedding.

    Returns:
        A one-row tidy frame holding ``silhouette_label``, whose value is NaN when separation is undefined for the object, which is when it holds one label or one row per label.

    Raises:
        ImportError: scib-metrics is not installed.
        KeyError: ``obsm`` holds nothing under ``use_rep``.
    """
    scib_metrics = require_scib_metrics()

    values = embedding(adata, use_rep)
    labels = adata.obs[label_key].to_numpy()

    n_labels = len(pd.unique(labels))
    if not 2 <= n_labels <= adata.n_obs - 1:
        # The silhouette needs between 2 and n_obs - 1 distinct labels; scib would raise here without naming the cause.
        warnings.warn(
            f"the label silhouette is undefined for obs[{label_key!r}]: it needs between 2 and "
            f"n_obs - 1 distinct labels, and this object has {n_labels} over {adata.n_obs} rows. "
            "One row per label, as a consensus object has, is the usual cause. Returning NaN.",
            UserWarning,
            stacklevel=2,
        )
        return tidy("silhouette_label", use_rep, label_key, np.nan)

    # rescale=True (the default) maps the average silhouette width into [0, 1].
    value = scib_metrics.silhouette_label(values, labels)
    return tidy("silhouette_label", use_rep, label_key, float(value))


def _silhouette_batch(adata: AnnData, label_key: str, batch_key: str, use_rep: str = "X_pca") -> pd.DataFrame:
    """How well mixed the batches are within each biological label.

    Computed by scib-metrics per label as ``1 - mean|silhouette over batch|`` and averaged over labels.
    Higher is better; a silhouette near zero means the batches are indistinguishable within that label.

    A label is skipped when mixing is undefined for it, which is when it has fewer than two batches or as many batches as rows (one well per plate on three plates, for example).

    Args:
        adata: Object with the embedding to measure in.
        label_key: ``obs`` column with the biological grouping, whose labels the batches are mixed within.
        batch_key: ``obs`` column with the nuisance grouping.
        use_rep: ``obsm`` key of the embedding.

    Returns:
        A one-row tidy frame holding ``silhouette_batch``, whose value is NaN when every label was skipped.

    Raises:
        ImportError: scib-metrics is not installed.
        KeyError: ``obsm`` holds nothing under ``use_rep``.
    """
    scib_metrics = require_scib_metrics()

    values = embedding(adata, use_rep)
    labels = adata.obs[label_key].to_numpy()
    batches = adata.obs[batch_key].to_numpy()

    # scib raises when every label group is undefined; check first so evaluate_correction gets NaN, not an error.
    # A group is scored only when it holds more than one batch but fewer batches than rows.
    def _scorable(label: object) -> bool:
        rows = labels == label
        return 1 < len(np.unique(batches[rows])) < int(rows.sum())

    if not any(_scorable(label) for label in pd.unique(labels)):
        return tidy("silhouette_batch", use_rep, batch_key, np.nan)

    value = scib_metrics.silhouette_batch(values, labels, batches)
    return tidy("silhouette_batch", use_rep, batch_key, float(value))
