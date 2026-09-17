"""Silhouette-based integration metrics, following scib's definitions."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_samples, silhouette_score

from mantispy.metrics._common import embedding, tidy

if TYPE_CHECKING:
    from anndata import AnnData


def silhouette_label(adata: AnnData, label_key: str, use_rep: str = "X_pca") -> pd.DataFrame:
    """How well separated the biological labels are, rescaled to ``[0, 1]``.

    Higher means tighter, better separated groups.
    """
    labels = adata.obs[label_key].to_numpy()
    value = (silhouette_score(embedding(adata, use_rep), labels) + 1.0) / 2.0
    return tidy("silhouette_label", use_rep, label_key, value)


def silhouette_batch(adata: AnnData, label_key: str, batch_key: str, use_rep: str = "X_pca") -> pd.DataFrame:
    """How well mixed the batches are within each biological label.

    Computed per label as ``1 - mean|silhouette over batch|`` and averaged over labels.
    Higher is better; a silhouette near zero means the batches are indistinguishable within
    that label.

    A label is skipped when mixing is undefined for it, which is when it has fewer than two
    batches or as many batches as rows (one well per plate on three plates, for example).
    The value is NaN when every label is skipped.
    """
    values = embedding(adata, use_rep)
    labels = adata.obs[label_key].to_numpy()
    batches = adata.obs[batch_key].to_numpy()

    scores = []
    for label in pd.unique(labels):
        rows = labels == label
        n_rows, n_batches = int(rows.sum()), len(np.unique(batches[rows]))
        if n_rows < 3 or not 2 <= n_batches <= n_rows - 1:
            continue
        scores.append(float(np.mean(1.0 - np.abs(silhouette_samples(values[rows], batches[rows])))))
    return tidy("silhouette_batch", use_rep, batch_key, float(np.mean(scores)) if scores else np.nan)
