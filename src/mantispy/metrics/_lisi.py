"""Local inverse Simpson's index (LISI), delegated to scib-metrics.

iLISI (over a batch key) and cLISI (over a label key) are computed by scib-metrics
:cite:p:`Korsunsky_2019` from a neighbour graph derived from ``use_rep``. The unscaled
median LISI is returned, so ``ilisi`` reads as the effective number of batches in a
neighbourhood (higher is better mixed) and ``clisi`` as the effective number of labels
(lower means the biological groups stay separated).
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from mantispy.metrics._common import embedding, require_scib_metrics, tidy

if TYPE_CHECKING:
    from anndata import AnnData

_BATCH_HINTS = ("batch", "plate", "source", "week", "run")


def _lisi(adata: AnnData, key: str, use_rep: str = "X_pca", perplexity: float = 30, kind: str = "auto") -> pd.DataFrame:
    """Median LISI over rows :cite:p:`Korsunsky_2019`, computed by scib-metrics.

    Args:
        adata: Object with the embedding to measure in.
        key: ``obs`` column whose labels the neighborhoods are scored over.
        use_rep: ``obsm`` key of the embedding.
        perplexity: Perplexity the kernel width is calibrated to.
            Each neighborhood holds ``3 * perplexity`` rows, so an object with no more rows than that cannot support it and the value is NaN.
        kind: ``"batch"`` names the result ``ilisi`` (higher is better mixed) and ``"label"`` names it ``clisi`` (lower means the biological groups stay separated).
            ``"auto"`` guesses from the column name: a key containing batch, plate, source, week or run is a batch and anything else a label, so ``"Metadata_Site"`` counts as a label.
            Pass ``kind`` explicitly when one table holds both, or the two rows get the same metric name.

    Returns:
        A one-row tidy frame holding ``ilisi`` or ``clisi``, whose value is NaN when the object holds too few rows for ``perplexity``, which is what a 48-well plate or a consensus object with one row per perturbation does.

    Raises:
        ImportError: scib-metrics is not installed.
        KeyError: ``obsm`` holds nothing under ``use_rep``.
        ValueError: ``kind`` is not one of the three accepted values.
        ValueError: ``obs[key]`` has missing values.
    """
    scib_metrics = require_scib_metrics()
    from scib_metrics.nearest_neighbors import NeighborsResults
    from sklearn.neighbors import NearestNeighbors

    if kind not in ("auto", "batch", "label"):
        raise ValueError(f"kind must be 'auto', 'batch' or 'label', got {kind!r}")

    values = embedding(adata, use_rep)
    labels = adata.obs[key].to_numpy()
    missing = int(pd.isna(labels).sum())
    if missing:
        raise ValueError(f"obs[{key!r}] has {missing} missing value(s); drop those rows or fill the column.")

    if kind == "auto":
        kind = "batch" if any(hint in key.lower() for hint in _BATCH_HINTS) else "label"
    metric = "ilisi" if kind == "batch" else "clisi"

    n_neighbors = int(perplexity * 3)
    if n_neighbors >= adata.n_obs:
        # Each neighborhood needs n_neighbors + 1 rows; with fewer, scib cannot calibrate the kernel and LISI is undefined.
        supported = (adata.n_obs - 1) // 3
        remedy = f"pass a perplexity of at most {supported}" if supported >= 2 else "measure on a larger object"
        warnings.warn(
            f"LISI over obs[{key!r}] is undefined at perplexity={perplexity}: the kernel is calibrated over "
            f"{n_neighbors} neighbors, which needs {n_neighbors + 1} rows, and this object has {adata.n_obs}. "
            f"A 48-well plate, or a consensus object with one row per perturbation, is the usual cause; "
            f"to measure it, {remedy}. Returning NaN.",
            UserWarning,
            stacklevel=2,
        )
        return tidy(metric, use_rep, key, np.nan)

    # Self is kept in the graph (index 0); scib masks it out when calibrating the kernel.
    distances, indices = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(values).kneighbors(values)
    neighbors = NeighborsResults(indices=indices, distances=distances)
    # scale=False returns the raw median LISI, so ilisi stays "higher is better" and clisi "lower is better".
    knn = scib_metrics.ilisi_knn if kind == "batch" else scib_metrics.clisi_knn
    value = knn(neighbors, labels, perplexity=perplexity, scale=False)
    return tidy(metric, use_rep, key, float(value))
