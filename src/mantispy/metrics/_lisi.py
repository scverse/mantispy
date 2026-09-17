"""Local inverse Simpson's index (LISI).

The effective number of distinct labels in a neighborhood, weighting each neighbor by
``exp(-beta * d)`` with ``beta`` calibrated so the neighborhood's entropy matches the
requested perplexity. On a batch key it is iLISI (higher is better mixed); on a label key
it is cLISI (lower means the biological groups stay separated).

Values match ``harmonypy.lisi.compute_lisi`` (Korsunsky et al. 2019) to machine precision,
so they are comparable with published LISI values.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from mantispy.metrics._common import embedding, tidy

if TYPE_CHECKING:
    from anndata import AnnData

_TOLERANCE = 1e-5
_MAX_STEPS = 50


def _inverse_simpson(distances: np.ndarray, labels: np.ndarray, perplexity: float) -> float:
    """Inverse Simpson index of one neighborhood, after calibrating the kernel width."""
    beta, lower, upper = 1.0, -np.inf, np.inf
    target = np.log(perplexity)
    weights = np.exp(-distances * beta)

    for _ in range(_MAX_STEPS):
        total = weights.sum()
        if total == 0:
            return 1.0
        entropy = np.log(total) + beta * np.sum(distances * weights) / total
        if abs(entropy - target) < _TOLERANCE:
            break
        if entropy > target:
            lower = beta
            beta = beta * 2 if upper == np.inf else (beta + upper) / 2
        else:
            upper = beta
            beta = beta / 2 if lower == -np.inf else (beta + lower) / 2
        weights = np.exp(-distances * beta)

    total = weights.sum()
    if total == 0:
        return 1.0
    shares = np.array([weights[labels == value].sum() for value in np.unique(labels)]) / total
    return float(1.0 / np.sum(shares**2))


#: Column-name fragments that mark a batch, used only when ``kind="auto"``.
_BATCH_HINTS = ("batch", "plate", "source", "week", "run")


def lisi(adata: AnnData, key: str, use_rep: str = "X_pca", perplexity: float = 30, kind: str = "auto") -> pd.DataFrame:
    """Median LISI over rows.

    Args:
        adata: Object with the embedding to measure in.
        key: ``obs`` column whose labels the neighborhoods are scored over.
        use_rep: ``obsm`` key of the embedding.
        perplexity: Perplexity the kernel width is calibrated to.
            Each neighborhood holds ``3 * perplexity`` rows, so the object needs more rows than that.
        kind: ``"batch"`` names the result ``ilisi`` (higher is better mixed) and ``"label"``
            names it ``clisi`` (lower means the biological groups stay separated). ``"auto"``
            guesses from the column name: a key containing batch, plate, source, week or run
            is a batch and anything else a label, so ``"Metadata_Site"`` counts as a label.
            Pass ``kind`` explicitly when one table holds both, or the two rows get the same
            metric name.

    Returns:
        A one-row tidy frame holding ``ilisi`` or ``clisi``.

    Raises:
        ValueError: ``kind`` is not one of the three accepted values.
        ValueError: ``obs[key]`` has missing values.
        ValueError: The object has too few rows for ``perplexity``.
    """
    if kind not in ("auto", "batch", "label"):
        raise ValueError(f"kind must be 'auto', 'batch' or 'label', got {kind!r}")

    values = embedding(adata, use_rep)
    labels = adata.obs[key].to_numpy()
    missing = int(pd.isna(labels).sum())
    if missing:
        raise ValueError(f"obs[{key!r}] has {missing} missing value(s); drop those rows or fill the column.")

    n_neighbors = int(perplexity * 3)
    if n_neighbors >= adata.n_obs:
        # A neighborhood of k rows tops out at an entropy of log(k), so once k falls below the
        # perplexity the bisection can never reach log(perplexity): beta halves all the way down,
        # the weights go uniform, and LISI collapses to the count of distinct labels.
        raise ValueError(
            f"perplexity={perplexity} calibrates the kernel over {n_neighbors} neighbors and so needs "
            f"{n_neighbors + 1} rows, but the object has {adata.n_obs}. "
            f"Pass a perplexity of at most {(adata.n_obs - 1) // 3}, or measure on a larger object."
        )
    distances, indices = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(values).kneighbors(values)

    # The kernel takes unsquared distances, as harmonypy does. Squared distances keep LISI
    # monotone in mixing but change every value; see tests/test_equivalence_harmonypy_lisi.py.
    scores = [_inverse_simpson(distances[row, 1:], labels[indices[row, 1:]], perplexity) for row in range(adata.n_obs)]
    if kind == "auto":
        kind = "batch" if any(hint in key.lower() for hint in _BATCH_HINTS) else "label"
    return tidy("ilisi" if kind == "batch" else "clisi", use_rep, key, float(np.median(scores)))
