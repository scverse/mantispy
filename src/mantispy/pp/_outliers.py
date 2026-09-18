"""Cell-level outlier detection."""

from __future__ import annotations

import math

import numpy as np
from anndata import AnnData

from mantispy._core._ecod import ecod_scores
from mantispy._core._reduce import get_matrix, group_codes, group_offsets
from mantispy._core._stats import robust_zscore
from mantispy._core.logging import get_logger
from mantispy._core.masks import feature_mask
from mantispy._core.mutation import inplace_or_copy

METHODS = ("ecod", "isolation_forest", "mad")


def _scores(X: np.ndarray, method: str, seed: int) -> np.ndarray:
    """Outlier score per row, higher meaning more outlying."""
    if method == "ecod":
        return ecod_scores(X)
    if method == "isolation_forest":
        from sklearn.ensemble import IsolationForest

        filled = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        model = IsolationForest(random_state=seed, n_estimators=200).fit(filled)
        return -model.score_samples(filled)

    return np.nanmax(np.abs(robust_zscore(X)), axis=1)


@inplace_or_copy()
def outliers(
    adata: AnnData,
    method: str = "ecod",
    contamination: float = 0.01,
    score_cutoff: float | None = None,
    key: str | None = "selected",
    by: str | None = None,
    seed: int = 0,
    key_added: str = "qc_outlier",
    copy: bool = False,
) -> AnnData | None:
    """Flag outlying cells.

    Every method produces a score where higher means more outlying, and the same thresholding applies to all of them, so ``contamination`` is the flagged fraction whichever method is used.

    Args:
        adata: Object to flag.
        method: ``"ecod"`` is parameter-free and interpretable per feature, ``"isolation_forest"`` catches outliers defined by feature interactions, and ``"mad"`` takes the largest robust z-score across features, which is easy to explain but sees each feature alone.
        contamination: Fraction of cells to flag, rounded up to a whole cell within each ``by`` group, so a non-empty group always flags its most outlying cell and the flagged fraction is higher than asked for in a group smaller than ``1 / contamination``. Ignored when ``score_cutoff`` is given.
        score_cutoff: Threshold the score absolutely instead of by quantile. With ``method="mad"`` the score is a robust z-score, so ``score_cutoff=5`` gives the usual rule.
        key: Restrict to features flagged by this boolean ``var`` column, usually ``"selected"``. Falls back to every feature when the column is absent.
        by: Threshold within each group of this ``obs`` column, e.g. per plate, rather than globally.
        seed: Seed for ``isolation_forest``.
        key_added: Prefix for the outputs: ``obs[key_added]`` and ``obs[key_added + "_score"]``.
        copy: Return a modified copy instead of mutating in place.

    Returns:
        ``None``, or the modified copy. Writes the boolean ``obs[key_added]`` and the score itself to ``obs[key_added + "_score"]``.

    Raises:
        ValueError: If ``method`` is unknown, or ``contamination`` is outside ``(0, 1)`` and no ``score_cutoff`` is given.
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    if not 0.0 < contamination < 1.0 and score_cutoff is None:
        raise ValueError(f"contamination must be between 0 and 1, got {contamination}")

    selected = feature_mask(adata, key)
    X = get_matrix(adata)[:, selected]

    scores = np.empty(adata.n_obs, dtype=np.float64)
    flagged = np.zeros(adata.n_obs, dtype=bool)
    codes, keys = group_codes(adata, by)
    order, offsets = group_offsets(codes, len(keys))
    for group in range(len(keys)):
        rows = order[offsets[group] : offsets[group + 1]]
        if not rows.size:
            continue
        block = _scores(X[rows], method, seed)
        scores[rows] = block
        if score_cutoff is not None:
            flagged[rows] = block > score_cutoff
        else:
            # Flag by rank: `> quantile` flags too few cells when scores tie, and none when an
            # infinite feature value makes the quantile infinite. The count is rounded up, so a
            # contamination the group is too small to express asks for the most outlying cell
            # rather than for none at all; pyod's threshold at the `1 - contamination`
            # percentile likewise flags the top cell of a sample that small.
            k = math.ceil(contamination * rows.size)
            flagged[rows[np.argsort(block)[::-1][:k]]] = True

    adata.obs[key_added] = flagged
    adata.obs[f"{key_added}_score"] = scores
    get_logger().info(
        "outliers(%s) flagged %d of %d cells using %d features",
        method,
        int(flagged.sum()),
        adata.n_obs,
        int(selected.sum()),
    )
    return None
