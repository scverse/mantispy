"""Principal-component regression: how much variance a covariate explains."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from mantispy._core._reduce import get_matrix
from mantispy._core.frames import as_frame
from mantispy.metrics._common import embedding, r_squared, tidy

if TYPE_CHECKING:
    from anndata import AnnData


def pc_regression(adata: AnnData, key: str, use_rep: str = "X_pca", n_comps: int | None = None) -> pd.DataFrame:
    """Variance-weighted R^2 of the principal components on ``key``.

    The value is the share of total variance the covariate explains, so for a batch key lower is better.

    Args:
        adata: Object with the embedding to measure in.
        key: ``obs`` column the components are regressed on.
        use_rep: ``obsm`` key of the embedding.
        n_comps: Use only the leading components, or ``None`` for every component the embedding holds.

    Returns:
        A one-row tidy frame holding ``pc_regression``.

    Raises:
        KeyError: ``obsm`` holds nothing under ``use_rep``.
    """
    values = embedding(adata, use_rep)
    if n_comps is not None:
        values = values[:, :n_comps]
    covariate = as_frame(adata.obs)[key]

    variances = values.var(axis=0, ddof=1)
    weights = variances / variances.sum()
    explained = np.array([r_squared(values[:, index], covariate) for index in range(values.shape[1])])
    return tidy("pc_regression", use_rep, key, float(np.sum(weights * explained)))


def batch_variance_explained(adata: AnnData, keys: Sequence[str], use_rep: str = "X_pca") -> pd.DataFrame:
    """:func:`~mantispy.metrics.pc_regression` for several covariates, stacked into one frame.

    Args:
        adata: Object with the embedding to measure in.
        keys: ``obs`` columns to score, one row of the result each.
        use_rep: ``obsm`` key of the embedding.

    Returns:
        A tidy frame holding one ``pc_regression`` row per entry of ``keys``.

    Raises:
        KeyError: ``obsm`` holds nothing under ``use_rep``.
    """
    return pd.concat([pc_regression(adata, key, use_rep) for key in keys], ignore_index=True)


def variance_carried(
    adata: AnnData,
    reference: AnnData,
    use_rep: str = "X_pca",
    groupby: str | None = "feature_group",
    n_splits: int = 5,
) -> pd.DataFrame:
    """How much of a named feature block a learned embedding linearly carries.

    A learned embedding has no ``var`` vocabulary, so a hit read off it is only a compound ID. This scores, per named feature, the out-of-fold R^2 of predicting that feature from the embedding with a cross-fit ridge, so a value near 1 means the embedding carries the feature and near 0 means it does not.

    Args:
        adata: Object holding the learned embedding in ``obsm``.
        reference: Object whose ``X`` holds the named block (e.g. CellProfiler features) to recover.
        use_rep: ``obsm`` key of the embedding used as the predictor.
        groupby: ``reference.var`` column to average over, one row of the result per group; ``None`` returns one row per feature instead.
        n_splits: Folds of the cross-fit that produces the out-of-fold predictions.

    Returns:
        A frame sorted by ``variance_carried`` descending with a reset index: columns ``["feature", "variance_carried"]`` when ``groupby`` is ``None``, else ``["<groupby>", "variance_carried", "n_features"]`` holding the mean over each group.

    Raises:
        KeyError: ``obsm`` holds nothing under ``use_rep``.
        ValueError: The two objects share no ``obs_names``.
        ValueError: ``adata`` or ``reference`` has non-unique ``obs_names``.
        ValueError: ``groupby`` is not ``None`` and not a column of ``reference.var``.

    Notes:
        The two blocks may hold the same wells in different row orders, so the rows are aligned on the shared ``obs_names`` before regressing; matching by position instead returns a near-zero R^2 for an informative block, which reads as a real negative result.
    """
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import KFold

    if groupby is not None and groupby not in as_frame(reference.var).columns:
        raise ValueError(f"groupby={groupby!r} is not a column of reference.var")
    if not adata.obs_names.is_unique:
        raise ValueError("adata.obs_names are not unique; call .obs_names_make_unique() first")
    if not reference.obs_names.is_unique:
        raise ValueError("reference.obs_names are not unique; call .obs_names_make_unique() first")

    shared = adata.obs_names[adata.obs_names.isin(reference.obs_names)]
    if len(shared) == 0:
        raise ValueError("adata and reference share no obs_names; the two blocks must hold the same wells")

    predictors = embedding(adata[shared], use_rep)
    targets = get_matrix(reference[shared]).astype(np.float64)

    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=0)
    scores = np.full(targets.shape[1], np.nan)
    for column in range(targets.shape[1]):
        # Drop rows missing this target rather than impute; a per-feature mask is simpler and unbiased.
        finite = ~np.isnan(targets[:, column])
        target = targets[finite, column]
        if target.size < n_splits or target.min() == target.max():
            continue  # all-NaN, too few rows to cross-fit, or constant (SS_tot == 0): leave NaN
        design = predictors[finite]
        predicted = np.empty_like(target)
        for train, test in splitter.split(design):
            predicted[test] = RidgeCV().fit(design[train], target[train]).predict(design[test])
        residual = float(np.sum((target - predicted) ** 2))
        total = float(np.sum((target - target.mean()) ** 2))
        scores[column] = 1.0 - residual / total

    if groupby is None:
        frame = pd.DataFrame({"feature": np.asarray(reference.var_names), "variance_carried": scores})
    else:
        labels = as_frame(reference.var)[groupby].to_numpy()
        rows = []
        for group in pd.Series(labels).dropna().unique():
            members = labels == group
            rows.append(
                {
                    groupby: group,
                    "variance_carried": float(np.nanmean(scores[members])),
                    "n_features": int(members.sum()),
                }
            )
        frame = pd.DataFrame(rows)
    return frame.sort_values("variance_carried", ascending=False).reset_index(drop=True)
