"""Principal-component regression: how much variance a covariate explains, computed natively."""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from numba import njit

from mantispy._core._reduce import get_matrix
from mantispy._core.frames import as_frame
from mantispy.metrics._common import embedding, is_categorical, tidy

if TYPE_CHECKING:
    from anndata import AnnData


@njit(cache=True, nogil=True)
def _anova_r2(values: np.ndarray, codes: np.ndarray, n_groups: int) -> np.ndarray:
    """One-way ANOVA R^2 of each column of ``values`` on the integer group ``codes``.

    The value is ``1 - SS_within / SS_total`` per column, the share of the column's variance the grouping explains, and ``0`` for a constant column where that share is undefined.
    """
    n_rows, n_cols = values.shape
    out = np.zeros(n_cols)
    counts = np.zeros(n_groups)
    for row in range(n_rows):
        counts[codes[row]] += 1.0

    sums = np.empty(n_groups)
    for col in range(n_cols):
        grand_sum = 0.0
        for row in range(n_rows):
            grand_sum += values[row, col]
        grand_mean = grand_sum / n_rows

        for group in range(n_groups):
            sums[group] = 0.0
        for row in range(n_rows):
            sums[codes[row]] += values[row, col]

        ss_total = 0.0
        ss_within = 0.0
        for row in range(n_rows):
            value = values[row, col]
            total_diff = value - grand_mean
            ss_total += total_diff * total_diff
            within_diff = value - sums[codes[row]] / counts[codes[row]]
            ss_within += within_diff * within_diff

        out[col] = 0.0 if ss_total == 0.0 else 1.0 - ss_within / ss_total
    return out


@njit(cache=True, nogil=True)
def _pearson_r2(values: np.ndarray, covariate: np.ndarray) -> np.ndarray:
    """Squared Pearson correlation of each column of ``values`` with the numeric ``covariate``.

    Returns ``0`` for a column or covariate with no variance, where the correlation is undefined.
    """
    n_rows, n_cols = values.shape
    out = np.zeros(n_cols)

    covariate_sum = 0.0
    for row in range(n_rows):
        covariate_sum += covariate[row]
    covariate_mean = covariate_sum / n_rows
    covariate_ss = 0.0
    for row in range(n_rows):
        centered = covariate[row] - covariate_mean
        covariate_ss += centered * centered

    for col in range(n_cols):
        column_sum = 0.0
        for row in range(n_rows):
            column_sum += values[row, col]
        column_mean = column_sum / n_rows

        cross = 0.0
        column_ss = 0.0
        for row in range(n_rows):
            centered = values[row, col] - column_mean
            cross += centered * (covariate[row] - covariate_mean)
            column_ss += centered * centered

        denominator = column_ss * covariate_ss
        out[col] = 0.0 if denominator == 0.0 else (cross * cross) / denominator
    return out


def pc_regression(adata: AnnData, key: str, use_rep: str = "X_pca", n_comps: int | None = None) -> pd.DataFrame:
    """Variance-weighted R^2 of the principal components on ``key``.

    Each component is regressed on ``key`` on its own, a categorical key through a one-way ANOVA R^2 and a numeric key through the squared Pearson correlation, and the per-component R^2 is weighted by that component's share of the total variance.
    The value is the share of total variance the covariate explains, so for a batch key lower is better.
    The per-component loop runs in a numba kernel and imports no scib.

    Args:
        adata: Object with the embedding to measure in.
        key: ``obs`` column the components are regressed on.
        use_rep: ``obsm`` key of the embedding.
        n_comps: Use only the leading components, or ``None`` for every component the embedding holds.

    Returns:
        A one-row tidy frame holding ``pc_regression``, whose value is NaN when ``key`` is constant (a single batch), where the share of variance it explains is undefined.

    Raises:
        KeyError: ``obsm`` holds nothing under ``use_rep``.
    """
    values = np.ascontiguousarray(embedding(adata, use_rep))
    if n_comps is not None:
        values = np.ascontiguousarray(values[:, :n_comps])
    covariate = as_frame(adata.obs)[key]

    if covariate.nunique(dropna=False) <= 1:
        # A constant covariate has no variance to regress against, so its share is undefined.
        warnings.warn(
            f"PC-regression over obs[{key!r}] is undefined: the covariate is constant (a single batch), "
            "so there is no variance to regress against. Returning NaN.",
            UserWarning,
            stacklevel=2,
        )
        return tidy("pc_regression", use_rep, key, np.nan)

    variances = values.var(axis=0, ddof=1)
    weights = variances / variances.sum()
    if is_categorical(covariate):
        # use_na_sentinel=False gives a missing value its own group rather than the -1 code a numba kernel cannot index.
        codes = pd.factorize(covariate, use_na_sentinel=False)[0].astype(np.int64)
        explained = _anova_r2(values, codes, int(codes.max()) + 1)
    else:
        explained = _pearson_r2(values, covariate.to_numpy(dtype=np.float64))
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

    A learned embedding has no ``var`` vocabulary, so a hit read off it is only a compound ID.
    This scores, per named feature, the out-of-fold R^2 of predicting that feature from the embedding with a cross-fit ridge, so a value near 1 means the embedding carries the feature and near 0 means it does not.

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
    # Built lazily, once the row-count guard below has proved there are enough rows to split.
    full_split = None
    scores = np.full(targets.shape[1], np.nan)
    for column in range(targets.shape[1]):
        finite = np.isfinite(targets[:, column])
        target = targets[finite, column]
        if target.size < n_splits or target.min() == target.max():
            continue
        if finite.all():
            if full_split is None:
                full_split = list(splitter.split(predictors))
            design, folds = predictors, full_split
        else:
            design = predictors[finite]
            folds = splitter.split(design)
        predicted = np.empty_like(target)
        for train, test in folds:
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
