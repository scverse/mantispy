"""Principal-component regression: how much variance a covariate explains, delegated to scib-metrics."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from mantispy._core._reduce import get_matrix
from mantispy._core.frames import as_frame
from mantispy.metrics._common import embedding, is_categorical, require_scib_metrics, tidy

if TYPE_CHECKING:
    from anndata import AnnData


def _pc_regression(adata: AnnData, key: str, use_rep: str = "X_pca", n_comps: int | None = None) -> pd.DataFrame:
    """Variance-weighted R^2 of the principal components on ``key``, computed by scib-metrics.

    The value is the share of total variance the covariate explains, so for a batch key lower is better.

    Args:
        adata: Object with the embedding to measure in.
        key: ``obs`` column the components are regressed on.
        use_rep: ``obsm`` key of the embedding.
        n_comps: Use only the leading components, or ``None`` for every component the embedding holds.

    Returns:
        A one-row tidy frame holding ``pc_regression``, whose value is NaN when ``key`` is constant (a single batch), where the share of variance it explains is undefined.

    Raises:
        ImportError: scib-metrics is not installed.
        KeyError: ``obsm`` holds nothing under ``use_rep``.
    """
    scib_metrics = require_scib_metrics()

    values = embedding(adata, use_rep)
    if n_comps is not None:
        values = values[:, :n_comps]
    covariate = as_frame(adata.obs)[key]

    if covariate.nunique(dropna=False) <= 1:
        # scib's PCR raises on a constant covariate; return NaN so one undefined metric doesn't abort the panel.
        warnings.warn(
            f"PC-regression over obs[{key!r}] is undefined: the covariate is constant (a single batch), "
            "so there is no variance to regress against. Returning NaN.",
            UserWarning,
            stacklevel=2,
        )
        return tidy("pc_regression", use_rep, key, np.nan)

    # A categorical or non-numeric covariate is one-hot encoded by scib; a numeric one enters as it is.
    value = scib_metrics.utils.principal_component_regression(
        values, covariate.to_numpy(), categorical=is_categorical(covariate)
    )
    return tidy("pc_regression", use_rep, key, float(value))


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
