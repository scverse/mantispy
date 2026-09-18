"""Principal-component regression: how much variance a covariate explains."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

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
