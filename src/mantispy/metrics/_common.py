"""Helpers shared by the metric implementations and the plots that mirror them."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from anndata import AnnData


def embedding(adata: AnnData, use_rep: str) -> np.ndarray:
    """Fetch a representation, with an error that says how to make one.

    Args:
        adata: Object holding the representation in ``obsm``.
        use_rep: ``obsm`` key of the representation.

    Returns:
        The representation as a float64 array, one row per observation.

    Raises:
        KeyError: ``obsm`` holds nothing under ``use_rep``.
    """
    if use_rep not in adata.obsm:
        raise KeyError(f"obsm has no {use_rep!r}; compute it first, e.g. sc.pp.pca(adata, n_comps=50)")
    return np.asarray(adata.obsm[use_rep], dtype=np.float64)


def tidy(metric: str, use_rep: str, key: str, value: float) -> pd.DataFrame:
    """One row in the shape every metric returns, so results from different metrics stack.

    Args:
        metric: Name of the metric, which is what :func:`~mantispy.pl.metrics` pivots the table on.
        use_rep: Representation the value was measured in, or ``"X"`` when it was measured on the matrix itself.
        key: ``obs`` column the metric was scored over.
        value: The measured value.

    Returns:
        A one-row frame with ``metric``, ``representation``, ``key`` and ``value``.
    """
    return pd.DataFrame([{"metric": metric, "representation": use_rep, "key": key, "value": float(value)}])


def is_categorical(covariate: pd.Series) -> bool:
    """Whether a covariate is treated as categorical (one-hot encoded) rather than entering as a number."""
    return isinstance(covariate.dtype, pd.CategoricalDtype) or not pd.api.types.is_numeric_dtype(covariate)
