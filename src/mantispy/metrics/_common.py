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
        metric: Name of the metric, which is what :func:`~mantispy.metrics.evaluate_correction` pivots the table on.
        use_rep: Representation the value was measured in, or ``"X"`` when it was measured on the matrix itself.
        key: ``obs`` column the metric was scored over.
        value: The measured value.

    Returns:
        A one-row frame with ``metric``, ``representation``, ``key`` and ``value``.
    """
    return pd.DataFrame([{"metric": metric, "representation": use_rep, "key": key, "value": float(value)}])


def r_squared(component: np.ndarray, covariate: pd.Series) -> float:
    """R^2 of one component regressed on one covariate, numeric or categorical.

    Args:
        component: One column of an embedding, one value per observation.
        covariate: The covariate to regress it on; a categorical one is expanded into dummy columns, a numeric one enters as it is.

    Returns:
        The share of the component's variance the covariate explains, and ``0.0`` for a constant component, where that share is undefined.
    """
    if pd.api.types.is_numeric_dtype(covariate) and not isinstance(covariate.dtype, pd.CategoricalDtype):
        design = np.column_stack([np.ones(component.size), covariate.to_numpy(dtype=float)])
    else:
        dummies = pd.get_dummies(covariate, drop_first=True, dtype=float).to_numpy()
        design = np.column_stack([np.ones(component.size), dummies])

    coefficients, *_ = np.linalg.lstsq(design, component, rcond=None)
    residual = component - design @ coefficients
    total = float(np.sum((component - component.mean()) ** 2))
    return 0.0 if total == 0 else float(1.0 - np.sum(residual**2) / total)
