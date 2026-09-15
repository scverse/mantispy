"""Helpers shared by the metric implementations and the plots that mirror them."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from anndata import AnnData


def embedding(adata: AnnData, use_rep: str) -> np.ndarray:
    """Fetch a representation, with an error that says how to make one."""
    if use_rep not in adata.obsm:
        raise KeyError(f"obsm has no {use_rep!r}; compute it first, e.g. sc.pp.pca(adata, n_comps=50)")
    return np.asarray(adata.obsm[use_rep], dtype=np.float64)


def tidy(metric: str, use_rep: str, key: str, value: float) -> pd.DataFrame:
    """One row in the shape every metric returns, so results from different metrics stack."""
    return pd.DataFrame([{"metric": metric, "representation": use_rep, "key": key, "value": float(value)}])


def r_squared(component: np.ndarray, covariate: pd.Series) -> float:
    """R^2 of one component regressed on one covariate, numeric or categorical."""
    if pd.api.types.is_numeric_dtype(covariate) and not isinstance(covariate.dtype, pd.CategoricalDtype):
        design = np.column_stack([np.ones(component.size), covariate.to_numpy(dtype=float)])
    else:
        dummies = pd.get_dummies(covariate, drop_first=True, dtype=float).to_numpy()
        design = np.column_stack([np.ones(component.size), dummies])

    coefficients, *_ = np.linalg.lstsq(design, component, rcond=None)
    residual = component - design @ coefficients
    total = float(np.sum((component - component.mean()) ** 2))
    return 0.0 if total == 0 else float(1.0 - np.sum(residual**2) / total)
