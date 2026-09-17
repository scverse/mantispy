"""Boolean masks over features and over reference rows."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from mantispy._core.frames import as_frame
from mantispy._core.logging import get_logger

if TYPE_CHECKING:
    from anndata import AnnData


def feature_mask(adata: AnnData, key: str | None) -> np.ndarray:
    """Boolean mask over ``var``: the features flagged by ``key``, or all of them.

    A missing column is not an error. Callers pass ``key="selected"`` by default, so running
    after :func:`~mantispy.pp.feature_select` uses the selection and running before it uses
    every feature.
    """
    if key is not None and key in adata.var:
        return as_frame(adata.var)[key].to_numpy(dtype=bool)
    if key is not None:
        get_logger().debug("var has no column %r; using every feature", key)
    return np.ones(adata.n_vars, dtype=bool)


def reference_mask(adata: AnnData, reference: str | None) -> np.ndarray:
    """Boolean mask over ``obs``: the rows a transform should be fitted on.

    ``None`` fits on everything, ``"negcon"`` on ``Metadata_Control``, and anything else
    names a boolean ``obs`` column.
    """
    if reference is None:
        return np.ones(adata.n_obs, dtype=bool)

    column = "Metadata_Control" if reference == "negcon" else reference
    if column not in adata.obs:
        extra = " Run mt.pp.annotate_controls to create it." if column == "Metadata_Control" else ""
        raise KeyError(f"obs has no column {column!r} to use as reference.{extra}")

    values = pd.Series(adata.obs[column])
    missing = int(values.isna().sum())
    if missing:
        raise ValueError(
            f"obs[{column!r}] has {missing} missing value(s) and cannot be used as a reference flag, "
            "because NaN coerces to True and would mark those rows as controls. Fill them, or check "
            "that the platemap covers every well."
        )

    known = set(values.unique())
    # An h5ad round trip can bring a bool column back as a category of "True"/"False".
    if known <= {"True", "False"}:
        return (values == "True").to_numpy()
    if not (pd.api.types.is_bool_dtype(values) or known <= {0, 1}):
        raise TypeError(
            f"obs[{column!r}] must be boolean to select reference rows, got dtype {values.dtype}. "
            "A string column would select every row."
        )
    return values.to_numpy(dtype=bool)
