"""Narrowing and normalizing the obs and var tables."""

from __future__ import annotations

from typing import Any, cast

import pandas as pd


def as_frame(obj: Any) -> pd.DataFrame:
    """Narrow ``adata.obs``/``adata.var`` to a DataFrame.

    anndata types these as ``DataFrame | Dataset2D`` because a backed object can hold a lazy table.
    mantispy works on in-memory tables, so the cast is made here once instead of at every call site.
    """
    return cast("pd.DataFrame", obj)


def categorize_metadata(obs: pd.DataFrame) -> pd.DataFrame:
    """Convert low-cardinality string ``obs`` columns to ``category``.

    Plate, well, perturbation and batch repeat across millions of rows.
    As objects they cost a pointer plus a string each; as categories, one int8 or int16 code.
    It also stops anndata printing "storing X as categorical" on every construction.
    """
    for column in obs.columns:
        values = obs[column]
        if isinstance(values.dtype, pd.CategoricalDtype):
            continue
        # pandas 3 infers StringDtype where pandas 2 gave object, so check both.
        if values.dtype == object or isinstance(values.dtype, pd.StringDtype):
            obs[column] = values.astype("category")
    return obs
