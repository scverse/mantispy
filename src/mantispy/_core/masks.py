"""Boolean masks over features and over reference rows."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from mantispy._core.frames import as_frame
from mantispy._core.logging import get_logger

if TYPE_CHECKING:
    from anndata import AnnData


def _flag_mask(values: pd.Series, label: str, subject: str, remedy: str) -> np.ndarray:
    """Read a flag column as a boolean mask, refusing the dtypes that would coerce to all-true.

    ``Series.to_numpy(dtype=bool)`` is true for every non-empty string and for NaN, so a column that is not boolean silently selects everything instead of failing.
    Both flag columns in mantispy go through here so that the two cannot drift apart.

    Args:
        values: The column to read.
        label: How to name the column in an error message, such as ``"obs['Metadata_Control']"``.
        subject: What a true entry means, such as ``"a control row"``, named in the error messages.
        remedy: What the caller can do about missing values, appended to that message.

    Returns:
        One boolean per entry of ``values``.

    Raises:
        ValueError: The column has missing values, which coerce to ``True``.
        TypeError: The column is neither boolean, nor 0/1, nor the ``"True"``/``"False"`` categorical an h5ad round trip produces.
    """
    missing = int(values.isna().sum())
    if missing:
        raise ValueError(
            f"{label} has {missing} missing value(s) and cannot be used as a flag, because NaN coerces "
            f"to True and would read as {subject}.{remedy}"
        )

    known = set(values.unique())
    # An h5ad round trip can bring a bool column back as a category of "True"/"False".
    if known <= {"True", "False"}:
        return (values == "True").to_numpy()
    if not (pd.api.types.is_bool_dtype(values) or known <= {0, 1}):
        raise TypeError(
            f"{label} must be boolean, got dtype {values.dtype}. A string column would read every entry as {subject}."
        )
    return values.to_numpy(dtype=bool)


def feature_mask(adata: AnnData, key: str | None) -> np.ndarray:
    """Boolean mask over ``var``: the features flagged by ``key``, or all of them.

    A missing column is not an error.
    Callers pass ``key="selected"`` by default, so running after :func:`~mantispy.pp.feature_select` uses the selection and running before it uses every feature.
    A column that is present must be boolean, on the same terms :func:`reference_mask` applies to ``obs``, because anything else would coerce to all-true and quietly use every feature.
    """
    if key is not None and key in adata.var:
        return _flag_mask(
            pd.Series(as_frame(adata.var)[key]),
            f"var[{key!r}]",
            "a selected feature",
            " Fill them, or re-run mt.pp.feature_select.",
        )
    if key is not None:
        get_logger().debug("var has no column %r; using every feature", key)
    return np.ones(adata.n_vars, dtype=bool)


def reference_mask(adata: AnnData, reference: str | None) -> np.ndarray:
    """Boolean mask over ``obs``: the rows a transform should be fitted on.

    ``None`` fits on everything, ``"negcon"`` on ``Metadata_Control``, and anything else names a boolean ``obs`` column.
    """
    if reference is None:
        return np.ones(adata.n_obs, dtype=bool)

    column = "Metadata_Control" if reference == "negcon" else reference
    if column not in adata.obs:
        extra = " Run mt.pp.annotate_controls to create it." if column == "Metadata_Control" else ""
        raise KeyError(f"obs has no column {column!r} to use as reference.{extra}")

    return _flag_mask(
        pd.Series(adata.obs[column]),
        f"obs[{column!r}]",
        "a control row",
        " Fill them, or check that the platemap covers every well.",
    )


def held_out_reference(adata: AnnData, is_control: np.ndarray, distance_key: str) -> np.ndarray:
    """Narrow a control mask to the rows that did not fit the transform ``distance_key`` is measured in.

    A row that fitted the centroid and the covariance sits closer to the centroid than one that did not, so a scale read off every control comes out low.
    :func:`~mantispy.tl.hit_calling` records the half it held out under the ``key_added`` the distance column already carries.
    A missing column is not an error, on the same terms :func:`feature_mask` applies: the distance came from somewhere else, and every control is kept.
    """
    column = f"{distance_key.removesuffix('_row_distance')}_reference_held_out"
    if column not in adata.obs:
        get_logger().debug("obs has no column %r; reading the control scale off every control row", column)
        return is_control
    return is_control & reference_mask(adata, column)
