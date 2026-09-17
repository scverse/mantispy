"""Per-group normalization."""

from __future__ import annotations

import warnings

import numpy as np
from anndata import AnnData

from mantispy._core._numba import MAD as MAD_STAT
from mantispy._core._numba import MEAN, MEDIAN, QUANTILE, STD
from mantispy._core._reduce import get_matrix, group_codes, reduce_grouped, transform_grouped
from mantispy._core._stats import MAD_TO_SIGMA
from mantispy._core._utils import inplace_or_copy, reference_mask

METHODS = ("mad_robustize", "standardize", "robustize")


def _center_and_scale(
    adata: AnnData, method: str, by, layer: str | None, mask: np.ndarray, epsilon: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-group center and scale, plus the features whose spread is zero somewhere.

    Returned as ``(center, scale, degenerate)``, the first two ``(n_groups, n_vars)``
    float32 and the last a boolean mask over features.
    """
    if method == "standardize":
        centre, _, _ = reduce_grouped(adata, by, MEAN, layer=layer, mask=mask)
        # ddof=0 matches pycytominer, which uses sklearn's StandardScaler (population SD).
        scale, _, _ = reduce_grouped(adata, by, STD, layer=layer, mask=mask, ddof=0)
    elif method == "mad_robustize":
        centre, _, _ = reduce_grouped(adata, by, MEDIAN, layer=layer, mask=mask)
        mad, _, _ = reduce_grouped(adata, by, MAD_STAT, layer=layer, mask=mask)
        scale = MAD_TO_SIGMA * mad + epsilon
    else:  # robustize: median and interquartile range, as sklearn's RobustScaler
        centre, _, _ = reduce_grouped(adata, by, MEDIAN, layer=layer, mask=mask)
        upper, _, _ = reduce_grouped(adata, by, QUANTILE, layer=layer, mask=mask, q=0.75)
        lower, _, _ = reduce_grouped(adata, by, QUANTILE, layer=layer, mask=mask, q=0.25)
        scale = upper - lower

    # A feature that is constant within a group has zero spread there. sklearn clamps such a
    # scale to 1.0, giving 0.0. mad_robustize divides by epsilon as pycytominer does, which
    # multiplies the feature by up to 1e18, so those features are flagged.
    degenerate = ((scale == 0) | ~np.isfinite(scale) | (scale <= epsilon)).any(axis=0)
    scale = np.where((scale == 0) | ~np.isfinite(scale), 1.0, scale)
    return centre.astype(np.float32), scale.astype(np.float32), degenerate


@inplace_or_copy()
def normalize(
    adata: AnnData,
    method: str = "mad_robustize",
    by: str | list[str] | None = "Metadata_Plate",
    reference: str | None = None,
    epsilon: float = 1e-18,
    keep_raw: bool = False,
    layer: str | None = None,
    key_added: str | None = None,
    copy: bool = False,
) -> AnnData | None:
    """Normalize features within groups, optionally fitting on reference rows only.

    Args:
        adata: Object to normalize.
        method: ``"mad_robustize"`` computes ``(x - median) / (1.4826 * MAD + epsilon)``,
            ``"standardize"`` computes ``(x - mean) / sd``, and ``"robustize"`` computes
            ``(x - median) / IQR``.
        by: Column(s) defining the groups statistics are computed within, usually the plate.
            ``None`` fits one set of statistics globally.
        reference: Rows to fit on: ``None`` for all, ``"negcon"`` for ``Metadata_Control``, or the
            name of a boolean ``obs`` column.
        epsilon: Added to the MAD, matching pycytominer's ``mad_robustize_epsilon``. Unused by
            the other methods.
        keep_raw: Store the pre-normalization matrix in ``layers["raw"]``. Off by default, because
            the layer doubles memory and the raw table is already on disk.
        layer: Read this layer instead of ``X``.
        key_added: Write to ``layers[key_added]`` instead of overwriting ``X``.
        copy: Return a normalized copy instead of normalizing in place.

    Returns:
        ``None``, or the normalized copy when ``copy=True``. Also writes
        ``var["degenerate_scale"]``, which flags features with no spread in some group and
        comes with a warning; drop those features before computing distances.
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")

    mask = reference_mask(adata, reference)
    if not mask.any():
        raise ValueError(f"no reference rows selected by reference={reference!r}")

    codes, keys = group_codes(adata, by)
    present = np.bincount(codes[mask], minlength=len(keys))
    if (present == 0).any():
        empty = [str(keys[int(index)]) for index in np.flatnonzero(present == 0)]
        raise ValueError(f"no reference rows in group(s): {empty[:5]}")

    centre, scale, degenerate = _center_and_scale(adata, method, by, layer, mask, epsilon)
    adata.var["degenerate_scale"] = degenerate
    if degenerate.any():
        warnings.warn(
            f"{int(degenerate.sum())} of {adata.n_vars} features have no spread in at least one "
            f"group of {by!r}"
            + (f" among the rows selected by reference={reference!r}" if reference is not None else "")
            + ". "
            + (
                f"Adding epsilon={epsilon:g} to their scale multiplies them by up to 1e18, so they "
                "dominate every distance downstream"
                if method == "mad_robustize"
                else "Their scale is clamped to 1, which sets them to 0"
            )
            + ". They are flagged in var['degenerate_scale']; drop them with "
            "adata = adata[:, ~adata.var['degenerate_scale']].copy(). Feature selection does not "
            "catch a feature that varies across a plate but is constant among its control wells.",
            UserWarning,
            stacklevel=3,
        )
    lookup = {key: position for position, key in enumerate(keys)}

    def rescale(key, block: np.ndarray) -> np.ndarray:
        index = lookup[key]
        return (block - centre[index]) / scale[index]

    if keep_raw and "raw" not in adata.layers:
        adata.layers["raw"] = get_matrix(adata, layer).copy()

    out = transform_grouped(adata, by, rescale, layer=layer)
    if key_added is None:
        adata.X = out
    else:
        adata.layers[key_added] = out
    return None
