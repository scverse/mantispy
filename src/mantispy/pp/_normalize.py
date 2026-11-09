from __future__ import annotations

import re
import warnings
from typing import Any

import numpy as np
import pandas as pd
from anndata import AnnData

from mantispy._core._numba import IQR, MEAN, STD, grouped_median_spread
from mantispy._core._numba import MAD as MAD_STAT
from mantispy._core._reduce import (
    get_matrix,
    group_codes,
    group_offsets,
    matrix_dtype,
    reduce_grouped,
    transform_grouped,
)
from mantispy._core._stats import MAD_TO_SIGMA
from mantispy._core.frames import as_frame
from mantispy._core.masks import reference_mask
from mantispy._core.mutation import inplace_or_copy
from mantispy._core.schema import resolve_perturbation_column

METHODS = ("mad_robustize", "standardize", "robustize", "center")
#: Names of obs columns that carry the treatment besides the recorded perturbation column, in mantispy's and in
#: the JUMP/pycytominer conventions. Matched whole, so ``Metadata_growth_condition`` is not taken for one.
_TREATMENT_COLUMN = re.compile(
    r"(Metadata_)?(perturbation|compound|treatment|condition|concentration|dose|moa|gene|gene_name|pert_iname|"
    r"pert_name|broad_sample|JCP2022|InChIKey|mmoles_per_liter|mg_per_ml|sgRNA|guide|allele)",
    re.IGNORECASE,
)


def _median_and_spread(
    adata: AnnData, spread: int, codes: np.ndarray, keys: pd.Index, layer: str | None, mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Per-group median and robust spread."""
    if adata.isbacked:
        # One group at a time, as the backed branch of reduce_grouped does, so a screen that does not fit in memory still normalizes.
        centre = np.full((len(keys), adata.n_vars), np.nan)
        scale = np.full((len(keys), adata.n_vars), np.nan)
        selected = np.flatnonzero(mask)
        order, offsets = group_offsets(codes[mask], len(keys))
        for index in range(len(keys)):
            rows = selected[order[offsets[index] : offsets[index + 1]]]
            if rows.size:
                block = get_matrix(adata, layer, rows=rows)
                group_centre, group_scale = grouped_median_spread(block, np.zeros(rows.size, np.int32), 1, spread)
                centre[index], scale[index] = group_centre[0], group_scale[0]
        return centre, scale

    matrix = get_matrix(adata, layer)
    # Masking copies the whole matrix, so it is skipped when the reference is every row.
    if not mask.all():
        codes, matrix = codes[mask], matrix[mask]
    return grouped_median_spread(matrix, codes, len(keys), spread)


def _center_and_scale(
    adata: AnnData,
    method: str,
    by: str | list[str] | None,
    codes: np.ndarray,
    keys: pd.Index,
    layer: str | None,
    mask: np.ndarray,
    epsilon: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-group center and scale, plus the features that cannot be normalized somewhere.

    Returned as ``(center, scale, no_spread, uncentred)``, the first two ``(n_groups, n_vars)`` in the dtype the matrix is read as and the last two boolean masks over features.
    ``no_spread`` flags a spread of zero in some group and ``uncentred`` a group whose reference rows hold no usable centre or spread at all.
    The two overlap: a feature can be constant in one group and unmeasured in another, and both facts have to reach the caller.
    """
    if method == "standardize":
        centre, _, _ = reduce_grouped(adata, by, MEAN, layer=layer, mask=mask)
        # ddof=0 matches pycytominer, which uses sklearn's StandardScaler (population SD).
        scale, _, _ = reduce_grouped(adata, by, STD, layer=layer, mask=mask, ddof=0)
    elif method == "mad_robustize":
        centre, mad = _median_and_spread(adata, MAD_STAT, codes, keys, layer, mask)
        scale = MAD_TO_SIGMA * mad + epsilon
    elif method == "center":
        centre, _ = _median_and_spread(adata, MAD_STAT, codes, keys, layer, mask)
        # Centring only: a feature with no spread is still centred correctly, so nothing is degenerate on scale.
        return _centre_only(centre, matrix_dtype(adata, layer))
    else:  # robustize: median and interquartile range, as sklearn's RobustScaler
        centre, scale = _median_and_spread(adata, IQR, codes, keys, layer, mask)

    # A NaN or infinite scale fails this comparison, so it is left to `uncentred`.
    no_spread = (scale <= epsilon).any(axis=0)
    # Repairing only the scale would subtract a NaN centre from every row of the group, wiping the non-reference values.
    uncentred = ~(np.isfinite(centre) & np.isfinite(scale)).all(axis=0)
    centre = np.where(np.isfinite(centre), centre, 0.0)
    scale = np.where((scale == 0) | ~np.isfinite(scale), 1.0, scale)
    # In the matrix's own dtype, so rescaling a float32 block makes no full-size float64 temporary.
    dtype = matrix_dtype(adata, layer)
    return centre.astype(dtype), scale.astype(dtype), no_spread, uncentred


def _centre_only(centre: np.ndarray, dtype: np.dtype) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """The ``_center_and_scale`` result for ``method="center"``: the median, a unit scale, no spread flags."""
    uncentred = ~np.isfinite(centre).all(axis=0)
    centre = np.where(np.isfinite(centre), centre, 0.0)
    return centre.astype(dtype), np.ones_like(centre, dtype=dtype), np.zeros(centre.shape[1], bool), uncentred


def _erased_treatments(adata: AnnData, by: str | list[str] | None) -> list[str]:
    """Treatment columns of obs that vary across groups of ``by`` but are constant within every group.

    Normalizing within such groups subtracts each group's own centre, and with it the treatment effect.
    The recorded perturbation column is checked first, then any column whose name is a known treatment name.
    A missing value counts as its own level: control wells often leave the concentration empty, and a plate
    whose treated wells share one concentration still holds two conditions, its controls and its treatment.
    """
    if by is None:
        return []
    keys = [by] if isinstance(by, str) else list(by)
    obs = as_frame(adata.obs)
    perturbation = resolve_perturbation_column(adata)
    candidates = [perturbation] if perturbation in obs else []
    candidates += [c for c in obs.columns if c != perturbation and _TREATMENT_COLUMN.fullmatch(str(c))]
    erased = []
    for column in candidates:
        if column in keys:
            continue
        if obs[column].nunique(dropna=False) < 2:
            continue
        if (obs.groupby(keys, observed=True)[column].nunique(dropna=False) <= 1).all():
            erased.append(column)
    return erased


@inplace_or_copy()
def normalize(
    adata: AnnData,
    method: str = "mad_robustize",
    by: str | list[str] | None = "Metadata_Plate",
    reference: str | None = "negcon",
    epsilon: float = 1e-18,
    keep_raw: bool = False,
    layer: str | None = None,
    key_added: str | None = None,
    copy: bool = False,
) -> AnnData | None:
    """Normalize features within groups, optionally fitting on reference rows only.

    Args:
        adata: Object to normalize.
        method: ``"mad_robustize"`` computes ``(x - median) / (1.4826 * MAD + epsilon)``, ``"standardize"`` computes ``(x - mean) / sd``, ``"robustize"`` computes ``(x - median) / IQR``, and ``"center"`` computes ``x - median``, which removes group offsets such as well-to-well shifts and keeps the units.
            Normalizing within groups that each hold one treatment removes the treatment effect along with the offset, so the recorded perturbation column, or another treatment column such as ``Metadata_Compound`` or ``Metadata_pert_iname``, that is constant within every group of ``by`` raises a warning, whatever the method.
        by: Column(s) defining the groups statistics are computed within, usually the plate.
            ``None`` fits one set of statistics globally.
        reference: Rows to fit on: ``"negcon"`` (the default) for the rows whose ``Metadata_Control_Type`` is ``"negcon"``, ``None`` for all rows, or the name of a boolean ``obs`` column.
        epsilon: Added to the MAD, matching pycytominer's ``mad_robustize_epsilon``.
            Unused by the other methods.
        keep_raw: Store the pre-normalization matrix in ``layers["raw"]``.
            Off by default, because the layer doubles memory and the raw table is already on disk.
        layer: Read this layer instead of ``X``.
        key_added: Write to ``layers[key_added]`` instead of overwriting ``X``.
        copy: Return a normalized copy instead of normalizing in place.

    Returns:
        ``None``, or the normalized copy when ``copy=True``.
        Writes ``X`` or ``layers[key_added]``, and ``var["degenerate_scale"]``, or ``var["degenerate_scale_<key_added>"]`` when writing to a layer, which flags features that have no spread in some group or no reference values to centre on there, and comes with a warning; drop those features before computing distances.

    Raises:
        ValueError: If ``method`` is unknown, or ``reference`` selects no rows at all or none in some group.
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")

    mask = reference_mask(adata, reference)
    if not mask.any():
        raise ValueError(
            f"no reference rows selected by reference={reference!r}. Check that obs['Metadata_Control_Type'] has "
            "'negcon' rows (mt.pp.annotate_controls), or pass reference=None to use all rows."
        )

    codes, keys = group_codes(adata, by)
    present = np.bincount(codes[mask], minlength=len(keys))
    if (present == 0).any():
        empty = [str(keys[int(index)]) for index in np.flatnonzero(present == 0)]
        raise ValueError(f"no reference rows in group(s): {empty[:5]}")

    erased = _erased_treatments(adata, by)
    if erased:
        warnings.warn(
            f"normalize: {erased} is constant within every group of {by!r}, so normalizing within those groups "
            "subtracts each group's own centre and removes the differences between them. Normalize by a "
            "coarser grouping if those columns are what you want to compare.",
            UserWarning,
            stacklevel=3,
        )
    centre, scale, no_spread, uncentred = _center_and_scale(adata, method, by, codes, keys, layer, mask, epsilon)
    degenerate = no_spread | uncentred
    # One flag column per output matrix, so a call writing another layer does not reset the flags describing the first.
    flag = "degenerate_scale" if key_added is None else f"degenerate_scale_{key_added}"
    adata.var[flag] = degenerate
    scope = f" among the rows selected by reference={reference!r}" if reference is not None else ""
    remedy = f"They are flagged in var[{flag!r}]; drop them with adata = adata[:, ~adata.var[{flag!r}]].copy()."
    if key_added is None:
        # feature_select reads the unsuffixed flag, which is the one describing X.
        remedy = (
            f"They are flagged in var[{flag!r}], and pp.feature_select drops them by default; to drop them "
            f"now, adata = adata[:, ~adata.var[{flag!r}]].copy()."
        )
    if uncentred.any():
        warnings.warn(
            f"{int(uncentred.sum())} of {adata.n_vars} features have no reference values to centre on "
            f"in at least one group of {by!r}{scope}, because every value there is missing or infinite. "
            "Their centre is set to 0 where it is missing and their scale to 1, so the values measured "
            "outside the reference rows pass through unnormalized instead of becoming NaN, and are not "
            f"comparable across groups. {remedy}",
            UserWarning,
            stacklevel=3,
        )
    # Not elif: a feature constant in one group and unmeasured in another is still multiplied by 1e18 in the first.
    if no_spread.any():
        warnings.warn(
            f"{int(no_spread.sum())} of {adata.n_vars} features have no spread in at least one "
            f"group of {by!r}{scope}. "
            + (
                f"Adding epsilon={epsilon:g} to their scale multiplies them by up to 1e18, so they "
                "dominate every distance downstream"
                if method == "mad_robustize"
                else "Their scale is clamped to 1, which sets them to 0"
            )
            + f". {remedy} A variance or outlier cut does not catch a feature that varies across a plate "
            "but is constant among its control wells; the flag does.",
            UserWarning,
            stacklevel=3,
        )
    lookup = {key: position for position, key in enumerate(keys)}

    def _rescale(key: Any, block: np.ndarray) -> np.ndarray:
        index = lookup[key]
        return (block - centre[index]) / scale[index]

    if keep_raw and "raw" not in adata.layers:
        adata.layers["raw"] = get_matrix(adata, layer).copy()

    out = transform_grouped(adata, by, _rescale, layer=layer)
    if key_added is None:
        adata.X = out
    else:
        adata.layers[key_added] = out
    return None
