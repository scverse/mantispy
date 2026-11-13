from __future__ import annotations

import re
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
from mantispy._core.logging import get_logger
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

    Returned as ``(center, scale, no_spread, uncentred)``, the first two ``(n_groups, n_vars)`` in the dtype the matrix is read as.
    ``no_spread`` and ``uncentred`` are ``(n_groups, n_vars)`` masks of the groups where a feature has a spread of zero, and where its reference rows hold no usable centre or spread at all.
    The two overlap: a feature can be constant in one group and unmeasured in another, and both facts have to reach the caller.
    """
    if method == "standardize":
        grouping = (codes, keys)
        centre, _, _ = reduce_grouped(adata, by, MEAN, layer=layer, mask=mask, grouping=grouping)
        # ddof=0 matches pycytominer, which uses sklearn's StandardScaler (population SD).
        scale, _, _ = reduce_grouped(adata, by, STD, layer=layer, mask=mask, ddof=0, grouping=grouping)
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
    no_spread = scale <= epsilon
    # Repairing only the scale would subtract a NaN centre from every row of the group, wiping the non-reference values.
    uncentred = ~(np.isfinite(centre) & np.isfinite(scale))
    centre = np.where(np.isfinite(centre), centre, 0.0)
    scale = np.where((scale == 0) | ~np.isfinite(scale), 1.0, scale)
    # In the matrix's own dtype, so rescaling a float32 block makes no full-size float64 temporary.
    dtype = matrix_dtype(adata, layer)
    return centre.astype(dtype), scale.astype(dtype), no_spread, uncentred


def _centre_only(centre: np.ndarray, dtype: np.dtype) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """The ``_center_and_scale`` result for ``method="center"``: the median, a unit scale, no spread flags."""
    uncentred = ~np.isfinite(centre)
    centre = np.where(np.isfinite(centre), centre, 0.0)
    return centre.astype(dtype), np.ones_like(centre, dtype=dtype), np.zeros(centre.shape, bool), uncentred


def _erased_treatments(adata: AnnData, by: str | list[str] | None, codes: np.ndarray) -> list[str]:
    """Treatment columns of obs that vary across groups of ``by`` but are constant within every group.

    Normalizing within such groups subtracts each group's own centre, and with it the treatment effect.
    The recorded perturbation column is checked first, then any column whose name is a known treatment name.
    A missing value counts as its own level: control wells often leave the concentration empty, and a plate
    whose treated wells share one concentration still holds two conditions, its controls and its treatment.

    ``codes`` is the row grouping already computed for ``by``; constancy within a group is read off it with one
    factorize per candidate instead of a pandas groupby per candidate, which scans every row each time.
    """
    if by is None:
        return []
    keys = [by] if isinstance(by, str) else list(by)
    obs = as_frame(adata.obs)
    perturbation = resolve_perturbation_column(adata)
    candidates = [perturbation] if perturbation in obs else []
    candidates += [c for c in obs.columns if c != perturbation and _TREATMENT_COLUMN.fullmatch(str(c))]
    group_codes64 = codes.astype(np.int64)
    n_present = np.unique(codes).size
    erased = []
    for column in candidates:
        if column in keys:
            continue
        # Missing values are their own level (dropna=False): factorize keeps them with use_na_sentinel=False.
        column_codes, uniques = pd.factorize(obs[column], use_na_sentinel=False)
        if len(uniques) < 2:
            continue
        # Constant within every present group iff the number of distinct (group, value) pairs equals the groups present.
        pairs = np.unique(group_codes64 * len(uniques) + column_codes)
        if pairs.size == n_present:
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
        Writes ``X`` or ``layers[key_added]``, and ``var["degenerate_scale"]``, or ``var["degenerate_scale_<key_added>"]`` when writing to a layer, which flags features that have no spread in some group or no reference values to centre on there, and comes with a warning; a feature with no spread in a group is set to NaN in that group, since it has no scale there. Drop the flagged features before computing distances.

    Raises:
        ValueError: If ``method`` is unknown, ``reference`` selects no rows at all or none in some group, or every feature has no spread in some group.
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

    erased = _erased_treatments(adata, by, codes)
    if erased:
        get_logger().warning(
            "normalize: %s is constant within every %s group, so normalizing removes its differences",
            erased,
            by,
            deep="normalize by a coarser grouping to compare them",
        )
    centre, scale, spreadless, unmeasured = _center_and_scale(adata, method, by, codes, keys, layer, mask, epsilon)
    no_spread, uncentred = spreadless.any(axis=0), unmeasured.any(axis=0)
    if adata.n_vars and no_spread.all():
        raise ValueError(
            f"every feature has no spread in at least one group of {by!r} among the reference rows, so none can "
            "be scaled. Check that the reference rows are the controls you meant, or normalize by a coarser grouping."
        )
    degenerate = no_spread | uncentred
    # One flag column per output matrix, so a call writing another layer does not reset the flags describing the first.
    flag = "degenerate_scale" if key_added is None else f"degenerate_scale_{key_added}"
    adata.var[flag] = degenerate
    remedy = f"drop them with adata = adata[:, ~adata.var[{flag!r}]].copy()"
    if uncentred.any():
        affected = np.flatnonzero(unmeasured.any(axis=1))
        get_logger().warning(
            "normalize: %d of %d features have no finite reference value in %d %s group(s) and stay unnormalized",
            int(uncentred.sum()),
            adata.n_vars,
            affected.size,
            by,
            deep=f"they are flagged in var[{flag!r}], and you can {remedy}",
        )
    # Not elif: a feature constant in one group and unmeasured in another is still unscalable in the first.
    if no_spread.any():
        get_logger().warning(
            "normalize: %d of %d features have no reference spread in some %s group and are NaN there",
            int(no_spread.sum()),
            adata.n_vars,
            by,
            deep=f"{', '.join(map(str, adata.var_names[no_spread][:5]))} are flagged in var[{flag!r}], and you can {remedy}",
        )
    lookup = {key: position for position, key in enumerate(keys)}

    def _rescale(key: Any, block: np.ndarray) -> np.ndarray:
        index = lookup[key]
        scaled = (block - centre[index]) / scale[index]
        # A feature with no spread among this group's reference rows has no scale, so its values are undefined here.
        scaled[:, spreadless[index]] = np.nan
        return scaled

    if keep_raw and "raw" not in adata.layers:
        adata.layers["raw"] = get_matrix(adata, layer).copy()

    out = transform_grouped(adata, by, _rescale, layer=layer, grouping=(codes, keys))
    if key_added is None:
        adata.X = out
    else:
        adata.layers[key_added] = out
    return None
