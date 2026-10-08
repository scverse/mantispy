"""Shared statistics: multiple-testing correction and permutation nulls."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from anndata import AnnData

from mantispy._core._corr import _blockwise
from mantispy._core._reduce import group_codes
from mantispy._core.frames import as_frame
from mantispy._core.schema import REQUIRED_OBS

#: Scale factor that makes the median absolute deviation estimate the standard deviation of a normal distribution.
MAD_TO_SIGMA = 1.4826


def _is_object_resolution(adata: AnnData) -> bool:
    """Whether each row is one primary segmented object, i.e. explicitly stamped ``object`` resolution.

    An unstamped object is not: the pseudoreplication safeguards must not assume object-level rows.
    """
    return adata.uns.get("mantispy", {}).get("resolution") == "object"


def _default_well_block(adata: AnnData, *, block: str | Sequence[str] | None = None) -> np.ndarray | None:
    """Per-row codes of the design's exchangeable unit, the well, or ``None`` when there is none to draw.

    A permutation null at object resolution must resample whole wells rather than objects, since objects within a
    well share the well and are not independent replicates. This centralizes the choice of that unit:

    - ``block`` given: its groups, whatever they are.
    - otherwise, on an object stamped ``object`` resolution that carries a complete ``(Metadata_Plate, Metadata_Well)``
      with more than one object in some well, the physical well.
    - otherwise ``None``, so the caller can warn and fall back to an object-level null.
    """
    if block is not None:
        return group_codes(adata, block)[0]
    if not _is_object_resolution(adata):
        return None
    # The exchangeable unit is the physical well (plate + well), not the object-resolution identity,
    # which now also carries the per-object image/type/number columns.
    well = list(REQUIRED_OBS["well"])
    obs = as_frame(adata.obs)
    if not set(well) <= set(adata.obs) or bool(obs[well].isna().to_numpy().any()):
        # group_codes rejects gaps, so a well column with any missing value is unusable.
        return None
    codes = group_codes(adata, well)[0]
    return codes if np.bincount(codes).max() > 1 else None


def _split_wells(codes: np.ndarray, generator: np.random.Generator) -> np.ndarray:
    """Boolean mask over ``codes`` selecting the rows of a random half of the wells they name.

    The block-aware sibling of :func:`split_reference`: it halves by whole wells, so a reference group's own row draws a strict subset of the wells like any other group rather than folding cells across the split.
    """
    wells = np.unique(codes)
    return np.isin(codes, generator.permutation(wells)[: wells.size // 2])


def benjamini_hochberg(pvalues: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg q-values.

    ``NaN`` in, ``NaN`` out, and excluded from the count.
    """
    pvalues = np.asarray(pvalues, dtype=np.float64)
    # Correct over every p-value whatever the shape, since a features-by-groups table is one family of tests.
    flat = pvalues.ravel()
    out = np.full(flat.shape, np.nan)
    finite = np.flatnonzero(np.isfinite(flat))
    if finite.size == 0:
        return out.reshape(pvalues.shape)

    values = flat[finite]
    order = np.argsort(values)
    ranked = values[order]
    n = ranked.size
    # q_(i) = min over j >= i of p_(j) * n / j, with i the 1-based ascending rank.
    adjusted = np.minimum.accumulate((ranked * n / np.arange(1, n + 1))[::-1])[::-1]
    out[finite[order]] = np.clip(adjusted, 0.0, 1.0)
    return out.reshape(pvalues.shape)


def permutation_pvalue(observed: np.ndarray, null: np.ndarray) -> np.ndarray:
    """Right-tailed permutation p-value, one row of ``null`` per observation.

    Uses ``(count + 1) / (n + 1)``, because a p-value of zero would claim more resolution than a finite number of permutations supports.
    """
    observed = np.asarray(observed, dtype=np.float64)
    null = np.atleast_2d(np.asarray(null, dtype=np.float64))
    at_least = (null >= observed[:, None]).sum(axis=1)
    pvalues = (at_least + 1.0) / (null.shape[1] + 1.0)
    # `nan >= x` is False, so an unmeasured observation, or one with no computable null, would otherwise get the smallest p-value the permutations can express.
    unmeasured = np.isnan(observed) | ~np.isfinite(null).any(axis=1)
    return np.where(unmeasured, np.nan, pvalues)


def robust_zscore(values: np.ndarray, axis: int = 0) -> np.ndarray:
    """Median/MAD z-score.

    A zero or non-finite spread gives zero rather than infinity, since a constant feature is not evidence of anything.
    An infinite value keeps its infinite z.
    CellProfiler ratio features produce such values, and mapping them to zero would score the most extreme cell as the most ordinary one.
    """
    values = np.asarray(values, dtype=np.float64)
    median = np.nanmedian(values, axis=axis, keepdims=True)
    mad = MAD_TO_SIGMA * np.nanmedian(np.abs(values - median), axis=axis, keepdims=True)
    mad = np.where((mad == 0) | ~np.isfinite(mad), np.nan, mad)
    with np.errstate(invalid="ignore", divide="ignore"):
        z = (values - median) / mad
    return np.nan_to_num(z, nan=0.0, posinf=np.inf, neginf=-np.inf)


def sorted_control(control: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Feature-major sorted reference, its measured counts, and its tie term per feature.

    Sorting the reference once makes :func:`mannwhitney_pvalues` cheap, because each group is then a binary search into it instead of another ranking of the whole reference.
    """
    values = np.ascontiguousarray(np.asarray(control, dtype=np.float64).T)
    values = np.sort(values, axis=1)  # missing values sort to the end
    # Infinities count as measured, because scipy ranks them as the extreme values they are.
    counts = (~np.isnan(values)).sum(axis=1)

    # sum(c ** 3 - c) over runs of equal values, the tie correction's reference half.
    position = np.arange(1, values.shape[1])[None, :]
    boundary = (values[:, 1:] != values[:, :-1]) | (position >= counts[:, None])
    ties = np.zeros(values.shape[0])
    for row in range(values.shape[0]):
        if counts[row]:
            edges = np.flatnonzero(boundary[row][: counts[row] - 1] if counts[row] > 1 else [])
            lengths = np.diff(np.concatenate([[-1], edges, [counts[row] - 1]]))
            ties[row] = float((lengths**3 - lengths).sum())
    return values, counts, ties


def mannwhitney_pvalues(treated: np.ndarray, reference: tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    """Two-sided Mann-Whitney p-values per feature, normal approximation with tie correction.

    Matches ``scipy.stats.mannwhitneyu(treated, control, axis=0, method="asymptotic")`` to machine precision, computed against a reference that was sorted once.
    scipy's ``method="auto"`` uses the exact distribution when the smaller sample has eight or fewer observations and there are no ties.
    Callers that can hit that case should send those groups to scipy, as :func:`mantispy.tl.effect_size` does.
    """
    from scipy.stats import norm

    from mantispy._core._numba import _mwu_moments

    values, counts, ties = reference
    block = np.ascontiguousarray(np.asarray(treated, dtype=np.float64).T)
    statistic, tie_term, n_treated = _mwu_moments(block, values, counts, ties)

    n_1, n_2 = n_treated.astype(np.float64), counts.astype(np.float64)
    total = n_1 + n_2
    with np.errstate(invalid="ignore", divide="ignore"):
        spread = np.sqrt(n_1 * n_2 / 12 * ((total + 1) - tie_term / (total * (total - 1))))
        numerator = statistic - n_1 * n_2 / 2
        numerator = numerator - 0.5 * np.sign(numerator)  # continuity correction
        return np.clip(2 * norm.sf(np.abs(numerator / spread)), 0.0, 1.0)


def split_reference(rows: np.ndarray, generator: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Halve the reference rows into a half to fit on and a half to draw the null from.

    A permutation null estimates how far out a group of this size would be if it were only controls.
    If the same control rows define the centroid, covariance or baseline the statistic is measured against, the null is in-sample while every real group is out-of-sample, and the null comes out too small.
    On pure noise, that called 12 of 12 groups hits.

    Splitting leaves half the rows for the estimate, in exchange for a null computed with the same arithmetic as the statistic.
    """
    rows = np.asarray(rows)
    if rows.size < 4:
        raise ValueError(
            f"a split reference needs at least four rows (half to fit the statistic, half for the null), "
            f"got {rows.size}. The caller should check the size of its reference before calling this."
        )
    order = generator.permutation(rows.size)
    half = rows.size // 2
    return np.sort(rows[order[:half]]), np.sort(rows[order[half:]])


def nanvar(X: np.ndarray, ddof: int = 0) -> np.ndarray:
    """Per-feature variance, ignoring missing values.

    A feature measured on no cell at all yields NaN rather than a warning: numpy raises "Degrees of freedom <= 0 for slice" through :mod:`warnings`, where ``np.errstate`` cannot reach it.
    ``ddof=0`` is the population variance, which is what sklearn's ``VarianceThreshold`` and pycytominer compare.
    """
    out_dtype = X.dtype if np.issubdtype(X.dtype, np.inexact) else np.float64
    return _blockwise(X, lambda block: np.nanvar(block, axis=0, ddof=ddof), out_dtype)
