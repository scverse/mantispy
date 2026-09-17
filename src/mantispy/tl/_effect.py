"""Per-feature effect sizes against a reference.

Where mAP asks whether a perturbation is active, these functions ask which features changed and by how much.
:func:`effect_size` gives a standardized difference in location, and :func:`wasserstein_features` a distance between whole distributions, which also detects a change in spread.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from anndata import AnnData

from mantispy._core._numba import _wasserstein_against
from mantispy._core._reduce import get_matrix, group_codes, group_offsets
from mantispy._core._stats import MAD_TO_SIGMA, benjamini_hochberg, mannwhitney_pvalues, sorted_control
from mantispy._core.logging import get_logger
from mantispy._core.masks import reference_mask
from mantispy._core.mutation import inplace_or_copy

METHODS = ("cohens_d", "robust_z")


def _mwu_small_samples(treated: np.ndarray, control: np.ndarray) -> np.ndarray:
    """Scipy's Mann-Whitney U, choosing the exact or asymptotic method per feature.

    ``scipy.stats.mannwhitneyu`` chooses once per call.
    It uses the exact null only when the smaller sample has eight or fewer observations and no column has ties, so one tied feature sends every other feature to the normal approximation.
    With three treated wells against 330 controls, an untied feature reaches 3.3e-07 under the exact null and 2.9e-03 under the approximation.
    Splitting the columns by ties takes 9.3 ms against 8.3 ms for 344 columns.
    """
    from scipy.stats import mannwhitneyu

    finite = np.vstack([treated, control])
    ordered = np.sort(finite, axis=0)
    # A difference involving NaN is never zero, so missing values do not count as ties.
    with np.errstate(invalid="ignore"):
        tied = np.any(np.diff(ordered, axis=0) == 0, axis=0)

    pvalue = np.full(treated.shape[1], np.nan)
    methods: tuple[tuple[np.ndarray, Literal["exact", "asymptotic"]], ...] = ((~tied, "exact"), (tied, "asymptotic"))
    for mask, method in methods:
        if not mask.any():
            continue
        with np.errstate(invalid="ignore"):
            pvalue[mask] = mannwhitneyu(
                treated[:, mask], control[:, mask], axis=0, nan_policy="omit", method=method
            ).pvalue
    return pvalue


def _cohens_d(treated: np.ndarray, control: np.ndarray) -> np.ndarray:
    """Difference in means divided by the pooled standard deviation."""
    n_t, n_c = treated.shape[0], control.shape[0]
    var_t = np.nanvar(treated, axis=0, ddof=1)
    var_c = np.nanvar(control, axis=0, ddof=1)
    pooled = np.sqrt(((n_t - 1) * var_t + (n_c - 1) * var_c) / max(n_t + n_c - 2, 1))
    pooled = np.where((pooled == 0) | ~np.isfinite(pooled), np.nan, pooled)
    with np.errstate(invalid="ignore"):
        return (np.nanmean(treated, axis=0) - np.nanmean(control, axis=0)) / pooled


def _robust_z(treated: np.ndarray, control: np.ndarray) -> np.ndarray:
    """Median difference expressed in control MADs."""
    centre = np.nanmedian(control, axis=0)
    mad = MAD_TO_SIGMA * np.nanmedian(np.abs(control - centre), axis=0)
    mad = np.where((mad == 0) | ~np.isfinite(mad), np.nan, mad)
    with np.errstate(invalid="ignore"):
        return (np.nanmedian(treated, axis=0) - centre) / mad


def _wasserstein_columns(treated: np.ndarray, control: np.ndarray, only: np.ndarray | None = None) -> np.ndarray:
    """Wasserstein-1 distance for every column at once.

    Computes ``W1 = integral |F(x) - G(x)| dx`` over the merged support, vectorized across columns instead of one ``scipy.stats.wasserstein_distance`` call per feature per group (700k calls for 3600 features and 200 perturbations).

    Columns holding a non-finite value fall back to the scalar function, which drops those rows.
    The split is ``isfinite``, the same one :func:`wasserstein_features` uses to choose the columns it sends here.
    Splitting on ``isnan`` instead left an infinite column in neither branch's remit, and it came back ``inf``.
    ``only`` restricts the work to a subset of columns and leaves the rest ``NaN``; callers that handle the complete columns on the pre-sorted path use it for the remaining ones.
    """
    from scipy.stats import wasserstein_distance

    n, m = treated.shape[0], control.shape[0]
    out = np.full(treated.shape[1], np.nan)
    if n == 0 or m == 0:
        return out

    ragged = ~(np.isfinite(treated).all(axis=0) & np.isfinite(control).all(axis=0))
    if only is not None:
        ragged &= only
    for column in np.flatnonzero(ragged):
        t = treated[np.isfinite(treated[:, column]), column]
        c = control[np.isfinite(control[:, column]), column]
        if t.size and c.size:
            out[column] = wasserstein_distance(t, c)

    keep = ~ragged if only is None else (only & ~ragged)
    if not keep.any():
        return out
    values = np.concatenate([treated[:, keep], control[:, keep]], axis=0)
    order = np.argsort(values, axis=0, kind="stable")
    ordered = np.take_along_axis(values, order, axis=0)

    # Both empirical CDFs, stepped along the merged support.
    from_treated = np.cumsum(order < n, axis=0)[:-1]
    positions = np.arange(1, n + m, dtype=np.float64)[:, None]
    gap = np.abs(from_treated / n - (positions - from_treated) / m)
    out[keep] = np.sum(gap * np.diff(ordered, axis=0), axis=0)
    return out


def _tidy(keys: pd.Index, features: np.ndarray, values: np.ndarray, name: str) -> pd.DataFrame:
    """A (n_vars, n_groups) matrix as a long frame, groups in matrix-column order."""
    return pd.DataFrame(
        {
            "group": np.repeat([str(key) for key in keys], features.size),
            "feature": np.tile(features, len(keys)),
            name: values.T.ravel(),
        }
    )


@inplace_or_copy()
def effect_size(
    adata: AnnData,
    groupby: str = "Metadata_Perturbation",
    reference: str | None = "negcon",
    method: str = "cohens_d",
    pvalues: bool = True,
    min_obs: int = 2,
    key_added: str = "effect",
    copy: bool = False,
) -> AnnData | None:
    """Per-feature effect size of each group against the reference.

    Args:
        adata: Object to score, at cell or profile resolution.
        groupby: Column defining the groups to score.
        reference: Rows to compare against: ``"negcon"``, ``None`` for everything, or a boolean ``obs`` column. The reference group is also scored against itself as a calibration check; its effects should be near zero.
        method: ``"cohens_d"`` is the difference in means over the pooled standard deviation. ``"robust_z"`` is the difference in medians in units of control MAD, which a few extreme cells cannot move.
        pvalues: Compute a Mann-Whitney p-value for each effect. At single-cell resolution nearly every feature is significant, so turn them off when ranking by effect.
        min_obs: Groups with fewer rows than this are left unscored as ``NaN``.
        key_added: Name for the outputs.
        copy: Return a modified copy instead of mutating in place.

    Returns:
        ``None``, or the modified copy.
        Writes ``varm[key_added]``, a ``(n_vars, n_groups)`` float32 array with columns in the order of ``uns["mantispy"][key_added + "_groups"]``, and ``uns["mantispy"][key_added]``, a tidy frame with ``group``, ``feature``, ``effect``, ``pvalue`` (Mann-Whitney U, two-sided, ``NaN`` when ``pvalues=False``) and ``qvalue`` (Benjamini-Hochberg over the whole table, since it is one family of tests).

    Raises:
        ValueError: ``method`` is not one of ``METHODS``, or ``reference`` selects fewer than two rows.

    Notes:
        The p-value tests whether the distributions differ, and its significance grows with the number of rows.
        The effect size measures by how much, and does not grow with the number of rows.
        At single-cell resolution nearly everything is significant, so rank by effect and use the q-value only to filter.
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")

    X = get_matrix(adata)
    control = X[reference_mask(adata, reference)]
    if control.shape[0] < 2:
        raise ValueError(
            f"effect sizes need at least two reference rows, reference={reference!r} selects {control.shape[0]}"
        )

    codes, keys = group_codes(adata, groupby)
    estimate = _cohens_d if method == "cohens_d" else _robust_z

    # Sorted once and searched by every group; scipy re-ranks the whole reference per group, which costs about three minutes on JUMP.
    ranked = sorted_control(control) if pvalues else None
    # Counted as sorted_control and scipy's nan_policy="omit" count, everything measured and infinities included, so that the branch chosen below is the branch scipy would choose.
    control_smallest = int((~np.isnan(control)).sum(axis=0).min()) if pvalues else 0

    effects = np.full((adata.n_vars, len(keys)), np.nan)
    significance = np.full((adata.n_vars, len(keys)), np.nan)
    skipped = []
    order, offsets = group_offsets(codes, len(keys))
    for index, key in enumerate(keys):
        treated = X[order[offsets[index] : offsets[index + 1]]]
        if treated.shape[0] < min_obs:
            skipped.append(str(key))
            continue
        effects[:, index] = estimate(treated, control)
        if not pvalues:
            continue
        # scipy's method="auto" uses the exact distribution when the smaller sample has eight or fewer observations.
        # Those groups go to scipy; larger ones, where "auto" would use the normal approximation, take the fast path.
        smallest = int((~np.isnan(treated)).sum(axis=0).min())
        if ranked is not None and smallest > 8 and control_smallest > 8:
            significance[:, index] = mannwhitney_pvalues(treated, ranked)
        else:
            significance[:, index] = _mwu_small_samples(treated, control)

    if skipped:
        get_logger().info("effect_size left %d group(s) unscored for having < %d rows", len(skipped), min_obs)

    table = _tidy(keys, adata.var_names.to_numpy(), effects, "effect")
    table["pvalue"] = significance.T.ravel()
    table["qvalue"] = benjamini_hochberg(table["pvalue"].to_numpy())

    adata.varm[key_added] = effects.astype(np.float32)
    store = adata.uns.setdefault("mantispy", {})
    store[key_added] = table
    store[f"{key_added}_groups"] = [str(key) for key in keys]
    return None


@inplace_or_copy()
def wasserstein_features(
    adata: AnnData,
    groupby: str = "Metadata_Perturbation",
    reference: str | None = "negcon",
    key_added: str = "wasserstein",
    copy: bool = False,
) -> AnnData | None:
    """Wasserstein-1 distance per feature between each group and the reference.

    Args:
        adata: Object to score. Most useful at single-cell resolution, where a group is a distribution rather than a point.
        groupby: As in :func:`effect_size`.
        reference: As in :func:`effect_size`.
        key_added: Name for the outputs.
        copy: Return a modified copy instead of mutating in place.

    Returns:
        ``None``, or the modified copy.
        Writes ``varm[key_added]``, a tidy ``uns["mantispy"][key_added]`` with ``group``, ``feature`` and ``distance``, and the group order in ``uns["mantispy"][key_added + "_groups"]``, laid out as in :func:`effect_size`.

    Raises:
        ValueError: ``reference`` selects no rows.

    Notes:
        The distance compares whole distributions, so a perturbation that widens a feature without moving its mean (a mixed response where only some cells react) is detected here but not by :func:`effect_size`.
        The distance is in the feature's units, so normalize first to compare features with each other.
    """
    X = get_matrix(adata)
    control = X[reference_mask(adata, reference)]
    if control.shape[0] == 0:
        raise ValueError(f"no reference rows selected by reference={reference!r}")

    codes, keys = group_codes(adata, groupby)
    # The reference is sorted once and reused for every group, by the same helper effect_size uses, so the two paths through this module cannot count it differently.
    # Columns holding a non-finite value take the general path, which drops those rows pairwise.
    ranked, counts, _ = sorted_control(control)
    clean = np.isfinite(control).all(axis=0)

    distances = np.empty((adata.n_vars, len(keys)))
    order, offsets = group_offsets(codes, len(keys))
    for index in range(len(keys)):
        treated = X[order[offsets[index] : offsets[index + 1]]]
        usable = clean & np.isfinite(treated).all(axis=0)
        distances[:, index] = _wasserstein_columns(treated, control, only=~usable)
        if usable.any() and treated.shape[0]:
            block = np.ascontiguousarray(np.asarray(treated[:, usable], dtype=np.float64).T)
            distances[usable, index] = _wasserstein_against(block, ranked[usable], counts[usable])

    adata.varm[key_added] = distances.astype(np.float32)
    store = adata.uns.setdefault("mantispy", {})
    store[key_added] = _tidy(keys, adata.var_names.to_numpy(), distances, "distance")
    store[f"{key_added}_groups"] = [str(key) for key in keys]
    return None
