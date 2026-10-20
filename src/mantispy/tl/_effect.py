from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from anndata import AnnData

from mantispy._core._numba import _mwu_exact_sf, _mwu_moments, _wasserstein_against
from mantispy._core._reduce import get_matrix, group_codes, group_offsets
from mantispy._core._stats import MAD_TO_SIGMA, benjamini_hochberg, mannwhitney_pvalues, sorted_control
from mantispy._core.logging import get_logger
from mantispy._core.masks import reference_mask
from mantispy._core.mutation import inplace_or_copy

METHODS = ("cohens_d", "robust_z")


def _mwu_exact_untied(
    treated: np.ndarray, control: np.ndarray, ranked: tuple[np.ndarray, np.ndarray, np.ndarray]
) -> np.ndarray:
    """Two-sided Mann-Whitney p-values: exact on untied features, the normal approximation on tied ones.

    Matches ``scipy.stats.mannwhitneyu(method="exact")`` on untied features and ``method="asymptotic")`` on tied ones,
    but builds the exact null once per distinct ``(n_treated, n_control)`` pair instead of once per feature, which is
    what scipy does and what makes it slow: the untied null depends only on the two sample sizes, so thousands of
    features share a handful of nulls. A feature is untied when its pooled sample has no repeated value.

    Tied features go to scipy's own asymptotic method rather than the in-house one, whose tie-corrected variance
    differs from scipy's at the eighth significant figure; they are rare, so the per-feature call costs nothing.
    """
    pvalue = mannwhitney_pvalues(treated, ranked)
    values, counts, control_ties = ranked
    block = np.ascontiguousarray(np.asarray(treated, dtype=np.float64).T)
    statistic, tie_term, n_treated = _mwu_moments(block, values, counts, control_ties)

    tied = (tie_term > 0) & (n_treated > 0) & (counts > 0)
    if tied.any():
        from scipy.stats import mannwhitneyu

        with np.errstate(invalid="ignore"):
            pvalue[tied] = mannwhitneyu(
                treated[:, tied], control[:, tied], axis=0, nan_policy="omit", method="asymptotic"
            ).pvalue

    untied = (tie_term == 0) & (n_treated > 0) & (counts > 0)
    if not untied.any():
        return pvalue
    n1, n2 = n_treated[untied], counts[untied]
    u_max = np.maximum(statistic[untied], n1 * n2 - statistic[untied]).astype(np.int64)
    out = pvalue[untied].copy()
    for pair in np.unique(np.stack([n1, n2], axis=1), axis=0):
        a, b = int(pair[0]), int(pair[1])
        same = (n1 == a) & (n2 == b)
        sf = _mwu_exact_sf(a, b)
        out[same] = np.clip(2.0 * sf[u_max[same]], 0.0, 1.0)
    pvalue[np.flatnonzero(untied)] = out
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

    Columns holding a non-finite value fall back to the scalar function, which drops those rows.
    The split is ``isfinite``, the same one :func:`wasserstein_features` uses to choose the columns it sends here.
    ``only`` restricts the work to a subset of columns and leaves the rest ``NaN``.
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
    exact: bool | Literal["auto"] = "auto",
    min_obs: int = 2,
    key_added: str = "effect",
    copy: bool = False,
) -> AnnData | None:
    """Per-feature effect size of each group against the reference.

    Args:
        adata: Object to score, at cell or profile resolution.
        groupby: Column defining the groups to score.
        reference: Rows to compare against: ``"negcon"``, ``None`` for everything, or a boolean ``obs`` column.
            The reference group is also scored against itself as a calibration check; its effects should be near zero.
        method: ``"cohens_d"`` is the difference in means over the pooled standard deviation.
            ``"robust_z"`` is the difference in medians in units of control MAD, which a few extreme cells cannot move.
        pvalues: Compute a Mann-Whitney p-value for each effect.
            At single-cell resolution nearly every feature is significant, so turn them off when ranking by effect.
        exact: How to compute the p-value's null for a group's untied features.
            ``"auto"`` uses the exact null when both the group and the reference have eight or fewer measured rows and
            the fast normal approximation otherwise, matching scipy.
            ``False`` always uses the normal approximation, which is the fast path and the sensible choice for a screen
            with many small groups, where the exact null dominates the runtime for no change a reader would see.
            ``True`` always uses the exact null. Tied features take the normal approximation whatever this is set to,
            because the exact null is only defined without ties.
        min_obs: Groups with fewer rows than this are left unscored as ``NaN``.
        key_added: Name for the outputs.
        copy: Return a modified copy instead of mutating in place.

    Returns:
        ``None``, or the modified copy.
        Writes ``varm[key_added]``, a ``(n_vars, n_groups)`` float32 array with columns in the order of ``uns["mantispy"][key_added + "_groups"]``, and ``uns["mantispy"][key_added]``, a tidy frame with ``group``, ``feature``, ``effect``, ``pvalue`` (Mann-Whitney U, two-sided, ``NaN`` when ``pvalues=False``) and ``qvalue`` (Benjamini-Hochberg over the whole table, since it is one family of tests).

    Raises:
        ValueError: ``method`` is not one of ``METHODS``, ``exact`` is not ``True``, ``False`` or ``"auto"``, or ``reference`` selects fewer than two rows.

    Notes:
        The p-value tests whether the distributions differ, and its significance grows with the number of rows.
        The effect size measures by how much, and does not grow with the number of rows.
        At single-cell resolution nearly everything is significant, so rank by effect and use the q-value only to filter.
        The exact null depends only on the two sample sizes, so it is built once per pair of sizes and reused across every feature that shares them.
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    if exact not in (True, False, "auto"):
        raise ValueError(f"exact must be True, False or 'auto', got {exact!r}")

    X = get_matrix(adata)
    control = X[reference_mask(adata, reference)]
    if control.shape[0] < 2:
        raise ValueError(
            f"effect sizes need at least two reference rows, reference={reference!r} selects {control.shape[0]}"
        )

    codes, keys = group_codes(adata, groupby)
    estimate = _cohens_d if method == "cohens_d" else _robust_z

    ranked = sorted_control(control) if pvalues else None
    # Infinities count as measured, as in scipy's nan_policy="omit", so the branch below is the one scipy would choose.
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
        if not pvalues or ranked is None:
            continue
        # "auto" takes the exact null only for small samples, the size at which scipy's method="auto" would.
        smallest = int((~np.isnan(treated)).sum(axis=0).min())
        use_exact = exact is True or (exact == "auto" and smallest <= 8 and control_smallest <= 8)
        if use_exact:
            significance[:, index] = _mwu_exact_untied(treated, control, ranked)
        else:
            significance[:, index] = mannwhitney_pvalues(treated, ranked)

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
        adata: Object to score.
            Most useful at single-cell resolution, where a group is a distribution rather than a point.
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
        raise ValueError(
            f"no reference rows selected by reference={reference!r}. Check that obs['Metadata_Control_Type'] has "
            "'negcon' rows (mt.pp.annotate_controls), or pass reference=None to use all rows."
        )

    codes, keys = group_codes(adata, groupby)
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
