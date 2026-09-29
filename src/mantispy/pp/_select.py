"""Feature selection, with pycytominer's semantics.

Every operation returns a boolean keep mask over ``var``.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence

import numpy as np
from anndata import AnnData

from mantispy._core._corr import _blockwise, correlated_pairs, rank_revealing_subset
from mantispy._core._reduce import get_matrix, group_codes, group_offsets
from mantispy._core._stats import nanvar
from mantispy._core.features import blocklist_hits
from mantispy._core.frames import as_frame
from mantispy._core.logging import get_logger
from mantispy._core.mutation import inplace_or_copy

#: Operations, in the order pycytominer applies them by default, after mantispy's own ``drop_degenerate``.
OPERATIONS = (
    "drop_degenerate",
    "variance_threshold",
    "frequency_threshold",
    "correlation_threshold",
    "drop_na_columns",
    "blocklist",
    "drop_outliers",
    "noise_removal",
)

DEFAULT_OPERATIONS = (
    "drop_degenerate",
    "variance_threshold",
    "correlation_threshold",
    "drop_na_columns",
    "blocklist",
)


def _op_drop_degenerate(adata: AnnData) -> np.ndarray:
    """Drop the features :func:`~mantispy.pp.normalize` flagged in ``var["degenerate_scale"]`` because it could not scale them.

    An object :func:`~mantispy.pp.normalize` never flagged has no such column and keeps every feature.
    """
    if "degenerate_scale" not in adata.var:
        return np.ones(adata.n_vars, dtype=bool)
    return ~as_frame(adata.var)["degenerate_scale"].to_numpy(dtype=bool)


def _op_variance_threshold(X: np.ndarray, min_variance: float = 1e-6) -> np.ndarray:
    """Drop features whose variance is at or below ``min_variance``.

    sklearn's ``VarianceThreshold`` keeps ``variance > threshold`` and uses the population variance, so ``ddof=0``.
    """
    return np.nan_to_num(nanvar(X), nan=0.0, posinf=0.0) > min_variance


def _op_frequency_threshold(X: np.ndarray, freq_cut: float = 0.05, unique_cut: float = 0.01) -> np.ndarray:
    """Drop near-constant features, by value frequency or by uniqueness.

    The two rules are independent and combined with OR, both with a strict ``<``.
    """
    n_obs, n_vars = X.shape
    keep = np.ones(n_vars, dtype=bool)
    for j in range(n_vars):
        column = X[:, j]
        counts = np.sort(np.unique(column[~np.isnan(column)], return_counts=True)[1])[::-1]
        frequency_ratio = counts[1] / counts[0] if counts.size >= 2 else 0.0
        unique_ratio = counts.size / n_obs
        keep[j] = not (frequency_ratio < freq_cut or unique_ratio < unique_cut)
    return keep


def _greedy_keep(pairs: np.ndarray, total: np.ndarray, n_vars: int) -> np.ndarray:
    """Keep mask that drops the more-connected member of every over-threshold pair."""
    ranking = np.argsort(total, kind="stable")
    rank = np.empty(n_vars, dtype=np.int64)
    rank[ranking] = np.arange(n_vars)
    keep = np.ones(n_vars, dtype=bool)
    if pairs.size:
        first, second = pairs[:, 0], pairs[:, 1]
        keep[np.where(rank[first] > rank[second], first, second)] = False
    return keep


def _op_correlation_threshold(
    X: np.ndarray,
    threshold: float = 0.9,
    method: str = "pearson",
    *,
    absolute: bool = False,
    iterative: bool = False,
    order: np.ndarray | None = None,
    window: int | None = None,
    stride: int | None = None,
) -> np.ndarray:
    """Drop one feature from every pair correlated above ``threshold``.

    With the defaults this matches pycytominer: each over-threshold pair is judged on its own against a ranking of total absolute correlation, the member ranked as more correlated overall is dropped, the comparison is on the signed correlation, and there is no re-sweep.
    """
    n_vars = X.shape[1]
    reduce = _iterative_correlation_drop if iterative else _greedy_keep
    if window is None:
        pairs, total = correlated_pairs(X, threshold, method=method, absolute=absolute)
        return reduce(pairs, total, n_vars)
    pairs, total = correlated_pairs(
        X, threshold, method=method, absolute=absolute, order=order, window=window, stride=stride
    )
    keep = reduce(pairs, total, n_vars)
    survivors = np.flatnonzero(keep)
    if survivors.size > 1:
        sub_pairs, sub_total = correlated_pairs(X[:, survivors], threshold, method=method, absolute=absolute)
        keep[survivors] = reduce(sub_pairs, sub_total, survivors.size)
    return keep


def _iterative_correlation_drop(pairs: np.ndarray, total: np.ndarray, n_vars: int) -> np.ndarray:
    """Remove the most-connected feature until no over-threshold pair is left.

    At each step the still-connected feature belonging to the most surviving pairs is dropped, ties broken by the largest total ``|r|`` and then the lowest index.
    """
    keep = np.ones(n_vars, dtype=bool)
    active = np.ones(len(pairs), dtype=bool)
    while active.any():
        connected, degree = np.unique(pairs[active].ravel(), return_counts=True)
        # lexsort reads keys last-first: degree, then total, then index.
        worst = connected[np.lexsort((connected, -total[connected], -degree))[0]]
        keep[worst] = False
        active &= (pairs[:, 0] != worst) & (pairs[:, 1] != worst)
    return keep


def _op_drop_na_columns(X: np.ndarray, cutoff: float = 0.05) -> np.ndarray:
    """Drop features missing in more than ``cutoff`` of rows."""
    return np.isnan(X).mean(axis=0) <= cutoff


def _op_drop_outliers(X: np.ndarray, outlier_cutoff: float = 500.0) -> np.ndarray:
    """Drop features whose largest absolute value exceeds ``outlier_cutoff``."""
    # One column block at a time (in _blockwise), so np.abs never copies more than one block.
    largest = _blockwise(X, lambda block: np.nanmax(np.abs(block), axis=0), np.float64)
    return ~(np.nan_to_num(largest, nan=0.0) > outlier_cutoff)


def _op_blocklist(adata: AnnData, blocklist: str | Sequence[str] = "default") -> np.ndarray:
    """Drop features named in the blocklist.

    Matched against the current names and, like :func:`~mantispy.pp.filter_features`, against ``var["original_name"]``.
    """
    names = [adata.var_names.to_numpy()]
    if "original_name" in adata.var:
        names.append(adata.var["original_name"].astype(str).to_numpy())
    return ~blocklist_hits(names, blocklist)


def _op_noise_removal(
    X: np.ndarray, codes: np.ndarray, stdev_cutoff: float = 0.8, *, mode: str = "absolute"
) -> np.ndarray:
    """Drop features that vary too much within a perturbation group.

    The within-group spread of a feature is the mean, over groups, of each group's population standard deviation (``ddof=0``).
    ``mode`` sets what ``stdev_cutoff`` is compared against:

    - ``"absolute"``: the spread itself, pycytominer's behaviour, meaningful only on the scale the normalization left.
    - ``"ratio"``: the spread divided by the feature's total standard deviation, so the cutoff means the same whichever normalization preceded it.
    - ``"quantile"``: ``stdev_cutoff`` is read as a quantile of the observed within-group spreads, keeping the features below it.
    """
    if mode not in ("absolute", "ratio", "quantile"):
        raise ValueError(f"noise_removal_cutoff_mode must be 'absolute', 'ratio' or 'quantile', got {mode!r}")
    n_groups = int(codes.max()) + 1
    order, offsets = group_offsets(codes, n_groups)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # "Degrees of freedom <= 0 for slice", once per all-NaN group
        deviations = np.stack(
            [np.nanstd(X[order[offsets[group] : offsets[group + 1]]], axis=0, ddof=0) for group in range(n_groups)]
        )
        within = np.nan_to_num(deviations.mean(axis=0), nan=0.0)
    cutoff = float(np.nanquantile(within, stdev_cutoff)) if mode == "quantile" else stdev_cutoff
    if mode == "ratio":
        total = np.nan_to_num(np.nanstd(X, axis=0, ddof=0), nan=0.0)
        # A feature with no total spread has no noise to remove, so its ratio is zero and it is kept.
        within = np.divide(within, total, out=np.zeros_like(within), where=total > 0)
    return ~(within > cutoff)


@inplace_or_copy()
def feature_select(
    adata: AnnData,
    operations: Sequence[str] = DEFAULT_OPERATIONS,
    min_variance: float = 1e-6,
    freq_cut: float = 0.05,
    unique_cut: float = 0.01,
    corr_threshold: float = 0.9,
    corr_method: str = "pearson",
    corr_window: int | None = None,
    corr_stride: int | None = None,
    na_cutoff: float = 0.05,
    outlier_cutoff: float = 500.0,
    blocklist: str | Sequence[str] = "default",
    noise_removal_perturb_groups: str = "Metadata_Perturbation",
    noise_removal_stdev_cutoff: float = 0.8,
    corr_absolute: bool = False,
    corr_iterative: bool = False,
    decorrelate: bool = False,
    decorr_threshold: float = 0.99,
    decorr_method: str = "pearson",
    key_added: str = "selected",
    copy: bool = False,
    *,
    noise_removal_cutoff_mode: str = "absolute",
) -> AnnData | None:
    """Flag the features worth keeping.

    Args:
        adata: Object to select features on.
            Usually well-level profiles.
        operations: Which operations to run, from ``OPERATIONS``.
            The default is pycytominer's own, which omits ``frequency_threshold``, ``drop_outliers`` and ``noise_removal``, plus ``drop_degenerate``, which removes nothing from an object :func:`~mantispy.pp.normalize` did not flag.
        min_variance: ``variance_threshold``: keep features with variance above this.
        freq_cut: ``frequency_threshold``: drop a feature when the count of its second most common value divided by the count of its most common is below this.
            Either this rule or ``unique_cut`` drops a feature.
        unique_cut: ``frequency_threshold``: drop a feature when its share of distinct values is below this.
        corr_threshold: ``correlation_threshold``: drop one member of every pair correlated above this.
        corr_method: ``correlation_threshold``: ``"pearson"`` or ``"spearman"``.
        corr_window: ``correlation_threshold``: ``None`` runs the exact pass (the default, matching pycytominer).
            An int switches to the two-pass fast path (prune redundancy within name-sorted windows of that size, then run the exact pass on the survivors); 500 is a good default, several times faster on large screens.
            It keeps a different set of features (a different member of each correlated group) but preserves the information and the downstream signal.
        corr_stride: ``correlation_threshold``: step between windows, default half the window.
        corr_absolute: ``correlation_threshold``: threshold ``|r|`` rather than the signed correlation, so a strongly anti-correlated pair is also reduced.
            Off by default, matching pycytominer.
        corr_iterative: ``correlation_threshold``: drop the most-connected feature one at a time until no pair is left, keeping at least as many features as the single pass.
            Off by default.
            With ``corr_absolute`` this approximates cytominer's R path (``caret::findCorrelation``).
        decorrelate: Run an extra, experimental redundancy step after the operations, on the features they keep.
            Unlike ``correlation_threshold`` it removes features that are a linear combination of several others, not just pairwise duplicates, by a rank-revealing QR.
            Off by default, and not part of pycytominer.
        decorr_threshold: ``decorrelate``: drop a feature once its multiple correlation with the kept set reaches this.
            The default 0.99 removes only near-collinear features, so the kept set spans almost the same space; lower it towards ``corr_threshold`` for a smaller, more aggressive set.
        decorr_method: ``decorrelate``: ``"pearson"`` or ``"spearman"``.
        na_cutoff: ``drop_na_columns``: drop features missing in more than this fraction of rows.
        outlier_cutoff: ``drop_outliers``: drop features whose absolute value exceeds this.
        blocklist: ``blocklist``: ``"default"`` for the bundled list, or explicit names.
            Matched against the current names and against ``var["original_name"]``, so it works either side of :func:`~mantispy.pp.standardize_feature_names`.
        noise_removal_perturb_groups: ``noise_removal``: ``obs`` column grouping replicates.
        noise_removal_stdev_cutoff: ``noise_removal``: drop features whose within-group standard deviation, averaged over groups, is above this.
            An absolute threshold on the scale :func:`~mantispy.pp.normalize` left the values on, so it is only meaningful next to the normalization that produced them; pycytominer's default assumes whole-plate standardization.
        key_added: Name of the boolean ``var`` column to write.
        copy: Return a modified copy instead of mutating in place.
        noise_removal_cutoff_mode: ``noise_removal``: what ``noise_removal_stdev_cutoff`` is compared against.
            ``"absolute"`` (the default) keeps pycytominer's behaviour, comparing the raw within-group spread and so tied to the normalization scale.
            ``"ratio"`` compares the within-group spread to the feature's total spread, and ``"quantile"`` reads the cutoff as a quantile of the observed within-group spreads; both keep a useful subset under the control-based or MAD normalization the package recommends, where the absolute cutoff keeps everything or nothing.

    Returns:
        ``None``, or the modified copy.
        Writes ``var[key_added]`` and a per-operation count of removals to ``uns["mantispy"]["feature_select"]``, each count being what that operation removes on its own.
        Nothing is dropped; use :func:`subset_features` for that.

    Raises:
        ValueError: If ``operations`` names an operation that is not in ``OPERATIONS``.
        KeyError: If ``noise_removal`` is requested but ``noise_removal_perturb_groups`` is not an ``obs`` column.

    Notes:
        ``drop_degenerate`` runs first, and the other operations judge only the features it keeps: a feature :func:`~mantispy.pp.normalize` could not scale can hold values large enough to decide the correlation ranking of every feature it is compared with.
        Every other operation judges that whole set, so each count in ``uns["mantispy"]["feature_select"]`` says what that operation alone would remove and is the same whatever order ``operations`` runs in.
        The counts therefore overlap: a feature that is both constant and mostly missing is counted by ``variance_threshold`` and by ``drop_na_columns``, and the counts sum to more than the number of features actually removed, which is ``n_vars`` minus ``var[key_added].sum()``.
        ``decorrelate`` is the exception: it runs last, on the features the operations kept, so its count is what it removes from those survivors.

        ``correlation_threshold`` is the most expensive operation.
        pycytominer uses ``pandas.DataFrame.corr``, one Cython pass per column pair.
        Here the pairs come from chunked matrix products over blocks of columns, so no ``n_vars ** 2`` array is held in memory.
    """
    unknown = set(operations) - set(OPERATIONS)
    if unknown:
        raise ValueError(f"unknown operation(s) {sorted(unknown)}; choose from {OPERATIONS}")

    full = get_matrix(adata)
    keep = np.ones(adata.n_vars, dtype=bool)
    removed: dict[str, int] = {}
    if "drop_degenerate" in operations:
        keep = _op_drop_degenerate(adata)
        removed["drop_degenerate"] = int((~keep).sum())
    judged = np.flatnonzero(keep)
    X = full[:, judged] if judged.size < adata.n_vars else full

    for operation in operations:
        if operation == "drop_degenerate":
            continue
        if operation == "variance_threshold":
            mask = _op_variance_threshold(X, min_variance)
        elif operation == "frequency_threshold":
            mask = _op_frequency_threshold(X, freq_cut, unique_cut)
        elif operation == "correlation_threshold":
            order = None
            if corr_window is not None:
                # Names of the columns of the sliced X, so order indexes X's columns 0..judged.size-1.
                names = np.asarray(adata.var_names[judged], dtype=str)
                order = np.argsort(names, kind="stable")
            mask = _op_correlation_threshold(
                X,
                corr_threshold,
                corr_method,
                absolute=corr_absolute,
                iterative=corr_iterative,
                order=order,
                window=corr_window,
                stride=corr_stride,
            )
        elif operation == "drop_na_columns":
            mask = _op_drop_na_columns(X, na_cutoff)
        elif operation == "blocklist":
            mask = _op_blocklist(adata, blocklist)[judged]
        elif operation == "drop_outliers":
            mask = _op_drop_outliers(X, outlier_cutoff)
        else:
            # Unknown names were rejected above, so what is left is `noise_removal`.
            if noise_removal_perturb_groups not in adata.obs:
                raise KeyError(f"obs has no column {noise_removal_perturb_groups!r} to group replicates by")
            codes, _ = group_codes(adata, noise_removal_perturb_groups)
            mask = _op_noise_removal(X, codes, noise_removal_stdev_cutoff, mode=noise_removal_cutoff_mode)
        removed[operation] = int((~mask).sum())
        keep[judged] &= mask

    if decorrelate:
        survivors = np.flatnonzero(keep)
        mask = rank_revealing_subset(full[:, survivors], decorr_threshold, decorr_method)
        removed["decorrelate"] = int((~mask).sum())
        keep[survivors[~mask]] = False

    noise_stripped = removed.get("noise_removal", 0)
    # Only the absolute cutoff is tied to the normalization scale, so only it earns the scale-relative advice.
    near_total_noise = noise_removal_cutoff_mode == "absolute" and noise_stripped > 0.95 * judged.size
    if not keep.any() and adata.n_vars:
        warnings.warn(
            f"feature_select flagged none of the {adata.n_vars} features as selected; "
            f"uns['mantispy']['feature_select'] says what each operation removed. Every cutoff here is an "
            f"absolute threshold on the scale pp.normalize left the values on, so check it against that "
            f"scale; noise_removal's stdev_cutoff is the usual cause.",
            UserWarning,
            stacklevel=3,
        )
    elif near_total_noise:
        warnings.warn(
            f"noise_removal alone flagged {noise_stripped} of the {judged.size} features it judged for removal "
            "(more than 95%). Its stdev_cutoff is an absolute threshold on the scale pp.normalize left the values "
            "on, so it strips nearly everything under the control-based or MAD normalization the package "
            "recommends. Set noise_removal_cutoff_mode='ratio' or 'quantile' for a scale-relative cutoff.",
            UserWarning,
            stacklevel=3,
        )

    adata.var[key_added] = keep
    adata.uns.setdefault("mantispy", {})["feature_select"] = removed
    get_logger().info(
        "feature_select kept %d of %d features (%s)",
        int(keep.sum()),
        adata.n_vars,
        ", ".join(f"{name} -{count}" for name, count in removed.items()),
    )
    return None


def subset_features(adata: AnnData, key: str = "selected") -> AnnData:
    """Return a new object holding only the features flagged by ``var[key]``.

    Args:
        adata: Object to subset.
            Never modified.
        key: Boolean ``var`` column naming the features to keep, as :func:`feature_select` writes.

    Returns:
        A new object holding only the flagged features.

    Raises:
        KeyError: If ``var`` has no column ``key``.
    """
    if key not in adata.var:
        raise KeyError(f"var has no column {key!r}; run mt.pp.feature_select first")
    return adata[:, as_frame(adata.var)[key].to_numpy(dtype=bool)].copy()
