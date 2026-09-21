"""Feature selection, with pycytominer's semantics.

Every operation returns a boolean keep mask over ``var``.
:func:`feature_select` combines them into one boolean column without dropping anything, and :func:`subset_features` does the subsetting.

Operation names and behavior follow pycytominer.
``variance_threshold`` is an sklearn-style variance cut, and the frequency and uniqueness rules are in the separate ``frequency_threshold``.
``correlation_threshold`` judges each pair against a ranking computed once from the full matrix instead of sweeping greedily, and thresholds the signed correlation, so two features correlated at -1.0 are both kept.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence

import numpy as np
from anndata import AnnData

from mantispy._core._corr import correlated_pairs
from mantispy._core._reduce import get_matrix, group_codes, group_offsets
from mantispy._core._stats import nanvar
from mantispy._core.features import blocklist_hits
from mantispy._core.frames import as_frame
from mantispy._core.logging import get_logger
from mantispy._core.mutation import inplace_or_copy

#: Operations, in the order pycytominer applies them by default.
OPERATIONS = (
    "variance_threshold",
    "frequency_threshold",
    "correlation_threshold",
    "drop_na_columns",
    "blocklist",
    "drop_outliers",
    "noise_removal",
)

DEFAULT_OPERATIONS = (
    "variance_threshold",
    "correlation_threshold",
    "drop_na_columns",
    "blocklist",
)


def _op_variance_threshold(X: np.ndarray, min_variance: float = 1e-6) -> np.ndarray:
    """Drop features whose variance is at or below ``min_variance``.

    sklearn's ``VarianceThreshold`` keeps ``variance > threshold`` and uses the population variance, so ``ddof=0``.
    """
    # The same helper pp.filter_features uses, so the two statements of this rule cannot drift apart.
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


def _op_correlation_threshold(X: np.ndarray, threshold: float = 0.9, method: str = "pearson") -> np.ndarray:
    """Drop one feature from every pair correlated above ``threshold``.

    Each over-threshold pair is judged on its own against a ranking of total absolute correlation computed once from the full matrix; the member ranked as more correlated overall is dropped.
    There is no iterative sweep, so a feature already dropped by one pair does not spare its partner in another.
    """
    n_vars = X.shape[1]
    # Signed correlation, as in pycytominer, which keeps a pair correlated at -1.0.
    # correlated_pairs never builds the full matrix, so this scales to tens of thousands of features.
    pairs, total = correlated_pairs(X, threshold, method=method)
    # Rank features by how correlated they are with everything else, ascending.
    order = np.argsort(total, kind="stable")
    rank = np.empty(n_vars, dtype=np.int64)
    rank[order] = np.arange(n_vars)

    keep = np.ones(n_vars, dtype=bool)
    if pairs.size:
        first, second = pairs[:, 0], pairs[:, 1]
        keep[np.where(rank[first] > rank[second], first, second)] = False
    return keep


def _op_drop_na_columns(X: np.ndarray, cutoff: float = 0.05) -> np.ndarray:
    """Drop features missing in more than ``cutoff`` of rows."""
    return np.isnan(X).mean(axis=0) <= cutoff


def _op_drop_outliers(X: np.ndarray, outlier_cutoff: float = 500.0) -> np.ndarray:
    """Drop features whose largest absolute value exceeds ``outlier_cutoff``.

    Ratios with a near-zero denominator blow up like this.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # "All-NaN slice encountered"
        largest = np.nanmax(np.abs(X), axis=0)
    return ~(np.nan_to_num(largest, nan=0.0) > outlier_cutoff)


def _op_blocklist(adata: AnnData, blocklist: str | Sequence[str] = "default") -> np.ndarray:
    """Drop features named in the blocklist.

    Matched against the current names and, like :func:`~mantispy.pp.filter_features`, against ``var["original_name"]``, so the blocklist still applies after :func:`~mantispy.pp.standardize_feature_names` has renamed the features.
    """
    names = [adata.var_names.to_numpy()]
    if "original_name" in adata.var:
        names.append(adata.var["original_name"].astype(str).to_numpy())
    return ~blocklist_hits(names, blocklist)


def _op_noise_removal(X: np.ndarray, codes: np.ndarray, stdev_cutoff: float = 0.8) -> np.ndarray:
    """Drop features that vary too much within a perturbation group.

    The statistic is the mean, over groups, of each group's population standard deviation (``ddof=0``).

    ``stdev_cutoff`` is an absolute threshold on whatever scale ``normalize`` left the values on, so it is
    only meaningful next to the normalization that produced them. pycytominer's 0.8 is calibrated for
    whole-plate standardization, where a feature's spread is 1 by construction. Normalizing against the
    controls instead, as :func:`~mantispy.pp.normalize` does by default and as the JUMP recipe does, measures
    every feature against the spread of the DMSO wells rather than of the plate, and treated wells vary more
    than controls do. Dividing by a MAD rather than a standard deviation rescales it again. Either choice
    puts the whole distribution above 0.8 and the operation then drops every feature.
    """
    n_groups = int(codes.max()) + 1
    order, offsets = group_offsets(codes, n_groups)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # "Degrees of freedom <= 0 for slice", once per all-NaN group
        deviations = np.stack(
            [np.nanstd(X[order[offsets[group] : offsets[group + 1]]], axis=0, ddof=0) for group in range(n_groups)]
        )
    return ~(np.nan_to_num(deviations.mean(axis=0), nan=0.0) > stdev_cutoff)


@inplace_or_copy()
def feature_select(
    adata: AnnData,
    operations: Sequence[str] = DEFAULT_OPERATIONS,
    min_variance: float = 1e-6,
    freq_cut: float = 0.05,
    unique_cut: float = 0.01,
    corr_threshold: float = 0.9,
    corr_method: str = "pearson",
    na_cutoff: float = 0.05,
    outlier_cutoff: float = 500.0,
    blocklist: str | Sequence[str] = "default",
    noise_removal_perturb_groups: str = "Metadata_Perturbation",
    noise_removal_stdev_cutoff: float = 0.8,
    key_added: str = "selected",
    copy: bool = False,
) -> AnnData | None:
    """Flag the features worth keeping.

    Args:
        adata: Object to select features on. Usually well-level profiles.
        operations: Which operations to run, from ``OPERATIONS``. The default omits ``frequency_threshold``, ``drop_outliers`` and ``noise_removal``, matching pycytominer's own default.
        min_variance: ``variance_threshold``: keep features with variance above this.
        freq_cut: ``frequency_threshold``: drop a feature when the count of its second most common value divided by the count of its most common is below this. Either this rule or ``unique_cut`` drops a feature.
        unique_cut: ``frequency_threshold``: drop a feature when its share of distinct values is below this.
        corr_threshold: ``correlation_threshold``: drop one member of every pair correlated above this.
        corr_method: ``correlation_threshold``: ``"pearson"`` or ``"spearman"``.
        na_cutoff: ``drop_na_columns``: drop features missing in more than this fraction of rows.
        outlier_cutoff: ``drop_outliers``: drop features whose absolute value exceeds this.
        blocklist: ``blocklist``: ``"default"`` for the bundled list, or explicit names. Matched against the current names and against ``var["original_name"]``, so it works either side of :func:`~mantispy.pp.standardize_feature_names`.
        noise_removal_perturb_groups: ``noise_removal``: ``obs`` column grouping replicates.
        noise_removal_stdev_cutoff: ``noise_removal``: drop features whose within-group standard deviation, averaged over groups, is above this. An absolute threshold on the scale ``normalize`` left the values on, so it is only meaningful next to the normalization that produced them; pycytominer's default assumes whole-plate standardization.
        key_added: Name of the boolean ``var`` column to write.
        copy: Return a modified copy instead of mutating in place.

    Returns:
        ``None``, or the modified copy. Writes ``var[key_added]`` and a per-operation count of removals to ``uns["mantispy"]["feature_select"]``, each count being what that operation removes on its own. Nothing is dropped; use :func:`subset_features` for that.

    Raises:
        ValueError: If ``operations`` names an operation that is not in ``OPERATIONS``.
        KeyError: If ``noise_removal`` is requested but ``noise_removal_perturb_groups`` is not an ``obs`` column.

    Notes:
        Every operation judges the full feature set, so each count in ``uns["mantispy"]["feature_select"]`` says what that operation alone would remove and is the same whatever order ``operations`` runs in.
        The counts therefore overlap: a feature that is both constant and mostly missing is counted by ``variance_threshold`` and by ``drop_na_columns``, and the counts sum to more than the number of features actually removed, which is ``n_vars`` minus ``var[key_added].sum()``.

        ``correlation_threshold`` is the most expensive operation.
        pycytominer uses ``pandas.DataFrame.corr``, one Cython pass per column pair.
        Here the pairs come from chunked matrix products over blocks of columns, so no ``n_vars ** 2`` array is held in memory.
    """
    unknown = set(operations) - set(OPERATIONS)
    if unknown:
        raise ValueError(f"unknown operation(s) {sorted(unknown)}; choose from {OPERATIONS}")

    X = get_matrix(adata)
    keep = np.ones(adata.n_vars, dtype=bool)
    removed: dict[str, int] = {}

    for operation in operations:
        if operation == "variance_threshold":
            mask = _op_variance_threshold(X, min_variance)
        elif operation == "frequency_threshold":
            mask = _op_frequency_threshold(X, freq_cut, unique_cut)
        elif operation == "correlation_threshold":
            mask = _op_correlation_threshold(X, corr_threshold, corr_method)
        elif operation == "drop_na_columns":
            mask = _op_drop_na_columns(X, na_cutoff)
        elif operation == "blocklist":
            mask = _op_blocklist(adata, blocklist)
        elif operation == "drop_outliers":
            mask = _op_drop_outliers(X, outlier_cutoff)
        else:
            # Unknown names were rejected above, so what is left is `noise_removal`.
            if noise_removal_perturb_groups not in adata.obs:
                raise KeyError(f"obs has no column {noise_removal_perturb_groups!r} to group replicates by")
            codes, _ = group_codes(adata, noise_removal_perturb_groups)
            mask = _op_noise_removal(X, codes, noise_removal_stdev_cutoff)
        # Counted against every feature rather than against the features its predecessors left, so the count does not depend on where the operation sits in `operations`.
        removed[operation] = int((~mask).sum())
        keep &= mask

    if not keep.any() and adata.n_vars:
        # Selecting nothing is almost always a cutoff set against the wrong scale rather than a screen with
        # no usable features, and on its own it surfaces further down as an empty matrix in whatever runs next.
        warnings.warn(
            f"feature_select flagged none of the {adata.n_vars} features as selected; "
            f"uns['mantispy']['feature_select'] says what each operation removed. Every cutoff here is an "
            f"absolute threshold on the scale pp.normalize left the values on, so check it against that "
            f"scale; noise_removal's stdev_cutoff is the usual cause.",
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
        adata: Object to subset. Never modified.
        key: Boolean ``var`` column naming the features to keep, as :func:`feature_select` writes.

    Returns:
        A new object holding only the flagged features.

    Raises:
        KeyError: If ``var`` has no column ``key``.
    """
    if key not in adata.var:
        raise KeyError(f"var has no column {key!r}; run mt.pp.feature_select first")
    return adata[:, as_frame(adata.var)[key].to_numpy(dtype=bool)].copy()
