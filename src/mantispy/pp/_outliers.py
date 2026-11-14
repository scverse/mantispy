"""Cell-level outlier detection."""

from __future__ import annotations

import math

import numpy as np
from anndata import AnnData
from numba import get_num_threads

from mantispy._core._ecod import AGGREGATIONS, ecod_scores
from mantispy._core._lof import lof_negative_outlier_factor
from mantispy._core._reduce import get_matrix, group_codes, group_offsets
from mantispy._core._stats import robust_zscore
from mantispy._core.frames import as_frame
from mantispy._core.logging import get_logger
from mantispy._core.masks import feature_mask
from mantispy._core.mutation import inplace_or_copy

METHODS = ("ecod", "isolation_forest", "lof", "mad")

#: Default neighbour count, matching sklearn's :class:`~sklearn.neighbors.LocalOutlierFactor`.
_LOF_NEIGHBORS = 20
#: The KD-tree stays exact in any dimension but loses to sklearn's optimised brute-force search as dimension
#: grows, so the fast path is kept to the few-feature regime the intensity filter actually uses and anything
#: wider falls back to sklearn.
_LOF_FAST_MAX_FEATURES = 16


def _lof_scores(X: np.ndarray, n_fit: int | None, rng: np.random.Generator) -> np.ndarray:
    """LOF score per row, higher meaning more outlying."""
    # sklearn, where it still runs, gets the same threads as the numba kernels.
    n_jobs = get_num_threads()
    scores = np.full(X.shape[0], np.nan)
    # LOF needs complete input; a row missing every selected feature, filled with zero, would collapse onto
    # the origin and inflate the density of nearby cells, so those rows are left out of the fit and unscored.
    present = ~np.isnan(X).all(axis=1)
    # X[present] already copies, so clean it in place rather than allocating a second array.
    fitted = np.nan_to_num(X[present], nan=0.0, posinf=0.0, neginf=0.0, copy=False)
    n = fitted.shape[0]
    if n < 2:
        scores[present] = 0.0
        return scores
    if n_fit is not None and n > n_fit:
        # Fit the density on a seeded random sample and score every present row against it in novelty mode,
        # so cost tracks the sample rather than the whole group.
        from sklearn.neighbors import LocalOutlierFactor

        sample = np.sort(rng.choice(n, size=n_fit, replace=False))
        model = LocalOutlierFactor(n_neighbors=min(20, n_fit - 1), novelty=True, n_jobs=n_jobs).fit(fitted[sample])
        scores[present] = -model.score_samples(fitted)
        return scores
    nof = None
    if fitted.shape[1] <= _LOF_FAST_MAX_FEATURES:
        nof = lof_negative_outlier_factor(fitted, _LOF_NEIGHBORS)
    if nof is None:
        from sklearn.neighbors import LocalOutlierFactor

        # Cap neighbours at group size - 1 so sklearn never raises on a small group.
        model = LocalOutlierFactor(n_neighbors=min(20, n - 1), n_jobs=n_jobs).fit(fitted)
        nof = model.negative_outlier_factor_
    scores[present] = -nof
    return scores


def _scores(
    X: np.ndarray,
    method: str,
    seed: int,
    ecod_aggregation: str,
    n_fit: int | None,
    rng: np.random.Generator,
) -> np.ndarray:
    """Outlier score per row, higher meaning more outlying."""
    if method == "ecod":
        return ecod_scores(X, ecod_aggregation)
    if method == "lof":
        return _lof_scores(X, n_fit, rng)
    if method == "isolation_forest":
        from sklearn.ensemble import IsolationForest

        filled = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        model = IsolationForest(random_state=seed, n_estimators=200).fit(filled)
        return -model.score_samples(filled)

    return np.nanmax(np.abs(robust_zscore(X)), axis=1)


@inplace_or_copy()
def outliers(
    adata: AnnData,
    method: str = "ecod",
    contamination: float = 0.01,
    score_cutoff: float | None = None,
    key: str | None = "selected",
    by: str | None = None,
    seed: int = 0,
    ecod_aggregation: str = "pyod",
    n_fit: int | None = None,
    key_added: str = "qc_outlier",
    copy: bool = False,
) -> AnnData | None:
    """Flag outlying cells.

    Every method produces a score where higher means more outlying, and the same thresholding applies to all of them, so ``contamination`` is the flagged fraction whichever method is used.

    Args:
        adata: Object to flag.
        method: ``"ecod"`` :cite:p:`Li_2023` needs no tuning and is interpretable per feature, ``"isolation_forest"`` catches outliers defined by feature interactions, ``"lof"`` is sklearn's local outlier factor with its default 20 neighbours, which scores a cell against the density of its neighbourhood, and ``"mad"`` takes the largest robust z-score across features, which is easy to explain but sees each feature alone.
        contamination: Fraction of cells to flag, rounded up to a whole cell within each ``by`` group, so a non-empty group always flags its most outlying cell and the flagged fraction is higher than asked for in a group smaller than ``1 / contamination``.
            Ignored when ``score_cutoff`` is given.
        score_cutoff: Threshold the score absolutely instead of by quantile.
            With ``method="mad"`` the score is a robust z-score, so ``score_cutoff=5`` gives the usual rule.
        key: Input: the name of a boolean ``var`` column; only the features it flags are scored, usually ``"selected"``.
            Brieflow's intensity filter is ``method="lof"`` with a ``key`` that flags the mean intensity of each channel.
            Falls back to every feature when the column is absent.
            Not the output name, which is ``key_added``.
        by: Threshold within each group of this ``obs`` column, e.g. per plate, rather than globally.
        seed: Seed for ``isolation_forest`` and the ``n_fit`` sample.
        ecod_aggregation: How ``"ecod"`` combines features.
            ``"pyod"`` sums each feature's larger tail, or both tails where its skewness is zero or undefined, as pyod and scmorph do, so no score depends on which way round a feature is measured.
            ``"paper"`` is Algorithm 1 of :cite:t:`Li_2023`, the largest of the left-tail, right-tail and skew-directed sums.
        n_fit: ``"lof"`` fits its density on a random sample of at most this many rows per ``by`` group and scores every row against that sample, rather than fitting on all of them, so its time and memory grow with the sample rather than the whole screen and stay bounded however large the screen is.
            A smaller sample sets a coarser scale for "normal" density, measuring each cell against a sparser neighbourhood than a full fit would, so the sampled run flags its most outlying cells by a different standard and can disagree substantially over which cells those are, more so the smaller the sample and the denser the data; it trades that fidelity for a run whose cost no longer rises with the number of cells.
            ``None``, the default, fits on every row, the exact local outlier factor, which runs on every thread numba is allowed.
            A group with no more than ``n_fit`` rows is fit in full either way.
        key_added: Output: the prefix for the results, written to ``obs[key_added]`` and ``obs[key_added + "_score"]``.
        copy: Return a modified copy instead of mutating in place.

    Returns:
        ``None``, or the modified copy.
        Writes the boolean ``obs[key_added]`` and the score itself to ``obs[key_added + "_score"]``.
        The flagged cells are also added to ``obs["qc_fail"]``, the union of the ``qc_*`` flags that :func:`~mantispy.pp.filter_cells` drops, which is created if absent; a cell already failing keeps failing.

    Raises:
        ValueError: If ``method`` or ``ecod_aggregation`` is unknown, ``contamination`` is outside ``(0, 1)`` and no ``score_cutoff`` is given, or ``n_fit`` is below 2.
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    if not 0.0 < contamination < 1.0 and score_cutoff is None:
        raise ValueError(f"contamination must be between 0 and 1, got {contamination}")
    if ecod_aggregation not in AGGREGATIONS:
        raise ValueError(f"ecod_aggregation must be one of {AGGREGATIONS}, got {ecod_aggregation!r}")
    if n_fit is not None and n_fit < 2:
        raise ValueError(f"n_fit must be at least 2, got {n_fit}")

    selected = feature_mask(adata, key)
    X = get_matrix(adata)[:, selected]

    scores = np.empty(adata.n_obs, dtype=np.float64)
    flagged = np.zeros(adata.n_obs, dtype=bool)
    rng = np.random.default_rng(seed)
    codes, keys = group_codes(adata, by)
    order, offsets = group_offsets(codes, len(keys))
    for group in range(len(keys)):
        rows = order[offsets[group] : offsets[group + 1]]
        if not rows.size:
            continue
        # One group is the whole matrix, so it is scored in place rather than copied.
        block = _scores(X if len(keys) == 1 else X[rows], method, seed, ecod_aggregation, n_fit, rng)
        scores[rows] = block
        if score_cutoff is not None:
            flagged[rows] = block > score_cutoff
        else:
            # Flag by rank: `> quantile` flags too few cells on ties, and none when an infinite value makes the quantile infinite.
            # Unscored rows (NaN, e.g. a cell LOF could not fit) are ranked out so they are never flagged.
            scorable = np.flatnonzero(~np.isnan(block))
            k = math.ceil(contamination * scorable.size)
            flagged[rows[scorable[np.argsort(block[scorable])[::-1][:k]]]] = True

    if method == "lof":
        unscored = int(np.isnan(scores).sum())
        if unscored:
            get_logger().warning(
                "outliers(lof): %d cell(s) have no selected feature measured and stay unscored and unflagged",
                unscored,
                deep="impute them first with mt.pp.impute, or drop them",
            )

    adata.obs[key_added] = flagged
    adata.obs[f"{key_added}_score"] = scores
    # qc_fail is the union of every qc_* flag and what filter_cells reads, so an outlier joins it.
    previous = as_frame(adata.obs)["qc_fail"].to_numpy(dtype=bool) if "qc_fail" in adata.obs else False
    adata.obs["qc_fail"] = previous | flagged
    get_logger().info(
        "outliers(%s) flagged %d of %d cells using %d features",
        method,
        int(flagged.sum()),
        adata.n_obs,
        int(selected.sum()),
    )
    return None
