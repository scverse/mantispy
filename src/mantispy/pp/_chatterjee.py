"""Chatterjee's rank correlation for feature selection.

Chatterjee's xi measures whether one variable is a function of another, monotonic or not.
A feature that is high at both extremes of a treatment and low in the middle has a near-zero Pearson correlation but a large xi, so this keeps features that a correlation filter drops.

References: :cite:t:`Chatterjee_2020` and :cite:t:`Lin_2022`, which generalizes xi to ``m`` right nearest neighbors.
At ``m=1`` the two differ by a term of order ``1/n``, and larger ``m`` has a lower noise floor.
"""

from __future__ import annotations

import numpy as np
from anndata import AnnData
from scipy.stats import rankdata

from mantispy._core._corr import CHUNK_BYTES
from mantispy._core._reduce import get_matrix, group_codes
from mantispy._core.logging import get_logger
from mantispy._core.mutation import inplace_or_copy


def _complete_columns(values: np.ndarray) -> np.ndarray:
    """Which columns hold a finite value in every row, reduced one column block at a time.

    One ``np.isfinite`` over the whole input is a boolean the size of the matrix, four gigabytes at a million cells by four thousand features, and pairing it with ``x`` needs a second one.
    """
    n_obs, n_vars = values.shape
    out = np.empty(n_vars, dtype=bool)
    width = max(int(CHUNK_BYTES / max(n_obs, 1) / values.itemsize), 1)
    for start in range(0, n_vars, width):
        block = slice(start, start + width)
        out[block] = np.isfinite(values[:, block]).all(axis=0)
    return out


def _xi(x: np.ndarray, values: np.ndarray, m: int, seed: int) -> np.ndarray:
    """Chatterjee's xi between ``x`` and every column of ``values``, all of which must be finite.

    The statistic of :cite:t:`Lin_2022`, centred on its exact mean under independence and scaled by the ``sum(l * (n - l))`` of :cite:t:`Chatterjee_2020`, which is how Chatterjee handles ties in ``y``.
    Without ties in ``y`` it is Lin and Han's statistic exactly.
    """
    n = x.size
    shuffled = np.random.default_rng(seed).permutation(n)
    order = shuffled[np.argsort(x[shuffled], kind="stable")]

    # How many values of y are at or below each one, read in the order x puts the rows in; tied values share it.
    # As integers: rankdata keeps float32 input float32, whose sums of ranks are inexact above 2**24.
    ranks = rankdata(values[order], method="max", axis=0).astype(np.int64)
    total = np.zeros(values.shape[1])
    for step in range(1, m + 1):
        total += np.minimum(ranks[: n - step], ranks[step:]).sum(axis=0) + ranks[n - step :].sum(axis=0)
    # How many are strictly below; the scale is zero only for a constant column, which has no xi.
    below = rankdata(values, method="min", axis=0).astype(np.float64) - 1
    scale = (below * (n - below)).sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        centred = n * (n - 1) * (total - m * ranks.sum(axis=0)) / scale
    return (centred + n * m - m * (m + 1) / 2) / (n * m + m * (m + 1) / 4)


def chatterjee_xi(x: np.ndarray, y: np.ndarray, m: int = 1, seed: int = 0, min_finite: int = 40) -> np.ndarray:
    """Chatterjee's xi between ``x`` and every column of ``y``.

    Args:
        x: The variable the others are tested against, such as a group code, a dose or a covariate.
        y: One column, or a matrix of them. Every column is scored against ``x`` in one pass.
        m: Right nearest neighbors, as in :cite:t:`Lin_2022`. Larger ``m`` has the same limit under dependence and a lower noise floor under independence, so a fixed threshold is more reliable.
        seed: Seed for breaking ties in ``x``.
        min_finite: Fewest finite pairs a column may be scored on, raised to ``m + 2`` when it is below that. Under independence xi has standard deviation ``sqrt(2 / (5 * n))``, which at 40 pairs equals the 0.1 threshold :func:`feature_select_chatterjee` selects on, so a column measured fewer times than this cannot be told from noise.

    Returns:
        One xi per column of ``y``, NaN for a constant column or one with fewer than ``min_finite`` finite pairs.

    Raises:
        ValueError: If ``m`` is below 1.

    Notes:
        Ties in ``x`` are broken at random, as the coefficient requires, so ``seed`` matters only when ``x`` has ties.
        A group label is almost all ties, and breaking them by row order would score the row order as structure.
        Ties in ``y`` are handled as :cite:t:`Chatterjee_2020` handles them, without randomness: tied values share a rank.
        A zero-inflated Zernike, Granularity or RadialDistribution column is mostly ties, and breaking them at random scores one that is a step function of ``x`` at 0.51 instead of 0.99.
        A constant column carries no information and scores NaN.

        xi is defined on complete pairs, so each column is scored on the rows where both it and ``x`` are finite, and two columns missing different rows are scored on different subsets.
        Ranking a column that still holds NaN sorts the missing rows last, which turns a feature that is merely unmeasured in one group into a step function of the group and scores it as dependence.
        How much of a feature is missing is a question for :func:`~mantispy.pp.feature_select` and its ``drop_na_columns`` operation; xi reports on the part that was measured.
    """
    if m < 1:
        raise ValueError(f"m must be at least 1, got {m}")
    values = np.atleast_2d(y.T).T if y.ndim > 1 else y[:, None]
    finite_x = np.isfinite(x)
    floor = max(min_finite, m + 2)
    complete = _complete_columns(values) if finite_x.all() else np.zeros(values.shape[1], dtype=bool)

    if complete.all():
        return _xi(x, values, m, seed) if x.size >= floor else np.full(values.shape[1], np.nan)

    # The complete columns share one ordering of x, so they are still ranked in one pass.
    # The others are missing different rows and no longer share it, so each is ranked over its own.
    scores = np.full(values.shape[1], np.nan)
    if complete.any() and x.size >= floor:
        scores[complete] = _xi(x, values[:, complete], m, seed)
    for column in np.flatnonzero(~complete):
        rows = np.flatnonzero(finite_x & np.isfinite(values[:, column]))
        if rows.size >= floor:
            scores[column] = _xi(x[rows], values[rows, column][:, None], m, seed)[0]
    return scores


@inplace_or_copy()
def feature_select_chatterjee(
    adata: AnnData,
    groupby: str = "Metadata_Perturbation",
    threshold: float = 0.1,
    m: int = 1,
    seed: int = 0,
    min_finite: int = 40,
    key_added: str = "selected_chatterjee",
    copy: bool = False,
) -> AnnData | None:
    """Keep features whose values depend on the group, monotonically or otherwise.

    Args:
        adata: Object to select features on.
        groupby: ``obs`` column the features are tested against.
        threshold: Keep features scoring above this. xi is near zero under independence and approaches one when the feature is a deterministic function of the group, so the threshold is comparable across datasets in a way a correlation cutoff is not.
        m: Right nearest neighbors :cite:p:`Lin_2022`. ``m=1`` is the coefficient of :cite:t:`Chatterjee_2020` up to a term of order ``1/n``; larger values lower the noise floor without changing what the statistic converges to.
        seed: Seed for breaking ties between rows of the same group.
        min_finite: Fewest finite values a feature may be scored on. A feature measured fewer times than this scores NaN, which is above no threshold and so is never selected.
        key_added: Name of the boolean ``var`` column written.
        copy: Return a modified copy instead of mutating in place.

    Returns:
        ``None``, or the modified copy. Writes ``var[key_added]`` and the statistic itself to ``var["chatterjee_xi"]``, which is NaN for a feature that is constant or too sparsely measured to score.

    Raises:
        ValueError: If ``groupby`` has a single group, so that no feature can depend on it.

    Notes:
        Run it after :func:`~mantispy.pp.feature_select`, which drops redundant or unmeasurable features; this keeps the features that carry information about the perturbation.
        Subset with ``mt.pp.subset_features(adata, key="selected_chatterjee")``.

        A feature is scored on the rows where it was measured, so one that is missing in a whole group is scored against the groups that do have it rather than against the pattern of what is missing.

        xi reaches one only for a noiseless function of the group, so real values are much lower.
        Over pki's 899 selected features and 38 treatments the largest was 0.32, and the default threshold of 0.1 kept about half of them.
        Check the distribution in ``var["chatterjee_xi"]`` before relying on a fixed cutoff.

        With shuffled group labels on the same data, the largest xi is 0.039 at ``m=1`` and 0.027 at ``m=5``, while the largest real value barely changes (0.322 and 0.321).
        ``m=1`` is the default because the threshold was calibrated there.
        scmorph uses ``m=5``, and on input free of ties the two implementations agree to 1e-9.
        scmorph breaks ties in ``y`` at random, so on tied input the two differ.
    """
    codes, keys = group_codes(adata, groupby)
    if len(keys) < 2:
        raise ValueError(f"{groupby!r} has one group, so no feature can depend on it")

    scores = chatterjee_xi(codes.astype(float), get_matrix(adata), m=m, seed=seed, min_finite=min_finite)
    adata.var["chatterjee_xi"] = scores
    # NaN is above no threshold, so a feature that cannot be scored drops out here.
    selected = scores > threshold
    adata.var[key_added] = selected
    get_logger().info(
        "chatterjee kept %d of %d features, %d constant or too sparsely measured to score",
        int(selected.sum()),
        adata.n_vars,
        int(np.isnan(scores).sum()),
    )
    return None
