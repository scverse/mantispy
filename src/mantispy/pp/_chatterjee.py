"""Chatterjee's rank correlation for feature selection.

Chatterjee's xi measures whether one variable is a function of another, monotonic or not.
A feature that is high at both extremes of a treatment and low in the middle has a near-zero Pearson correlation but a large xi, so this keeps features that a correlation filter drops.

References: Chatterjee (2021), "A new coefficient of correlation", JASA 116:2009, and Lin & Han (2023), "On boosting the power of Chatterjee's rank correlation", Biometrika 110:283, which generalizes xi to ``m`` right nearest neighbors.
The two agree at ``m=1``, and larger ``m`` has a lower noise floor.
"""

from __future__ import annotations

import numpy as np
from anndata import AnnData

from mantispy._core._reduce import get_matrix, group_codes
from mantispy._core.logging import get_logger
from mantispy._core.mutation import inplace_or_copy


def _xi(x: np.ndarray, values: np.ndarray, m: int, seed: int) -> np.ndarray:
    """Chatterjee's xi between ``x`` and every column of ``values``, all of which must be finite."""
    n = x.size
    generator = np.random.default_rng(seed)
    shuffled = generator.permutation(n)
    order = shuffled[np.argsort(x[shuffled], kind="stable")]

    # Ranks of y, read in the order x puts the rows in. Ties in y are broken at random too:
    # a stable sort breaks them by position in x-order, which reads the x ordering back out of
    # a tied column and scores a constant feature as a perfect function of x.
    jumble = generator.permutation(n)
    jumbled = np.argsort(np.argsort(values[order][jumble], axis=0, kind="stable"), axis=0) + 1
    ranks = np.empty_like(jumbled)
    ranks[jumble] = jumbled

    total = np.zeros(values.shape[1])
    for step in range(1, m + 1):
        total += np.minimum(ranks[: n - step], ranks[step:]).sum(axis=0) + ranks[n - step :].sum(axis=0)
    return -2.0 + 6.0 * total / ((n + 1) * (n * m + m * (m + 1) / 4))


def chatterjee_xi(x: np.ndarray, y: np.ndarray, m: int = 1, seed: int = 0, min_finite: int = 40) -> np.ndarray:
    """Chatterjee's xi between ``x`` and every column of ``y``.

    Args:
        x: The variable the others are tested against, such as a group code, a dose or a covariate.
        y: One column, or a matrix of them. Every column is scored against ``x`` in one pass.
        m: Right nearest neighbors, as in Lin & Han (2023). ``m=1`` is Chatterjee's original coefficient. Larger ``m`` has the same limit under dependence and a lower noise floor under independence, so a fixed threshold is more reliable.
        seed: Seed for the tie-breaking.
        min_finite: Fewest finite pairs a column may be scored on, raised to ``m + 2`` when it is below that. Under independence xi has standard deviation ``sqrt(2 / (5 * n))``, which at 40 pairs equals the 0.1 threshold :func:`feature_select_chatterjee` selects on, so a column measured fewer times than this cannot be told from noise.

    Returns:
        One xi per column of ``y``, NaN for a column with fewer than ``min_finite`` finite pairs.

    Raises:
        ValueError: If ``m`` is below 1.

    Notes:
        Ties in both ``x`` and ``y`` are broken at random, as the coefficient requires.
        A group label is almost all ties, and breaking them by row order would score the row order as structure.
        Ties in ``y`` matter just as much: breaking them in ``x``-order scores an all-zero column at 0.996 instead of 0.007, which is what a zero-inflated Zernike, Granularity or RadialDistribution column looks like.

        xi is defined on complete pairs, so each column is scored on the rows where both it and ``x`` are finite, and two columns missing different rows are scored on different subsets.
        Ranking a column that still holds NaN sorts the missing rows last, which turns a feature that is merely unmeasured in one group into a step function of the group and scores it as dependence.
        How much of a feature is missing is a question for :func:`~mantispy.pp.feature_select` and its ``drop_na_columns`` operation; xi reports on the part that was measured.
    """
    if m < 1:
        raise ValueError(f"m must be at least 1, got {m}")
    values = np.atleast_2d(y.T).T if y.ndim > 1 else y[:, None]
    usable = np.isfinite(x)[:, None] & np.isfinite(values)
    floor = max(min_finite, m + 2)

    if usable.all():
        return _xi(x, values, m, seed) if x.size >= floor else np.full(values.shape[1], np.nan)

    # Columns missing different rows no longer share one ordering of x, so each is ranked over its own rows.
    scores = np.full(values.shape[1], np.nan)
    for column in range(values.shape[1]):
        rows = np.flatnonzero(usable[:, column])
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
        m: Right nearest neighbors (Lin & Han 2023). ``m=1`` is Chatterjee's original coefficient; larger values lower the noise floor without changing what the statistic converges to.
        seed: Seed for the random tie-breaking.
        min_finite: Fewest finite values a feature may be scored on. A feature measured fewer times than this scores NaN, which is above no threshold and so is never selected.
        key_added: Name of the boolean ``var`` column written.
        copy: Return a modified copy instead of mutating in place.

    Returns:
        ``None``, or the modified copy. Writes ``var[key_added]`` and the statistic itself to ``var["chatterjee_xi"]``, which is NaN for a feature too sparsely measured to score.

    Raises:
        ValueError: If ``groupby`` has a single group, so that no feature can depend on it.

    Notes:
        Run it after :func:`~mantispy.pp.feature_select`, which drops redundant or unmeasurable features; this keeps the features that carry information about the perturbation.
        Subset with ``mt.pp.subset_features(adata, key="selected_chatterjee")``.

        A feature is scored on the rows where it was measured, so one that is missing in a whole group is scored against the groups that do have it rather than against the pattern of what is missing.

        xi reaches one only for a noiseless function of the group, so real values are much lower.
        Over pki's 852 selected features and 38 treatments the largest was 0.32, and the default threshold of 0.1 kept about half of them.
        Check the distribution in ``var["chatterjee_xi"]`` before relying on a fixed cutoff.

        With shuffled group labels on the same data, the largest xi is 0.035 at ``m=1`` and 0.018 at ``m=5``, while the largest real value barely changes (0.322 and 0.321).
        ``m=1`` is the default because the threshold was calibrated there.
        scmorph uses ``m=5``, and on input free of ties the two implementations agree to 1e-9.
        Tied input cannot agree that closely, because each breaks its ties from its own draws, so what is pinned there is that neither reads structure out of the ties (``tests/test_equivalence_scmorph.py``).
    """
    codes, keys = group_codes(adata, groupby)
    if len(keys) < 2:
        raise ValueError(f"{groupby!r} has one group, so no feature can depend on it")

    scores = chatterjee_xi(codes.astype(float), get_matrix(adata), m=m, seed=seed, min_finite=min_finite)
    adata.var["chatterjee_xi"] = scores
    # NaN is above no threshold, so a feature too sparsely measured to score drops out here.
    selected = scores > threshold
    adata.var[key_added] = selected
    get_logger().info(
        "chatterjee kept %d of %d features, %d too sparsely measured to score",
        int(selected.sum()),
        adata.n_vars,
        int(np.isnan(scores).sum()),
    )
    return None
