"""Rank-based inverse normal transformation.

After per-plate normalization, each feature is replaced by the normal quantile of its
rank. This is the ``_int`` step of the JUMP consortium's recipe and the third step of the
baseline in Arevalo et al. (2024). Morphology features are heavy-tailed and differ in
shape, so a few extreme wells can dominate distances; ranking removes both the shape
differences and the outliers, at the cost of the original units.

Reference: the ``rank_int_array`` implementation in ``broadinstitute/jump-profiling-recipe``,
which this reproduces on data with no missing values.
"""

from __future__ import annotations

import numpy as np
from anndata import AnnData
from scipy.special import ndtri
from scipy.stats import rankdata

from mantispy._core._reduce import get_matrix, group_codes
from mantispy._core._utils import get_logger, inplace_or_copy

#: Blom's constant, the default the field uses for the quantile estimate.
BLOM = 3.0 / 8.0


def rank_inverse_normal(X: np.ndarray, c: float = BLOM, stochastic: bool = True, seed: int = 0) -> np.ndarray:
    """Map each column onto a standard normal by rank.

    Args:
        X: Values to transform, one feature per column.
        c: Blom's constant in ``(rank - c) / (n - 2c + 1)``.
        stochastic: Break ties at random, as the reference implementation does. With ``False``, tied
            values share the mid-rank and get the same output, which suits real ties (such as a
            feature that is zero in half the wells) but not ties from rounding.
        seed: Seed for the tie-breaking.

    Returns:
        The transformed values, ``NaN`` where the input was missing.

    Notes:
        The finite values of each column are ranked among themselves, and only missing
        entries come back missing. The reference implementation passes the column to
        ``scipy.stats.rankdata``, which returns all NaN for a column with any missing value.
    """
    values = np.atleast_2d(np.asarray(X, dtype=np.float64).T).T
    out = np.full(values.shape, np.nan)

    generator = np.random.default_rng(seed)
    order = generator.permutation(values.shape[0])

    for column in range(values.shape[1]):
        finite = np.flatnonzero(np.isfinite(values[:, column]))
        if finite.size < 2:
            continue
        present = values[finite, column]
        if stochastic:
            shuffle = order[np.isin(order, finite)]
            ranks = np.empty(finite.size)
            ranks[np.searchsorted(finite, shuffle)] = rankdata(values[shuffle, column], method="ordinal")
        else:
            ranks = rankdata(present, method="average")
        out[finite, column] = ndtri((ranks - c) / (finite.size - 2 * c + 1))

    return out.reshape(np.shape(X))


@inplace_or_copy()
def rank_int(
    adata: AnnData,
    by: str | None = None,
    c: float = BLOM,
    stochastic: bool = True,
    seed: int = 0,
    key_added: str | None = None,
    copy: bool = False,
) -> AnnData | None:
    """Replace every feature by the normal quantile of its rank.

    Args:
        adata: Object to transform. Run it after :func:`~mantispy.pp.normalize`, as the JUMP
            recipe and the batch-correction benchmark do.
        by: Rank within each group of this column. ``None`` ranks globally, as the reference
            implementation does, which keeps every feature comparable across the screen.
            Ranking per plate also removes plate-level differences in distribution shape, but
            can hide a plate that failed.
        c: Blom's constant.
        stochastic: Tie handling; see ``rank_inverse_normal``.
        seed: Tie handling; see ``rank_inverse_normal``.
        key_added: Write to ``layers[key_added]`` instead of overwriting ``X``.
        copy: Return a modified copy instead of transforming in place.

    Returns:
        ``None``, or the modified copy.

    Notes:
        Every feature comes out standard normal, so no feature dominates a distance through
        its units. Effect sizes are lost as well: a feature that doubled and one that moved by
        one percent look the same if they reorder the same wells. Keep the untransformed values
        with ``key_added`` for effect sizes and dose-response curves.
    """
    X = get_matrix(adata)
    out = np.empty_like(X, dtype=np.float32)

    codes, keys = group_codes(adata, by)
    for index in range(len(keys)):
        rows = np.flatnonzero(codes == index)
        if rows.size:
            out[rows] = rank_inverse_normal(X[rows], c=c, stochastic=stochastic, seed=seed).astype(np.float32)

    if key_added is None:
        adata.X = out
    else:
        adata.layers[key_added] = out
    get_logger().info("rank_int transformed %d features within %s", adata.n_vars, by or "the whole object")
    return None
