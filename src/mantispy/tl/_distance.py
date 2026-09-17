"""Energy distance between perturbation populations."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from anndata import AnnData

from mantispy._core._distance import energy_distance, pairwise_sqeuclidean
from mantispy._core._reduce import group_codes, representation
from mantispy._core._stats import benjamini_hochberg, permutation_pvalue, split_reference
from mantispy._core.logging import get_logger
from mantispy._core.masks import reference_mask
from mantispy._core.mutation import inplace_or_copy


def _energy_from_membership(membership: np.ndarray, distances: np.ndarray, size: int) -> np.ndarray:
    """Energy distance between each 0/1 row of ``membership`` and its complement.

    Computes ``2E|a-b| - E|a-a'| - E|b-b'|`` over a pooled distance matrix for a batch of
    labelings at once. As a membership matrix the null is three matrix products, instead of
    ``n_permutations * n**2`` gathers to index out each permutation's blocks.
    """
    n = distances.shape[0]
    other = 1.0 - membership
    within = np.einsum("pi,pi->p", membership @ distances, membership) / (size * (size - 1))
    rest = np.einsum("pi,pi->p", other @ distances, other) / ((n - size) * (n - size - 1))
    cross = np.einsum("pi,pi->p", membership @ distances, other) / (size * (n - size))
    return 2.0 * cross - within - rest


@inplace_or_copy()
def edistance(
    adata: AnnData,
    groupby: str = "Metadata_Perturbation",
    reference: str | None = "negcon",
    use_rep: str | None = None,
    n_permutations: int = 1000,
    threshold: float = 0.05,
    max_reference: int = 2000,
    seed: int = 0,
    key_added: str = "edistance",
    copy: bool = False,
) -> AnnData | None:
    """Energy distance between each group and the controls, or between every pair.

    Energy distance, ``2E|a-b| - E|a-a'| - E|b-b'|``, compares whole distributions and is zero
    only when they are equal. It makes no distributional assumption and responds to changes in
    spread or shape as well as shifts, which suits single-cell resolution, where a perturbation
    is a population.

    Args:
        adata: Object to score. Most informative at cell resolution.
        groupby: Column defining the populations.
        reference: Rows to compare against. Each group is compared with them and gets a
            permutation p-value in ``uns["mantispy"][key_added]``. With ``None``, the
            group-by-group distance matrix is written to ``uns["mantispy"][key_added + "_pairwise"]``
            as a square frame labeled by group, without p-values. Its cost is quadratic in the
            number of groups, which is expensive for a whole screen at cell level.
            ``max_reference`` and ``seed`` apply on this path too, capping the rows taken from each group.
        use_rep: Score ``obsm[use_rep]`` instead of ``X``.
        n_permutations: Number of label permutations in the null.
        threshold: q-value below which a group is marked ``is_hit``.
        max_reference: Maximum number of rows sampled from the reference and, separately, from each
            group. The pooled distance matrix is quadratic in their sum; 2000 of each takes 128 MB.
        seed: Seed for the subsampling and the permutations.
        key_added: Name for the outputs.
        copy: Return a modified copy instead of mutating in place.

    Returns:
        ``None``, or the modified copy. With a reference, writes ``uns["mantispy"][key_added]``
        with ``group``, ``n_obs``, ``distance``, ``pvalue``, ``qvalue`` and ``is_hit``.

    Notes:
        The null permutes the group and reference labels over the pooled rows and recomputes the
        statistic, the standard permutation test for a two-sample quantity.

        A group that is the reference itself is scored by splitting its rows in half. A group
        with fewer than two rows on either side, or a reference group with fewer than four
        rows, gets a NaN p-value and a warning.

        Check the false positive rate on your own screen with
        :func:`~mantispy.metrics.diagnose_testing`, which relabels control wells as
        pseudo-treatments of the same size and reports the fraction that are called.
    """
    values = representation(adata, use_rep)
    codes, keys = group_codes(adata, groupby)
    generator = np.random.default_rng(seed)

    if reference is None:
        blocks = []
        for index in range(len(keys)):
            rows = np.flatnonzero(codes == index)
            # Each pair builds a distance matrix quadratic in the two groups, so the cap applies
            # here as well; without it this path had no memory bound.
            if rows.size > max_reference:
                get_logger().info(
                    "edistance sampled %d of %d rows of %r for the pairwise matrix",
                    max_reference,
                    rows.size,
                    keys[index],
                )
                rows = np.sort(generator.choice(rows, size=max_reference, replace=False))
            blocks.append(values[rows])
        matrix = np.zeros((len(keys), len(keys)))
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                matrix[i, j] = matrix[j, i] = energy_distance(blocks[i], blocks[j])
        labels = [str(key) for key in keys]
        adata.uns.setdefault("mantispy", {})[f"{key_added}_pairwise"] = pd.DataFrame(
            matrix, index=labels, columns=labels
        )
        return None

    is_control = reference_mask(adata, reference)
    if int(is_control.sum()) < 2:
        raise ValueError(
            f"e-distance needs at least two reference rows, reference={reference!r} selects {int(is_control.sum())}"
        )

    control_rows = np.flatnonzero(is_control)
    if control_rows.size > max_reference:
        get_logger().info("edistance sampled %d of %d reference rows for the null", max_reference, control_rows.size)
        control_rows = np.sort(generator.choice(control_rows, size=max_reference, replace=False))

    observed = np.full(len(keys), np.nan)
    sizes = np.empty(len(keys), dtype=int)
    null = np.full((len(keys), n_permutations), np.nan)
    unscorable_reference = []
    unscorable_size = []
    for index in range(len(keys)):
        rows = np.flatnonzero(codes == index)
        # n_obs is the number of rows the statistic uses, so it is updated after sampling and
        # after splitting the reference.
        sizes[index] = rows.size
        if rows.size > max_reference:
            get_logger().info(
                "edistance sampled %d of %d rows of %r for the null", max_reference, rows.size, keys[index]
            )
            rows = np.sort(generator.choice(rows, size=max_reference, replace=False))
            sizes[index] = rows.size
        against = np.setdiff1d(control_rows, rows)
        if against.size < 2:
            # The reference group has no other reference rows to compare against, so its own
            # rows are split in half. That needs at least four rows.
            if rows.size < 4:
                unscorable_reference.append(str(keys[index]))
                continue
            rows, against = split_reference(rows, generator)
            sizes[index] = rows.size

        pooled = np.vstack([values[rows], values[against]])
        size, total = rows.size, pooled.shape[0]
        if size < 2 or total - size < 2:
            # Energy distance needs at least two rows on each side. One-row groups are common
            # (single-well screens, tl.consensus output), so they are collected for a warning.
            unscorable_size.append(str(keys[index]))
            continue

        # Permuting labels over the pooled rows gives the null the same arithmetic as the
        # statistic and works when a group is as large as the reference.
        distances = np.sqrt(pairwise_sqeuclidean(pooled, pooled))
        membership = np.zeros((n_permutations + 1, total))
        membership[0, :size] = 1.0
        for draw in range(1, n_permutations + 1):
            membership[draw, generator.choice(total, size=size, replace=False)] = 1.0
        statistics = _energy_from_membership(membership, distances, size)
        observed[index] = statistics[0]
        null[index] = statistics[1:]

    if unscorable_reference:
        warnings.warn(
            f"{len(unscorable_reference)} group(s) are the reference itself with fewer than four rows, too few "
            f"to split in half, so their p-values are NaN: {', '.join(unscorable_reference)}.",
            UserWarning,
            stacklevel=3,
        )
    if unscorable_size:
        warnings.warn(
            f"{len(unscorable_size)} group(s) have fewer than two rows on one side of the comparison, which "
            f"energy distance needs, so their p-values are NaN: {', '.join(unscorable_size)}.",
            UserWarning,
            stacklevel=3,
        )

    pvalues = permutation_pvalue(observed, null)
    qvalues = benjamini_hochberg(pvalues)
    adata.uns.setdefault("mantispy", {})[key_added] = pd.DataFrame(
        {
            "group": [str(key) for key in keys],
            "n_obs": sizes,
            "distance": observed,
            "pvalue": pvalues,
            "qvalue": qvalues,
            "is_hit": qvalues < threshold,
        }
    )
    return None
