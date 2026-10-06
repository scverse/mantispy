from __future__ import annotations

import warnings
from collections.abc import Sequence

import numpy as np
import pandas as pd
from anndata import AnnData

from mantispy._core._distance import energy_distance, pairwise_sqeuclidean
from mantispy._core._reduce import group_codes, group_offsets, representation
from mantispy._core._stats import (
    _default_well_block,
    _is_object_resolution,
    _split_wells,
    benjamini_hochberg,
    permutation_pvalue,
    split_reference,
)
from mantispy._core.logging import get_logger
from mantispy._core.masks import reference_mask
from mantispy._core.mutation import inplace_or_copy


def _energy_from_membership(membership: np.ndarray, distances: np.ndarray, size: int) -> np.ndarray:
    """Energy distance between each 0/1 row of ``membership`` and its complement.

    Computes ``2E|a-b| - E|a-a'| - E|b-b'|`` over a pooled distance matrix for a batch of labelings at once.
    """
    n = distances.shape[0]
    other = 1.0 - membership
    within = np.einsum("pi,pi->p", membership @ distances, membership) / (size * (size - 1))
    rest = np.einsum("pi,pi->p", other @ distances, other) / ((n - size) * (n - size - 1))
    cross = np.einsum("pi,pi->p", membership @ distances, other) / (size * (n - size))
    return 2.0 * cross - within - rest


def _edistance_block_null(
    values: np.ndarray,
    block_codes: np.ndarray,
    group_rows: np.ndarray,
    against_rows: np.ndarray,
    n_permutations: int,
    generator: np.random.Generator,
) -> np.ndarray:
    """Well-block permutation null for the energy distance.

    Each permutation draws as many whole wells as the group spans from the pool of the group and the reference wells, and takes the energy distance between the drawn rows and the rest, so the resampling unit is the well rather than the cell.
    """
    pool = np.concatenate([group_rows, against_rows])
    pool_blocks = block_codes[pool]
    uniq = np.unique(pool_blocks)
    if uniq.size < 2:
        return np.full(n_permutations, np.nan)
    # Cap the draw so the complement always keeps at least one well; a group spanning the whole pool (a reference
    # group compared with a fresh subsample of its own wells) would otherwise leave an empty side and an all-NaN null.
    n_draw = min(int(np.unique(block_codes[group_rows]).size), uniq.size - 1)
    pool_values = values[pool]
    null = np.empty(n_permutations)
    for permutation in range(n_permutations):
        drawn = np.isin(pool_blocks, uniq[generator.choice(uniq.size, size=n_draw, replace=False)])
        null[permutation] = energy_distance(pool_values[drawn], pool_values[~drawn])
    return null


def _split_reference_wells(
    rows: np.ndarray, block_codes: np.ndarray | None, generator: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Halve a reference group's rows, by whole wells when a block is in use, else by row.

    Splitting by well keeps every well complete so the block null draws a strict subset of the pool's wells rather than nearly all of them, the same halving :func:`~mantispy.tl.hit_calling` uses for its reference group.
    """
    if block_codes is None:
        return split_reference(rows, generator)
    in_left = _split_wells(block_codes[rows], generator)
    return np.sort(rows[in_left]), np.sort(rows[~in_left])


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
    *,
    block: str | Sequence[str] | None = None,
) -> AnnData | None:
    """Energy distance between each group and the controls, or between every pair.

    Energy distance, ``2E|a-b| - E|a-a'| - E|b-b'|``, compares whole distributions and is zero only when they are equal.
    It makes no distributional assumption and responds to changes in spread or shape as well as shifts, which suits single-cell resolution, where a perturbation is a population.

    Args:
        adata: Object to score.
            Most informative at cell resolution.
        groupby: Column defining the populations.
        reference: Rows to compare against.
            Each group is compared with them and gets a permutation p-value in ``uns["mantispy"][key_added]``.
            With ``None``, the group-by-group distance matrix is written to ``uns["mantispy"][key_added + "_pairwise"]`` as a square frame labeled by group, without p-values.
            Its cost is quadratic in the number of groups, which is expensive for a whole screen at cell level.
            ``max_reference`` and ``seed`` apply on this path too, capping the rows taken from each group.
        use_rep: Score ``obsm[use_rep]`` instead of ``X``.
        n_permutations: Number of label permutations in the null.
        threshold: q-value below which a group is marked ``is_hit``.
        max_reference: Maximum number of rows sampled from the reference and, separately, from each group.
            The pooled distance matrix is quadratic in their sum; 2000 of each takes 128 MB.
        seed: Seed for the subsampling and the permutations.
        key_added: Name for the outputs.
        copy: Return a modified copy instead of mutating in place.
        block: ``obs`` column, or sequence of columns, whose groups are the design's exchangeable unit, normally the well.
            The permutation null then draws whole wells rather than cells, since cells within a well are not independent replicates (see Notes).
            Left ``None``, it defaults to the physical well on an object stamped cell resolution with replicated wells; it warns and permutes cells when no complete well column is present.

    Returns:
        ``None``, or the modified copy.
        With a reference, writes ``uns["mantispy"][key_added]`` with ``group``, ``n_obs``, ``distance``, ``pvalue``, ``qvalue`` and ``is_hit``, where ``n_obs`` counts the rows the statistic actually used, after any subsampling and after splitting the reference.
        With ``reference=None``, writes only the square, group-labeled matrix at ``uns["mantispy"][key_added + "_pairwise"]``.

    Raises:
        ValueError: ``reference`` selects fewer than two rows.

    Notes:
        The null permutes the group and reference labels over the pooled rows and recomputes the statistic, the standard permutation test for a two-sample quantity.
        At cell resolution the exchangeable unit is the well, not the cell, so ``block`` draws whole wells for the pseudo-group instead; without a usable well column it warns and permutes cells, which is anti-conservative.

        A group that is the reference itself is scored by splitting its rows in half, by whole wells when a block is in use.
        A group with fewer than two rows on either side, or a reference group with fewer than four rows, gets a NaN p-value and a warning.

        Check the false positive rate on your own screen with :func:`~mantispy.metrics.diagnose_testing`, which relabels control wells as pseudo-treatments of the same size and reports the fraction that are called.
    """
    values = representation(adata, use_rep)
    codes, keys = group_codes(adata, groupby)
    order, offsets = group_offsets(codes, len(keys))
    generator = np.random.default_rng(seed)

    if reference is None:
        blocks = []
        for index in range(len(keys)):
            rows = order[offsets[index] : offsets[index + 1]]
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

    block_codes = _default_well_block(adata, block=block)
    if block_codes is None and block is None and _is_object_resolution(adata):
        warnings.warn(
            "edistance is at cell resolution and no usable block was given, so the null permutes single cells. "
            "Cells within a well are not independent replicates (they share the well, its plate position, seeding "
            "and focus), so the null is anti-conservative. Pass block= a well column, add a complete "
            "Metadata_Well, or aggregate to wells with mt.tl.aggregate.",
            UserWarning,
            stacklevel=3,
        )

    observed = np.full(len(keys), np.nan)
    sizes = np.empty(len(keys), dtype=int)
    null = np.full((len(keys), n_permutations), np.nan)
    unscorable_reference = []
    unscorable_size = []
    for index in range(len(keys)):
        rows = order[offsets[index] : offsets[index + 1]]
        sizes[index] = rows.size
        if rows.size > max_reference:
            get_logger().info(
                "edistance sampled %d of %d rows of %r for the null", max_reference, rows.size, keys[index]
            )
            rows = np.sort(generator.choice(rows, size=max_reference, replace=False))
            sizes[index] = rows.size
        against = np.setdiff1d(control_rows, rows)
        if against.size < 2:
            if rows.size < 4:
                unscorable_reference.append(str(keys[index]))
                continue
            rows, against = _split_reference_wells(rows, block_codes, generator)
            sizes[index] = rows.size

        size = rows.size
        if size < 2 or against.size < 2:
            unscorable_size.append(str(keys[index]))
            continue

        if block_codes is not None:
            observed[index] = energy_distance(values[rows], values[against])
            null[index] = _edistance_block_null(values, block_codes, rows, against, n_permutations, generator)
            continue

        pooled = np.vstack([values[rows], values[against]])
        total = pooled.shape[0]
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
