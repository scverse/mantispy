"""Hit calling: which perturbations moved away from the controls, and by how much."""

from __future__ import annotations

import warnings
from collections.abc import Sequence

import numpy as np
import pandas as pd
from anndata import AnnData

from mantispy._core._distance import mahalanobis_transform
from mantispy._core._reduce import group_codes, group_offsets, representation
from mantispy._core._stats import (
    _default_replicate,
    _is_object_resolution,
    _split_wells,
    benjamini_hochberg,
    permutation_pvalue,
    split_reference,
)
from mantispy._core.frames import as_frame
from mantispy._core.logging import get_logger
from mantispy._core.masks import reference_mask
from mantispy._core.mutation import inplace_or_copy

METHODS = ("mahalanobis", "ks", "hellinger", "mp_value")

COVARIANCES = ("empirical", "robust")

NULLS = ("permutation", "pseudogenes")

PSEUDOGENE_FUNCS = ("median", "mean")

_REPLICATE_MIN = 16

#: The floor warning fires only when Benjamini-Hochberg cannot call a set smaller than this many groups ...
_FLOOR_MIN_CALLS = 10
#: ... and than this share of them.
_FLOOR_SHARE = 0.01

#: Pseudo-genes profiled per block, which bounds the memory of the draws.
_PSEUDOGENE_BLOCK = 1024

#: The cell-count column of guide profiles, as mt.tl.aggregate writes it.
_CELL_COUNT = "Metadata_CellCount"
#: Guide profiles whose control guides carry more than this many times the tested guides' median cell count raise.
_CELL_RATIO_MAX = 1.25


def ks_statistic(samples: np.ndarray, reference: np.ndarray, *, reference_sorted: bool = False) -> np.ndarray:
    """Two-sample KS statistic of each row of ``samples`` against ``reference``.

    Vectorized over rows, where ``scipy.stats.ks_2samp`` takes one pair at a time.
    The supremum of ``|F_s - F_r|`` is attained at a jump of either step function, and both one-sided differences evaluated at the points of ``samples`` cover every such jump.
    With the reference sorted once, all rows take two ``searchsorted`` calls.

    Args:
        samples: One sample per row; a one-dimensional array is treated as a single row.
        reference: The sample to compare against, sorted here rather than by the caller.
        reference_sorted: Skip sorting ``reference``, for a caller that passes the same sorted reference many times (a permutation null).

    Returns:
        One statistic per row of ``samples``, or ``NaN`` for every row when either side is empty.
    """
    samples = np.atleast_2d(samples)
    reference = reference if reference_sorted else np.sort(reference)
    n, m = samples.shape[1], reference.size
    if n == 0 or m == 0:
        return np.full(samples.shape[0], np.nan)

    ordered = np.sort(samples, axis=1)
    below = np.searchsorted(reference, ordered, side="left") / m
    at_or_below = np.searchsorted(reference, ordered, side="right") / m
    steps = np.arange(1, n + 1) / n
    upper = np.max(steps[None, :] - at_or_below, axis=1)
    lower = np.max(below - (steps - 1.0 / n)[None, :], axis=1)
    return np.maximum(np.maximum(upper, lower), 0.0)


def _statistic(distances: np.ndarray, control_distances: np.ndarray, method: str) -> np.ndarray:
    """One number per row of ``distances``: a median, or a KS statistic."""
    if method == "mahalanobis":
        return np.median(np.atleast_2d(distances), axis=1)
    return ks_statistic(distances, control_distances)


def _replicate_null(
    to_control: np.ndarray,
    replicate_codes: np.ndarray,
    pool: np.ndarray,
    tested: np.ndarray,
    n_permutations: int,
    seed: int,
    index: int,
) -> np.ndarray:
    """Permutation null that resamples whole replicates (normally wells) rather than cells.

    The pool is the replicates the tested group and the held-out controls span; each permutation draws as many whole replicates as the tested group spans, without replacement, and takes all of the pool's rows in them.
    Only ``method="mahalanobis"`` reaches here, so the statistic is the median distance.
    """
    pool_replicates = replicate_codes[pool]
    uniq = np.unique(pool_replicates)
    n_draw = int(np.unique(replicate_codes[tested]).size)
    dist_by_replicate = [to_control[pool[pool_replicates == code]] for code in uniq]
    generator = np.random.default_rng([seed, index])
    null = np.empty(n_permutations)
    for permutation in range(n_permutations):
        chosen = generator.choice(uniq.size, size=n_draw, replace=False)
        null[permutation] = float(np.median(np.concatenate([dist_by_replicate[position] for position in chosen])))
    return null


def _ks_replicate_null(
    values: np.ndarray,
    replicate_codes: np.ndarray,
    pool: np.ndarray,
    tested: np.ndarray,
    reference: np.ndarray,
    n_permutations: int,
    seed: int,
    index: int,
) -> np.ndarray:
    """Replicate permutation null for the KS statistic.

    Like :func:`_replicate_null` the pool is the replicates the tested group and the held-out controls span, and each permutation draws as many whole replicates as the tested group spans; the statistic recomputed per draw is the KS distance of the drawn rows against the fixed ``reference`` rather than a median.
    """
    pool_replicates = replicate_codes[pool]
    uniq = np.unique(pool_replicates)
    n_draw = int(np.unique(replicate_codes[tested]).size)
    value_by_replicate = [values[pool[pool_replicates == code]] for code in uniq]
    reference = np.sort(reference)  # fixed across draws, so sort it once rather than inside ks_statistic
    generator = np.random.default_rng([seed, index])
    null = np.empty(n_permutations)
    for permutation in range(n_permutations):
        chosen = generator.choice(uniq.size, size=n_draw, replace=False)
        drawn = np.concatenate([value_by_replicate[position] for position in chosen])
        null[permutation] = float(ks_statistic(drawn[None, :], reference, reference_sorted=True)[0])
    return null


def _hellinger_null(
    x: np.ndarray,
    tested: np.ndarray,
    against: np.ndarray,
    replicate_codes: np.ndarray | None,
    settings: tuple,
    min_against: int,
    n_permutations: int,
    seed: int,
    index: int,
) -> tuple[float, np.ndarray]:
    """Observed Hellinger distance or mp-value statistic and its null, from the same kernel so that the observed draw ties with itself.

    The null draws as many whole replicates as the tested group spans, or as many rows without a replicate, from the tested and reference rows pooled, against the rest of them.
    """
    import numba

    from mantispy._core import _hellinger

    pool = np.concatenate([tested, against])
    if replicate_codes is not None:
        units = np.unique(replicate_codes[pool], return_inverse=True)[1].astype(np.int64)
        observed_draw = np.unique(units[: tested.size])
    else:
        units, observed_draw = np.arange(pool.size, dtype=np.int64), np.arange(tested.size)
    n_units, n_draw = (int(units.max()) + 1 if pool.size else 0), observed_draw.size
    # A side too small for its covariance leaves the group unscored, so there is no null to draw.
    min_rows = settings[-1]
    if n_draw == 0 or n_draw >= n_units or tested.size < min_rows or against.size < min_against:
        return np.nan, np.full(n_permutations, np.nan)
    generator = np.random.default_rng([seed, index])
    draws = np.stack(
        [observed_draw] + [generator.choice(n_units, n_draw, replace=False) for _ in range(n_permutations)]
    )
    threads = numba.get_num_threads()
    # ponytail: wide matrices run the permutations on one thread, because concurrent LAPACK calls on them crashed the
    # BLAS with numba's threads; a reduced use_rep, which the method wants anyway, keeps the parallel path.
    if x.shape[1] > 64:
        numba.set_num_threads(1)
    try:
        out = _hellinger.null(np.ascontiguousarray(x[pool]), units, draws, *settings)
    finally:
        numba.set_num_threads(threads)
    return float(out[0]), out[1:]


def _profile_distances(
    values: np.ndarray,
    unit_rows: list[np.ndarray],
    draws: np.ndarray,
    centre: np.ndarray,
    whitening: np.ndarray,
    func: str,
    generator: np.random.Generator,
    cell_counts: np.ndarray | None = None,
    n_matched: int | None = None,
) -> tuple[np.ndarray, int, int]:
    """Distance from the control centre of each draw's profile, its units' rows combined by ``func``.

    ``draws`` holds one row of positions in ``unit_rows`` per profile.
    The profile is taken in feature space with the kernel :func:`~mantispy.tl.aggregate` uses, and whitened after, since a median does not commute with the whitening.
    With ``cell_counts``, each drawn unit below position ``n_matched`` keeps a fresh random subset of its rows, of a size drawn from ``cell_counts``, or all of them when it has fewer.
    Also returns how many drawn units were matched, and how many of them had fewer rows than their draw.
    """
    from mantispy._core._numba import MEAN, MEDIAN, _grouped, _kernel_input

    matrix = _kernel_input(values)
    stat = MEDIAN if func == "median" else MEAN
    n_matched = len(unit_rows) if n_matched is None else n_matched
    out = np.empty(draws.shape[0])
    short = matched = 0
    for start in range(0, draws.shape[0], _PSEUDOGENE_BLOCK):
        block = draws[start : start + _PSEUDOGENE_BLOCK]
        targets = (
            cell_counts[generator.integers(cell_counts.size, size=block.shape)] if cell_counts is not None else None
        )
        # Each unit's rows are shuffled once per block; a window at a random start of that order is a uniform subset.
        shuffled: dict[int, np.ndarray] = {}
        pieces = []
        lengths = np.zeros(block.shape[0], dtype=np.int64)
        for position, draw in enumerate(block):
            for column, unit in enumerate(draw):
                rows = unit_rows[unit]
                if targets is not None and unit < n_matched:
                    matched += 1
                    target = int(targets[position, column])
                    if rows.size > target:
                        order = shuffled.get(unit)
                        if order is None:
                            order = shuffled[unit] = generator.permutation(rows)
                        first = int(generator.integers(rows.size))
                        rows = order[first : first + target]
                        if rows.size < target:
                            rows = np.concatenate([rows, order[: target - rows.size]])
                    elif rows.size < target:
                        short += 1
                pieces.append(rows)
                lengths[position] += rows.size
        offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
        profiles = _grouped(matrix, np.concatenate(pieces).astype(np.int64), offsets, block.shape[0], stat, 0.5, 1)
        out[start : start + block.shape[0]] = np.linalg.norm(np.nan_to_num(profiles - centre) @ whitening, axis=1)
    return out, short, matched


def _n_draws(n: int, k: int, limit: int) -> int:
    """The number of ways to choose ``k`` of ``n``, or ``limit + 1`` once it exceeds ``limit``."""
    if k < 0 or k > n:
        return 0
    k = min(k, n - k)
    count = 1
    for step in range(k):
        count = count * (n - step) // (step + 1)
        if count > limit:
            return limit + 1
    return count


def _draw_units(n_units: int, size: int, n_draws: int, generator: np.random.Generator) -> np.ndarray:
    """``n_draws`` sets of ``size`` distinct units, drawn independently, so a unit recurs across sets but never within one."""
    # One draw at a time, so the memory does not grow with the number of units.
    return np.sort([generator.choice(n_units, size, replace=False, shuffle=False) for _ in range(n_draws)], axis=1)


def _own_share(n_pool: int, size: int) -> float:
    """The share of the ``size``-unit subsets of a pool plus a group's own ``size`` units that hold none of the group's units."""
    from math import comb

    return comb(n_pool, size) / comb(n_pool + size, size)


def _draws_with_own(n_pool: int, size: int, n_draws: int, generator: np.random.Generator) -> np.ndarray:
    """``size``-unit subsets of a pool (positions below ``n_pool``) and a group's own units (the ``size`` positions after it), drawn uniformly among those that hold at least one own unit."""
    from math import comb

    weights = np.array([comb(size, own) * comb(n_pool, size - own) for own in range(1, size + 1)], dtype=float)
    own = generator.choice(np.arange(1, size + 1), size=n_draws, p=weights / weights.sum())
    own_order = np.argsort(generator.random((n_draws, size)), axis=1) + n_pool
    pool_order = np.argpartition(generator.random((n_draws, n_pool)), size - 1, axis=1)[:, :size]
    column = np.arange(size)
    # The first ``own`` positions take own units and the rest pool units.
    pool_column = np.clip(column[None, :] - own[:, None], 0, size - 1)
    return np.where(column[None, :] < own[:, None], own_order, np.take_along_axis(pool_order, pool_column, axis=1))


def _pseudogene_pvalue(
    observed: float,
    shared: np.ndarray,
    values: np.ndarray,
    pool_rows: list[np.ndarray],
    own_rows: list[np.ndarray],
    centre: np.ndarray,
    whitening: np.ndarray,
    func: str,
    generator: np.random.Generator,
    cell_counts: np.ndarray | None,
    match_own: bool,
) -> tuple[float, float, int, int]:
    """Permutation p-value of a group against every way to draw its number of units from the pool and its own units.

    Draws holding none of the group's units come from ``shared``, computed once for every group; draws holding some are drawn here, as many as their share calls for, so each draw counts the same.
    Returns the p-value, its floor, and how many drawn units were matched and fell short of their cell count.
    """
    size = len(own_rows)
    share = _own_share(len(pool_rows), size)
    n_own = int(np.ceil(shared.size * (1.0 - share) / share))
    draws = _draws_with_own(len(pool_rows), size, n_own, generator)
    with_own, short, matched = _profile_distances(
        values,
        [*pool_rows, *own_rows],
        draws,
        centre,
        whitening,
        func,
        generator,
        cell_counts,
        n_matched=len(pool_rows) + (size if match_own else 0),
    )
    total = shared.size + n_own
    at_least = np.sum(shared >= observed) + np.sum(with_own >= observed)
    return (at_least + 1.0) / (total + 1.0), 1.0 / (total + 1.0), matched, short


def _pseudogene_null(
    values: np.ndarray,
    unit_codes: np.ndarray,
    is_control: np.ndarray,
    null_rows: np.ndarray,
    codes: np.ndarray,
    n_groups: int,
    centre: np.ndarray,
    whitening: np.ndarray,
    func: str,
    n_pseudogenes: int,
    seed: int,
    match_cells: bool,
    cell_count: np.ndarray | None,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Score every tested group against pseudo-genes drawn from the held-out control units, and score calibration pseudo-genes.

    A group of ``K`` units is ranked among the profiles of ``K`` units drawn from the held-out control units and its own units pooled, as the permutation null pools a group with the controls.
    The held-out units are also cut into disjoint pseudo-genes of the median group size, each tested the same way against the other held-out units.

    Returns the table without q-values, real groups first, and each row's smallest reachable p-value.
    """
    generator = np.random.default_rng([seed, 1])
    null_units, null_inverse = np.unique(unit_codes[null_rows], return_inverse=True)
    by_unit = np.argsort(null_inverse, kind="stable")
    null_unit_rows = np.split(null_rows[by_unit], np.flatnonzero(np.diff(null_inverse[by_unit])) + 1)
    n_null = null_units.size

    order, offsets = group_offsets(codes, n_groups)
    tested_rows = [order[offsets[index] : offsets[index + 1]] for index in range(n_groups)]
    tested_rows = [rows[~is_control[rows]] for rows in tested_rows]
    sizes = np.array([np.unique(unit_codes[rows]).size for rows in tested_rows])
    scored = sizes > 0
    if scored.any() and (sizes[scored] == 1).all():
        raise ValueError(
            "null='pseudogenes' builds each pseudo-gene from the guides a gene pools, but every tested group here is "
            "a single guide or a single row. A gene-level object cannot say how its rows were aggregated, so score "
            "guide profiles instead: aggregate the cells with mt.tl.aggregate(by=[gene, guide]) and pass the gene "
            "column as groupby."
        )
    calibration_size = int(np.floor(np.median(sizes[scored]) + 0.5)) if scored.any() else 1
    wanted = sorted(set(sizes[scored].tolist()) | {calibration_size})
    too_large = [size for size in wanted if size > n_null]
    if too_large:
        raise ValueError(
            f"a group pools {max(too_large)} guides, more than the {n_null} held-out reference guides a pseudo-gene "
            "draws from"
        )

    tested_units = np.concatenate([unit_codes[rows] for rows in tested_rows])
    target_counts = _match_cells(
        match_cells, null_unit_rows, tested_units, null_rows, np.concatenate(tested_rows), cell_count
    )
    draws = {size: _draw_units(n_null, size, n_pseudogenes, generator) for size in wanted}
    shared, short, drawn = {}, 0, 0
    for size in wanted:
        shared[size], missing, matched = _profile_distances(
            values, null_unit_rows, draws[size], centre, whitening, func, generator, target_counts
        )
        short, drawn = short + missing, drawn + matched

    observed = np.full(n_groups, np.nan)
    pvalues = np.full(n_groups, np.nan)
    floors = np.full(n_groups, np.nan)
    combine = np.nanmedian if func == "median" else np.nanmean
    for index in np.flatnonzero(scored):
        rows = tested_rows[index]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            profile = combine(values[rows], axis=0)
        observed[index] = float(np.linalg.norm(np.nan_to_num(profile - centre) @ whitening))
        unit_order = np.argsort(unit_codes[rows], kind="stable")
        own_rows = np.split(rows[unit_order], np.flatnonzero(np.diff(unit_codes[rows][unit_order])) + 1)
        pvalues[index], floors[index], matched, missing = _pseudogene_pvalue(
            observed[index],
            shared[sizes[index]],
            values,
            null_unit_rows,
            own_rows,
            centre,
            whitening,
            func,
            generator,
            target_counts,
            match_own=False,
        )
        short, drawn = short + missing, drawn + matched

    # Calibration pseudo-genes: disjoint sets of held-out units, each tested against the other held-out units.
    n_calibration = n_null // calibration_size
    cut = generator.permutation(n_null)[: n_calibration * calibration_size].reshape(n_calibration, calibration_size)
    cut = np.sort(cut, axis=1)
    calibration_observed = _profile_distances(
        values, null_unit_rows, cut, centre, whitening, func, generator, target_counts
    )[0]
    calibration_p = np.full(n_calibration, np.nan)
    calibration_floor = np.full(n_calibration, np.nan)
    reference_draws, reference = draws[calibration_size], shared[calibration_size]
    for position, units in enumerate(cut):
        disjoint = ~np.isin(reference_draws, units).any(axis=1)
        if disjoint.any() and n_null > calibration_size:
            own = set(units.tolist())
            pool = [rows for unit, rows in enumerate(null_unit_rows) if unit not in own]
            calibration_p[position], calibration_floor[position], _, _ = _pseudogene_pvalue(
                calibration_observed[position],
                reference[disjoint],
                values,
                pool,
                [null_unit_rows[unit] for unit in units],
                centre,
                whitening,
                func,
                generator,
                target_counts,
                match_own=True,
            )
    if target_counts is not None and short:
        get_logger().warning(
            "hit_calling: %.0f%% of the control guides drawn into pseudo-genes have fewer cells than the tested "
            "guide count drawn for them and keep every cell",
            100.0 * short / max(drawn, 1),
            deep="a guide is never upsampled, so pseudo-genes of those guides carry fewer cells than the genes",
        )
    width = max(2, len(str(n_calibration)))
    table = pd.DataFrame(
        {
            "n_obs": np.concatenate(
                [
                    [rows.size for rows in tested_rows],
                    [sum(null_unit_rows[unit].size for unit in units) for units in cut],
                ]
            ).astype(int),
            "distance": np.concatenate([observed, calibration_observed]),
            "pvalue": np.concatenate([pvalues, calibration_p]),
            "is_pseudogene": np.r_[np.zeros(n_groups, dtype=bool), np.ones(n_calibration, dtype=bool)],
        }
    )
    table.insert(
        0,
        "group",
        [None] * n_groups + [f"pseudogene_{position + 1:0{width}d}" for position in range(n_calibration)],
    )
    return table, np.concatenate([floors, calibration_floor])


def _match_cells(
    match_cells: bool,
    null_unit_rows: list[np.ndarray],
    tested_units: np.ndarray,
    null_rows: np.ndarray,
    tested_rows: np.ndarray,
    cell_count: np.ndarray | None,
) -> np.ndarray | None:
    """The cell counts a drawn control guide is subsampled to, or ``None`` when no subsampling is needed or wanted.

    On cells, the tested guides' cell counts.
    On guide profiles the cells are gone, so the control guides must carry about as many cells as the tested guides, read from ``cell_count``.
    """
    if not match_cells:
        return None
    if all(rows.size == 1 for rows in null_unit_rows) and np.unique(tested_units).size == tested_units.size:
        if cell_count is None:
            raise ValueError(
                f"match_cells=True needs obs[{_CELL_COUNT!r}] on guide profiles, to check that the control guides "
                "carry about as many cells as the tested guides; mt.tl.aggregate writes it. Pass match_cells=False to "
                "skip the check."
            )
        control, tested = np.nanmedian(cell_count[null_rows]), np.nanmedian(cell_count[tested_rows])
        if control > _CELL_RATIO_MAX * tested:
            raise ValueError(
                f"the control guides carry a median of {control:g} cells and the tested guides {tested:g}, so a "
                "pseudo-gene has less noise than a gene and the test calls too much. Score the cells instead, with "
                "replicate= naming the guide, so that each pseudo-gene guide is subsampled to the tested guides' cell "
                "counts. Pass match_cells=False to skip the check."
            )
        return None
    return np.bincount(np.unique(tested_units, return_inverse=True)[1])


def _warn_on_floor(floors: np.ndarray, threshold: float, limited_by_draws: bool) -> None:
    """Warn when the smallest p-value the null can reach leaves Benjamini-Hochberg no small call set."""
    from math import ceil

    finite = floors[np.isfinite(floors)]
    if finite.size == 0:
        return
    floor = float(finite.min())
    smallest = ceil(floor * finite.size / threshold - 1e-9)
    if smallest > max(_FLOOR_MIN_CALLS, _FLOOR_SHARE * finite.size):
        get_logger().warning(
            "hit_calling: the p-value floor is %.2g, so Benjamini-Hochberg over %d groups can call no set smaller than %d",
            floor,
            finite.size,
            smallest,
            deep=(
                "the null has few distinct draws: groups of one row against single reference rows; score guide "
                "profiles with null='pseudogenes'"
                if limited_by_draws
                else "raise n_permutations"
            ),
        )


@inplace_or_copy()
def hit_calling(
    adata: AnnData,
    groupby: str = "Metadata_Perturbation",
    reference: str | None = "negcon",
    method: str = "mahalanobis",
    covariance: str = "empirical",
    use_rep: str | None = None,
    replicate: str | Sequence[str] | None = None,
    null: str = "permutation",
    pseudogene_func: str = "median",
    match_cells: bool = True,
    n_permutations: int = 1000,
    threshold: float = 0.05,
    seed: int = 0,
    key_added: str = "hits",
    copy: bool = False,
) -> AnnData | None:
    """Call hits by testing each group's distance from the controls.

    Each group is scored against the half of the reference rows that did not fit the centroid and covariance, and its null is the other ways to draw a group of its size from the group and those controls pooled.
    Under the null the two are exchangeable, so the null is calibrated; bootstrapping the controls alone is not, because it is centred on that sample's own median rather than the population's and leaves the error in that centre out of its spread.

    This calibrates each group against control wells; :func:`mantispy.tl.empirical_fdr` instead calibrates a per-gene score against control genes that cannot respond, for a genetic screen where non-expressed genes are the better null.

    This wants a well-replicated design.
    The statistic is a group's median distance, so a group of one or two wells is dominated by whichever wells it holds and no number of permutations recovers that; :func:`mantispy.tl.map` with ``mode="activity"`` ranks replicate pairs instead and is the usual readout on screens with little replication.
    JUMP-Target-2 read as a single plate gives every compound one well and is the common way to land in that regime, while the same plate map read across several of them gives one well per plate.

    Distance from the controls also rises when a treatment kills cells.
    Read the calls beside a cell count, or beside :func:`mantispy.tl.cytotoxicity`, before taking them for morphology.

    Args:
        adata: Object to score, at cell or well resolution.
        groupby: Column defining the groups to test.
        reference: Which rows are the controls.
            They are split in half, one half to fit the covariance and the other to form the null (see Notes).
        method: ``"mahalanobis"`` scores the median distance of the group's rows from the control centroid, measured in the controls' covariance so that directions the controls already vary in count for less.
            ``"ks"`` scores the Kolmogorov-Smirnov statistic between the group's and the controls' distance distributions, which detects a shifted subpopulation that leaves the median unchanged.
            Use it at cell resolution.
            Its p-value comes from a permutation null over whole replicates when one is in use (see ``replicate``), and otherwise from ``scipy.stats.ks_2samp``.
            ``"hellinger"`` fits a Gaussian to the group's rows and one to the controls', and scores the Hellinger distance between the two, so a change in the group's own spread or shape counts as well as a shift of its centre; a treatment that sends only some cells into a new state widens the group without moving its median.
            It needs each side to have at least twice as many rows as features, so use it at cell resolution on a reduced ``use_rep``; a group short of that is left unscored with a warning.
            It is opt-in; the default stays ``"mahalanobis"``.
            A single Gaussian describes a cloud with several cell states poorly, so on a real screen it can call fewer groups than ``"mahalanobis"`` or ``"ks"``; read it beside them rather than instead of them.
            With ``covariance="robust"`` it is the robust Hellinger distance of BioProfiling.jl (see Notes), which resists stray cells but trims a partial responder and runs far slower.
            ``"mp_value"`` scores the squared Mahalanobis distance between the group's centroid and the controls', measured in the average of the two sides' covariances, the multidimensional perturbation value of Hutz et al. (see Notes).
            Unlike ``"mahalanobis"`` it counts the group's own spread, so a group whose replicates disagree is measured in a wider covariance and moves less.
            The average is positive definite whenever the controls' covariance is, so the group needs only two rows and the controls more rows than features; it works on well profiles with a few replicates, where ``"hellinger"`` cannot.
            A group with fewer rows is left unscored with a warning.
        covariance: Scatter the Mahalanobis distance is measured in.
            ``"empirical"`` uses every fitting control row.
            ``"robust"`` uses the minimum covariance determinant subset, so a few stray control wells stop widening the covariance in their own direction and masking real hits there; it needs more control rows than features, so pair it with ``use_rep``.
            Under ``method="hellinger"`` and ``method="mp_value"`` it is the location and scatter both sides are fitted with, refitted on every permutation; under ``method="mp_value"`` the group then needs more rows than features too.
        use_rep: Score ``obsm[use_rep]`` instead of ``X``.
            When the covariance-fitting half of the controls has no more rows than there are features, the covariance is singular and a warning suggests a PCA representation.
        replicate: ``obs`` column, or sequence of columns, whose groups are the design's exchangeable unit: the well in an arrayed screen, the guide (``"Metadata_sgRNA"``) in a pooled one.
            The permutation null then draws whole replicates rather than cells, since cells within a well are not independent replicates (see Notes).
            Left ``None``, it defaults to the physical well, the ``(Metadata_Plate, Metadata_Well)`` pair, on an object explicitly stamped cell resolution that has replicated wells; a same well name on two plates is then two replicates, not one.
            On such an object where some well holds both reference and tested cells, a pooled design, there is no default and it raises instead.
            It stays off when the object is not stamped cell resolution and when it has no complete well column, in which case the cell path warns; under ``method="ks"`` without a replicate the p-value then falls back to ``scipy.stats.ks_2samp``.
            Under ``null="pseudogenes"`` it names the guide, the unit a pseudo-gene is built from.
        null: What each group is ranked against.
            ``"permutation"`` draws same-size sets of rows, or of replicates, from the group and the held-out controls pooled (see Notes).
            ``"pseudogenes"`` ranks each group against pseudo-genes, each built from as many held-out control guides as the group has guides and combined the same way, for a pooled screen whose genes pool several guides (see Notes).
            It needs ``method="mahalanobis"``, and groups made of several guides: one row per guide, or cells with ``replicate`` naming the guide.
        pseudogene_func: How the rows of a group, and of a pseudo-gene, are combined into the profile whose distance is scored under ``null="pseudogenes"``: ``"median"`` or ``"mean"``, as in :func:`~mantispy.tl.aggregate`.
        match_cells: Under ``null="pseudogenes"``, match the pseudo-gene guides to the tested guides in cells (see Notes).
            On cells, each drawn control guide keeps a random subset of its cells, of a size drawn from the tested guides' cell counts.
            On guide profiles it checks ``obs["Metadata_CellCount"]`` instead and raises when the control guides carry clearly more cells than the tested guides.
            ``False`` skips both.
        n_permutations: Size of the permutation null, or under ``null="pseudogenes"`` the number of pseudo-genes without the group's own guides, drawn once for each group size.
            Applies to ``method="mahalanobis"``, ``method="hellinger"`` and ``method="mp_value"``, and to ``method="ks"`` when a replicate is in use.
            The smallest p-value a group can reach is one over this number plus one, so it sets how small a set of groups Benjamini-Hochberg can call (see Notes).
        threshold: q-value below which a group is called a hit in ``is_hit``.
        seed: Seed for the control split and the permutation null.
        key_added: Name for the outputs.
        copy: Return a modified copy instead of mutating in place.

    Returns:
        ``None``, or the modified copy.
        Writes ``uns["mantispy"][key_added]`` with ``group``, ``n_obs``, ``distance``, ``pvalue``, ``qvalue`` and ``is_hit``, where ``n_obs`` counts the rows the statistic used rather than the rows the group has, and joins ``obs[key_added + "_distance"]`` and ``obs[key_added + "_qvalue"]`` back onto the rows.
        Also writes ``obs[key_added + "_row_distance"]``, each row's own distance from the control centroid rather than its group's.
        That is the column a dose-response fit wants: the group statistic is one number repeated over the group's rows, so the controls show no spread and nothing downstream can read a scale off them.
        Rows that fitted the centroid sit a little closer to it than the held-out controls do, by the same split the Notes describe.
        Which rows those are is written to ``obs[key_added + "_reference_held_out"]``, true for the controls that did not fit, so that a scale taken from the controls can be taken from the honest half.
        Under ``null="pseudogenes"`` the table has no row for a group of reference rows only, and adds the calibration pseudo-genes as rows marked by an ``is_pseudogene`` column (see Notes).

    Raises:
        ValueError: ``method`` is not one of ``METHODS``, ``null`` is not one of ``NULLS``, ``reference`` selects fewer than four rows, or ``replicate`` is not given on a pooled design at cell resolution.
            Under ``null="pseudogenes"``: ``method`` is not ``"mahalanobis"``, every tested group is a single guide, a group has more guides than the held-out controls, or ``match_cells`` finds guide profiles without ``obs["Metadata_CellCount"]`` or with control guides that carry clearly more cells than the tested guides.

    Notes:
        The controls are split in half.
        One half estimates the centroid and the covariance, and the other supplies the null.
        A null drawn from the rows that defined the centroid would be in-sample while every tested group is out-of-sample, so it would come out too small and call pure noise as hits.
        Because only half the controls fit the covariance, the singular-covariance warning fires when there are fewer than about twice as many controls as features.

        A row that fitted the centroid and the covariance sits closer to the centroid than any other group's row can, so it stays off the tested side of every group.
        The controls carry a perturbation label of their own, and that group is therefore left with the held-out half, which is split once more, at random, so that the rows tested and the rows they are tested against are different rows.
        Its row of the table is a draw from the null rather than a sample compared with part of itself measured against a centroid half of it placed, and ``n_obs`` reports about a quarter of the controls for it.
        Its null is the other ways to halve those rows rather than a bootstrap of them, which would be too wide because the sample is part of what it is drawn from.
        That row is therefore an honest draw from the null: its p-value is uniform and falls below any cutoff about as often as the cutoff says, so a control group appearing in a hit list is this test working rather than a fault.

        The null is drawn from the controls only, and asks whether a group is further out than the same number of control rows would be.
        Drawing from every row would put real hits into the null, and a screen with many hits would look like one with none.

        Calibration degrades with few controls.
        On pure-noise screens the false positive rate at a nominal 0.05 is inflated with few controls and approaches the nominal rate as controls are added.
        This is not a bound for another screen, and it was checked with ``method="mahalanobis"``.
        Under ``method="ks"`` with a replicate the null draws whole replicates for the pseudo-treated side and recomputes the KS statistic, the same unit ``replicate`` fixes for the median; without one each group's distances are compared with those of the null half by ``scipy.stats.ks_2samp``.

        At cell resolution the calibration above assumes a null unit that matches the design.
        Cells within a well share the well, its plate position, seeding and focus, so they are not independent replicates: the exchangeable unit is the well, and a null that shuffles cells shrinks the statistic's spread by the cell count rather than the well count and calls pure noise far above nominal.
        ``replicate`` fixes this by drawing whole wells.
        The controls are then split into whole wells too, one half fitting the centroid and covariance and the other forming the null, so every well in the null pool is complete and the reference group draws whole wells like any other group.
        ``replicate`` defaults to the physical well, the ``(Metadata_Plate, Metadata_Well)`` pair, on an object explicitly stamped cell resolution with replicated wells; it warns and leaves the cell-level test in place when no complete well column is present.
        In a pooled screen every well holds every perturbation, so drawing whole wells cannot separate a group from the reference; the exchangeable unit is the guide, and the call raises until ``replicate`` names it.
        With fewer than about sixteen wells in the null half the well-level null is coarse and runs conservative rather than anti-conservative; for an exact well-level test, aggregate with :func:`~mantispy.tl.aggregate` first.

        To check the rate on your own screen, :func:`~mantispy.metrics.diagnose_testing` relabels control wells as pseudo-treatments of your group sizes and reports the fraction called.

        The permutation null can be no finer than its distinct draws.
        A group of one row against single reference rows has only as many draws as there are held-out rows, so its p-value cannot fall below one over that count, however large ``n_permutations`` is.
        Benjamini-Hochberg over many groups then calls either a large set or nothing, and a warning says how small a set it could call.

        ``null="pseudogenes"`` is for a pooled screen whose genes each pool several guides while every control guide stands alone.
        A gene's profile pools the noise of several guides and sits closer to the control centre than a single control guide does, so ranking it against single guides calls almost nothing.
        A pseudo-gene removes that mismatch, because it draws as many held-out control guides as the gene has guides and combines their rows with ``pseudogene_func``, as the gene's own rows are combined, and its profile's distance from the control centre is the null draw.
        The control guides are split as above, half placing the centre and covariance and half supplying the pseudo-genes.
        A pseudo-gene never holds one guide twice, but the draws are independent, so the same guide recurs across pseudo-genes and their number is not capped by the number of control guides.
        A group's own guides join the pool its pseudo-genes are drawn from, as the permutation null pools a group with the controls, so the group's profile is one draw among the ways to pick its number of guides from the pool.
        Without them, the pseudo-genes of a finite pool share their guides and spread a little less than a new gene does, which in many dimensions calls above the nominal rate.
        The pseudo-genes without any of the group's guides are drawn once and shared by every group of that size, and those with some of its guides are drawn for each group, as many as their share of the pool calls for.
        A group's p-value cannot fall below one over the number of its pseudo-genes plus one, which is ``n_permutations`` and the group's own draws together.
        The rows of a group are combined into one profile, so a gene-level object, one row per gene, cannot be tested this way. Aggregate the cells to one row per guide with :func:`~mantispy.tl.aggregate` and pass the gene column as ``groupby``.
        From guide profiles, a gene and a pseudo-gene are built alike; a gene aggregated from pooled cells is not, since a median over cells is a different statistic from a median over guide medians.
        A pseudo-gene must also match a gene in cells per guide.
        A guide's profile is a median over its cells, so a control guide with more cells than a gene guide has less noise, and a pseudo-gene of such guides sits closer to the centre than a gene with no phenotype, which calls too much.
        On cells, with ``replicate`` naming the guide, ``match_cells`` subsamples each drawn control guide to a cell count drawn from the tested guides', afresh in every pseudo-gene, so the null averages over the subsampling and a seed moves the calls little.
        The centre and covariance are still fitted on every cell of the fitting guides.
        A guide with fewer cells than its draw keeps all of them and is never upsampled, and a warning counts how many fell short.
        Guide profiles no longer hold the cells, so there the control guides cannot be subsampled, and a clear excess of cells in the control guides raises and points to the cells.

        The held-out control guides are also cut into disjoint pseudo-genes of the typical group size, and each is tested like a group against the other held-out guides.
        These are the ``is_pseudogene`` rows of the table.
        Their q-values are corrected among themselves, so the real groups' q-values are the ones they get without them.
        The share of them below a p-value cutoff is a check on the false positive rate, though it cannot see a bias the pseudo-genes and their null share, such as the cells per guide above.

        ``method="hellinger"`` follows the robust morphological perturbation value of BioProfiling.jl (Vulliard et al., Bioinformatics, 2022).
        It does not split the controls, because both Gaussians are refitted on every draw and no row is spent on a fixed centroid: a group is tested against every control row it does not hold, and its null draws whole replicates, or rows without a replicate, from the two pooled.
        Rows with a missing value are left out of both fits.
        With ``covariance="robust"`` each fit is the reweighted minimum covariance determinant estimate with the defaults of R robustbase's ``covMcd``, which BioProfiling.jl calls, so the distance agrees with BioProfiling.jl's wherever the two searches reach the same subset.
        That robustness costs a partial responder: the MCD fits the most concentrated half of a side's rows, so a responding subpopulation smaller than half of the group is trimmed as outlying, where the empirical fit keeps it.
        The robust fit is also refitted on every permutation, which makes it far slower than the empirical one; the empirical fit is the default for that reason.

        ``method="mp_value"`` follows the multidimensional perturbation value of Hutz et al. (J Biomol Screen, 2013), with the covariance averaged over the group and the controls as its Methods describe; cytominer-eval's ``mp_value`` uses the controls' covariance alone.
        Like ``method="hellinger"`` it does not split the controls, and its null draws whole replicates, or rows without a replicate, from the group and the controls pooled.
        Hutz et al. first project each group and the controls onto the principal components that carry 90% of their variance; that step is left out, and ``use_rep`` sets the representation as for the other methods.
    """
    from scipy.stats import ks_2samp

    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    if covariance not in COVARIANCES:
        raise ValueError(f"covariance must be one of {COVARIANCES}, got {covariance!r}")
    if null not in NULLS:
        raise ValueError(f"null must be one of {NULLS}, got {null!r}")
    if pseudogene_func not in PSEUDOGENE_FUNCS:
        raise ValueError(f"pseudogene_func must be one of {PSEUDOGENE_FUNCS}, got {pseudogene_func!r}")
    if null == "pseudogenes" and method != "mahalanobis":
        raise ValueError(
            f"null='pseudogenes' scores a profile's distance, so it needs method='mahalanobis', got {method!r}"
        )

    is_control = reference_mask(adata, reference)
    replicate_codes = _default_replicate(adata, replicate=replicate, reference=is_control)
    if replicate_codes is None and replicate is None and _is_object_resolution(adata):
        warnings.warn(
            "replicate=None at cell resolution treats cells as independent replicates, which makes p-values too "
            "small. Pass replicate= a well column, or aggregate with mt.tl.aggregate.",
            UserWarning,
            stacklevel=3,
        )

    values = representation(adata, use_rep)
    n_control = int(is_control.sum())
    if n_control < 4:
        raise ValueError(
            f"hit calling needs at least four reference rows, and reference={reference!r} selects {n_control}; "
            "half fit the covariance and half form the null. On a consensus object the controls were probably "
            "dropped before mt.tl.consensus; keep them through consensus and drop the control row afterwards."
        )

    generator = np.random.default_rng(seed)
    control_rows = np.flatnonzero(is_control)
    control_replicates = (
        np.unique(replicate_codes[control_rows]) if replicate_codes is not None else np.empty(0, dtype=int)
    )
    refit = method in ("hellinger", "mp_value")
    if refit:
        # Both sides are refitted on every draw, so no control is spent on a fixed centroid.
        fit_rows = null_rows = control_rows
    elif replicate_codes is not None and control_replicates.size >= 4:
        fit_replicates = split_reference(control_replicates, generator)[0]
        in_fit = np.isin(replicate_codes[control_rows], fit_replicates)
        fit_rows, null_rows = control_rows[in_fit], control_rows[~in_fit]
    else:
        fit_rows, null_rows = split_reference(control_rows, generator)
        replicate_codes = None
    if not refit and fit_rows.size <= values.shape[1]:
        get_logger().warning(
            "hit_calling: covariance from %d of %d reference rows for %d features is singular",
            fit_rows.size,
            n_control,
            values.shape[1],
            deep="reduce features with mt.pp.feature_select, or score use_rep='X_pca'",
        )

    if refit and covariance == "robust":
        # The row distances use the same MCD the statistic does, rather than scikit-learn's.
        from mantispy._core._mcd import cov_mcd

        reference_rows = values[fit_rows]
        centre, scatter = cov_mcd(reference_rows[~np.isnan(reference_rows).any(axis=1)], seed=seed)
        eigenvalues, eigenvectors = np.linalg.eigh(scatter)
        whitening = eigenvectors @ np.diag(1.0 / np.sqrt(np.clip(eigenvalues, 1e-6, None))) @ eigenvectors.T
    else:
        centre, whitening = mahalanobis_transform(values[fit_rows], robust=covariance == "robust", seed=seed)
    # Fill gaps before whitening, which mixes columns; one NaN would otherwise zero the row's distance.
    to_control = np.linalg.norm(np.nan_to_num(values - centre) @ whitening, axis=1)
    fitted = np.zeros(adata.n_obs, dtype=bool)
    if not refit:
        fitted[fit_rows] = True
    held_out = is_control & ~fitted
    if refit:
        from scipy.stats import chi2

        from mantispy._core._mcd import N_SAMP, mcd_constants

        x = np.ascontiguousarray(values, dtype=np.float64)
        complete = ~np.isnan(x).any(axis=1)
        robust = covariance == "robust"
        # MCD constants by sample size, looked up inside numba where scipy is unavailable; unused for the empirical fit.
        sizes_up_to = np.arange(adata.n_obs + 1) if robust else np.zeros(1, dtype=np.int64)
        tables = mcd_constants(sizes_up_to, x.shape[1])
        n_features = x.shape[1]
        if method == "hellinger":
            # BioProfiling leaves a side with fewer than twice as many rows as features unscored.
            min_rows = min_against = 2 * n_features
        else:
            # The pooled covariance only needs the controls' to be full rank; the group needs two rows for a covariance,
            # or more rows than features for an MCD fit.
            min_rows, min_against = (n_features + 1 if robust else 2), n_features + 1
        cutoff = float(chi2.ppf(0.975, n_features))
        settings = (method == "mp_value", robust, *tables, cutoff, N_SAMP, seed, min_rows)

    if replicate_codes is not None and null == "permutation":
        # Only the null half enters the permutation pool, so it alone sets how finely the null calibrates.
        null_replicates = np.unique(replicate_codes[null_rows]).size
        if null_replicates < _REPLICATE_MIN:
            where = "the reference" if refit else "the null half"
            get_logger().warning(
                "hit_calling: the permutation test draws from only %d replicate(s) in %s, so its p-values can take only a few values",
                null_replicates,
                where,
                deep="the test is conservative, and aggregating to wells with mt.tl.aggregate gives an exact test",
            )

    codes, keys = group_codes(adata, groupby)

    if null == "pseudogenes":
        table, floors = _pseudogene_null(
            values,
            replicate_codes if replicate_codes is not None else np.arange(adata.n_obs),
            is_control,
            null_rows,
            codes,
            len(keys),
            centre,
            whitening,
            pseudogene_func,
            n_pseudogenes=n_permutations,
            seed=seed,
            match_cells=match_cells,
            cell_count=(
                pd.to_numeric(as_frame(adata.obs)[_CELL_COUNT], errors="coerce").to_numpy(dtype=float)
                if _CELL_COUNT in adata.obs
                else None
            ),
        )
        is_pseudo = table["is_pseudogene"].to_numpy()
        table.loc[~is_pseudo, "group"] = [str(key) for key in keys]
        # A group of reference rows only is part of the null, not a test.
        keep = is_pseudo | (table["n_obs"].to_numpy() > 0)
        table, floors, is_pseudo = table[keep].reset_index(drop=True), floors[keep], is_pseudo[keep]
        _warn_on_floor(floors[~is_pseudo], threshold, limited_by_draws=False)
        qvalues = np.empty(len(table))
        for family in (~is_pseudo, is_pseudo):
            qvalues[family] = benjamini_hochberg(table["pvalue"].to_numpy()[family])
        table = table.assign(qvalue=qvalues, is_hit=qvalues < threshold)[
            ["group", "n_obs", "distance", "pvalue", "qvalue", "is_hit", "is_pseudogene"]
        ]
        _write_results(adata, table, groupby, key_added, to_control, held_out, method)
        return None

    observed = np.empty(len(keys))
    sizes = np.empty(len(keys), dtype=int)
    pvalues = np.empty(len(keys))
    null_draws = np.empty((len(keys), n_permutations))
    # The smallest p-value each group's null can reach, and whether the null's distinct draws rather than n_permutations set it.
    floors = np.full(len(keys), 1.0 / (n_permutations + 1))
    draw_limited = np.zeros(len(keys), dtype=bool)
    order, offsets = group_offsets(codes, len(keys))
    for index in range(len(keys)):
        rows = order[offsets[index] : offsets[index + 1]]
        keep = ~fitted[rows]
        # The reference group's held-out rows are halved at random because rows are ordered by plate and well.
        shared = np.flatnonzero(held_out[rows])
        if replicate_codes is not None:
            keep[shared[_split_wells(replicate_codes[rows[shared]], generator)]] = False
        else:
            keep[generator.permutation(shared)[: shared.size // 2]] = False
        tested = rows[keep]
        # A row is never on both sides: the reference's sample comes out of the held-out rows.
        in_sample = np.zeros(adata.n_obs, dtype=bool)
        in_sample[tested] = True
        against_rows = null_rows[~in_sample[null_rows]]
        against = to_control[against_rows]
        sizes[index] = tested.size
        if refit:
            tested, against_rows = tested[complete[tested]], against_rows[complete[against_rows]]
            sizes[index] = tested.size
            observed[index], null_draws[index] = _hellinger_null(
                x, tested, against_rows, replicate_codes, settings, min_against, n_permutations, seed, index
            )
            continue
        observed[index] = _statistic(to_control[tested], against, method)[0]
        # The pool must include the tested rows; bootstrapping the controls alone is anti-conservative.
        pool = np.concatenate([tested, against_rows])
        if replicate_codes is not None:
            distinct = _n_draws(
                np.unique(replicate_codes[pool]).size, np.unique(replicate_codes[tested]).size, n_permutations
            )
        else:
            distinct = _n_draws(pool.size, max(tested.size, 1), n_permutations)
        if distinct <= n_permutations:
            floors[index], draw_limited[index] = 1.0 / max(distinct, 1), True
        if method == "ks":
            if replicate_codes is not None:
                null_draws[index] = _ks_replicate_null(
                    to_control, replicate_codes, pool, tested, against, n_permutations, seed, index
                )
                pvalues[index] = float(permutation_pvalue(observed[[index]], null_draws[[index]])[0])
            else:
                # A permutation null from the reference rows is too small here, since each draw is a subset of its own reference.
                pvalues[index] = float(ks_2samp(to_control[tested], against).pvalue)
                floors[index] = np.nan
            continue
        if replicate_codes is not None:
            null_draws[index] = _replicate_null(to_control, replicate_codes, pool, tested, n_permutations, seed, index)
            continue
        spread = np.random.default_rng([seed, index]).random((n_permutations, pool.size))
        draws = pool[np.argsort(spread, axis=1)[:, : max(tested.size, 1)]]
        null_draws[index] = _statistic(to_control[draws], against, method)

    if refit and np.isnan(observed).any():
        unscored = [str(key) for key, value in zip(keys, observed, strict=True) if np.isnan(value)]
        if method == "hellinger":
            reason = "a side had fewer than twice as many complete rows as features, or a singular covariance"
        else:
            reason = (
                f"the group had fewer than {min_rows} complete rows, the reference no more than features, or the "
                "averaged covariance was singular"
            )
        get_logger().warning(
            "hit_calling: %d group(s) left unscored (%s)",
            len(unscored),
            ", ".join(unscored[:5]) + (", ..." if len(unscored) > 5 else ""),
            deep=f"{reason}, so score a reduced representation with use_rep='X_pca'",
        )
    thin = int(np.sum(sizes < 3))
    if not refit and thin > len(keys) // 2:
        get_logger().warning(
            "hit_calling: %d of %d groups have fewer than three rows",
            thin,
            len(keys),
            deep="their statistic is the median of one or two wells, so aggregate more replicates or use "
            "mt.tl.map(mode='activity')",
        )

    if method != "ks":
        pvalues = permutation_pvalue(observed, null_draws)
    floors[np.isnan(pvalues)] = np.nan
    finite = np.isfinite(floors)
    _warn_on_floor(
        floors, threshold, limited_by_draws=bool(finite.any() and draw_limited[finite][np.argmin(floors[finite])])
    )
    qvalues = benjamini_hochberg(pvalues)
    table = pd.DataFrame(
        {
            "group": [str(key) for key in keys],
            "n_obs": sizes,
            "distance": observed,
            "pvalue": pvalues,
            "qvalue": qvalues,
            "is_hit": qvalues < threshold,
        }
    )
    _write_results(adata, table, groupby, key_added, to_control, held_out, method)
    return None


def _write_results(
    adata: AnnData,
    table: pd.DataFrame,
    groupby: str,
    key_added: str,
    to_control: np.ndarray,
    held_out: np.ndarray,
    method: str,
) -> None:
    """Store the table and join each row's group distance and q-value, and its own distance, onto ``obs``."""
    adata.uns.setdefault("mantispy", {})[key_added] = table
    real = table[~table["is_pseudogene"]] if "is_pseudogene" in table else table
    lookup = real.set_index("group")
    labels = as_frame(adata.obs)[groupby].astype(str)
    adata.obs[f"{key_added}_distance"] = lookup["distance"].reindex(labels).to_numpy()
    adata.obs[f"{key_added}_qvalue"] = lookup["qvalue"].reindex(labels).to_numpy()
    adata.obs[f"{key_added}_row_distance"] = to_control
    adata.obs[f"{key_added}_reference_held_out"] = held_out
    get_logger().info("hit_calling(%s) called %d of %d groups", method, int(table["is_hit"].sum()), len(table))
