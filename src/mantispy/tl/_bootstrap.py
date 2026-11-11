"""Per-feature hit calling at cell level, against a null resampled from the control guides.

This is the bootstrap test of Brieflow (``workflow/lib/aggregate/bootstrap.py``), the optical pooled screen pipeline of :cite:t:`Di_Bernardo_2026`.
"""

from __future__ import annotations

import zlib
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from anndata import AnnData
from numba import njit

from mantispy._core._numba import MEDIAN, _kernel_input, grouped_stat
from mantispy._core._reduce import get_matrix, group_codes
from mantispy._core._stats import benjamini_hochberg
from mantispy._core.frames import as_frame
from mantispy._core.logging import get_logger
from mantispy._core.masks import reference_mask
from mantispy._core.mutation import inplace_or_copy

if TYPE_CHECKING:
    from builtins import range as prange
else:
    from numba import prange

#: Where each guide's null is drawn from; see :func:`bootstrap_features`.
CONTROL_SCOPES = ("pooled", "within_group", "reference_group", "within_perturbation")

#: Simulations per block, as in Brieflow; fixed so a gene's p-values do not depend on how many guides it has.
_BLOCK = 1000


@njit(cache=True, nogil=True)
def _median_inplace(buffer: np.ndarray, n: int) -> float:
    """Median of ``buffer[:n]`` by in-place selection, which reorders it; Brieflow's ``_median_inplace``."""
    k = n // 2
    lo = 0
    hi = n - 1
    while lo < hi:
        pivot = buffer[k]
        i = lo
        j = hi
        while i <= j:
            while buffer[i] < pivot:
                i += 1
            while pivot < buffer[j]:
                j -= 1
            if i <= j:
                tmp = buffer[i]
                buffer[i] = buffer[j]
                buffer[j] = tmp
                i += 1
                j -= 1
        if j < k:
            lo = i
        if k < i:
            hi = j
    upper = buffer[k]
    if n % 2 == 1:
        return upper
    lower = buffer[0]
    for t in range(1, k):
        if buffer[t] > lower:
            lower = buffer[t]
    return (lower + upper) / 2


@njit(cache=True, nogil=True)
def _tally(value: float, observed: float, greater: np.ndarray, equal: np.ndarray, row: int, j: int) -> None:
    if value > observed:
        greater[row, j] += 1
    elif value == observed:
        equal[row, j] += 1


@njit(parallel=True, cache=True, nogil=True)
def _bootstrap(
    pool_t: np.ndarray,
    row_sub: np.ndarray,
    sub_start: np.ndarray,
    sub_stop: np.ndarray,
    pool_start: np.ndarray,
    pool_stop: np.ndarray,
    gene_offsets: np.ndarray,
    construct_pool: np.ndarray,
    construct_size: np.ndarray,
    construct_observed: np.ndarray,
    gene_observed: np.ndarray,
    seeds: np.ndarray,
    n_sims: int,
    block: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Count, per guide and per gene, the null medians above and equal to the observed median.

    Genes run in parallel, each seeded on its own so the counts do not depend on the thread that ran it.
    Within a gene the random draws come in Brieflow's order: block by block, guide by guide, simulation by simulation.
    """
    n_features = pool_t.shape[0]
    n_constructs = construct_size.size
    n_genes = gene_offsets.size - 1
    c_greater = np.zeros((n_constructs, n_features), dtype=np.int64)
    c_equal = np.zeros((n_constructs, n_features), dtype=np.int64)
    g_greater = np.zeros((n_genes, n_features), dtype=np.int64)
    g_equal = np.zeros((n_genes, n_features), dtype=np.int64)
    largest = 1
    for c in range(n_constructs):
        largest = max(largest, construct_size[c])

    for g in prange(n_genes):
        first, last = gene_offsets[g], gene_offsets[g + 1]
        n_here = last - first
        nulls = np.empty((n_here, min(block, n_sims), n_features), dtype=np.float64)
        rows = np.empty(largest, dtype=np.int64)
        values = np.empty(largest, dtype=np.float64)
        across = np.empty(n_here, dtype=np.float64)
        np.random.seed(seeds[g])
        done = 0
        while done < n_sims:
            n_block = min(block, n_sims - done)
            for ci in range(n_here):
                c = first + ci
                pool = construct_pool[c]
                start_row = pool_start[pool]
                n_rows = pool_stop[pool] - start_row
                size = construct_size[c]
                for s in range(n_block):
                    sub = row_sub[start_row + np.random.randint(0, n_rows)]
                    start = sub_start[sub]
                    width = sub_stop[sub] - start
                    for k in range(size):
                        rows[k] = start + np.random.randint(0, width)
                    for j in range(n_features):
                        m = 0
                        for k in range(size):
                            value = pool_t[j, rows[k]]
                            if not np.isnan(value):
                                values[m] = value
                                m += 1
                        median = _median_inplace(values, m) if m else np.nan
                        nulls[ci, s, j] = median
                        _tally(median, construct_observed[c, j], c_greater, c_equal, c, j)
            for s in range(n_block):
                for j in range(n_features):
                    m = 0
                    for ci in range(n_here):
                        value = nulls[ci, s, j]
                        if not np.isnan(value):
                            across[m] = value
                            m += 1
                    median = _median_inplace(across, m) if m else np.nan
                    _tally(median, gene_observed[g, j], g_greater, g_equal, g, j)
            done += n_block
    return c_greater, c_equal, g_greater, g_equal


def _pvalues(greater: np.ndarray, equal: np.ndarray, observed: np.ndarray, n_sims: int) -> np.ndarray:
    """Two-sided p-values, ``2 * min(p, 1 - p)``, floored at ``1 / n_sims``; a tie counts half."""
    one_sided = (greater + 0.5 * equal) / n_sims
    pvalue = np.maximum(2 * np.minimum(one_sided, 1 - one_sided), 1.0 / n_sims)
    return np.where(np.isfinite(observed), pvalue, np.nan)


def _seed(key: object, seed: int) -> int:
    """Brieflow's per-gene seed, the CRC32 of the gene's name, shifted by ``seed``."""
    return (zlib.crc32(str(key).encode("utf-8")) + seed) % 2**32


@inplace_or_copy(expects="object")
def bootstrap_features(
    adata: AnnData,
    groupby: str | list[str] = "Metadata_Gene",
    replicate: str = "Metadata_sgRNA",
    reference: str | None = "negcon",
    control_scope: str = "pooled",
    condition: str | None = None,
    reference_condition: str | None = None,
    layer: str | None = None,
    n_sims: int = 100_000,
    seed: int = 0,
    key_added: str = "bootstrap",
    copy: bool = False,
) -> AnnData | None:
    """Test each feature of each gene against a null resampled from the control guides' cells.

    Each guide (each ``replicate`` within a ``groupby`` group) is summarised by the median of its cells, per feature, and each gene by the median of its guides' medians.
    One simulation of a guide's null picks a random control cell, which picks that cell's control guide, so a control guide is picked with probability proportional to its cell count.
    It then draws as many cells as the tested guide has, with replacement, from that one control guide and takes their median.
    The gene's null is, simulation by simulation, the median across its guides' nulls.
    Because each simulation draws from one control guide, the null carries the spread between control guides as well as the sampling error of a median.

    The test is Brieflow's bootstrap for optical pooled screens.
    Every feature of ``adata`` is tested, so subset ``adata`` to the features you want tested.
    The output has the layout of :func:`~mantispy.tl.differential_features`, so :func:`~mantispy.pl.feature_volcano` and :func:`~mantispy.pl.effect_sizes` read it with ``key=key_added``.
    :func:`~mantispy.tl.differential_features` fits wells and refuses cells, while this function works on cells and uses the guide as the replicate, the unit of a pooled screen in which every well holds every guide.

    Args:
        adata: Cell-level object.
        groupby: ``obs`` column, or columns, naming the gene each cell's guide targets; each group is tested.
        replicate: ``obs`` column naming the guide, the unit the null resamples.
        reference: The control cells: ``"negcon"``, or the name of a boolean ``obs`` column.
            Their guides form the null and are not tested.
            Unused under ``control_scope="within_perturbation"``.
        control_scope: Which control cells a guide's null is drawn from.
            ``"pooled"`` uses every control cell.
            ``"within_group"`` uses the control cells that share the guide's ``condition``.
            ``"reference_group"`` uses the control cells in ``reference_condition``, so every guide, whatever its condition, is tested against the controls of one condition.
            ``"within_perturbation"`` uses the cells of the same gene in ``reference_condition``, so each gene is its own control, and tests the gene's other conditions against it.
            It needs ``condition`` among the ``groupby`` columns, so that a gene in each condition is its own group.
        condition: ``obs`` column naming a treatment or other condition the scope pins the null to.
            With it set, a guide's cells in two conditions are two guides, and so are a control guide's.
        reference_condition: The value of ``condition`` that ``"reference_group"`` and ``"within_perturbation"`` draw the null from.
        layer: Read this layer instead of ``X``.
        n_sims: Simulations per guide.
            No p-value falls below ``1 / n_sims``.
        seed: Shifts every gene's seed.
            Each gene is seeded from its own name, so its p-values do not change when other genes are added or removed; ``0`` gives Brieflow's seeds.
        key_added: Name for the output tables.
        copy: Return a modified copy instead of mutating in place.

    Returns:
        ``None``, or the modified copy.
        Writes ``uns["mantispy"][key_added]``, a tidy frame of ``group``, ``feature``, ``effect`` (the gene's median minus the median of its guides' control pools, in the units of ``X``), ``pvalue`` and ``qvalue`` (Benjamini-Hochberg over every gene and feature at once).
        Also writes ``uns["mantispy"][key_added + "_guides"]``, the same per guide, with ``replicate`` and ``n_cells`` columns and its own q-values.

    Raises:
        ValueError: ``control_scope`` is unknown, a scope that needs ``condition`` or ``reference_condition`` lacks it, ``"within_perturbation"`` is used without ``condition`` in ``groupby``, the controls in ``reference_condition`` are empty, ``n_sims`` is not positive, or no guide has a null to draw from.

    Notes:
        The p-value is two-sided, ``2 * min(p, 1 - p)`` with ``p`` the share of null medians above the observed median, and is floored at ``1 / n_sims``.
        A null median equal to the observed one counts half, so a feature where most cells share one value, such as a count that is mostly zero, is not called because its median cannot move; Brieflow counts such ties as below.
        Medians skip missing values, and a feature whose observed median is missing is left ``NaN``.

        The null resamples a control guide's own cells, so it carries that sample's error on top of the spread between control guides, while a tested guide carries only its own.
        The test therefore runs conservative when guides barely differ from one another, and near the nominal rate when they differ by more than the error of a guide's median.
        Check the rate on your own screen by grouping held-out control guides into pseudo-genes and testing them against the other control guides.

        The run time grows with ``n_sims`` times the number of tested cells times the number of features.
    """
    if control_scope not in CONTROL_SCOPES:
        raise ValueError(f"control_scope must be one of {CONTROL_SCOPES}, got {control_scope!r}")
    if n_sims < 1:
        raise ValueError(f"n_sims must be positive, got {n_sims}")
    if control_scope != "pooled" and condition is None:
        raise ValueError(f"control_scope={control_scope!r} needs condition= the column to scope the controls by")
    if control_scope in ("reference_group", "within_perturbation") and reference_condition is None:
        raise ValueError(f"control_scope={control_scope!r} needs reference_condition=")
    by = [groupby] if isinstance(groupby, str) else list(groupby)
    if control_scope == "within_perturbation" and condition not in by:
        raise ValueError(
            f"control_scope='within_perturbation' needs condition={condition!r} among the groupby columns, so a gene "
            "in each condition is its own group"
        )

    obs = as_frame(adata.obs)
    logger = get_logger()
    values = _kernel_input(get_matrix(adata, layer))
    unit_codes, unit_keys = group_codes(adata, by)
    sub_codes, sub_keys = group_codes(adata, [replicate] if condition is None else [replicate, condition])

    if control_scope == "within_perturbation":
        in_reference = (obs[condition].astype(str) == str(reference_condition)).to_numpy()
        perturbation_codes = group_codes(adata, [column for column in by if column != condition] or None)[0]
        tested = ~in_reference & np.isin(perturbation_codes, perturbation_codes[in_reference])
        unreferenced = ~in_reference & ~tested
        if unreferenced.any():
            logger.warning(
                "bootstrap_features: %d group(s) have no %r arm to draw their null from and are not tested",
                len(np.unique(unit_codes[unreferenced])),
                reference_condition,
            )
        pool_of_row = np.where(in_reference, perturbation_codes, -1)
    else:
        is_control = reference_mask(adata, reference)
        tested = ~is_control
        if control_scope == "pooled":
            pool_of_row = np.where(is_control, 0, -1)
        else:
            condition_values = obs[condition].astype(str).to_numpy()
            if control_scope == "reference_group":
                in_reference = is_control & (condition_values == str(reference_condition))
                if not in_reference.any():
                    present = sorted(set(condition_values[is_control]))
                    raise ValueError(
                        f"no control cell has {condition}={reference_condition!r}; control conditions: {present}"
                    )
                pool_of_row = np.where(in_reference, 0, -1)
            else:
                condition_codes = pd.factorize(condition_values, sort=True)[0].astype(np.int64)
                pool_of_row = np.where(is_control, condition_codes, -1)

    # The pool a tested row's guide draws its null from, by the scope's key.
    if control_scope in ("pooled", "reference_group"):
        row_pool_key = np.zeros(adata.n_obs, dtype=np.int64)
    elif control_scope == "within_group":
        row_pool_key = condition_codes
    else:
        row_pool_key = perturbation_codes

    # Lay the pool cells out pool by pool, and within a pool control guide by control guide.
    n_subs = len(sub_keys)
    pool_rows = np.flatnonzero(pool_of_row >= 0)
    pool_rows = pool_rows[np.lexsort((pool_rows, sub_codes[pool_rows], pool_of_row[pool_rows]))]
    pool_keys, pool_starts = np.unique(pool_of_row[pool_rows], return_index=True)
    pool_stops = np.append(pool_starts[1:], pool_rows.size)
    pair = pool_of_row[pool_rows].astype(np.int64) * n_subs + sub_codes[pool_rows]
    sub_breaks = np.flatnonzero(np.diff(pair)) + 1
    sub_start = np.concatenate([[0], sub_breaks]).astype(np.int64)
    sub_stop = np.append(sub_breaks, pool_rows.size).astype(np.int64)
    row_sub = np.repeat(np.arange(sub_start.size, dtype=np.int64), sub_stop - sub_start)

    no_pool = tested & ~np.isin(row_pool_key, pool_keys)
    if no_pool.any():
        logger.warning(
            "bootstrap_features: %d cell(s) have no control cells in their %r scope and are not tested",
            int(no_pool.sum()),
            control_scope,
        )
    tested &= ~no_pool
    if not tested.any():
        raise ValueError("no guide has control cells in its scope to draw a null from")

    # A construct is one guide of one tested group; sorting by its id sorts the constructs by group.
    rows_kept = np.flatnonzero(tested)
    constructs, row_construct = np.unique(
        unit_codes[rows_kept].astype(np.int64) * n_subs + sub_codes[rows_kept], return_inverse=True
    )
    construct_unit, construct_sub = constructs // n_subs, constructs % n_subs
    first_row = rows_kept[np.unique(row_construct, return_index=True)[1]]
    construct_pool = np.searchsorted(pool_keys, row_pool_key[first_row]).astype(np.int64)
    construct_observed = grouped_stat(values[rows_kept], row_construct, constructs.size, MEDIAN)
    construct_size = np.bincount(row_construct, minlength=constructs.size).astype(np.int64)
    pool_t = np.ascontiguousarray(values[pool_rows].T)
    pool_median = grouped_stat(
        values[pool_rows], np.searchsorted(pool_keys, pool_of_row[pool_rows]), pool_keys.size, MEDIAN
    )

    genes, gene_starts = np.unique(construct_unit, return_index=True)
    gene_offsets = np.append(gene_starts, construct_unit.size).astype(np.int64)
    gene_of_construct = np.searchsorted(genes, construct_unit)
    gene_observed = grouped_stat(construct_observed, gene_of_construct, genes.size, MEDIAN)
    gene_baseline = grouped_stat(pool_median[construct_pool], gene_of_construct, genes.size, MEDIAN)
    seeds = np.array([_seed(unit_keys[unit], seed) for unit in genes], dtype=np.int64)

    c_greater, c_equal, g_greater, g_equal = _bootstrap(
        pool_t,
        row_sub,
        sub_start,
        sub_stop,
        pool_starts.astype(np.int64),
        pool_stops.astype(np.int64),
        gene_offsets,
        construct_pool,
        construct_size,
        construct_observed,
        gene_observed,
        seeds,
        int(n_sims),
        _BLOCK,
    )

    features = adata.var_names.to_numpy()
    n_features = features.size
    gene_labels = [str(unit_keys[unit]) for unit in genes]
    table = pd.DataFrame(
        {
            "group": np.repeat(gene_labels, n_features),
            "feature": np.tile(features, genes.size),
            "effect": (gene_observed - gene_baseline).ravel(),
            "pvalue": _pvalues(g_greater, g_equal, gene_observed, n_sims).ravel(),
        }
    )
    table["qvalue"] = benjamini_hochberg(table["pvalue"].to_numpy())

    sub_labels = [str(key[0] if isinstance(key, tuple) else key) for key in sub_keys]
    guides = pd.DataFrame(
        {
            "group": np.repeat([gene_labels[g] for g in gene_of_construct], n_features),
            "replicate": np.repeat([sub_labels[s] for s in construct_sub], n_features),
            "n_cells": np.repeat(construct_size, n_features),
            "feature": np.tile(features, constructs.size),
            "effect": (construct_observed - pool_median[construct_pool]).ravel(),
            "pvalue": _pvalues(c_greater, c_equal, construct_observed, n_sims).ravel(),
        }
    )
    if condition is not None:
        guides.insert(2, condition, np.repeat([str(sub_keys[s][1]) for s in construct_sub], n_features))
    guides["qvalue"] = benjamini_hochberg(guides["pvalue"].to_numpy())

    logger.info(
        "bootstrap_features: %d of %d gene-feature pair(s) at q < 0.05 over %d gene(s)",
        int((table["qvalue"] < 0.05).sum()),
        len(table),
        genes.size,
    )
    store = adata.uns.setdefault("mantispy", {})
    store[key_added] = table
    store[f"{key_added}_guides"] = guides
    return None
