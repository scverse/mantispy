"""Null distributions of a group's mean average precision, and the p-values against them.

Three nulls, as copairs defines them:

``copied``
    Each member's AP null is that of a random rank list with its numbers of positives and of pairs; the group's null
    averages its members' nulls, so members with equal counts share one draw. ``sampler="exact"`` draws each
    configuration's null from numpy's ``Generator`` stream copairs draws from, so the p-values equal copairs';
    ``sampler="floyd"`` draws the positions of the positives alone, which is cheaper and has the same distribution.
``joint``
    The members' APs are drawn together: a positive pair inside the group is one uniform in both members' rank
    lists. This is copairs' Philox sampler compiled, so its p-values equal copairs'.
``permutation``
    See :mod:`mantispy._core._map_permutation`.

A null mAP counts against the group when it is at least the group's mAP, within the scores' rounding.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from numba import njit

from mantispy._core._average_precision import _pairwise_sum, summation_plan
from mantispy._core._numba import N_CHUNKS

if TYPE_CHECKING:
    from builtins import range as prange
else:
    from numba import prange

# Far above float64 summation-order differences of an AP; a null value at or above ``score - tol`` is a tie.
TIE_TOL = 1e-10
# Rounding, in units of eps * |score|, allowed for scores below float64 precision.
TIE_ULPS = 8


def tie_thresholds(scores: np.ndarray) -> np.ndarray:
    """float64 ``scores - tol``: a null value at or above its threshold counts against the score."""
    scores = np.asarray(scores)
    dtype = scores.dtype if scores.dtype.kind == "f" else np.dtype(np.float64)
    scores = scores.astype(np.float64)
    return scores - np.maximum(TIE_TOL, TIE_ULPS * float(np.finfo(dtype).eps) * np.abs(scores))


def _conf_seed(seed: int, num_pos: int, total: int) -> int:
    """The seed of one configuration's copied null, a function of its counts alone."""
    return int(np.random.SeedSequence([seed, num_pos, total]).generate_state(1)[0])


def group_scores(ap: pd.DataFrame, sameby: list[str]) -> tuple[pd.DataFrame, list[np.ndarray]]:
    """Each group's mean AP and normalized AP, and its rows, over the rows with an AP and a positive pair.

    Groups and their rows come out in copairs' order: sorted keys, rows in table order, as positions in the filtered
    table.
    """
    valid = ap[ap["average_precision"].notna() & (ap["n_pos_pairs"] > 0)].reset_index(drop=True)
    grouped = valid.groupby(sameby, observed=True, sort=True)
    table = grouped[["average_precision", "normalized_average_precision"]].mean().reset_index()
    table = table.rename(
        columns={
            "average_precision": "mean_average_precision",
            "normalized_average_precision": "mean_normalized_average_precision",
        }
    )
    indices = grouped.indices
    keys = table[sameby].itertuples(index=False, name=None)
    groups = [np.asarray(indices[key if len(sameby) > 1 else key[0]], dtype=np.int64) for key in keys]
    return table, groups


# ---------------------------------------------------------------------------------------------------------------------
# copied null


@njit(nogil=True, cache=True)
def _random_ap_exact(rng, null_size, num_pos, total, out, leaf_lo, leaf_n, ops, ones32, ones64):
    """Copairs' ``random_ap``: each draw shuffles ``range(total)`` with ``rng`` and scores its first ``num_pos``.

    numba's ``Generator.shuffle`` consumes numpy's stream exactly as ``Generator.permuted`` does row by row, and the
    float32 precisions are summed with numpy's pairwise tree, so every draw is copairs' value.
    """
    buf = np.empty(total, np.int64)
    terms32 = np.empty(num_pos, np.float32)
    terms64 = np.empty(num_pos, np.float64)
    stack32 = np.empty(64, np.float32)
    stack64 = np.empty(64, np.float64)
    small = total < 2**16  # copairs indexes with uint16 then, whose ``+ 1`` stays an integer float32 divides
    for d in range(null_size):
        for t in range(total):
            buf[t] = t
        rng.shuffle(buf)
        head = buf[:num_pos]
        head.sort()
        if small:
            for r in range(num_pos):
                terms32[r] = np.float32(r + 1) / np.float32(head[r] + 1)
            s = _pairwise_sum(terms32, ones32, 0, leaf_lo, leaf_n, ops, stack32)
            out[d] = s / np.float32(num_pos)
        else:
            for r in range(num_pos):
                terms64[r] = np.float64(np.float32(r + 1)) / np.float64(head[r] + 1)
            s64 = _pairwise_sum(terms64, ones64, 0, leaf_lo, leaf_n, ops, stack64)
            out[d] = np.float32(s64 / num_pos)


def _null_exact(confs: np.ndarray, null_size: int, seed: int, n_threads: int) -> np.ndarray:
    out = np.empty((len(confs), null_size), np.float32)

    def one(c: int) -> None:
        k, total = int(confs[c, 0]), int(confs[c, 1])
        leaf_lo, leaf_n, ops = summation_plan(k)
        rng = np.random.default_rng(_conf_seed(seed, k, total))
        _random_ap_exact(rng, null_size, k, total, out[c], leaf_lo, leaf_n, ops, np.ones(k, np.float32), np.ones(k))

    with ThreadPoolExecutor(max(1, n_threads)) as pool:
        list(pool.map(one, range(len(confs))))
    return out


@njit(nogil=True, cache=True, inline="always")
def _splitmix(state):
    state = state + np.uint64(0x9E3779B97F4A7C15)
    z = state
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return state, z ^ (z >> np.uint64(31))


@njit(nogil=True, cache=True, inline="always")
def _uniform(state):
    """A double in ``[0, 1)`` with 53 random bits, and the advanced state."""
    state, z = _splitmix(state)
    return state, np.float64(z >> np.uint64(11)) * (1.0 / 9007199254740992.0)


@njit(parallel=True, nogil=True, cache=True)
def _null_floyd_kernel(ks, totals, seeds, null_size, max_total, max_k, n_chunks):
    """AP of ``k`` random positives among ``total``, by Floyd's ``k``-of-``total`` sampling: O(k) per draw."""
    n_conf = ks.shape[0]
    out = np.empty((n_conf, null_size), np.float32)
    for chunk in prange(n_chunks):
        mark = np.zeros(max_total, np.bool_)
        pos = np.empty(max_k, np.int64)
        for c in range(chunk, n_conf, n_chunks):
            k = ks[c]
            total = totals[c]
            state = np.uint64(seeds[c])
            for d in range(null_size):
                m = 0
                for j in range(total - k, total):
                    state, u = _uniform(state)
                    t = min(np.int64(u * (j + 1)), j)
                    if mark[t]:
                        t = j
                    mark[t] = True
                    pos[m] = t
                    m += 1
                head = pos[:k]
                head.sort()
                s = 0.0
                for r in range(k):
                    mark[head[r]] = False
                    s += (r + 1) / (head[r] + 1)
                out[c, d] = s / k
    return out


def _null_floyd(confs: np.ndarray, null_size: int, seed: int) -> np.ndarray:
    ks = confs[:, 0].astype(np.int64)
    totals = confs[:, 1].astype(np.int64)
    seeds = np.array([_conf_seed(seed, int(k), int(n)) for k, n in confs], dtype=np.uint64)
    return _null_floyd_kernel(
        ks, totals, seeds, null_size, int(totals.max(initial=1)), int(ks.max(initial=1)), min(N_CHUNKS, len(ks))
    )


@njit(parallel=True, nogil=True, cache=True)
def _copied_counts(null_dists, member_conf, group_ptr, thresholds, n_chunks):
    """Null draws of each group's mean of member nulls at or above its threshold."""
    n_groups = group_ptr.shape[0] - 1
    null_size = null_dists.shape[1]
    hits = np.zeros(n_groups, np.int64)
    for chunk in prange(n_chunks):
        for g in range(chunk, n_groups, n_chunks):
            lo = group_ptr[g]
            hi = group_ptr[g + 1]
            m = hi - lo
            for d in range(null_size):
                s = 0.0
                for t in range(lo, hi):
                    s += null_dists[member_conf[t], d]
                if s / m >= thresholds[g]:
                    hits[g] += 1
    return hits


def copied_pvalues(
    scores: np.ndarray,
    group_ptr: np.ndarray,
    n_pos: np.ndarray,
    n_total: np.ndarray,
    null_size: int,
    seed: int,
    sampler: str = "exact",
    n_threads: int = 1,
) -> np.ndarray:
    """P-values of the groups' mAPs (members ``group_ptr[g]:group_ptr[g + 1]`` of the count arrays) under the copied null."""
    confs, member_conf = np.unique(np.stack([n_pos, n_total], axis=1).astype(np.int64), axis=0, return_inverse=True)
    if sampler == "exact":
        nulls = _null_exact(confs, null_size, seed, n_threads)
    else:
        nulls = _null_floyd(confs, null_size, seed)
    thresholds = tie_thresholds(np.asarray(scores, dtype=np.float32))
    hits = _copied_counts(
        nulls, member_conf.ravel().astype(np.int64), group_ptr, thresholds, min(N_CHUNKS, len(scores))
    )
    return (hits + 1) / (null_size + 1)
