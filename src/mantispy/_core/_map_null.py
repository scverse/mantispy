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

import hashlib
import math
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from numba import njit

from mantispy._core._average_precision import APResult, _pairwise_sum, summation_plan
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


def group_scores(ap: pd.DataFrame, sameby: list[str]) -> tuple[pd.DataFrame, list[np.ndarray], pd.DataFrame]:
    """Each group's mean AP and normalized AP, and its rows, over the rows with an AP and a positive pair.

    Returns the group table, each group's rows as positions in the tested rows, and the tested rows, whose ``index``
    keeps ``ap``'s (the AP row, when ``ap`` is a filtered :func:`~mantispy._core._average_precision.ap_frame`).
    Groups and rows come out in copairs' order: sorted keys, rows in table order.
    """
    tested = ap[ap["average_precision"].notna() & (ap["n_pos_pairs"] > 0)]
    valid = tested.reset_index(drop=True)
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
    return table, groups, tested


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


# ---------------------------------------------------------------------------------------------------------------------
# joint null: copairs' Philox4x32-10 sampler (copairs.nulls.joint), compiled


_M32 = np.uint64(0xFFFFFFFF)
_S32 = np.uint64(32)
_INV_2_53 = 1.0 / 9007199254740992.0
#: Below this mean, binomials are drawn by inversion; above it by BTRS.
_INVERSION_MEAN = 10.0
#: log(k!) for small k; larger k use Stirling's series.
LOG_FACTORIAL = np.array([math.lgamma(k + 1.0) for k in range(256)])
#: Coefficients ``1 / (2k + 1)`` of ``log(m) = 2 atanh(s)``.
LOG_SERIES = np.array([1.0 / (2 * k + 1) for k in range(12)])
_HALF_LOG_2PI = 0.5 * math.log(2 * math.pi)
_SQRT_HALF = math.sqrt(0.5)
_LN2 = math.log(2.0)
#: A member sorts up to this many positives by insertion, more by quicksort.
_INSERTION = 32
#: Draws per pass of the joint null; bounds the samples held at once.
_JOINT_CHUNK = 1024


@njit(nogil=True, cache=True)
def _philox(c0, c1, c2, c3, k0, k1):
    m0 = np.uint64(0xD2511F53)
    m1 = np.uint64(0xCD9E8D57)
    w0 = np.uint64(0x9E3779B9)
    w1 = np.uint64(0xBB67AE85)
    mask = np.uint64(0xFFFFFFFF)
    s32 = np.uint64(32)
    for r in range(10):
        if r > 0:
            k0 = (k0 + w0) & mask
            k1 = (k1 + w1) & mask
        p0 = m0 * c0
        p1 = m1 * c2
        c0, c1, c2, c3 = (p1 >> s32) ^ c1 ^ k0, p1 & mask, (p0 >> s32) ^ c3 ^ k1, p0 & mask
    return c0, c1, c2, c3


@njit(nogil=True, cache=True, inline="always")
def _uniform53(a, b):
    return np.float64(((a >> np.uint64(5)) << np.uint64(26)) | (b >> np.uint64(6))) * (1.0 / 9007199254740992.0)


@njit(nogil=True, cache=True)
def _uniform_pair(c0, c1, block, stream, k0, k1):
    w0, w1, w2, w3 = _philox(c0, c1, np.uint64(block), np.uint64(stream), k0, k1)
    return _uniform53(w0, w1), _uniform53(w2, w3)


@njit(nogil=True, cache=True)
def _log(x, series):
    """Natural log of ``x >= 0`` from a fixed series, as copairs computes it (not libm)."""
    if x == 0.0:
        return -np.inf
    m, e = math.frexp(x)
    if m < 0.7071067811865476:
        m *= 2.0
        e -= 1
    s = (m - 1.0) / (m + 1.0)
    z = s * s
    p = series[11]
    for k in range(10, -1, -1):
        p = p * z + series[k]
    return e * 0.6931471805599453 + 2.0 * s * p


@njit(nogil=True, cache=True)
def _powi(x, n):
    r = 1.0
    while n > 0:
        if n & 1:
            r *= x
        x *= x
        n >>= 1
    return r


@njit(nogil=True, cache=True)
def _log_factorial(k, table, series):
    if k < 256:
        return table[int(k)]
    x = k + 1.0
    r = 1.0 / (x * x)
    return (x - 0.5) * _log(x, series) - x + 0.9189385332046727 + (1.0 / 12.0 - r * (1.0 / 360.0 - r / 1260.0)) / x


@njit(nogil=True, cache=True)
def _binomial(n, p, c0, c1, block, stream, k0, k1, table, series):
    """Binomial(``n``, ``p``) from the stream's blocks starting at ``block``: ``(draw, next_block)``."""
    if n <= 0 or p <= 0.0:
        return 0, block
    if p >= 1.0:
        return n, block
    flip = p > 0.5
    q = 1.0 - p if flip else p
    if n * q < 10.0:
        u, _ = _uniform_pair(c0, c1, block, stream, k0, k1)
        block += 1
        s = q / (1.0 - q)
        a = (n + 1) * s
        r = _powi(1.0 - q, n)
        x = 0
        while u > r and x < n:
            u -= r
            x += 1
            r *= a / x - s
    else:
        spq = math.sqrt(n * q * (1.0 - q))
        b = 1.15 + 2.53 * spq
        a = -0.0873 + 0.0248 * b + 0.01 * q
        c = n * q + 0.5
        v_r = 0.92 - 4.2 / b
        alpha = (2.83 + 5.1 / b) * spq
        lpq = _log(q / (1.0 - q), series)
        m = math.floor((n + 1) * q)
        h = _log_factorial(m, table, series) + _log_factorial(n - m, table, series)
        while True:
            u, v = _uniform_pair(c0, c1, block, stream, k0, k1)
            block += 1
            u -= 0.5
            us = 0.5 - abs(u)
            if us <= 0.0:
                continue
            kf = math.floor((2.0 * a / us + b) * u + c)
            if kf < 0.0 or kf > n:
                continue
            if us >= 0.07 and v <= v_r:
                break
            v = _log(v * alpha / (a / (us * us) + b), series)
            if v <= h - _log_factorial(kf, table, series) - _log_factorial(n - kf, table, series) + (kf - m) * lpq:
                break
        x = int(kf)
    return (n - x if flip else x), block


@njit(nogil=True, cache=True)
def _sort_head(vals, n):
    if n > 32:
        vals[:n].sort()
        return
    for i in range(1, n):
        v = vals[i]
        k = i - 1
        while k >= 0 and vals[k] > v:
            vals[k + 1] = vals[k]
            k -= 1
        vals[k + 1] = v


@njit(nogil=True, cache=True)
def _member_ap(vals, num_pos, num_neg, c0, c1, stream, k0, k1, table, series):
    _sort_head(vals, num_pos)
    remaining = num_neg
    prev = 1.0
    above = 0
    acc = 0.0
    block = 0
    for t in range(num_pos):
        v = vals[num_pos - 1 - t]
        if remaining > 0 and prev > 0.0:
            x, block = _binomial(remaining, (prev - v) / prev, c0, c1, block, stream, k0, k1, table, series)
            above += x
            remaining -= x
        prev = v
        acc += (t + 1) / (t + 1 + above)
    return acc / num_pos


@njit(nogil=True, cache=True)
def _joint_sample(s, j, mem_ptr, num_pos, num_neg, edge_ptr, inc_ptr, inc_edge, k0, k1, ev, vals, table, series):
    """The mAP of draw ``j`` of signature ``s``."""
    c0 = np.uint64(j) & np.uint64(0xFFFFFFFF)
    c1 = np.uint64(j) >> np.uint64(32)
    key0, key1 = k0[s], k1[s]
    n_edge = edge_ptr[s + 1] - edge_ptr[s]
    for e in range(0, n_edge, 2):
        u0, u1 = _uniform_pair(c0, c1, e // 2, 0, key0, key1)
        ev[e] = u0
        if e + 1 < n_edge:
            ev[e + 1] = u1
    m0, m1 = mem_ptr[s], mem_ptr[s + 1]
    total = 0.0
    for i in range(m0, m1):
        p = num_pos[i]
        shared = inc_ptr[i + 1] - inc_ptr[i]
        for t in range(shared):
            vals[t] = ev[inc_edge[inc_ptr[i] + t]]
        stream = 1 + 2 * (i - m0)
        for t in range(shared, p, 2):
            u0, u1 = _uniform_pair(c0, c1, (t - shared) // 2, stream, key0, key1)
            vals[t] = u0
            if t + 1 < p:
                vals[t + 1] = u1
        total += _member_ap(vals, p, num_neg[i], c0, c1, stream + 1, key0, key1, table, series)
    return total / (m1 - m0)


@njit(parallel=True, nogil=True, cache=True)
def _joint_counts(
    plan_mem_ptr, num_pos, num_neg, edge_ptr, inc_ptr, inc_edge, k0, k1, group_sig, sig_group_ptr, sig_groups,
    thresholds, null_size, max_edges, max_pos, table, series, n_chunks,
):  # fmt: skip
    """Null draws of each group's signature at or above the group's threshold, draws split over the threads."""
    n_sig = k0.shape[0]
    n_groups = group_sig.shape[0]
    hits = np.zeros((n_chunks, n_groups), np.int64)
    for chunk in prange(n_chunks):
        ev = np.empty(max(max_edges, 1), np.float64)
        vals = np.empty(max(max_pos, 1), np.float64)
        lo = null_size * chunk // n_chunks
        hi = null_size * (chunk + 1) // n_chunks
        for s in range(n_sig):
            g_lo = sig_group_ptr[s]
            g_hi = sig_group_ptr[s + 1]
            for j in range(lo, hi):
                value = _joint_sample(
                    s, j, plan_mem_ptr, num_pos, num_neg, edge_ptr, inc_ptr, inc_edge, k0, k1, ev, vals, table, series
                )
                for t in range(g_lo, g_hi):
                    g = sig_groups[t]
                    if value >= thresholds[g]:
                        hits[chunk, g] += 1
    return hits.sum(axis=0)


def canonical_signature(members, edges):
    """``(members, edges)`` with members sorted when every pair is an edge, as copairs keys a joint null."""
    members = tuple((int(a), int(b)) for a, b in members)
    edges = tuple(sorted({(min(int(i), int(j)), max(int(i), int(j))) for i, j in edges}))
    n = len(members)
    if len(edges) == n * (n - 1) // 2:
        members = tuple(sorted(members))
    return members, edges


def _stream_key(seed: int, obj) -> tuple[np.uint64, np.uint64]:
    """Philox key of ``(seed, obj)``, from a hash of its representation, as copairs derives it."""
    digest = hashlib.blake2b(repr((int(seed), obj)).encode(), digest_size=8)
    word = int.from_bytes(digest.digest(), "little")
    return np.uint64(word & 0xFFFFFFFF), np.uint64(word >> 32)


def joint_pvalues(
    scores: np.ndarray,
    groups: list[np.ndarray],
    n_pos: np.ndarray,
    n_neg: np.ndarray,
    edges: list[np.ndarray],
    null_size: int,
    seed: int,
) -> np.ndarray:
    """P-values of the groups' mAPs under the joint null.

    ``groups[g]`` lists the group's AP rows, ``edges[g]`` its positive pairs between them as local member indices.
    Groups with equal signatures share draws, as in copairs.
    """
    signatures = [
        canonical_signature(zip(n_pos[ix], n_neg[ix], strict=True), e) for ix, e in zip(groups, edges, strict=True)
    ]
    unique = sorted(set(signatures))
    index = {sig: k for k, sig in enumerate(unique)}
    group_sig = np.array([index[s] for s in signatures], dtype=np.int64)
    mem_ptr, edge_ptr, inc_ptr = [0], [0], [0]
    num_pos, num_neg, inc_edge = [], [], []
    k0 = np.empty(len(unique), np.uint64)
    k1 = np.empty(len(unique), np.uint64)
    for s, (members, sig_edges) in enumerate(unique):
        incident: list[list[int]] = [[] for _ in members]
        for e, (i, j) in enumerate(sig_edges):
            incident[i].append(e)
            incident[j].append(e)
        for i, (p, m) in enumerate(members):
            num_pos.append(p)
            num_neg.append(m)
            inc_edge.extend(incident[i])
            inc_ptr.append(len(inc_edge))
        mem_ptr.append(len(num_pos))
        edge_ptr.append(edge_ptr[-1] + len(sig_edges))
        k0[s], k1[s] = _stream_key(seed, (members, sig_edges))
    order = np.argsort(group_sig, kind="stable")
    sig_group_ptr = np.searchsorted(group_sig[order], np.arange(len(unique) + 1)).astype(np.int64)
    as64 = lambda a: np.asarray(a, dtype=np.int64)
    n_chunks = max(1, min(N_CHUNKS, null_size))
    hits = _joint_counts(
        as64(mem_ptr), as64(num_pos), as64(num_neg), as64(edge_ptr), as64(inc_ptr), as64(inc_edge), k0, k1, group_sig,
        sig_group_ptr, order.astype(np.int64), tie_thresholds(np.asarray(scores, dtype=np.float64)), null_size,
        int(np.diff(edge_ptr).max(initial=1)), int(max(num_pos, default=1)), LOG_FACTORIAL, LOG_SERIES, n_chunks,
    )  # fmt: skip
    return (hits + 1) / (null_size + 1)


def group_edges(groups: list[np.ndarray], rows: np.ndarray, result: APResult) -> list[np.ndarray]:
    """Each group's positive pairs between its members, as local member indices.

    ``rows[k]`` is the AP row of position ``k`` of the tested table; a positive pair joins two AP rows of one label.
    """
    n_rows = len(result.row_profile)
    n_labels = int(result.row_label.max(initial=0)) + 1
    keys = result.row_profile * n_labels + result.row_label
    owner = np.repeat(np.arange(n_rows), np.diff(result.pos_ptr))
    other = np.searchsorted(keys, result.partner * n_labels + result.row_label[owner])
    group_of = np.full(n_rows, -1, np.int64)
    local = np.zeros(n_rows, np.int64)
    for g, ix in enumerate(groups):
        group_of[rows[ix]] = g
        local[rows[ix]] = np.arange(len(ix))
    other = np.minimum(other, n_rows - 1)
    inside = (
        (group_of[owner] >= 0)
        & (group_of[owner] == group_of[other])
        & (keys[other] == result.partner * n_labels + result.row_label[owner])
    )
    inside &= local[owner] < local[other]
    a, b = owner[inside], other[inside]
    order = np.argsort(group_of[a], kind="stable")
    ptr = np.searchsorted(group_of[a][order], np.arange(len(groups) + 1))
    pairs = np.stack([local[a], local[b]], axis=1)[order]
    return [pairs[ptr[g] : ptr[g + 1]] for g in range(len(groups))]
