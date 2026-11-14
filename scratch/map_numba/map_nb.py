"""Scratch prototype: fused numba mAP (AP per query + copied-null p-values) without materialising pairs.

Not wired into mantispy. Mirrors copairs 0.6.0 semantics for single-label queries:

* pairs: j is a positive of i if they share every ``pos_sameby`` column and differ in every ``pos_diffby`` column;
  a negative if they share every ``neg_sameby`` column and differ in every ``neg_diffby`` column. A row with a
  missing value in a column never matches on it (SQL NULL semantics; copairs raises on NaN anyway).
* similarity: cosine in float32, the products summed with numpy's pairwise-summation tree, so every similarity is
  bit-identical to ``copairs.compute.pairwise_cosine``.
* ranking: copairs sorts by ``float32(1) - sim`` and breaks ties positives-first (stable lexsort of [pos..., neg...]).
  AP only needs, per positive, the negatives ranked above it, so negatives are binned against the sorted positives
  (O(N log P)) instead of sorting the full rank list.
* copied null: per (n_pos, n_total) configuration, either numpy's exact copairs stream (identical p-values) or a
  numba Floyd sampler (O(k) per draw instead of O(n_total); same distribution, different random stream).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numba import get_num_threads, njit, prange

PW_BLOCKSIZE = 128


# ---------------------------------------------------------------------------------------------------------------------
# similarity, replicating numpy's float32 pairwise summation of x_norm * y_norm


@njit(nogil=True, inline="always")
def _pw_dot(x, y, lo, n):
    if n < 8:
        res = np.float32(0.0)
        for i in range(lo, lo + n):
            res += x[i] * y[i]
        return res
    if n <= PW_BLOCKSIZE:
        r0 = x[lo] * y[lo]
        r1 = x[lo + 1] * y[lo + 1]
        r2 = x[lo + 2] * y[lo + 2]
        r3 = x[lo + 3] * y[lo + 3]
        r4 = x[lo + 4] * y[lo + 4]
        r5 = x[lo + 5] * y[lo + 5]
        r6 = x[lo + 6] * y[lo + 6]
        r7 = x[lo + 7] * y[lo + 7]
        i = 8
        stop = n - (n % 8)
        while i < stop:
            b = lo + i
            r0 += x[b] * y[b]
            r1 += x[b + 1] * y[b + 1]
            r2 += x[b + 2] * y[b + 2]
            r3 += x[b + 3] * y[b + 3]
            r4 += x[b + 4] * y[b + 4]
            r5 += x[b + 5] * y[b + 5]
            r6 += x[b + 6] * y[b + 6]
            r7 += x[b + 7] * y[b + 7]
            i += 8
        res = ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7))
        while i < n:
            res += x[lo + i] * y[lo + i]
            i += 1
        return res
    return np.float32(0.0)  # unreachable, the caller splits


@njit(nogil=True)
def _dot_rec(x, y, lo, n):
    """``np.sum(x * y)`` in float32 over ``[lo, lo + n)``, with numpy's pairwise-summation tree."""
    if n <= PW_BLOCKSIZE:
        return _pw_dot(x, y, lo, n)
    n2 = n // 2
    n2 -= n2 % 8
    return _dot_rec(x, y, lo, n2) + _dot_rec(x, y, lo + n2, n - n2)


def normalize_rows(feats: np.ndarray) -> np.ndarray:
    """``x / np.linalg.norm(x, axis=1)[:, None]`` exactly as copairs.pairwise_cosine computes it, in float32."""
    feats = np.ascontiguousarray(feats, dtype=np.float32)
    return np.ascontiguousarray(feats / np.linalg.norm(feats, axis=1)[:, np.newaxis])


# ---------------------------------------------------------------------------------------------------------------------
# AP kernel


@njit(nogil=True)
def _bucket(codes, n_groups):
    n = codes.shape[0]
    off = np.zeros(n_groups + 1, np.int64)
    for i in range(n):
        if codes[i] >= 0:
            off[codes[i] + 1] += 1
    for g in range(n_groups):
        off[g + 1] += off[g]
    order = np.empty(off[n_groups], np.int64)
    cur = off[:n_groups].copy()
    for i in range(n):
        c = codes[i]
        if c >= 0:
            order[cur[c]] = i
            cur[c] += 1
    return order, off


@njit(nogil=True, inline="always")
def _differs(diff, i, j):
    for c in range(diff.shape[1]):
        a = diff[i, c]
        b = diff[j, c]
        if a < 0 or b < 0 or a == b:
            return False
    return True


#: Queries scored together against each candidate, so a candidate row is read once per block, not once per query.
BLOCK = 32


@njit(parallel=True, nogil=True, cache=True)
def _ap_kernel(xn, pos_same, pos_order, pos_off, pos_diff, neg_diff, q_order, items, max_pos, max_neg):
    """``items[k] = (q_lo, q_hi, c_lo, c_hi)``: queries ``q_order[q_lo:q_hi]`` share the negative candidates
    ``q_order[c_lo:c_hi]`` (their ``neg_sameby`` bucket). Positives are few and scored per query."""
    n = xn.shape[0]
    f = xn.shape[1]
    ap = np.full(n, np.nan)
    npos = np.zeros(n, np.int64)
    ntot = np.zeros(n, np.int64)
    n_items = items.shape[0]
    n_chunks = min(n_items, get_num_threads())
    for chunk in prange(n_chunks):
        pkeys = np.empty(max_pos, np.float32)
        nkeys = np.empty((BLOCK, max_neg), np.float32)
        cnt = np.zeros(BLOCK, np.int64)
        bins = np.empty(max_pos + 1, np.int64)
        one = np.float32(1.0)
        for k in range(chunk, n_items, n_chunks):
            q_lo, q_hi, c_lo, c_hi = items[k, 0], items[k, 1], items[k, 2], items[k, 3]
            nb = q_hi - q_lo
            for b in range(nb):
                cnt[b] = 0
            for t in range(c_lo, c_hi):
                j = q_order[t]
                xj = xn[j]
                for b in range(nb):
                    i = q_order[q_lo + b]
                    if j != i and _differs(neg_diff, i, j):
                        nkeys[b, cnt[b]] = one - _dot_rec(xn[i], xj, 0, f)
                        cnt[b] += 1
            for b in range(nb):
                i = q_order[q_lo + b]
                xi = xn[i]
                p = 0
                g = pos_same[i]
                if g >= 0:
                    for t in range(pos_off[g], pos_off[g + 1]):
                        j = pos_order[t]
                        if j != i and _differs(pos_diff, i, j):
                            pkeys[p] = one - _dot_rec(xi, xn[j], 0, f)
                            p += 1
                q = cnt[b]
                npos[i] = p
                ntot[i] = p + q
                if p == 0:
                    continue
                ps = pkeys[:p]
                ps.sort()
                for m in range(p + 1):
                    bins[m] = 0
                for t in range(q):
                    # ties rank positives first: a negative sits after every positive with key <= its key
                    bins[np.searchsorted(ps, nkeys[b, t], side="right")] += 1
                above = 0
                s = 0.0
                for r in range(p):
                    above += bins[r]
                    s += (r + 1) / (r + 1 + above)
                ap[i] = s / p
    return ap, npos, ntot


def _work_items(neg_codes: np.ndarray, n_groups: int) -> tuple[np.ndarray, np.ndarray]:
    """Query order (rows bucketed by negative code, rows without one last) and blocks of BLOCK queries per bucket."""
    order, off = _bucket(neg_codes, n_groups)
    lone = np.flatnonzero(neg_codes < 0)
    q_order = np.concatenate([order, lone]).astype(np.int64)
    items = []
    for g in range(n_groups):
        for lo in range(off[g], off[g + 1], BLOCK):
            items.append((lo, min(lo + BLOCK, off[g + 1]), off[g], off[g + 1]))
    for lo in range(len(order), len(q_order), BLOCK):
        items.append((lo, min(lo + BLOCK, len(q_order)), 0, 0))
    return q_order, np.array(items, dtype=np.int64).reshape(-1, 4)


def _codes(meta: pd.DataFrame, columns: list[str]) -> tuple[np.ndarray, int]:
    """One code per row for the combination of ``columns``; -1 where any is missing; all zero for no columns."""
    if not columns:
        return np.zeros(len(meta), np.int64), 1
    codes = meta.groupby(columns, observed=True, sort=False, dropna=True).ngroup().to_numpy()
    codes = np.where(np.isnan(codes.astype(float)), -1, codes).astype(np.int64)
    return codes, int(codes.max()) + 1


def _diff_codes(meta: pd.DataFrame, columns: list[str]) -> np.ndarray:
    out = np.empty((len(meta), len(columns)), np.int64)
    for c, column in enumerate(columns):
        out[:, c] = pd.factorize(meta[column], use_na_sentinel=True)[0]
    return out


def average_precision(meta, feats, pos_sameby, pos_diffby, neg_sameby, neg_diffby, xn=None) -> pd.DataFrame:
    """Per-row AP, n_pos_pairs, n_total_pairs, as copairs.map.average_precision (cosine only)."""
    meta = meta.reset_index(drop=True)
    xn = normalize_rows(feats) if xn is None else xn
    ps, n_ps = _codes(meta, list(pos_sameby))
    ns, n_ns = _codes(meta, list(neg_sameby))
    po, poff = _bucket(ps, n_ps)
    q_order, items = _work_items(ns, n_ns)
    max_pos = int(np.diff(poff).max()) if n_ps else 1
    max_neg = int((items[:, 3] - items[:, 2]).max()) if len(items) else 1
    ap, npos, ntot = _ap_kernel(
        xn, ps, po, poff, _diff_codes(meta, list(pos_diffby)), _diff_codes(meta, list(neg_diffby)), q_order, items,
        max(max_pos, 1), max(max_neg, 1),
    )  # fmt: skip
    out = meta.copy()
    out["n_pos_pairs"] = npos
    out["n_total_pairs"] = ntot
    out["average_precision"] = ap
    return out


# ---------------------------------------------------------------------------------------------------------------------
# copied null


def _conf_seed(seed: int, num_pos: int, total: int) -> int:
    return int(np.random.SeedSequence([seed, num_pos, total]).generate_state(1)[0])


def null_numpy(confs: np.ndarray, null_size: int, seed: int, n_jobs: int | None = None) -> np.ndarray:
    """copairs' exact per-configuration AP nulls (same RNG stream), threaded over configurations."""
    from concurrent.futures import ThreadPoolExecutor

    from copairs.compute import random_ap

    out = np.empty((len(confs), null_size), np.float32)

    def one(c):
        k, n = int(confs[c, 0]), int(confs[c, 1])
        out[c] = random_ap(null_size, k, n, _conf_seed(seed, k, n))

    with ThreadPoolExecutor(n_jobs) as ex:
        list(ex.map(one, range(len(confs))))
    return out


@njit(nogil=True, inline="always")
def _splitmix(state):
    state = (state + np.uint64(0x9E3779B97F4A7C15)) & np.uint64(0xFFFFFFFFFFFFFFFF)
    z = state
    z = ((z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)) & np.uint64(0xFFFFFFFFFFFFFFFF)
    z = ((z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)) & np.uint64(0xFFFFFFFFFFFFFFFF)
    return state, z ^ (z >> np.uint64(31))


@njit(parallel=True, nogil=True, cache=True)
def _null_floyd(ks, ns, seeds, null_size, max_n, max_k):
    """AP of k random positives among n, per configuration, by Floyd's k-of-n sampling (O(k) per draw)."""
    n_conf = ks.shape[0]
    out = np.empty((n_conf, null_size), np.float32)
    for c in prange(n_conf):
        k = ks[c]
        n = ns[c]
        mark = np.zeros(max_n, np.bool_)
        pos = np.empty(max_k, np.int64)
        state = np.uint64(seeds[c])
        for d in range(null_size):
            m = 0
            for j in range(n - k, n):
                state, r = _splitmix(state)
                t = np.int64(r % np.uint64(j + 1))
                if mark[t]:
                    t = j
                mark[t] = True
                pos[m] = t
                m += 1
            sel = pos[:k]
            sel.sort()
            s = 0.0
            for r_ in range(k):
                mark[sel[r_]] = False
                s += (r_ + 1) / (sel[r_] + 1)
            out[c, d] = s / k
    return out


def null_floyd(confs: np.ndarray, null_size: int, seed: int) -> np.ndarray:
    ks = confs[:, 0].astype(np.int64)
    ns = confs[:, 1].astype(np.int64)
    seeds = np.array([_conf_seed(seed, int(k), int(n)) for k, n in confs], dtype=np.uint64)
    return _null_floyd(ks, ns, seeds, null_size, int(ns.max()), int(ks.max()))


@njit(parallel=True, nogil=True, cache=True)
def _group_pvalues(null_dists, member_conf, group_off, thresholds):
    n_groups = group_off.shape[0] - 1
    null_size = null_dists.shape[1]
    out = np.empty(n_groups)
    for g in prange(n_groups):
        lo = group_off[g]
        hi = group_off[g + 1]
        m = hi - lo
        hits = 0
        for d in range(null_size):
            s = 0.0
            for t in range(lo, hi):
                s += null_dists[member_conf[t], d]
            if s / m >= thresholds[g]:
                hits += 1
        out[g] = (hits + 1) / (null_size + 1)
    return out


def tie_thresholds_f32(scores: np.ndarray) -> np.ndarray:
    s32 = np.asarray(scores, dtype=np.float32)
    s = s32.astype(np.float64)
    return s - np.maximum(1e-10, 8 * float(np.finfo(np.float32).eps) * np.abs(s))


def mean_average_precision(ap: pd.DataFrame, sameby: list[str], null_size: int, seed: int, null_kind="numpy",
                           n_jobs=None) -> pd.DataFrame:  # fmt: skip
    """Group mAP and copied-null p-values, as copairs.map.mean_average_precision(null="copied") without FDR."""
    valid = ap["average_precision"].notna() & (ap["n_pos_pairs"] > 0)
    tested = ap[valid].reset_index(drop=True)
    gcodes, n_groups = _codes(tested, sameby)
    order, off = _bucket(gcodes, n_groups)
    confs_all = tested[["n_pos_pairs", "n_total_pairs"]].to_numpy(np.int64)
    confs, inv = np.unique(confs_all, axis=0, return_inverse=True)
    inv = inv.ravel()
    sums = np.bincount(gcodes, weights=tested["average_precision"].to_numpy(), minlength=n_groups)
    scores = sums / np.diff(off)
    if null_kind == "numpy":
        nulls = null_numpy(confs, null_size, seed, n_jobs)
    else:
        nulls = null_floyd(confs, null_size, seed)
    pvals = _group_pvalues(nulls, inv[order], off, tie_thresholds_f32(scores))
    keys = tested.loc[order[off[:-1]], sameby].reset_index(drop=True)
    keys["mean_average_precision"] = scores
    keys["p_value"] = pvals
    return keys
