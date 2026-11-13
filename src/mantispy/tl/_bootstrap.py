"""Per-feature hit calling at cell level, against the null of resampling the control guides.

This is the bootstrap test of Brieflow (``workflow/lib/aggregate/bootstrap.py``), the optical pooled screen pipeline of :cite:t:`Di_Bernardo_2026`, but the null is evaluated exactly rather than by simulation, so the p-values are the ones an infinite bootstrap converges to.
"""

from __future__ import annotations

from math import exp, lgamma, log
from typing import TYPE_CHECKING

import numba
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

#: Atoms of a guide null whose probability is below this are dropped when the gene null needs
#: them. The dropped mass is a far order-statistic tail of comparable size, well under the
#: Monte-Carlo error of any feasible simulation; small problems keep every atom, so the result
#: still matches a brute-force enumeration. :func:`_gene_caps` sizes its buffer to the same band.
_ATOM_EPS = 1e-8


@njit(cache=True, nogil=True)
def _binom_cdf(k: int, n: int, p: float) -> float:
    """``P(Binomial(n, p) <= k)`` by the pmf recurrence."""
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    if p <= 0.0:
        return 1.0
    if p >= 1.0:
        return 0.0
    term = (1.0 - p) ** n
    acc = term
    r = p / (1.0 - p)
    for i in range(1, k + 1):
        term = term * r * (n - i + 1) / i
        acc += term
    return acc


@njit(cache=True, nogil=True)
def _median_gt(v: np.ndarray, lo: int, n: int, m: int, x: float) -> float:
    """``P(median of m draws with replacement from v[lo:lo+n] (sorted asc, no nan) > x)``."""
    if n == 0:
        return 0.0
    cle = np.searchsorted(v[lo : lo + n], x, side="right")  # count <= x
    f_le = cle / n
    t = m // 2
    if m % 2 == 1:
        return _binom_cdf(t, m, f_le)  # P(#{<=x} <= t)
    out = _binom_cdf(t - 1, m, f_le)  # both central order stats > x
    if cle == 0 or cle == n:
        return out
    pk = _binom_cdf(t, m, f_le) - _binom_cdf(t - 1, m, f_le)  # P(exactly t draws <= x)
    if pk <= 0.0:
        return out
    nhigh = n - cle
    contrib = 0.0
    i = lo
    end_low = lo + cle
    while i < end_low:
        a = v[i]
        j = i
        while j < end_low and v[j] == a:
            j += 1
        c_le = j - lo
        c_prev = i - lo
        pmax = (c_le / cle) ** t - (c_prev / cle) ** t  # P(max of t low draws == a)
        if pmax > 0.0:
            # first high value b with the actual midpoint (a+b)/2 > x
            left = int(end_low)
            right = int(lo + n)
            while left < right:
                mid = (left + right) // 2
                if (a + v[mid]) / 2.0 > x:
                    right = mid
                else:
                    left = mid + 1
            if left < lo + n:
                ge = n - np.searchsorted(v[lo : lo + n], v[left], side="left")  # count >= v[left]
                contrib += pmax * (ge / nhigh) ** t
        i = j
    return out + pk * contrib


@njit(cache=True, nogil=True)
def _median_lt(v: np.ndarray, lo: int, n: int, m: int, x: float) -> float:
    """``P(median of m draws from v[lo:lo+n] (sorted asc, no nan) < x)``. Mirror of :func:`_median_gt`."""
    if n == 0:
        return 0.0
    hi = np.searchsorted(v[lo : lo + n], x, side="left")  # count < x
    f_lt = hi / n
    t = m // 2
    if m % 2 == 1:
        return 1.0 - _binom_cdf(t, m, f_lt)  # P(#{<x} >= t+1)
    out = 1.0 - _binom_cdf(t, m, f_lt)  # both central order stats < x
    nlow = hi
    nge = n - hi
    if nlow == 0 or nge == 0:
        return out
    pk = _binom_cdf(t, m, f_lt) - _binom_cdf(t - 1, m, f_lt)  # P(exactly t draws < x)
    if pk <= 0.0:
        return out
    contrib = 0.0
    i = lo + hi
    end = lo + n
    while i < end:
        b = v[i]
        j = i
        while j < end and v[j] == b:
            j += 1
        c_ge = end - i
        c_gt = end - j
        pmin = (c_ge / nge) ** t - (c_gt / nge) ** t  # P(min of t high draws == b)
        if pmin > 0.0:
            # largest low value a with the actual midpoint (a+b)/2 < x
            left = int(lo)
            right = int(lo + hi)
            while left < right:
                mid = (left + right) // 2
                if (v[mid] + b) / 2.0 < x:
                    left = mid + 1
                else:
                    right = mid
            take = left - lo
            if take > 0:
                contrib += pmin * (take / nlow) ** t
        i = j
    return out + pk * contrib


@njit(cache=True, nogil=True)
def _sub_tail(v: np.ndarray, lo: int, nn: int, width: int, m: int, x: float):
    """Tail probabilities of one control guide's resampled median at threshold ``x``.

    Returns ``(P(median > x), P(median < x), P(median is nan))`` for the median of m draws from the
    guide's ``width`` cells, skipping the nan ones; ``v[lo:lo+nn]`` are its sorted non-nan values.
    """
    if nn == 0:
        return 0.0, 0.0, 1.0
    if nn == width:
        return _median_gt(v, lo, nn, m, x), _median_lt(v, lo, nn, m, x), 0.0
    p_nn = nn / width
    nan_prob = (1.0 - p_nn) ** m
    # number of non-nan draws r ~ Binomial(m, p_nn); median is over the r non-nan draws
    gt = 0.0
    lt = 0.0
    term = (1.0 - p_nn) ** m
    ratio = p_nn / (1.0 - p_nn)
    for r in range(1, m + 1):
        term = term * ratio * (m - r + 1) / r  # P(r non-nan draws)
        if term > 0.0:
            gt += term * _median_gt(v, lo, nn, r, x)
            lt += term * _median_lt(v, lo, nn, r, x)
    return gt, lt, nan_prob


@njit(parallel=True, cache=True, nogil=True)
def _guide_pvalues(
    pool_sorted: np.ndarray,
    sub_nn: np.ndarray,
    sub_start: np.ndarray,
    sub_stop: np.ndarray,
    pool_sub_start: np.ndarray,
    pool_sub_stop: np.ndarray,
    pool_width: np.ndarray,
    construct_pool: np.ndarray,
    construct_size: np.ndarray,
    construct_observed: np.ndarray,
) -> np.ndarray:
    """Two-sided p-value of every guide against its control pool, exactly."""
    n_constructs, n_features = construct_observed.shape
    out = np.empty((n_constructs, n_features), dtype=np.float64)
    for c in prange(n_constructs):
        p = construct_pool[c]
        m = construct_size[c]
        s0 = pool_sub_start[p]
        s1 = pool_sub_stop[p]
        width_total = pool_width[p]
        for j in range(n_features):
            obs = construct_observed[c, j]
            if np.isnan(obs):
                out[c, j] = np.nan
                continue
            row = pool_sorted[j]
            gt = 0.0
            lt = 0.0
            nanp = 0.0
            for sub in range(s0, s1):
                a = sub_start[sub]
                width = sub_stop[sub] - a
                nn = sub_nn[j, sub]
                w = width / width_total
                g, l_, na = _sub_tail(row, a, nn, width, m, obs)
                gt += w * g
                lt += w * l_
                nanp += w * na
            eq = 1.0 - gt - lt - nanp
            one = gt + 0.5 * eq
            # clamp tiny round-off below 0 or above 1 from the eq = 1 - gt - lt - nan subtraction
            out[c, j] = min(1.0, max(0.0, 2.0 * min(one, 1.0 - one)))
    return out


@njit(cache=True, nogil=True)
def _add_sub_atoms(v, lo, n, r, scale, eps, out_val, out_prob, cnt, sval, sfl, sgt):
    """Append the atoms of the median of r draws from ``v[lo:lo+n]`` (sorted, no nan).

    Each atom is ``(value, scale*prob)``. Prunes atoms below ``eps`` and stops each unimodal run
    once its probability drops back below it. For even r the distinct values and their tail powers
    are precomputed into the scratch arrays ``sval``/``sfl``/``sgt`` so the off-diagonal inner loop
    is a plain product, O(distinct) powers rather than O(distinct^2).
    """
    if n == 0:
        return cnt
    t = r // 2
    if r % 2 == 1:
        i = lo
        end = lo + n
        seen = False
        while i < end:
            a = v[i]
            j = i
            while j < end and v[j] == a:
                j += 1
            pr = _binom_cdf(t, r, (i - lo) / n) - _binom_cdf(t, r, (j - lo) / n)
            if pr > eps:
                if cnt >= out_val.shape[0]:
                    return cnt
                out_val[cnt] = a
                out_prob[cnt] = scale * pr
                cnt += 1
                seen = True
            elif seen:
                break
            i = j
        return cnt
    # even r = 2t: pass 1 records each distinct value, its CDF, and (1-F)^(r-t)
    log_c = lgamma(r + 1.0) - lgamma(t + 1.0) - lgamma(r - t + 1.0)
    nd = 0
    i = lo
    end = lo + n
    while i < end:
        a = v[i]
        j = i
        while j < end and v[j] == a:
            j += 1
        f = (j - lo) / n
        sval[nd] = a
        sfl[nd] = f
        sgt[nd] = (1.0 - f) ** (r - t)
        nd += 1
        i = j
    # pass 2: diagonal and off-diagonal atoms from the precomputed tables
    fp_prev = 0.0
    outer_seen = False
    for ka in range(nd):
        fp = sfl[ka] ** t
        la = fp - fp_prev
        fp_prev = fp
        f_prev = sfl[ka - 1] if ka > 0 else 0.0
        marg = _binom_cdf(t - 1, r, f_prev) - _binom_cdf(t - 1, r, sfl[ka])  # P(t-th order stat == a)
        if marg > eps:
            outer_seen = True
            diag = marg
            if la > 0.0:
                diag -= exp(log_c + log(la)) * sgt[ka]  # C(r,t)*la*(1-F)^(r-t)
            if diag > eps:
                if cnt >= out_val.shape[0]:
                    return cnt
                out_val[cnt] = sval[ka]
                out_prob[cnt] = scale * diag
                cnt += 1
            if la > 0.0:
                coef = exp(log_c + log(la))  # C(r,t)*la
                inner_seen = False
                g_prev = sgt[ka]
                for lb in range(ka + 1, nd):
                    rb = g_prev - sgt[lb]  # (1-F_{lb-1})^(r-t) - (1-F_lb)^(r-t)
                    g_prev = sgt[lb]
                    if rb > 0.0:
                        pr = coef * rb
                        if pr > eps:
                            if cnt >= out_val.shape[0]:
                                return cnt
                            out_val[cnt] = (sval[ka] + sval[lb]) / 2.0
                            out_prob[cnt] = scale * pr
                            cnt += 1
                            inner_seen = True
                        elif inner_seen:
                            break
        elif outer_seen:
            break
    return cnt


@njit(cache=True, nogil=True)
def _sort2(val, prob, lo, hi, stack):
    """In-place Hoare quicksort of ``val[lo:hi]`` carrying ``prob`` alongside.

    ``stack`` is scratch for the pending inclusive ranges; recursing on the smaller side keeps its
    depth O(log n). The pre-increment partition always leaves both sides strictly smaller.
    """
    sp = 0
    stack[sp] = lo
    stack[sp + 1] = hi - 1
    sp += 2
    while sp > 0:
        sp -= 2
        left = stack[sp]
        right = stack[sp + 1]
        while left < right:
            mid = (left + right) // 2  # median of three, moved to `left` as the pivot
            if val[mid] < val[left]:
                val[mid], val[left] = val[left], val[mid]
                prob[mid], prob[left] = prob[left], prob[mid]
            if val[right] < val[left]:
                val[right], val[left] = val[left], val[right]
                prob[right], prob[left] = prob[left], prob[right]
            if val[right] < val[mid]:
                val[right], val[mid] = val[mid], val[right]
                prob[right], prob[mid] = prob[mid], prob[right]
            val[left], val[mid] = val[mid], val[left]
            prob[left], prob[mid] = prob[mid], prob[left]
            pv = val[left]
            i = left - 1
            j = right + 1
            while True:
                i += 1
                while val[i] < pv:
                    i += 1
                j -= 1
                while val[j] > pv:
                    j -= 1
                if i >= j:
                    break
                val[i], val[j] = val[j], val[i]
                prob[i], prob[j] = prob[j], prob[i]
            # [left..j] and [j+1..right]; recurse on the smaller, loop on the larger
            if (j - left) < (right - j):
                if left < j:
                    stack[sp] = left
                    stack[sp + 1] = j
                    sp += 2
                left = j + 1
            else:
                if j + 1 < right:
                    stack[sp] = j + 1
                    stack[sp + 1] = right
                    sp += 2
                right = j


@njit(cache=True, nogil=True)
def _collapse(val_buf, prob_buf, lo, hi, stack):
    """Sort ``[lo, hi)`` by value and merge equal values in place; returns the new end."""
    if hi - lo <= 0:
        return lo
    _sort2(val_buf, prob_buf, lo, hi, stack)
    write = lo
    cur_v = val_buf[lo]
    acc = prob_buf[lo]
    for z in range(lo + 1, hi):
        if val_buf[z] == cur_v:
            acc += prob_buf[z]
        else:
            val_buf[write] = cur_v
            prob_buf[write] = acc
            write += 1
            cur_v = val_buf[z]
            acc = prob_buf[z]
    val_buf[write] = cur_v
    prob_buf[write] = acc
    write += 1
    return write


@njit(cache=True, nogil=True)
def _build_construct(
    pool_sorted_row,
    sub_nn_row,
    sub_start,
    sub_stop,
    s0,
    s1,
    width_total,
    m,
    eps,
    val_buf,
    prob_buf,
    start,
    stack,
    sval,
    sfl,
    sgt,
):
    """Build one guide null as merged, sorted atoms in ``val_buf``/``prob_buf`` from ``start``.

    Mixes every control guide's exact median-atom distribution, then sorts and merges equal values.
    Returns ``(end, nan_prob)``; the atom probabilities sum to ``1 - nan_prob``.
    """
    cnt = start
    nan_prob = 0.0
    for sub in range(s0, s1):
        a = sub_start[sub]
        width = sub_stop[sub] - a
        nn = sub_nn_row[sub]
        w = width / width_total
        if nn == 0:
            nan_prob += w
            continue
        p_nn = nn / width
        if p_nn >= 1.0:
            cnt = _add_sub_atoms(pool_sorted_row, a, nn, m, w, eps, val_buf, prob_buf, cnt, sval, sfl, sgt)
        else:
            nan_prob += w * (1.0 - p_nn) ** m
            term = (1.0 - p_nn) ** m
            ratio = p_nn / (1.0 - p_nn)
            for r in range(1, m + 1):
                term = term * ratio * (m - r + 1) / r
                if term > 0.0:
                    cnt = _add_sub_atoms(
                        pool_sorted_row, a, nn, r, w * term, eps, val_buf, prob_buf, cnt, sval, sfl, sgt
                    )
    if cnt == start:
        return start, nan_prob
    return _collapse(val_buf, prob_buf, start, cnt, stack), nan_prob


@njit(cache=True, nogil=True)
def _poisbin_cdf(j, probs, k):
    """``P(sum of k independent Bernoulli(probs[:k]) <= j)`` by DP."""
    if j < 0:
        return 0.0
    if j >= k:
        return 1.0
    dp = np.zeros(k + 1)
    dp[0] = 1.0
    size = 1
    for i in range(k):
        p = probs[i]
        for q in range(size, 0, -1):
            dp[q] = dp[q] * (1.0 - p) + dp[q - 1] * p
        dp[0] = dp[0] * (1.0 - p)
        size += 1
    s = 0.0
    for q in range(j + 1):
        s += dp[q]
    return s


@njit(cache=True, nogil=True)
def _even_side(memb, nmem, off, vals, cum, totmass, obs, s, tv, ple):
    """``P(median of the nmem listed guides > obs)`` for an even count ``2s``.

    ``ple[a]`` is the exact ``P(guide <= obs | not nan)`` (the count part is kept exact so a coarsened
    atom list cannot move probability across ``obs``); the atoms give only the boundary shape where
    exactly ``s`` guides are at most ``obs``, classified by the actual float midpoint.
    """
    f_obs_m = np.empty(nmem)  # cum_le(obs) per low member; s_obs per high member
    le_end = np.empty(nmem, dtype=np.int64)  # low: one past the last atom <= obs
    gt_start = np.empty(nmem, dtype=np.int64)  # high: first atom > obs
    ptr = np.empty(nmem, dtype=np.int64)  # sweeping pointer (low: next >= aval; high: current cut)
    mp = np.empty(nmem, dtype=np.int64)  # merge pointer for gathering sorted low atoms
    g = _poisbin_cdf(s - 1, ple, nmem)
    for lowmask in range(1 << nmem):
        pc = 0
        mm = lowmask
        while mm:
            pc += mm & 1
            mm >>= 1
        if pc != s:
            continue
        pat = 1.0
        for a in range(nmem):
            if (lowmask >> a) & 1:
                pat *= ple[a]
            else:
                pat *= 1.0 - ple[a]
        if pat <= 0.0:
            continue
        nt = 0
        for a in range(nmem):
            c = memb[a]
            lo = off[c]
            hi = off[c + 1]
            cnt_le = np.searchsorted(vals[lo:hi], obs, side="right")
            if (lowmask >> a) & 1:
                le_end[a] = lo + cnt_le
                f_obs_m[a] = cum[lo + cnt_le - 1] - (cum[lo - 1] if lo > 0 else 0.0) if cnt_le > 0 else 0.0
                ptr[a] = lo  # next atom with value >= current aval (used in the CDF sweep below)
                mp[a] = lo  # merge pointer for gathering the sorted low atoms
                nt += cnt_le
            else:
                gt_start[a] = lo + cnt_le
                f_obs_m[a] = totmass[c] - (
                    cum[lo + cnt_le - 1] - (cum[lo - 1] if lo > 0 else 0.0) if cnt_le > 0 else 0.0
                )
                ptr[a] = hi  # current cut: atoms [cut, hi) have midpoint (aval+.)/2 > obs
        if nt == 0:
            continue
        # k-way merge of the already-sorted low-guide atom slices into tv (O(nt * s), no sort)
        for pos in range(nt):
            best = -1
            bv = 0.0
            for a in range(nmem):
                if (lowmask >> a) & 1 and mp[a] < le_end[a] and (best == -1 or vals[mp[a]] < bv):
                    bv = vals[mp[a]]
                    best = a
            tv[pos] = bv
            mp[best] += 1
        bg = 0.0
        i = 0
        while i < nt:
            aval = tv[i]
            jj = i
            while jj < nt and tv[jj] == aval:
                jj += 1
            prod_le = 1.0
            prod_lt = 1.0
            prod_s = 1.0
            for a in range(nmem):
                c = memb[a]
                base = cum[off[c] - 1] if off[c] > 0 else 0.0
                if (lowmask >> a) & 1:  # low: sweep pointer to first atom >= aval, then past == aval
                    p = ptr[a]
                    while p < le_end[a] and vals[p] < aval:
                        p += 1
                    f_lt = (cum[p - 1] - base) if p > off[c] else 0.0
                    while p < le_end[a] and vals[p] == aval:
                        p += 1
                    f_le = (cum[p - 1] - base) if p > off[c] else 0.0
                    ptr[a] = p
                    prod_lt *= f_lt / f_obs_m[a]
                    prod_le *= f_le / f_obs_m[a]
                else:  # high: move cut left while the midpoint with the atom below it exceeds obs
                    cut = ptr[a]
                    while cut > gt_start[a] and (aval + vals[cut - 1]) / 2.0 > obs:
                        cut -= 1
                    ptr[a] = cut
                    hi = off[c + 1]
                    gt_mass = cum[hi - 1] - (cum[cut - 1] if cut > 0 else 0.0) if cut < hi else 0.0
                    prod_s *= gt_mass / f_obs_m[a]
            pmax = prod_le - prod_lt
            if pmax > 0.0:
                bg += pmax * prod_s
            i = jj
        g += pat * bg
    return g


@njit(cache=True, nogil=True)
def _gene_core(k_all, off, vals, cum, totmass, nanp, le_arr, lt_arr, rvals, rcum, obs, tv, memb):
    """``(P(gene median > obs), P(gene median == obs))`` over the gene's guides.

    Sums over which guides come out nan, with the odd case a Poisson-binomial count and the even
    case handled via :func:`_even_side` and reflection. ``le_arr``/``lt_arr`` are the exact
    ``P(guide <= obs)`` / ``P(guide < obs)`` (from the sub-tail mixture), so the count that decides
    significance is exact; the atoms serve only the boundary shape.
    """
    p_gt = 0.0
    p_eq = 0.0
    ple = np.empty(k_all)
    pge = np.empty(k_all)
    for mask in range(1, 1 << k_all):
        weight = 1.0
        k = 0
        for c in range(k_all):
            if (mask >> c) & 1:
                weight *= totmass[c]
                ple[k] = le_arr[c] / totmass[c]  # P(guide <= obs | not nan)
                pge[k] = (totmass[c] - lt_arr[c]) / totmass[c]  # P(guide >= obs | not nan)
                memb[k] = c
                k += 1
            else:
                weight *= nanp[c]
        if weight <= 0.0:
            continue
        if k % 2 == 1:
            s = k // 2
            g = _poisbin_cdf(s, ple, k)
            lt = _poisbin_cdf(s, pge, k)
        else:
            s = k // 2
            g = _even_side(memb, k, off, vals, cum, totmass, obs, s, tv, ple)
            lt = _even_side(memb, k, off, rvals, rcum, totmass, -obs, s, tv, pge)
        e = 1.0 - g - lt
        p_gt += weight * g
        p_eq += weight * e
    return p_gt, p_eq


@njit(parallel=True, cache=True, nogil=True)
def _gene_pvalues(
    pool_sorted,
    sub_nn,
    sub_start,
    sub_stop,
    pool_sub_start,
    pool_sub_stop,
    pool_width,
    construct_pool,
    construct_size,
    gene_offsets,
    gene_observed,
    gene_caps,
    eps,
    n_chunks,
):
    """Two-sided p-value of every gene against its guides' control pools, exactly."""
    n_genes = gene_offsets.size - 1
    n_features = gene_observed.shape[1]
    out = np.empty((n_genes, n_features), dtype=np.float64)
    max_cap = 1
    max_k = 1
    for g in range(n_genes):
        if gene_caps[g] > max_cap:
            max_cap = gene_caps[g]
        if gene_offsets[g + 1] - gene_offsets[g] > max_k:
            max_k = gene_offsets[g + 1] - gene_offsets[g]
    max_width = 1
    for sub in range(sub_start.size):
        if sub_stop[sub] - sub_start[sub] > max_width:
            max_width = sub_stop[sub] - sub_start[sub]
    # one set of scratch buffers per gene chunk, reused across its genes, so a large screen does not
    # churn the allocator with a fresh buffer per gene
    chunk = (n_genes + n_chunks - 1) // n_chunks
    for ch in prange(n_chunks):
        val_buf = np.empty(max_cap, dtype=np.float64)
        prob_buf = np.empty(max_cap, dtype=np.float64)
        rval = np.empty(max_cap, dtype=np.float64)
        cum = np.empty(max_cap, dtype=np.float64)
        rcum = np.empty(max_cap, dtype=np.float64)
        tv = np.empty(max_cap, dtype=np.float64)
        off = np.empty(max_k + 1, dtype=np.int64)
        totmass = np.empty(max_k, dtype=np.float64)
        nanp = np.empty(max_k, dtype=np.float64)
        le_arr = np.empty(max_k, dtype=np.float64)  # exact P(guide <= obs)
        lt_arr = np.empty(max_k, dtype=np.float64)  # exact P(guide < obs)
        memb = np.empty(max_k, dtype=np.int64)
        stack = np.empty(128, dtype=np.int64)
        sval = np.empty(max_width, dtype=np.float64)
        sfl = np.empty(max_width, dtype=np.float64)
        sgt = np.empty(max_width, dtype=np.float64)
        glo = ch * chunk
        ghi = min(glo + chunk, n_genes)
        for g in range(glo, ghi):
            c0 = gene_offsets[g]
            c1 = gene_offsets[g + 1]
            k_all = c1 - c0
            for j in range(n_features):
                obs = gene_observed[g, j]
                if np.isnan(obs):
                    out[g, j] = np.nan
                    continue
                row = pool_sorted[j]
                off[0] = 0
                end = 0
                for ci in range(k_all):
                    c = c0 + ci
                    p = construct_pool[c]
                    m = construct_size[c]
                    new_end, nprob = _build_construct(
                        row,
                        sub_nn[j],
                        sub_start,
                        sub_stop,
                        pool_sub_start[p],
                        pool_sub_stop[p],
                        pool_width[p],
                        m,
                        eps,
                        val_buf,
                        prob_buf,
                        end,
                        stack,
                        sval,
                        sfl,
                        sgt,
                    )
                    off[ci + 1] = new_end
                    totmass[ci] = 0.0
                    for z in range(end, new_end):
                        totmass[ci] += prob_buf[z]
                    nanp[ci] = nprob
                    end = new_end
                    # exact P(guide <= obs) / P(guide < obs) from the sub-tail mixture, so the gene
                    # count that decides significance never depends on the atom coarsening
                    le = 0.0
                    lt = 0.0
                    s0 = pool_sub_start[p]
                    s1 = pool_sub_stop[p]
                    wtot = pool_width[p]
                    for sub in range(s0, s1):
                        aa = sub_start[sub]
                        width = sub_stop[sub] - aa
                        w = width / wtot
                        gsub, lsub, nasub = _sub_tail(row, aa, sub_nn[j, sub], width, m, obs)
                        le += w * (1.0 - gsub - nasub)
                        lt += w * lsub
                    le_arr[ci] = le
                    lt_arr[ci] = lt
                # global prefix sums and reflected (negated, reversed) atoms for the < obs side
                running = 0.0
                for z in range(end):
                    running += prob_buf[z]
                    cum[z] = running
                for ci in range(k_all):
                    lo = off[ci]
                    hi = off[ci + 1]
                    nreff = hi - lo
                    for z in range(nreff):
                        rval[lo + z] = -val_buf[hi - 1 - z]
                running = 0.0
                for ci in range(k_all):
                    lo = off[ci]
                    hi = off[ci + 1]
                    nreff = hi - lo
                    for z in range(nreff):
                        running += prob_buf[hi - 1 - z]
                        rcum[lo + z] = running
                gt, eq = _gene_core(k_all, off, val_buf, cum, totmass, nanp, le_arr, lt_arr, rval, rcum, obs, tv, memb)
                one = gt + 0.5 * eq
                out[g, j] = min(1.0, max(0.0, 2.0 * min(one, 1.0 - one)))
    return out


#: Hard limit on a gene's per-thread atom buffer, so a huge screen cannot ask for unbounded memory;
#: the probability pruning keeps the real count far below this, and small problems never approach it.
_ATOM_CAP_MAX = 40_000_000


def _gene_caps(
    sub_nn, sub_start, sub_stop, pool_sub_start, pool_sub_stop, construct_pool, construct_size, gene_offsets
):
    """Per-gene atom-buffer size: an upper bound on the atoms its guides generate before merging.

    A control guide of ``w`` cells resampled ``m`` at a time produces median atoms only where the
    central order statistic's empirical CDF lies within the band ``|F - 1/2| < ~3/sqrt(m)`` in which
    its marginal exceeds ``_ATOM_EPS``; that band holds at most ``nw = min(w, 7*w/sqrt(m))`` distinct
    values, so at most ``nw^2 + nw`` even-median atoms, never more than the ``w(w+1)/2 + w`` pairs.
    """
    widths = (sub_stop - sub_start).astype(np.int64)
    full = widths * (widths + 1) // 2 + widths
    n_genes = gene_offsets.size - 1
    caps = np.empty(n_genes, dtype=np.int64)
    for g in range(n_genes):
        total = 0
        for c in range(gene_offsets[g], gene_offsets[g + 1]):
            p = construct_pool[c]
            m = max(int(construct_size[c]), 1)
            w = widths[pool_sub_start[p] : pool_sub_stop[p]]
            nw = np.minimum(w, (7 * w) // int(np.sqrt(m)) + 4)
            bound = np.minimum(full[pool_sub_start[p] : pool_sub_stop[p]], nw * nw + nw)
            total += int(bound.sum())
        caps[g] = min(total + 16, _ATOM_CAP_MAX)
    return caps


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
    key_added: str = "bootstrap",
    copy: bool = False,
) -> AnnData | None:
    """Test each feature of each gene against the null of resampling the control guides' cells.

    Each guide (each ``replicate`` within a ``groupby`` group) is summarised by the median of its cells, per feature, and each gene by the median of its guides' medians.
    The guide's null draws one control guide with probability proportional to its cell count, draws as many of that guide's cells as the tested guide has, with replacement, and takes their median; the gene's null is the median across its guides' such nulls.
    Because each draw comes from one control guide, the null carries the spread between control guides as well as the sampling error of a median.

    The null is not simulated but computed exactly: the median of a resample is an order statistic (a binomial tail in the guide's empirical distribution, and for an even cell count the mean of two consecutive order statistics), the guide null is the cell-count-weighted mixture of its control guides' such distributions, and the gene null is the order-statistic distribution of its independent guide nulls.
    The p-values are therefore the ones an infinite bootstrap converges to, with no simulation noise and no lower limit.

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
        key_added: Name for the output tables.
        copy: Return a modified copy instead of mutating in place.

    Returns:
        ``None``, or the modified copy.
        Writes ``uns["mantispy"][key_added]``, a tidy frame of ``group``, ``feature``, ``effect`` (the gene's median minus the median of its guides' control pools, in the units of ``X``), ``pvalue`` and ``qvalue`` (Benjamini-Hochberg over every gene and feature at once).
        Also writes ``uns["mantispy"][key_added + "_guides"]``, the same per guide, with ``replicate`` and ``n_cells`` columns and its own q-values.

    Raises:
        ValueError: ``control_scope`` is unknown, a scope that needs ``condition`` or ``reference_condition`` lacks it, ``"within_perturbation"`` is used without ``condition`` in ``groupby``, the controls in ``reference_condition`` are empty, or no guide has a null to draw from.

    Notes:
        The p-value is two-sided, ``2 * min(p, 1 - p)`` with ``p`` the probability that a null median exceeds the observed one.
        It is exact rather than simulated, so it has no lower floor and can be zero when the observed median lies beyond every attainable null median.
        A null median equal to the observed one counts half, so a feature where most cells share one value, such as a count that is mostly zero, is not called because its median cannot move; Brieflow counts such ties as below.
        Medians skip missing values, and a feature whose observed median is missing is left ``NaN``.

        The null resamples a control guide's own cells, so it carries that sample's error on top of the spread between control guides, while a tested guide carries only its own.
        The test therefore runs conservative when guides barely differ from one another, and near the nominal rate when they differ by more than the error of a guide's median.
        Check the rate on your own screen by grouping held-out control guides into pseudo-genes and testing them against the other control guides.
    """
    if control_scope not in CONTROL_SCOPES:
        raise ValueError(f"control_scope must be one of {CONTROL_SCOPES}, got {control_scope!r}")
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

    # Sort each control guide's cells, per feature, so the kernels can read order statistics; a guide's
    # non-nan values come first (nan sorts last) and sub_nn counts them. Lay out the pool-to-guide index.
    pool_sorted = pool_t.copy()
    n_features = pool_sorted.shape[0]
    n_subs_total = sub_start.size
    sub_nn = np.empty((n_features, n_subs_total), dtype=np.int64)
    for sub in range(n_subs_total):
        a, b = int(sub_start[sub]), int(sub_stop[sub])
        block = pool_sorted[:, a:b]
        block.sort(axis=1)
        sub_nn[:, sub] = (~np.isnan(block)).sum(axis=1)
    sub_pool_idx = np.searchsorted(pool_keys, pool_of_row[pool_rows[sub_start]])
    pool_sub_start = np.searchsorted(sub_pool_idx, np.arange(pool_keys.size), side="left").astype(np.int64)
    pool_sub_stop = np.searchsorted(sub_pool_idx, np.arange(pool_keys.size), side="right").astype(np.int64)
    pool_width = (pool_stops - pool_starts).astype(np.int64)

    genes, gene_starts = np.unique(construct_unit, return_index=True)
    gene_offsets = np.append(gene_starts, construct_unit.size).astype(np.int64)
    gene_of_construct = np.searchsorted(genes, construct_unit)
    gene_observed = grouped_stat(construct_observed, gene_of_construct, genes.size, MEDIAN)
    gene_baseline = grouped_stat(pool_median[construct_pool], gene_of_construct, genes.size, MEDIAN)

    guide_p = _guide_pvalues(
        pool_sorted,
        sub_nn,
        sub_start,
        sub_stop,
        pool_sub_start,
        pool_sub_stop,
        pool_width,
        construct_pool,
        construct_size,
        construct_observed,
    )
    gene_caps = _gene_caps(
        sub_nn, sub_start, sub_stop, pool_sub_start, pool_sub_stop, construct_pool, construct_size, gene_offsets
    )
    gene_p = _gene_pvalues(
        pool_sorted,
        sub_nn,
        sub_start,
        sub_stop,
        pool_sub_start,
        pool_sub_stop,
        pool_width,
        construct_pool,
        construct_size,
        gene_offsets,
        gene_observed,
        gene_caps,
        _ATOM_EPS,
        numba.get_num_threads(),
    )

    features = adata.var_names.to_numpy()
    n_features = features.size
    gene_labels = [str(unit_keys[unit]) for unit in genes]
    table = pd.DataFrame(
        {
            "group": np.repeat(gene_labels, n_features),
            "feature": np.tile(features, genes.size),
            "effect": (gene_observed - gene_baseline).ravel(),
            "pvalue": np.where(np.isfinite(gene_observed), gene_p, np.nan).ravel(),
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
            "pvalue": np.where(np.isfinite(construct_observed), guide_p, np.nan).ravel(),
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
