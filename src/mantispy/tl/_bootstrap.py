"""Per-feature hit calling at cell level, against the null of resampling the control guides.

This is the bootstrap test of Brieflow (``workflow/lib/aggregate/bootstrap.py``), the optical pooled screen pipeline of :cite:t:`Di_Bernardo_2026`, but the null is evaluated exactly rather than by simulation, so the p-values are the ones an infinite bootstrap converges to.
"""

from __future__ import annotations

from math import exp, expm1, lgamma, log, log1p
from typing import TYPE_CHECKING

import numba
import numpy as np
import pandas as pd
from anndata import AnnData
from numba import njit

from mantispy._core._numba import MEDIAN, N_CHUNKS, _kernel_input, _median_select, grouped_stat
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
#: still matches a brute-force enumeration. :func:`_null_caps` sizes its buffer to the same band.
_ATOM_EPS = 1e-8


@njit(cache=True, nogil=True)
def _binom_pmf(k: int, n: int, p: float) -> float:
    """``P(Binomial(n, p) == k)`` for ``0 < p < 1``, in log space so it does not underflow for large ``n``."""
    return exp(lgamma(n + 1.0) - lgamma(k + 1.0) - lgamma(n - k + 1.0) + k * log(p) + (n - k) * log1p(-p))


@njit(cache=True, nogil=True)
def _binom_cdf(k: int, n: int, p: float) -> float:
    """``P(Binomial(n, p) <= k)`` by the pmf recurrence from ``pmf(0) = (1 - p) ** n``."""
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    if p <= 0.0:
        return 1.0
    if p >= 1.0:
        return 0.0
    term = (1.0 - p) ** n
    if term < 1e-280:  # pmf(0) underflows, or is a subnormal with too few digits, past ~1000 draws
        return _binom_cdf_from_k(k, n, p)
    acc = term
    r = p / (1.0 - p)
    for i in range(1, k + 1):
        term = term * r * (n - i + 1) / i
        acc += term
    return acc


@njit(cache=True, nogil=True)
def _binom_cdf_from_k(k: int, n: int, p: float) -> float:
    """:func:`_binom_cdf` by the recurrence from ``pmf(k)`` (or ``pmf(k + 1)``, for the upper tail) outward."""
    if k < (n + 1) * p:  # lower tail: pmf decreases from k down to 0
        term = _binom_pmf(k, n, p)
        acc = term
        q = (1.0 - p) / p
        for i in range(k, 0, -1):
            term *= i * q / (n - i + 1)  # pmf(i - 1) from pmf(i)
            acc += term
            if term <= acc * 1e-17:
                break
        return min(acc, 1.0)
    term = _binom_pmf(k + 1, n, p)  # upper tail: pmf decreases from k + 1 up to n
    acc = term
    r = p / (1.0 - p)
    for i in range(k + 1, n):
        term *= (n - i) * r / (i + 1)  # pmf(i + 1) from pmf(i)
        acc += term
        if term <= acc * 1e-17:
            break
    return max(1.0 - acc, 0.0)


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
    for r in range(1, m + 1):
        term = _binom_pmf(r, m, p_nn)  # P(r non-nan draws)
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
    pool_subs: np.ndarray,
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
            for si in range(s0, s1):
                sub = pool_subs[si]
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
    # even r = 2t: pass 1 records each distinct value, its CDF, and (1-F)^(r-t). Over the band of
    # values that carry atoms these powers and F^t stay normal floats up to a thousand draws; past
    # that they underflow while C(r, t) overflows, so the atoms are formed in log space instead.
    direct = r <= 1000
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
        if direct:
            sgt[nd] = (1.0 - f) ** (r - t)
        else:
            sgt[nd] = (r - t) * log1p(-f) if f < 1.0 else -np.inf
        nd += 1
        i = j
    if not direct:
        # sgt[lb] becomes log rb[lb], rb[lb] = (1-F_{lb-1})^(r-t) - (1-F_lb)^(r-t), then rb[lb] / rb[lb-1]
        for lb in range(nd - 1, 0, -1):
            sgt[lb] = sgt[lb - 1] + log(-expm1(sgt[lb] - sgt[lb - 1]))
        for lb in range(nd - 1, 1, -1):
            sgt[lb] = exp(sgt[lb] - sgt[lb - 1])
    # pass 2: diagonal and off-diagonal atoms from the precomputed tables
    outer_seen = False
    for ka in range(nd):
        f_prev = sfl[ka - 1] if ka > 0 else 0.0
        marg = _binom_cdf(t - 1, r, f_prev) - _binom_cdf(t - 1, r, sfl[ka])  # P(t-th order stat == a)
        if marg > eps:
            outer_seen = True
            # coef = C(r,t) * (F^t - F_prev^t); the diagonal takes off coef * (1-F)^(r-t), and the row's
            # off-diagonal atoms are coef * rb[lb]
            pr = 0.0
            coef = 0.0
            if direct:
                la = sfl[ka] ** t - f_prev**t
                coef = exp(log_c + log(la)) if la > 0.0 else 0.0
                diag = marg - coef * sgt[ka]
                if ka + 1 < nd:
                    pr = coef * (sgt[ka] - sgt[ka + 1])
            else:
                log_coef = log_c + t * log(sfl[ka])
                if f_prev > 0.0:
                    log_coef += log(-expm1(t * log(f_prev / sfl[ka])))
                log_g = (r - t) * log1p(-sfl[ka]) if sfl[ka] < 1.0 else -np.inf
                diag = marg - exp(log_coef + log_g)
                if ka + 1 < nd:
                    log_g_next = (r - t) * log1p(-sfl[ka + 1]) if sfl[ka + 1] < 1.0 else -np.inf
                    pr = exp(log_coef + log_g + log(-expm1(log_g_next - log_g)))
            if diag > eps:
                if cnt >= out_val.shape[0]:
                    return cnt
                out_val[cnt] = sval[ka]
                out_prob[cnt] = scale * diag
                cnt += 1
            inner_seen = False
            for lb in range(ka + 1, nd):
                if lb > ka + 1:
                    pr = coef * (sgt[lb - 1] - sgt[lb]) if direct else pr * sgt[lb]
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
    pool_subs,
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

    Mixes the exact median-atom distributions of the control guides ``pool_subs[s0:s1]``, then sorts and merges equal values.
    Returns ``(end, nan_prob)``; the atom probabilities sum to ``1 - nan_prob``.
    """
    cnt = start
    nan_prob = 0.0
    for si in range(s0, s1):
        sub = pool_subs[si]
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
            for r in range(1, m + 1):
                term = _binom_pmf(r, m, p_nn)
                if term > 0.0:
                    cnt = _add_sub_atoms(
                        pool_sorted_row, a, nn, r, w * term, eps, val_buf, prob_buf, cnt, sval, sfl, sgt
                    )
    if cnt == start:
        return start, nan_prob
    return _collapse(val_buf, prob_buf, start, cnt, stack), nan_prob


@njit(cache=True, nogil=True)
def _poisbin_cdf(j, probs, k, dp):
    """``P(sum of k independent Bernoulli(probs[:k]) <= j)`` by DP, in the scratch ``dp`` (size > k)."""
    if j < 0:
        return 0.0
    if j >= k:
        return 1.0
    dp[: k + 1] = 0.0
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
def _even_side(memb, nmem, lo_c, hi_c, vals, cum, obs, s, ple, fs, ints, pats, patw, bg, dp):
    """``P(median of the nmem listed guides > obs)`` for an even count ``2s``.

    Guide ``c``'s atoms are ``vals[lo_c[c]:hi_c[c]]``, sorted and distinct, with ``cum`` their running
    mass from that list's start. ``ple[a]`` is the exact ``P(guide <= obs | not nan)`` (the count part
    is kept exact so a coarsened atom list cannot move probability across ``obs``); the atoms give only
    the boundary shape where exactly ``s`` guides are at most ``obs`` (the low ones), classified by the
    actual float midpoint of the largest low and the smallest high guide. One ascending sweep over all
    the guides' atoms at most ``obs`` serves every split into low and high guides at once, so each
    atom list is read once. ``fs``/``ints`` are float/int scratch rows, ``pats``/``patw``/``bg`` hold
    one entry per split.
    """
    f_low, f_high, rle, rlt, rh, inv_low, inv_high, top = fs[0], fs[1], fs[2], fs[3], fs[4], fs[5], fs[6], fs[7]
    le_end, gt_start, lptr, cut = ints[0], ints[1], ints[2], ints[3]
    g = _poisbin_cdf(s - 1, ple, nmem, dp)
    for a in range(nmem):
        c = memb[a]
        lo = lo_c[c]
        hi = hi_c[c]
        cnt_le = np.searchsorted(vals[lo:hi], obs, side="right")
        at_obs = cum[lo + cnt_le - 1] if cnt_le > 0 else 0.0
        f_low[a] = at_obs  # mass at most obs, for the guide as a low one
        top[a] = cum[hi - 1] if hi > lo else 0.0
        f_high[a] = top[a] - at_obs  # mass above obs, as a high one
        inv_low[a] = 1.0 / at_obs if at_obs > 0.0 else 0.0
        inv_high[a] = 1.0 / f_high[a] if f_high[a] > 0.0 else 0.0
        rle[a] = 0.0
        rh[a] = 0.0
        le_end[a] = lo + cnt_le  # one past the last atom <= obs
        gt_start[a] = lo + cnt_le  # first atom > obs
        lptr[a] = lo  # next atom <= obs not yet swept
        cut[a] = hi  # atoms [cut, hi) have midpoint (aval + .)/2 > obs
    # the splits with s low guides; a guide with no atom on its side of obs gives the boundary no
    # shape, and its exact probability of being there is below the pruned atoms', so such a split,
    # at most that probable, is left out
    npat = 0
    for lowmask in range(1 << nmem):
        pc = 0
        mm = lowmask
        while mm:
            pc += mm & 1
            mm >>= 1
        if pc != s:
            continue
        pat = 1.0
        shapeless = False
        for a in range(nmem):
            if (lowmask >> a) & 1:
                pat *= ple[a]
                shapeless |= f_low[a] <= 0.0
            else:
                pat *= 1.0 - ple[a]
                shapeless |= f_high[a] <= 0.0
        if pat <= 0.0 or shapeless:
            continue
        pats[npat] = lowmask
        patw[npat] = pat
        bg[npat] = 0.0
        npat += 1
    while npat > 0:
        aval = np.inf
        for a in range(nmem):
            if lptr[a] < le_end[a] and vals[lptr[a]] < aval:
                aval = vals[lptr[a]]
        if aval == np.inf:
            break
        hit = 0  # the guides with an atom at aval
        for a in range(nmem):
            p = lptr[a]
            if p < le_end[a] and vals[p] == aval:
                hit |= 1 << a
                lo = lo_c[memb[a]]
                rlt[a] = (cum[p - 1] if p > lo else 0.0) * inv_low[a]
                rle[a] = cum[p] * inv_low[a]
                lptr[a] = p + 1
        for a in range(nmem):
            if (hit >> a) & 1:
                # a guide at its own atom is low in every split scored here, unless another guide ties
                # with it; its cut catches up on the next atom that needs it
                if hit == 1 << a:
                    continue
            k = cut[a]
            if k > gt_start[a] and (aval + vals[k - 1]) / 2.0 > obs:
                k -= 1
                while k > gt_start[a] and (aval + vals[k - 1]) / 2.0 > obs:
                    k -= 1
                cut[a] = k
                lo = lo_c[memb[a]]
                rh[a] = (top[a] - (cum[k - 1] if k > lo else 0.0)) * inv_high[a]
        for q in range(npat):
            lowmask = pats[q]
            if not lowmask & hit:  # no low guide's maximum can be at aval
                continue
            prod_le = 1.0
            prod_lt = 1.0
            prod_s = 1.0
            for a in range(nmem):
                if (lowmask >> a) & 1:
                    prod_lt *= rlt[a] if (hit >> a) & 1 else rle[a]
                    prod_le *= rle[a]
                else:
                    prod_s *= rh[a]
            pmax = prod_le - prod_lt
            if pmax > 0.0:
                bg[q] += pmax * prod_s
    for q in range(npat):
        g += patw[q] * bg[q]
    return g


@njit(cache=True, nogil=True)
def _gene_core(
    k_all, lo_c, hi_c, vals, cum, totmass, nanp, le_arr, lt_arr, rvals, rcum, obs, scratch, iscratch, pats, patw, bg
):
    """``(P(gene median > obs), P(gene median == obs))`` over the gene's guides.

    Sums over which guides come out nan, with the odd case a Poisson-binomial count and the even
    case handled via :func:`_even_side` and reflection. ``le_arr``/``lt_arr`` are the exact
    ``P(guide <= obs)`` / ``P(guide < obs)`` (from the sub-tail mixture), so the count that decides
    significance is exact; the atoms serve only the boundary shape. ``scratch`` (11 rows) and
    ``iscratch`` (5 rows) have at least ``k_all + 1`` columns, ``pats``/``patw``/``bg`` ``2 ** k_all``.
    """
    p_gt = 0.0
    p_eq = 0.0
    ple, pge, dp, fs = scratch[0], scratch[1], scratch[2], scratch[3:]
    memb, ints = iscratch[0], iscratch[1:]
    if nanp[:k_all].max() <= 0.0:  # no guide can come out nan: only the all-guides mask has weight
        first = (1 << k_all) - 1
    else:
        first = 1
    for mask in range(first, 1 << k_all):
        weight = 1.0
        k = 0
        for c in range(k_all):
            if (mask >> c) & 1:
                if totmass[c] <= 0.0:  # no atoms, so never non-nan in this mask
                    weight = 0.0
                    break
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
            g = _poisbin_cdf(s, ple, k, dp)
            lt = _poisbin_cdf(s, pge, k, dp)
        else:
            s = k // 2
            g = _even_side(memb, k, lo_c, hi_c, vals, cum, obs, s, ple, fs, ints, pats, patw, bg, dp)
            lt = _even_side(memb, k, lo_c, hi_c, rvals, rcum, -obs, s, pge, fs, ints, pats, patw, bg, dp)
        e = 1.0 - g - lt
        p_gt += weight * g
        p_eq += weight * e
    return p_gt, p_eq


@njit(cache=True, nogil=True)
def _deal(cost, n_slots):
    """Deal items to ``n_slots`` slots, costliest first onto the least-loaded slot.

    Returns ``(order, start)``: slot ``s`` runs ``order[start[s]:start[s + 1]]``. The parallel loops
    run one slot per thread, so the threads finish together however unequal the items are.
    """
    n = cost.size
    load = np.zeros(n_slots)
    slot = np.empty(n, dtype=np.int64)
    for i in np.argsort(-cost):
        s = np.argmin(load)
        slot[i] = s
        load[s] += cost[i]
    order = np.argsort(slot, kind="mergesort")
    start = np.zeros(n_slots + 1, dtype=np.int64)
    for i in range(n):
        start[slot[i] + 1] += 1
    for s in range(n_slots):
        start[s + 1] += start[s]
    return order, start


@njit(parallel=True, cache=True, nogil=True)
def _gene_pvalues(
    pool_sorted,
    sub_nn,
    sub_start,
    sub_stop,
    pool_subs,
    pool_sub_start,
    pool_sub_stop,
    pool_width,
    construct_size,
    construct_null,
    null_pool,
    null_size,
    null_caps,
    gene_offsets,
    gene_observed,
    eps,
    n_slots,
):
    """Two-sided p-value of every gene against its guides' control pools, exactly.

    A guide's null depends only on its pool and cell count, so per feature each distinct
    ``(null_pool, null_size)`` null is built once into a shared atom store, and the genes then only
    read it. Work is dealt to ``n_slots`` slots (see :func:`_deal`), each with its own scratch.
    """
    n_genes = gene_offsets.size - 1
    n_features = gene_observed.shape[1]
    n_nulls = null_pool.size
    out = np.empty((n_genes, n_features), dtype=np.float64)
    max_cap = max(1, null_caps.max())
    max_k = max(1, np.diff(gene_offsets).max())
    max_width = max(1, (sub_stop - sub_start).max())
    val_t = np.empty((n_slots, max_cap), dtype=np.float64)
    prob_t = np.empty((n_slots, max_cap), dtype=np.float64)
    stack_t = np.empty((n_slots, 128), dtype=np.int64)
    sval_t = np.empty((n_slots, max_width), dtype=np.float64)
    sfl_t = np.empty((n_slots, max_width), dtype=np.float64)
    sgt_t = np.empty((n_slots, max_width), dtype=np.float64)
    # per gene: lo_c, hi_c, its guides' masses and tails, and the _gene_core scratch; rows written at
    # every atom, padded by a cache line so that slots on different cores share none
    row_k = max_k + 1 + 8
    lohi_t = np.empty((n_slots, 2, row_k), dtype=np.int64)
    small_t = np.empty((n_slots, 4, row_k), dtype=np.float64)
    scratch_t = np.empty((n_slots, 11, row_k), dtype=np.float64)
    iscratch_t = np.empty((n_slots, 5, row_k), dtype=np.int64)
    pats_t = np.empty((n_slots, (1 << max_k) + 8), dtype=np.int64)
    patw_t = np.empty((n_slots, (1 << max_k) + 8), dtype=np.float64)
    bg_t = np.empty((n_slots, (1 << max_k) + 8), dtype=np.float64)
    counts = np.empty(n_nulls, dtype=np.int64)
    offs = np.zeros(n_nulls + 1, dtype=np.int64)
    totmass_u = np.empty(n_nulls, dtype=np.float64)
    nanp_u = np.empty(n_nulls, dtype=np.float64)
    null_order, null_start = _deal(null_caps.astype(np.float64), n_slots)
    gene_cost = np.empty(n_genes, dtype=np.float64)
    for j in range(n_features):
        row = pool_sorted[j]
        # size each shared null's merged atom list, then build it again into its slot of the store
        for t in prange(n_slots):
            for ui in range(null_start[t], null_start[t + 1]):
                u = null_order[ui]
                p = null_pool[u]
                end, _ = _build_construct(
                    row,
                    sub_nn[j],
                    sub_start,
                    sub_stop,
                    pool_subs,
                    pool_sub_start[p],
                    pool_sub_stop[p],
                    pool_width[p],
                    null_size[u],
                    eps,
                    val_t[t],
                    prob_t[t],
                    0,
                    stack_t[t],
                    sval_t[t],
                    sfl_t[t],
                    sgt_t[t],
                )
                counts[u] = end
        for u in range(n_nulls):
            offs[u + 1] = offs[u] + counts[u]
        vals = np.empty(offs[n_nulls], dtype=np.float64)
        cum = np.empty(offs[n_nulls], dtype=np.float64)  # running mass from the list's start
        rvals = np.empty(offs[n_nulls], dtype=np.float64)  # the list negated and reversed, for the < side
        rcum = np.empty(offs[n_nulls], dtype=np.float64)
        for t in prange(n_slots):
            for ui in range(null_start[t], null_start[t + 1]):
                u = null_order[ui]
                p = null_pool[u]
                end, nprob = _build_construct(
                    row,
                    sub_nn[j],
                    sub_start,
                    sub_stop,
                    pool_subs,
                    pool_sub_start[p],
                    pool_sub_stop[p],
                    pool_width[p],
                    null_size[u],
                    eps,
                    val_t[t],
                    prob_t[t],
                    0,
                    stack_t[t],
                    sval_t[t],
                    sfl_t[t],
                    sgt_t[t],
                )
                nanp_u[u] = nprob
                lo = offs[u]
                running = 0.0
                for z in range(end):
                    vals[lo + z] = val_t[t, z]
                    running += prob_t[t, z]
                    cum[lo + z] = running
                totmass_u[u] = running
                running = 0.0
                for z in range(end):
                    rvals[lo + z] = -val_t[t, end - 1 - z]
                    running += prob_t[t, end - 1 - z]
                    rcum[lo + z] = running
        # a gene costs about its guides' atom count, times the patterns of an even guide count
        for g in range(n_genes):
            tot = 0
            for c in range(gene_offsets[g], gene_offsets[g + 1]):
                tot += counts[construct_null[c]]
            k = gene_offsets[g + 1] - gene_offsets[g]
            gene_cost[g] = tot * (1.0 + (k % 2 == 0) * k) + 1.0
        gene_order, gene_start = _deal(gene_cost, n_slots)
        for t in prange(n_slots):
            for gi in range(gene_start[t], gene_start[t + 1]):
                g = gene_order[gi]
                obs = gene_observed[g, j]
                if np.isnan(obs):
                    out[g, j] = np.nan
                    continue
                lo_c, hi_c = lohi_t[t, 0], lohi_t[t, 1]
                totmass, nanp, le_arr, lt_arr = small_t[t, 0], small_t[t, 1], small_t[t, 2], small_t[t, 3]
                c0 = gene_offsets[g]
                k_all = gene_offsets[g + 1] - c0
                for ci in range(k_all):
                    c = c0 + ci
                    u = construct_null[c]
                    lo_c[ci] = offs[u]
                    hi_c[ci] = offs[u + 1]
                    totmass[ci] = totmass_u[u]
                    nanp[ci] = nanp_u[u]
                    # exact P(guide <= obs) / P(guide < obs) from the sub-tail mixture, so the gene
                    # count that decides significance never depends on the atom coarsening
                    p = null_pool[u]
                    m = construct_size[c]
                    le = 0.0
                    lt = 0.0
                    for si in range(pool_sub_start[p], pool_sub_stop[p]):
                        sub = pool_subs[si]
                        aa = sub_start[sub]
                        width = sub_stop[sub] - aa
                        w = width / pool_width[p]
                        gsub, lsub, nasub = _sub_tail(row, aa, sub_nn[j, sub], width, m, obs)
                        le += w * (1.0 - gsub - nasub)
                        lt += w * lsub
                    le_arr[ci] = le
                    lt_arr[ci] = lt
                gt, eq = _gene_core(
                    k_all,
                    lo_c,
                    hi_c,
                    vals,
                    cum,
                    totmass,
                    nanp,
                    le_arr,
                    lt_arr,
                    rvals,
                    rcum,
                    obs,
                    scratch_t[t],
                    iscratch_t[t],
                    pats_t[t],
                    patw_t[t],
                    bg_t[t],
                )
                one = gt + 0.5 * eq
                out[g, j] = min(1.0, max(0.0, 2.0 * min(one, 1.0 - one)))
    return out


@njit(parallel=True, cache=True, nogil=True)
def _pool_medians(pool_t, sub_start, sub_stop, pool_subs, pool_sub_start, pool_sub_stop, first, longest):
    """Median of each feature over the cells of pools ``first`` onwards, ``(n_new_pools, n_features)``.

    A pool's cells are those of its guides ``pool_subs[pool_sub_start[p]:pool_sub_stop[p]]``, read straight
    from ``pool_t`` (features by pool cells) without gathering a pool's rows out. The median is the selection
    :func:`~mantispy._core._numba.grouped_stat` uses, so the value is the same; ``NaN`` is skipped.
    """
    n_features = pool_t.shape[0]
    n_new = pool_sub_start.size - first
    out = np.empty((n_new, n_features), dtype=np.float64)
    n_tasks = n_new * n_features
    n_chunks = min(N_CHUNKS, n_tasks)
    for chunk in prange(n_chunks):
        buffer = np.empty(longest, dtype=np.float64)
        for task in range(chunk, n_tasks, n_chunks):
            p, j = divmod(task, n_features)
            n = 0
            for si in range(pool_sub_start[first + p], pool_sub_stop[first + p]):
                sub = pool_subs[si]
                for i in range(sub_start[sub], sub_stop[sub]):
                    value = pool_t[j, i]
                    if not np.isnan(value):
                        buffer[n] = value
                        n += 1
            out[p, j] = _median_select(buffer, n)
    return out


def _leave_own_out(
    pool_t,
    unit_codes,
    unit_keys,
    pool_rows,
    sub_start,
    sub_stop,
    pool_subs,
    pool_sub_start,
    pool_sub_stop,
    pool_width,
    pool_median,
    construct_unit,
    construct_pool,
):
    """Give each tested group whose own guides sit in its pool a copy of that pool without them.

    Only pseudo-genes, made of control guides, have guides in a pool; for them each new pool is appended
    and ``construct_pool`` is repointed to it in place. Every other group keeps its pool untouched.
    Returns the pool arrays, extended.
    """
    sub_of_position = np.repeat(np.arange(sub_start.size), sub_stop - sub_start)
    tested_units = np.unique(construct_unit)
    in_tested = np.isin(unit_codes[pool_rows], tested_units)
    if not in_tested.any():
        return pool_subs, pool_sub_start, pool_sub_stop, pool_width, pool_median
    own_unit, own_sub = unit_codes[pool_rows][in_tested], sub_of_position[in_tested]

    subs, starts, stops, widths = [pool_subs], [pool_sub_start], [pool_sub_stop], [pool_width]
    n_pools, n_entries = pool_sub_start.size, pool_subs.size
    sub_width = (sub_stop - sub_start).astype(np.int64)
    new_pool: dict[tuple[int, int], int] = {}
    for c in np.flatnonzero(np.isin(construct_unit, own_unit)):
        key = (int(construct_unit[c]), int(construct_pool[c]))
        if key not in new_pool:
            unit, pool = key
            base = pool_subs[pool_sub_start[pool] : pool_sub_stop[pool]]
            keep = base[~np.isin(base, own_sub[own_unit == unit])]
            if keep.size == 0:
                raise ValueError(
                    f"group {unit_keys[unit]!r} holds every control guide its null is drawn from, so leaving "
                    "its own guides out leaves no null"
                )
            subs.append(keep)
            starts.append(np.array([n_entries], dtype=np.int64))
            stops.append(np.array([n_entries + keep.size], dtype=np.int64))
            widths.append(np.array([sub_width[keep].sum()], dtype=np.int64))
            new_pool[key] = n_pools + len(new_pool)
            n_entries += keep.size
        construct_pool[c] = new_pool[key]
    pool_subs, pool_sub_start, pool_sub_stop, pool_width = (
        np.concatenate(parts) for parts in (subs, starts, stops, widths)
    )
    new_medians = _pool_medians(
        pool_t, sub_start, sub_stop, pool_subs, pool_sub_start, pool_sub_stop, n_pools, int(pool_width[n_pools:].max())
    )
    return pool_subs, pool_sub_start, pool_sub_stop, pool_width, np.concatenate([pool_median, new_medians])


#: The q-value cutoff the summary log and the pseudo-gene calibration read.
_ALPHA = 0.05


def _family_qvalues(pvalues: np.ndarray, is_pseudo: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg q-values of the real genes among themselves, and of the pseudo-genes among themselves."""
    qvalues = np.full(pvalues.shape, np.nan)
    for family in (~is_pseudo, is_pseudo):
        qvalues[family] = benjamini_hochberg(pvalues[family])
    return qvalues


def _calibration(table: pd.DataFrame, alpha: float) -> str:
    """One line on how often the pseudo-genes are called at the p-value cutoff the real genes get at ``alpha``.

    Benjamini-Hochberg calls a real gene-feature pair at ``q < alpha`` exactly when its p-value is at most the
    largest p-value it calls, so that p-value is the real genes' cutoff, and the pseudo-genes are judged by it.
    """
    from scipy.stats import binomtest

    real, pseudo = table[~table["is_pseudogene"]], table[table["is_pseudogene"] & table["pvalue"].notna()]
    n_genes = table.loc[table["is_pseudogene"], "group"].nunique()
    called_real = real["qvalue"] < alpha
    if not called_real.any():
        return (
            f"no real gene-feature pair is at q < {alpha:g}, so there is no cutoff to call the "
            f"{n_genes} pseudo-gene(s) at"
        )
    cutoff = float(real.loc[called_real, "pvalue"].max())
    called = pseudo["pvalue"] <= cutoff
    n_called, n_tests = int(called.sum()), len(pseudo)
    genes_called = int(called.groupby(pseudo["group"]).any().sum())
    interval = binomtest(n_called, n_tests).proportion_ci(0.95) if n_tests else None
    low, high = (np.nan, np.nan) if interval is None else (interval.low, interval.high)
    return (
        f"at p <= {cutoff:.3g}, the real genes' cutoff for q < {alpha:g}, {genes_called} of {n_genes} pseudo-gene(s) "
        f"have a feature called ({genes_called / n_genes:.1%}), and {n_called} of {n_tests} pseudo-gene test(s) are "
        f"called ({n_called / max(n_tests, 1):.2%}, 95% interval {low:.2%} to {high:.2%})"
    )


#: Hard limit on a guide null's per-thread atom buffer, so a huge screen cannot ask for unbounded memory;
#: the probability pruning keeps the real count far below this, and small problems never approach it.
_ATOM_CAP_MAX = 40_000_000


def _null_caps(sub_start, sub_stop, pool_subs, pool_sub_start, pool_sub_stop, null_pool, null_size):
    """Per-null atom-buffer size: an upper bound on the atoms a guide null generates before merging.

    A control guide of ``w`` cells resampled ``m`` at a time produces median atoms only where the
    central order statistic's empirical CDF lies within the band ``|F - 1/2| < ~3/sqrt(m)`` in which
    its marginal exceeds ``_ATOM_EPS``; that band holds at most ``nw = min(w, 7*w/sqrt(m))`` distinct
    values, so at most ``nw^2 + nw`` even-median atoms, never more than the ``w(w+1)/2 + w`` pairs.
    """
    widths = (sub_stop - sub_start).astype(np.int64)[pool_subs]
    full = widths * (widths + 1) // 2 + widths
    caps = np.empty(null_pool.size, dtype=np.int64)
    for u, (p, m) in enumerate(zip(null_pool, null_size, strict=True)):
        w = widths[pool_sub_start[p] : pool_sub_stop[p]]
        nw = np.minimum(w, (7 * w) // int(np.sqrt(max(int(m), 1))) + 4)
        bound = np.minimum(full[pool_sub_start[p] : pool_sub_stop[p]], nw * nw + nw)
        caps[u] = min(int(bound.sum()) + 16, _ATOM_CAP_MAX)
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
            Their guides form the null and are not tested, except those :func:`~mantispy.pp.pseudogenes` relabelled (see Notes).
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
        Writes ``uns["mantispy"][key_added]``, a tidy frame of ``group``, ``feature``, ``effect`` (the gene's median minus the median of its guides' control pools, in the units of ``X``), ``pvalue``, ``qvalue`` (Benjamini-Hochberg over every gene and feature at once) and ``is_pseudogene``.
        Also writes ``uns["mantispy"][key_added + "_guides"]``, the same per guide, with ``replicate`` and ``n_cells`` columns and its own q-values.
        Pseudo-genes get their q-values among themselves, so the real genes' q-values are the ones they get without pseudo-genes.

    Raises:
        ValueError: ``control_scope`` is unknown, a scope that needs ``condition`` or ``reference_condition`` lacks it, ``"within_perturbation"`` is used without ``condition`` in ``groupby``, the controls in ``reference_condition`` are empty, or no guide has a null to draw from.

    Notes:
        The p-value is two-sided, ``2 * min(p, 1 - p)`` with ``p`` the probability that a null median exceeds the observed one.
        It is exact rather than simulated, so it has no lower floor and can be zero when the observed median lies beyond every attainable null median.
        A null median equal to the observed one counts half, so a feature where most cells share one value, such as a count that is mostly zero, is not called because its median cannot move; Brieflow counts such ties as below.
        Medians skip missing values, and a feature whose observed median is missing is left ``NaN``.

        The null resamples a control guide's own cells, so it carries that sample's error on top of the spread between control guides, while a tested guide carries only its own.
        The test therefore runs conservative when guides barely differ from one another, and near the nominal rate when they differ by more than the error of a guide's median.
        To check the rate on your own screen, group the control guides into pseudo-genes with :func:`~mantispy.pp.pseudogenes` and pass its column as ``groupby``.
        A pseudo-gene that function records for a ``groupby`` column is tested even though its cells are controls, against every control guide but its own; its guides stay in the null of every other group, so the real genes get the p-values they get without pseudo-genes.
        The pseudo-genes are judged at the real genes' operating point: the largest p-value Benjamini-Hochberg calls among the real genes at ``q < 0.05``.
        A log line gives the per-gene rate, the share of pseudo-genes with any feature at or below that cutoff, and the per-test rate, the share of pseudo-gene features there, with a binomial 95% interval.
        A calibrated test calls a share of the pseudo-gene tests close to the cutoff itself.
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

    # Control cells that mt.pp.pseudogenes relabelled, as it records, are tested as well as pooled.
    pseudo = np.zeros(adata.n_obs, dtype=bool)
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
        recorded = adata.uns.get("mantispy", {}).get("pseudogenes", {})
        for column in by:
            if column in recorded:
                pseudo |= obs[column].astype(str).isin([str(label) for label in recorded[column]]).to_numpy()
        tested = ~is_control | (is_control & pseudo)
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
    pool_subs = np.arange(n_subs_total, dtype=np.int64)
    pool_subs, pool_sub_start, pool_sub_stop, pool_width, pool_median = _leave_own_out(
        pool_t,
        unit_codes,
        unit_keys,
        pool_rows,
        sub_start,
        sub_stop,
        pool_subs,
        pool_sub_start,
        pool_sub_stop,
        pool_width,
        pool_median,
        construct_unit,
        construct_pool,
    )

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
        pool_subs,
        pool_sub_start,
        pool_sub_stop,
        pool_width,
        construct_pool,
        construct_size,
        construct_observed,
    )
    # a guide's null depends only on its pool and cell count, so guides sharing both share it
    null_keys, construct_null = np.unique(
        construct_pool * (construct_size.max() + 1) + construct_size, return_inverse=True
    )
    null_pool, null_size = np.divmod(null_keys, construct_size.max() + 1)
    gene_p = _gene_pvalues(
        pool_sorted,
        sub_nn,
        sub_start,
        sub_stop,
        pool_subs,
        pool_sub_start,
        pool_sub_stop,
        pool_width,
        construct_size,
        construct_null.astype(np.int64),
        null_pool.astype(np.int64),
        null_size.astype(np.int64),
        _null_caps(sub_start, sub_stop, pool_subs, pool_sub_start, pool_sub_stop, null_pool, null_size),
        gene_offsets,
        gene_observed,
        _ATOM_EPS,
        numba.get_num_threads(),
    )

    features = adata.var_names.to_numpy()
    n_features = features.size
    gene_labels = [str(unit_keys[unit]) for unit in genes]
    unit_is_pseudo = np.zeros(len(unit_keys), dtype=bool)
    unit_is_pseudo[unit_codes[pseudo & tested]] = True
    gene_is_pseudo = unit_is_pseudo[genes]
    table = pd.DataFrame(
        {
            "group": np.repeat(gene_labels, n_features),
            "feature": np.tile(features, genes.size),
            "effect": (gene_observed - gene_baseline).ravel(),
            "pvalue": np.where(np.isfinite(gene_observed), gene_p, np.nan).ravel(),
        }
    )
    table["qvalue"] = _family_qvalues(table["pvalue"].to_numpy(), np.repeat(gene_is_pseudo, n_features))

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
    guides["qvalue"] = _family_qvalues(
        guides["pvalue"].to_numpy(), np.repeat(gene_is_pseudo[gene_of_construct], n_features)
    )
    table["is_pseudogene"] = np.repeat(gene_is_pseudo, n_features)
    guides["is_pseudogene"] = np.repeat(gene_is_pseudo[gene_of_construct], n_features)

    real = table[~table["is_pseudogene"]]
    logger.info(
        "bootstrap_features: %d of %d gene-feature pair(s) at q < 0.05 over %d gene(s)",
        int((real["qvalue"] < 0.05).sum()),
        len(real),
        genes.size - int(gene_is_pseudo.sum()),
    )
    if gene_is_pseudo.any():
        logger.info("bootstrap_features: %s", _calibration(table, _ALPHA))
    store = adata.uns.setdefault("mantispy", {})
    store[key_added] = table
    store[f"{key_added}_guides"] = guides
    return None
