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

from mantispy._core._numba import MEDIAN, _counting_order, _kernel_input, grouped_stat
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
def _count_below(v, lo, hi, x, inclusive):
    """How many of the sorted ``v[lo:hi]`` are below ``x`` (or at most ``x``, if ``inclusive``).

    ``np.searchsorted`` on that slice gives the same, but a slice of an array the threads share touches
    its reference count from every thread, which serializes them in the innermost loops.
    """
    left = lo
    right = hi
    while left < right:
        mid = (left + right) // 2
        if v[mid] < x or (inclusive and v[mid] == x):
            left = mid + 1
        else:
            right = mid
    return left - lo


@njit(cache=True, nogil=True)
def _median_gt(v: np.ndarray, lo: int, n: int, m: int, x: float) -> float:
    """``P(median of m draws with replacement from v[lo:lo+n] (sorted asc, no nan) > x)``."""
    if n == 0:
        return 0.0
    cle = _count_below(v, lo, lo + n, x, True)  # count <= x
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
        a = np.float64(v[i])  # a float32 midpoint would round differently from the float64 observed median
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
                ge = n - _count_below(v, lo, lo + n, v[left], False)  # count >= v[left]
                contrib += pmax * (ge / nhigh) ** t
        i = j
    return out + pk * contrib


@njit(cache=True, nogil=True)
def _median_lt(v: np.ndarray, lo: int, n: int, m: int, x: float) -> float:
    """``P(median of m draws from v[lo:lo+n] (sorted asc, no nan) < x)``. Mirror of :func:`_median_gt`."""
    if n == 0:
        return 0.0
    hi = _count_below(v, lo, lo + n, x, False)  # count < x
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
        b = np.float64(v[i])
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
    pool_sub_start: np.ndarray,
    pool_sub_stop: np.ndarray,
    pool_width: np.ndarray,
    construct_pool: np.ndarray,
    construct_nn: np.ndarray,
    construct_observed: np.ndarray,
) -> np.ndarray:
    """Two-sided p-value of every guide against its control pool, exactly.

    A guide's null draws as many cells as the guide has measured in the feature, and is conditioned on a
    measured null median; with no measured control cell the p-value is ``NaN``.
    """
    n_constructs, n_features = construct_observed.shape
    out = np.empty((n_constructs, n_features), dtype=np.float64)
    for c in prange(n_constructs):
        p = construct_pool[c]
        s0 = pool_sub_start[p]
        s1 = pool_sub_stop[p]
        width_total = pool_width[p]
        for j in range(n_features):
            obs = construct_observed[c, j]
            if np.isnan(obs):
                out[c, j] = np.nan
                continue
            row = pool_sorted[j]
            m = construct_nn[c, j]
            gt, lt, _, _, _, measured = _pool_tails(row, sub_nn[j], sub_start, sub_stop, s0, s1, width_total, m, obs)
            out[c, j] = _two_sided(gt, lt, measured)
    return out


@njit(cache=True, nogil=True)
def _pool_tails(row, sub_nn_row, sub_start, sub_stop, s0, s1, width_total, m, x):
    """A guide null's exact tails at ``x``: the cell-count-weighted mixture of its control guides' :func:`_sub_tail`.

    Returns ``P(> x)``, ``P(< x)``, ``P(<= x)``, ``P(>= x)``, ``P(nan)`` and ``P(measured)``, each summed
    directly so that a small one keeps its digits.
    """
    gt = lt = le = ge = na = meas = 0.0
    for sub in range(s0, s1):
        a = sub_start[sub]
        width = sub_stop[sub] - a
        w = width / width_total
        g, l_, n_ = _sub_tail(row, a, sub_nn_row[sub], width, m, x)
        gt += w * g
        lt += w * l_
        le += w * (1.0 - g - n_)
        ge += w * (1.0 - l_ - n_)
        na += w * n_
        meas += w * (1.0 - n_)
    return gt, lt, le, ge, na, meas


@njit(cache=True, nogil=True)
def _two_sided(gt: float, lt: float, measured: float) -> float:
    """``2 * min(P(> obs), P(< obs))`` with ties counting half, given the null median is measured.

    ``measured`` is the probability that it is; ``NaN`` if it never is. The clamp catches round-off
    from forming the tie mass as ``measured - gt - lt``.
    """
    if measured <= 0.0:
        return np.nan
    half_eq = 0.5 * (measured - gt - lt)
    return min(1.0, max(0.0, 2.0 * min(gt + half_eq, lt + half_eq) / measured))


@njit(cache=True, nogil=True)
def _add_sub_atoms(v, lo, n, r, scale, eps, out_val, out_prob, cnt, sval, sfl, sgt):
    """Append the atoms of the median of r draws from ``v[lo:lo+n]`` (sorted, no nan).

    Each atom is ``(value, scale*prob)``. Prunes atoms below ``eps`` and stops each unimodal run
    once its probability drops back below it. For even r the distinct values and their tail powers
    are precomputed into the scratch arrays ``sval``/``sfl``/``sgt`` so the off-diagonal inner loop
    is a plain product, O(distinct) powers rather than O(distinct^2). Returns ``-1`` if ``out_val`` fills up.
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
                    return -1
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
                    return -1
                out_val[cnt] = sval[ka]
                out_prob[cnt] = scale * diag
                cnt += 1
            inner_seen = False
            for lb in range(ka + 1, nd):
                if lb > ka + 1:
                    pr = coef * (sgt[lb - 1] - sgt[lb]) if direct else pr * sgt[lb]
                if pr > eps:
                    if cnt >= out_val.shape[0]:
                        return -1
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
    stack,
    sval,
    sfl,
    sgt,
):
    """Build one guide null as merged, sorted atoms in ``val_buf``/``prob_buf``; returns their count.

    Mixes every control guide's exact median-atom distribution, then sorts and merges equal values. A
    control guide with missing cells adds one band per number of measured draws, skipping the bands
    whose probability is below ``eps``. Returns ``-1`` if the atoms do not fit the buffer.
    """
    cnt = 0
    for sub in range(s0, s1):
        a = sub_start[sub]
        width = sub_stop[sub] - a
        nn = sub_nn_row[sub]
        if nn == 0:
            continue
        w = width / width_total
        if nn == width:
            cnt = _add_sub_atoms(pool_sorted_row, a, nn, m, w, eps, val_buf, prob_buf, cnt, sval, sfl, sgt)
        else:
            # the number of measured draws is Binomial(m, nn / width); walk out from its mode while the
            # band is worth keeping, as the pmf is unimodal
            p_nn = nn / width
            mode = min(m, max(1, int((m + 1) * p_nn)))
            for step in (-1, 1):
                r = mode if step == -1 else mode + 1
                while 1 <= r <= m and cnt >= 0:
                    term = w * _binom_pmf(r, m, p_nn)
                    if term <= eps:
                        break
                    cnt = _add_sub_atoms(pool_sorted_row, a, nn, r, term, eps, val_buf, prob_buf, cnt, sval, sfl, sgt)
                    r += step
        if cnt < 0:
            return -1
    if cnt == 0:
        return 0
    return _collapse(val_buf, prob_buf, 0, cnt, stack)


@njit(cache=True, nogil=True)
def _step(dp, src, dst, nlo, nhi, stay, up, down, stride=1):
    """Add one guide to the distribution of ``#high - #low``, from row ``src`` of ``dp`` into ``dst[nlo:nhi+1]``.

    The guide is nan with probability ``stay``, high with ``up`` and low with ``down``. Counts outside
    ``[nlo, nhi]`` are dropped, which prunes those that can no longer reach the one asked for; row ``src``
    is zero for two entries past its own range, and ``dst`` is left so. Rows are indexed rather than
    sliced, since a slice of the shared scratch would touch its reference count from every thread.
    With no guide ever nan, the counts after a given number of guides share one parity, so
    ``stride=2`` visits only those.
    """
    if stride == 2:
        for d in range(nlo, nhi + 1, 2):
            dp[dst, d] = dp[src, d - 1] * up + dp[src, d + 1] * down
    else:
        for d in range(nlo, nhi + 1):
            dp[dst, d] = dp[src, d] * stay + dp[src, d - 1] * up + dp[src, d + 1] * down
    _pad(dp, dst, nlo, nhi, stride)


@njit(cache=True, nogil=True)
def _pad(dp, row, lo, hi, stride):
    """Zero the entries past ``[lo, hi]`` that the next :func:`_step` reads, of the parity it reads."""
    dp[row, lo - 2] = dp[row, hi + 2] = 0.0
    if stride == 1:
        dp[row, lo - 1] = dp[row, hi + 1] = 0.0


@njit(cache=True, nogil=True)
def _start(dp, row, o):
    """A count distribution at zero, ``dp[row, o] = 1``, zero-padded as :func:`_step` expects."""
    dp[row, o - 2] = dp[row, o - 1] = dp[row, o + 1] = dp[row, o + 2] = 0.0
    dp[row, o] = 1.0


@njit(cache=True, nogil=True)
def _count_above(k, up, down, stay, dp):
    """``P(#high > #low)`` over ``k`` independent guides, by a dynamic program over the guides.

    The distribution of ``#high - #low`` sits in rows 0 and 1 of ``dp``, centred at ``k + 2``.
    """
    o = k + 2
    _start(dp, 0, o)
    for c in range(k):
        _step(dp, c % 2, 1 - c % 2, o - c - 1, o + c + 1, stay[c], up[c], down[c])
    total = 0.0
    for d in range(o + 1, o + k + 1):
        total += dp[k % 2, d]
    return total


@njit(cache=True, nogil=True)
def _count_at(k, skip, target, up, down, stay, dp, stride=1):
    """``P(#high - #low == target)`` over the ``k`` guides other than ``skip`` (``-1`` for none).

    ``stride=2`` (see :func:`_step`) needs the count of guides taken and ``target`` of equal parity.
    """
    rem = k - 1 if skip >= 0 else k
    if abs(target) > rem:
        return 0.0
    o = k + 2
    _start(dp, 0, o)
    row = 0
    lo = hi = o
    for c in range(k):
        if c == skip:
            continue
        rem -= 1
        lo = max(lo - 1, o + target - rem)
        hi = min(hi + 1, o + target + rem)
        _step(dp, row, 1 - row, lo, hi, stay[c], up[c], down[c], stride)
        row = 1 - row
    return dp[row, o + target]


@njit(cache=True, nogil=True)
def _count_grown(k, below, grown, up, stay, dp, stride):
    """``P(#high == #low and some low guide in its grown part)`` over ``k`` guides.

    A low guide lies in ``below`` with that probability and in ``grown`` with that one, so the result is
    ``F(below + grown) - F(below)`` for ``F`` the probability of as many low as high guides, summed as
    positive terms. Rows 0 and 1 of ``dp`` hold the counts with no low guide in its grown part yet, rows
    2 and 3 those with one. ``stride`` as in :func:`_step`, for an even ``k``.
    """
    o = k + 2
    _start(dp, 0, o)
    _start(dp, 2, o)
    dp[2, o] = 0.0
    src = 0
    lo = hi = o
    for c in range(k):
        rem = k - c - 1
        lo = max(lo - 1, o - rem)
        hi = min(hi + 1, o + rem)
        st = stay[c]
        hi_w = up[c]
        lt = below[c]
        eq = grown[c]
        dst = 1 - src
        if stride == 2:
            for d in range(lo, hi + 1, 2):
                dp[dst, d] = dp[src, d - 1] * hi_w + dp[src, d + 1] * lt
                dp[dst + 2, d] = dp[src + 2, d - 1] * hi_w + dp[src + 2, d + 1] * (lt + eq) + dp[src, d + 1] * eq
        else:
            for d in range(lo, hi + 1):
                dp[dst, d] = dp[src, d] * st + dp[src, d - 1] * hi_w + dp[src, d + 1] * lt
                dp[dst + 2, d] = (
                    dp[src + 2, d] * st
                    + dp[src + 2, d - 1] * hi_w
                    + dp[src + 2, d + 1] * (lt + eq)
                    + dp[src, d + 1] * eq
                )
        _pad(dp, dst, lo, hi, stride)
        _pad(dp, dst + 2, lo, hi, stride)
        src = dst
    return dp[src + 2, o]


@njit(cache=True, nogil=True)
def _close_run(k, lo_c, cum, plow, inv_low, run_ptr, lptr, wh, pnan, below, grown, dp, stride):
    """The terms of a run of swept atoms under one cut: ``F`` at its end minus ``F`` at its start.

    Guide ``a`` swept the atoms ``[run_ptr[a], lptr[a])`` in the run, so its low weight grew from the
    mass below ``run_ptr[a]`` by the mass of those atoms; ``below``/``grown`` are scratch for both.
    """
    n_grown = 0
    last = 0
    for a in range(k):
        lo = lo_c[a]
        p0 = run_ptr[a]
        start = cum[p0 - 1] if p0 > lo else 0.0
        below[a] = plow[a] * start * inv_low[a]
        grown[a] = plow[a] * ((cum[lptr[a] - 1] if lptr[a] > lo else 0.0) - start) * inv_low[a]
        if grown[a] > 0.0:
            n_grown += 1
            last = a
    if n_grown == 1:
        # F is linear in one guide's low weight: its growth times the others balancing it, one high to spare
        return grown[last] * _count_at(k, last, 1, wh, below, pnan, dp, stride)
    return _count_grown(k, below, grown, wh, pnan, dp, stride)


@njit(cache=True, nogil=True)
def _any_positive(x, k):
    """Whether any of ``x[:k]`` is positive, without slicing ``x``."""
    for i in range(k):
        if x[i] > 0.0:
            return True
    return False


@njit(cache=True, nogil=True)
def _boundary(k, lo_c, hi_c, vals, cum, obs, plow, phigh, pnan, fs, ints, dp):
    """``P(#high == #low >= 1 and the midpoint of the largest low and smallest high guide > obs)``.

    This is the part of ``P(gene median > obs)`` that the counts alone do not settle: an even number of
    measured guides, half of them at most ``obs`` (low). Guide ``a``'s atoms are ``vals[lo_c[a]:hi_c[a]]``,
    sorted and distinct, with ``cum`` their running mass from that list's start. ``plow``/``phigh``/``pnan``
    are its exact probabilities of being low, high or nan, so the count part stays exact; the atoms give
    only the shape within each side.

    One ascending sweep over the atoms at most ``obs`` takes each value ``a`` in turn as the largest low
    one. The high guides must then sit past the cut, the atoms whose float midpoint with ``a`` exceeds
    ``obs``. With ``F`` the probability of as many low as high guides, the low ones at most a given value
    and the high ones past the cut, the term of ``a`` is ``F(at most a) - F(below a)``. While the cut stays
    put these terms telescope, so a run of atoms under one cut is summed at once, by a dynamic program
    over the guides (:func:`_count_grown`) that keeps every term positive. ``fs`` (6 rows), ``ints``
    (6 rows) and ``dp`` (4 rows of ``2 k + 5``) are scratch.
    """
    inv_low, inv_high, top, wl, wh, grown = fs[0], fs[1], fs[2], fs[3], fs[4], fs[5]
    le_end, gt_start, lptr, cut, run_ptr, twin = ints[0], ints[1], ints[2], ints[3], ints[4], ints[5]
    for a in range(k):
        lo = lo_c[a]
        hi = hi_c[a]
        # guides that share a null (the same pool and cell count) sweep alike, so the later ones copy the first
        twin[a] = -1
        for b in range(a):
            if lo_c[b] == lo and hi_c[b] == hi and twin[b] < 0:
                twin[a] = b
                break
        cnt_le = _count_below(vals, lo, hi, obs, True)
        at_obs = cum[lo + cnt_le - 1] if cnt_le > 0 else 0.0
        top[a] = cum[hi - 1] if hi > lo else 0.0
        inv_low[a] = 1.0 / at_obs if at_obs > 0.0 else 0.0
        inv_high[a] = 1.0 / (top[a] - at_obs) if top[a] > at_obs else 0.0
        wl[a] = plow[a] if inv_low[a] > 0.0 else 0.0  # for the check below: low with an atom to sweep
        wh[a] = phigh[a] if inv_high[a] > 0.0 else 0.0
        le_end[a] = lo + cnt_le  # one past the last atom <= obs
        gt_start[a] = lo + cnt_le  # first atom > obs
        lptr[a] = lo  # next atom <= obs not yet swept
        run_ptr[a] = lo  # where the current run of atoms under one cut began
        cut[a] = hi  # atoms [cut, hi) have midpoint (aval + .)/2 > obs
    # with no guide ever nan the count here is even, and every count the program reaches keeps a parity
    stride = 1 if _any_positive(pnan, k) else 2
    # no split into as many low as high guides, each with atoms on its side, can happen: nothing to sweep
    all_nan = 1.0
    for a in range(k):
        all_nan *= pnan[a]
    if _count_at(k, -1, 0, wh, wl, pnan, dp, stride) <= all_nan:
        return 0.0
    for a in range(k):
        wh[a] = 0.0  # P(high and past the current cut)
    total = 0.0
    n_high = 0  # guides with an atom past the cut; until one has, every term is zero
    # the largest high atom not yet past any cut: until an atom's midpoint with it exceeds obs, no cut moves,
    # so each run under one cut is found guide by guide rather than atom value by atom value
    vtop = -np.inf
    for a in range(k):
        if cut[a] > gt_start[a]:
            vtop = max(vtop, vals[cut[a] - 1])
    while True:
        aval = np.inf  # the first atom, over the guides, where a cut moves
        for a in range(k):
            if twin[a] >= 0:
                lptr[a] = lptr[twin[a]]
                continue
            p = lptr[a]
            while p < le_end[a] and (vals[p] + vtop) / 2.0 <= obs:
                p += 1
            lptr[a] = p
            if p < le_end[a]:
                aval = min(aval, vals[p])
        if aval == np.inf:
            break
        if n_high > 0:
            total += _close_run(k, lo_c, cum, plow, inv_low, run_ptr, lptr, wh, pnan, wl, grown, dp, stride)
        vtop = -np.inf
        for a in range(k):
            run_ptr[a] = lptr[a]
            q = cut[a]
            if twin[a] >= 0:
                q = cut[twin[a]]
            while q > gt_start[a] and (aval + vals[q - 1]) / 2.0 > obs:
                q -= 1
            if q < cut[a]:
                cut[a] = q
                past = (
                    wh[twin[a]]
                    if twin[a] >= 0
                    else phigh[a] * (top[a] - (cum[q - 1] if q > lo_c[a] else 0.0)) * inv_high[a]
                )
                n_high += wh[a] == 0.0 and past > 0.0
                wh[a] = past
            if q > gt_start[a]:
                vtop = max(vtop, vals[q - 1])
    if n_high > 0:
        total += _close_run(k, lo_c, cum, plow, inv_low, run_ptr, lptr, wh, pnan, wl, grown, dp, stride)
    return total


@njit(cache=True, nogil=True)
def _build_into(
    row, nn_row, sub_start, sub_stop, pool_sub_start, pool_sub_stop, pool_width, p, m, eps,
    val_buf, prob_buf, stack, sval, sfl, sgt, vals, cum, rvals, rcum, lo, room,
):  # fmt: skip
    """Build the null of pool ``p`` at ``m`` draws and, if its merged atoms fit ``room``, write them at ``lo``.

    The store keeps the atom values, their running mass from the bottom (``cum``) and, negated and reversed,
    from the top (``rvals``/``rcum``). Returns the merged count, or ``-1`` if the atoms overflow the buffer.
    """
    end = _build_construct(
        row, nn_row, sub_start, sub_stop, pool_sub_start[p], pool_sub_stop[p], pool_width[p], m, eps,
        val_buf, prob_buf, stack, sval, sfl, sgt,
    )  # fmt: skip
    if 0 <= end <= room:
        running = 0.0
        for z in range(end):
            vals[lo + z] = val_buf[z]
            running += prob_buf[z]
            cum[lo + z] = running
        running = 0.0
        for z in range(end):
            rvals[lo + z] = -val_buf[end - 1 - z]
            running += prob_buf[end - 1 - z]
            rcum[lo + z] = running
    return end


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
    return _counting_order(slot, n_slots)


@njit(parallel=True, cache=True, nogil=True)
def _gene_pvalues(
    pool_sorted,
    sub_nn,
    sub_start,
    sub_stop,
    pool_sub_start,
    pool_sub_stop,
    pool_width,
    construct_nn,
    construct_null,
    null_pool,
    null_size,
    null_caps,
    gene_offsets,
    gene_observed,
    eps,
    n_slots,
    n_gene_slots,
):
    """Two-sided p-value of every gene against its guides' control pools, exactly.

    A guide's null depends only on its pool and measured cell count, so per feature each distinct
    ``(null_pool, null_size)`` null is built once into a shared atom store, and the genes then only
    read it; ``construct_null[j, c]`` names guide ``c``'s null in feature ``j``, ``-1`` where the guide
    measures nothing and so leaves the gene. The nulls are dealt to ``n_slots`` slots and the genes to
    ``n_gene_slots`` (see :func:`_deal`), each with its own scratch; more gene slots than threads let a
    thread that finishes early take another. Also returns which nulls had more atoms than their buffer holds; those
    keep no atoms, so the even-count boundary they enter counts as a tie.
    """
    n_genes = gene_offsets.size - 1
    n_features = gene_observed.shape[1]
    n_constructs = construct_null.shape[1]
    n_nulls = null_pool.size
    out = np.empty((n_genes, n_features), dtype=np.float64)
    overflow = np.zeros(n_nulls, dtype=np.bool_)
    max_cap = max(1, null_caps.max())
    max_k = max(1, np.diff(gene_offsets).max())
    max_width = max(1, (sub_stop - sub_start).max())
    val_t = np.empty((n_slots, max_cap), dtype=np.float64)
    prob_t = np.empty((n_slots, max_cap), dtype=np.float64)
    stack_t = np.empty((n_slots, 128), dtype=np.int64)
    sval_t = np.empty((n_slots, max_width), dtype=np.float64)
    sfl_t = np.empty((n_slots, max_width), dtype=np.float64)
    sgt_t = np.empty((n_slots, max_width), dtype=np.float64)
    # per gene: lo_c, hi_c, its guides' tails, and the _boundary scratch; rows written at every atom,
    # padded by a cache line so that slots on different cores share none
    row_k = max_k + 8
    lohi_t = np.empty((n_gene_slots, 2, row_k), dtype=np.int64)
    tails_t = np.empty((n_gene_slots, 6, row_k), dtype=np.float64)
    fs_t = np.empty((n_gene_slots, 6, row_k), dtype=np.float64)
    ints_t = np.empty((n_gene_slots, 6, row_k), dtype=np.int64)
    dp_t = np.empty((n_gene_slots, 4, 2 * max_k + 5 + 8), dtype=np.float64)
    used = np.empty(n_nulls, dtype=np.bool_)
    counts = np.empty(n_nulls, dtype=np.int64)
    offs = np.zeros(n_nulls + 1, dtype=np.int64)
    reserve = np.zeros(n_nulls, dtype=np.int64)  # the merged length each null's stretch of the store holds
    grow = np.empty(n_nulls, dtype=np.int64)
    null_order, null_start = _deal(null_caps.astype(np.float64), n_slots)
    gene_cost = np.empty(n_genes, dtype=np.float64)
    for j in range(n_features):
        row = pool_sorted[j]
        nn_row = sub_nn[j]
        used[:] = False
        for c in range(n_constructs):
            if construct_null[j, c] >= 0:
                used[construct_null[j, c]] = True
        # build each used null once into its reserved stretch of the store; a null longer than its reserve
        # (every null, at the first feature) is built again once the store has grown to hold it
        for u in range(n_nulls):
            offs[u + 1] = offs[u] + (reserve[u] if used[u] else 0)
        vals = np.empty(offs[n_nulls], dtype=np.float64)
        cum = np.empty(offs[n_nulls], dtype=np.float64)  # running mass from the list's start
        rvals = np.empty(offs[n_nulls], dtype=np.float64)  # the list negated and reversed, for the < side
        rcum = np.empty(offs[n_nulls], dtype=np.float64)
        for t in prange(n_slots):
            for ui in range(null_start[t], null_start[t + 1]):
                u = null_order[ui]
                counts[u] = 0
                if used[u]:
                    counts[u] = _build_into(
                        row, nn_row, sub_start, sub_stop, pool_sub_start, pool_sub_stop, pool_width, null_pool[u],
                        null_size[u], eps, val_t[t], prob_t[t], stack_t[t], sval_t[t], sfl_t[t], sgt_t[t],
                        vals, cum, rvals, rcum, offs[u], reserve[u],
                    )  # fmt: skip
        n_grow = 0
        for u in range(n_nulls):
            if counts[u] < 0:  # more atoms than the null's buffer holds
                overflow[u] = True
                counts[u] = 0
            elif counts[u] > reserve[u]:
                grow[n_grow] = u
                n_grow += 1
        if n_grow > 0:
            for i in range(n_grow):
                reserve[grow[i]] = counts[grow[i]]
            offs2 = np.zeros(n_nulls + 1, dtype=np.int64)
            for u in range(n_nulls):
                offs2[u + 1] = offs2[u] + (reserve[u] if used[u] else 0)
            vals2 = np.empty(offs2[n_nulls], dtype=np.float64)
            cum2 = np.empty(offs2[n_nulls], dtype=np.float64)
            rvals2 = np.empty(offs2[n_nulls], dtype=np.float64)
            rcum2 = np.empty(offs2[n_nulls], dtype=np.float64)
            for u in prange(n_nulls):
                if used[u] and counts[u] <= offs[u + 1] - offs[u]:
                    for z in range(counts[u]):
                        vals2[offs2[u] + z] = vals[offs[u] + z]
                        cum2[offs2[u] + z] = cum[offs[u] + z]
                        rvals2[offs2[u] + z] = rvals[offs[u] + z]
                        rcum2[offs2[u] + z] = rcum[offs[u] + z]
            vals, cum, rvals, rcum = vals2, cum2, rvals2, rcum2
            offs[:] = offs2
            for t in prange(n_slots):
                for i in range(t, n_grow, n_slots):
                    u = grow[i]
                    _build_into(
                        row, nn_row, sub_start, sub_stop, pool_sub_start, pool_sub_stop, pool_width, null_pool[u],
                        null_size[u], eps, val_t[t], prob_t[t], stack_t[t], sval_t[t], sfl_t[t], sgt_t[t],
                        vals, cum, rvals, rcum, offs[u], reserve[u],
                    )  # fmt: skip
        # a gene costs about its guides' atom count, more when an even count sends it through the boundary sweep
        for g in range(n_genes):
            tot = 0
            k = 0
            for c in range(gene_offsets[g], gene_offsets[g + 1]):
                if construct_null[j, c] >= 0:
                    tot += counts[construct_null[j, c]]
                    k += 1
            gene_cost[g] = tot * (1.0 + (k % 2 == 0) * k) + 1.0
        gene_order, gene_start = _deal(gene_cost, n_gene_slots)
        for t in prange(n_gene_slots):
            # the slot's scratch views, taken once: each view touches the shared reference count
            lo_c, hi_c = lohi_t[t, 0], lohi_t[t, 1]
            gt_arr, lt_arr, le_arr, ge_arr, nan_arr, meas_arr = (
                tails_t[t, 0],
                tails_t[t, 1],
                tails_t[t, 2],
                tails_t[t, 3],
                tails_t[t, 4],
                tails_t[t, 5],
            )
            fs, ints, dp = fs_t[t], ints_t[t], dp_t[t]
            for gi in range(gene_start[t], gene_start[t + 1]):
                g = gene_order[gi]
                obs = gene_observed[g, j]
                if np.isnan(obs):
                    out[g, j] = np.nan
                    continue
                k = 0
                for c in range(gene_offsets[g], gene_offsets[g + 1]):
                    u = construct_null[j, c]
                    if u < 0:  # its observed median is nan, so it is not in the gene's median either
                        continue
                    lo_c[k] = offs[u]
                    hi_c[k] = offs[u] + counts[u]
                    # exact tails from the sub-tail mixture, so the gene count that decides
                    # significance never depends on the atom coarsening
                    p = null_pool[u]
                    gt_arr[k], lt_arr[k], le_arr[k], ge_arr[k], nan_arr[k], meas_arr[k] = _pool_tails(
                        row,
                        nn_row,
                        sub_start,
                        sub_stop,
                        pool_sub_start[p],
                        pool_sub_stop[p],
                        pool_width[p],
                        construct_nn[c, j],
                        obs,
                    )
                    k += 1
                # more high than low guides puts the median above obs; as many of each leaves the
                # midpoint of the two central guides to decide, possible for an even measured count
                above = _count_above(k, gt_arr, le_arr, nan_arr, dp)
                below = _count_above(k, lt_arr, ge_arr, nan_arr, dp)
                all_nan = 1.0
                any_measured = False
                for ci in range(k):
                    all_nan *= nan_arr[ci]
                    any_measured |= meas_arr[ci] > 0.0
                if k >= 2 and (k % 2 == 0 or _any_positive(nan_arr, k)):
                    above += _boundary(k, lo_c, hi_c, vals, cum, obs, le_arr, gt_arr, nan_arr, fs, ints, dp)
                    below += _boundary(k, lo_c, hi_c, rvals, rcum, -obs, ge_arr, lt_arr, nan_arr, fs, ints, dp)
                out[g, j] = _two_sided(above, below, 1.0 - all_nan if any_measured else 0.0)
    return out, overflow


#: Hard limit on a guide null's per-thread atom buffer, so a huge screen cannot ask for unbounded memory;
#: the probability pruning keeps the real count far below this, and small problems never approach it.
_ATOM_CAP_MAX = 40_000_000


def _sub_atom_bound(nn: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Upper bound on the median atoms of ``r`` draws from ``nn`` distinct-or-not values, before merging.

    The central order statistic's empirical CDF carries atoms above :data:`_ATOM_EPS` only within the band
    ``|F - 1/2| < ~3/sqrt(r)``; that band holds at most ``nw = min(nn, 7*nn/sqrt(r))`` distinct values, so
    at most ``nw^2 + nw`` even-median atoms, never more than the ``nn(nn+1)/2 + nn`` pairs.
    """
    nw = np.minimum(nn, (7 * nn) // np.sqrt(np.maximum(r, 1)).astype(np.int64) + 4)
    return np.minimum(nn * (nn + 1) // 2 + nn, nw * nw + nw)


def _null_caps(sub_nn, sub_start, sub_stop, pool_sub_start, pool_sub_stop, null_pool, null_size, eps):
    """Per-null atom-buffer size: an upper bound on the atoms a guide null generates before merging.

    A control guide measured in every cell adds one band, for the null's draw count ``m``. One with
    missing cells adds a band per number ``r`` of measured draws whose probability tops ``eps``; by
    Hoeffding's bound those lie within ``sqrt(m log(1/eps) / 2)`` of ``m`` times its measured share.
    """
    widths = (sub_stop - sub_start).astype(np.int64)
    # the measured counts short of the full guide, for the control guides with any missing cell
    partial = {
        sub: np.unique(sub_nn[(sub_nn[:, sub] > 0) & (sub_nn[:, sub] < widths[sub]), sub])
        for sub in np.flatnonzero(((sub_nn > 0) & (sub_nn < widths)).any(axis=0))
    }
    caps = np.empty(null_pool.size, dtype=np.int64)
    for u, (p, m) in enumerate(zip(null_pool, null_size, strict=True)):
        lo, hi = pool_sub_start[p], pool_sub_stop[p]
        bound = _sub_atom_bound(widths[lo:hi], np.int64(m))
        for sub in (sub for sub in partial if lo <= sub < hi):
            nn = partial[sub]
            spread = np.sqrt(m * np.log(1.0 / eps) / 2.0)
            r_lo = np.maximum(1, np.floor(m * nn / widths[sub] - spread)).astype(np.int64)
            r_hi = np.minimum(m, np.ceil(m * nn / widths[sub] + spread)).astype(np.int64)
            bands = np.maximum(r_hi - r_lo + 1, 0) * _sub_atom_bound(nn, r_lo)
            bound[sub - lo] = max(bound[sub - lo], int(bands.max()))
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

    The null is not simulated but computed exactly: the median of a resample is an order statistic (a binomial tail in the guide's empirical distribution, and for an even cell count the mean of two consecutive order statistics), the guide null is the cell-count-weighted mixture of its control guides' such distributions, and the gene null is the order-statistic distribution of its independent guide nulls, summed over its guides by a dynamic program so that a gene of many guides stays cheap.
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
        Medians skip missing values. A guide's null draws as many cells as the guide measures in the feature, a guide that measures none leaves its gene's median and null, and a null median that comes out missing counts neither way.
        A feature whose observed median is missing, or that no control cell measures, is left ``NaN``.

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
    # a guide is its group and its label, so control guides of two control genes that share a label stay apart
    sub_by = [*by, replicate, *([condition] if condition is not None and condition not in by else [])]
    sub_codes, sub_keys = group_codes(adata, sub_by)

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
    construct_of_row = np.zeros(adata.n_obs, dtype=np.int64)
    construct_of_row[rows_kept] = row_construct
    construct_observed = grouped_stat(values, construct_of_row, constructs.size, MEDIAN, mask=tested)
    construct_size = np.bincount(row_construct, minlength=constructs.size).astype(np.int64)
    # the cells each guide measures per feature: its null draws that many, as its median is over them
    has_nan = bool(np.isnan(values).any())
    if has_nan:
        by_construct = np.argsort(row_construct, kind="stable")
        construct_nn = np.add.reduceat(
            ~np.isnan(values[rows_kept[by_construct]]),
            np.searchsorted(row_construct[by_construct], np.arange(constructs.size)),
            axis=0,
        ).astype(np.int64)
    else:
        construct_nn = np.repeat(construct_size[:, None], values.shape[1], axis=1)
    in_pool = pool_of_row >= 0
    pool_median = grouped_stat(values, np.searchsorted(pool_keys, pool_of_row), pool_keys.size, MEDIAN, mask=in_pool)

    # Sort each control guide's cells, per feature, so the kernels can read order statistics; a guide's
    # non-nan values come first (nan sorts last) and sub_nn counts them. Lay out the pool-to-guide index.
    pool_sorted = np.ascontiguousarray(values[pool_rows].T)
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
        construct_nn,
        construct_observed,
    )
    # a guide's null depends only on its pool and measured cell count, so guides sharing both share it
    stride = construct_nn.max() + 1
    if has_nan:
        null_of = (construct_pool[:, None] * stride + construct_nn).T
        measured = null_of % stride > 0
        null_keys, inverse = np.unique(null_of[measured], return_inverse=True)
        construct_null = np.full(null_of.shape, -1, dtype=np.int64)
        construct_null[measured] = inverse
    else:  # every guide measures every feature, so its null is the same in each
        null_keys, inverse = np.unique(construct_pool * stride + construct_size, return_inverse=True)
        construct_null = np.tile(inverse.astype(np.int64), (n_features, 1))
    null_pool, null_size = np.divmod(null_keys, stride)
    n_threads = numba.get_num_threads()
    # threads take gene slots one at a time as they free up, so one costly gene does not hold up the rest
    with numba.parallel_chunksize(1):
        gene_p, overflow = _gene_pvalues(
            pool_sorted,
            sub_nn,
            sub_start,
            sub_stop,
            pool_sub_start,
            pool_sub_stop,
            pool_width,
            construct_nn,
            construct_null,
            null_pool.astype(np.int64),
            null_size.astype(np.int64),
            _null_caps(sub_nn, sub_start, sub_stop, pool_sub_start, pool_sub_stop, null_pool, null_size, _ATOM_EPS),
            gene_offsets,
            gene_observed,
            _ATOM_EPS,
            n_threads,
            8 * n_threads,
        )
    if overflow.any():
        logger.warning(
            "bootstrap_features: %d guide null(s) have too many distinct medians to hold, so a gene whose "
            "null median would fall between two of their guides counts that case as a tie",
            int(overflow.sum()),
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

    sub_labels = [str(sub_keys[s][len(by)]) for s in range(len(sub_keys))]
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
        guides.insert(
            2, condition, np.repeat([str(sub_keys[s][sub_by.index(condition)]) for s in construct_sub], n_features)
        )
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
