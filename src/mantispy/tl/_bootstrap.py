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

from mantispy._core._numba import MEDIAN, N_CHUNKS, _counting_order, _kernel_input, _masked_offsets, grouped_stat
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

#: The atoms the gene null needs leave out each far tail of a control guide's resampled median up to this
#: mass, and each row of even-count midpoints past this share of its lower value's mass, a few times this in
#: all, well under the Monte-Carlo error of any feasible simulation; small problems keep every atom, so the
#: result still matches a brute-force enumeration. :func:`_null_caps` sizes its buffer to the same band.
_ATOM_EPS = 1e-8
#: A gene whose even-count boundary reads a guide null that lacks more than this of its mass, which only a
#: null whose atoms overflow their buffer does, is left ``NaN`` rather than given a p-value off by that much.
_MAX_LOST = 1e-6
#: A gene's boundary below its observed median, read as its split's mass less the part above, is swept directly
#: when it comes out under this share of the split, so the subtraction cannot cost a small p-value its digits.
_SWEEP_BOTH = 1e-6


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


@njit(parallel=True, cache=True, nogil=True)
def _guide_pvalues(
    pool_sorted: np.ndarray,
    sub_nn: np.ndarray,
    sub_start: np.ndarray,
    pool_sub_start: np.ndarray,
    pool_sub_stop: np.ndarray,
    construct_pool: np.ndarray,
    construct_nn: np.ndarray,
    construct_observed: np.ndarray,
) -> np.ndarray:
    """Two-sided p-value of every guide against its control pool, exactly.

    A guide's null draws as many cells as the guide has measured in the feature, from the control cells
    that measure it; with no such control cell the p-value is ``NaN``.
    """
    n_constructs, n_features = construct_observed.shape
    out = np.empty((n_constructs, n_features), dtype=np.float64)
    for c in prange(n_constructs):
        p = construct_pool[c]
        s0 = pool_sub_start[p]
        s1 = pool_sub_stop[p]
        for j in range(n_features):
            obs = construct_observed[c, j]
            if np.isnan(obs):
                out[c, j] = np.nan
                continue
            gt, lt, _, _, measured = _pool_tails(pool_sorted[j], sub_nn[j], sub_start, s0, s1, construct_nn[c, j], obs)
            out[c, j] = _two_sided(gt, lt, measured)
    return out


@njit(cache=True, nogil=True)
def _pool_tails(row, sub_nn_row, sub_start, s0, s1, m, x):
    """A guide null's exact tails at ``x``: the mixture of its control guides' resampled medians.

    Each control guide is weighted by the cells it measures in the feature, and its median is of ``m``
    draws from those. Returns ``P(> x)``, ``P(< x)``, ``P(<= x)``, ``P(>= x)``, each summed directly so
    that a small one keeps its digits, and ``P(measured)``: one, or zero when no control cell measures it.
    """
    total = 0
    for sub in range(s0, s1):
        total += sub_nn_row[sub]
    if total == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    gt = lt = le = ge = 0.0
    for sub in range(s0, s1):
        nn = sub_nn_row[sub]
        if nn == 0:
            continue
        w = nn / total
        g = _median_gt(row, sub_start[sub], nn, m, x)
        l_ = _median_lt(row, sub_start[sub], nn, m, x)
        gt += w * g
        lt += w * l_
        le += w * (1.0 - g)
        ge += w * (1.0 - l_)
    return gt, lt, le, ge, 1.0


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

    Each atom is ``(value, scale*prob)``. Only the two far tails of the median's distribution are left out,
    each of mass at most ``eps``, and for even r the far end of each row of midpoints, at most ``eps`` of the
    row's lower value's mass; the atoms between are all kept, however small, as many of them can carry much
    mass between them, and ties make their sizes rise and fall. For even r the distinct values and their tail
    powers are precomputed into the scratch arrays ``sval``/``sfl``/``sgt`` so the off-diagonal inner loop
    is a plain product, O(distinct) powers rather than O(distinct^2). Returns ``-1`` if ``out_val`` fills up.
    """
    if n == 0:
        return cnt
    t = r // 2
    if r % 2 == 1:
        i = lo
        end = lo + n
        above = 1.0  # P(median >= v[i])
        while i < end and above > eps:
            a = v[i]
            j = i
            while j < end and v[j] == a:
                j += 1
            beyond = _binom_cdf(t, r, (j - lo) / n)  # P(median > a)
            pr = above - beyond
            if 1.0 - beyond > eps and pr > 0.0:  # not in the far lower tail
                if cnt >= out_val.shape[0]:
                    return -1
                out_val[cnt] = a
                out_prob[cnt] = scale * pr
                cnt += 1
            above = beyond
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
    above = 1.0  # P(t-th order stat >= sval[ka])
    for ka in range(nd):
        if above <= eps:  # the far upper tail
            break
        f_prev = sfl[ka - 1] if ka > 0 else 0.0
        beyond = _binom_cdf(t - 1, r, sfl[ka])  # P(t-th order stat > a)
        marg = above - beyond  # P(t-th order stat == a)
        above = beyond
        if 1.0 - beyond <= eps or marg <= 0.0:  # the far lower tail
            continue
        # coef = C(r,t) * (F^t - F_prev^t); the diagonal takes off coef * (1-F)^(r-t), and the row's
        # off-diagonal atoms are coef * rb[lb]
        pr = 0.0
        coef = 0.0
        log_coef = 0.0
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
        if diag > 0.0:
            if cnt >= out_val.shape[0]:
                return -1
            out_val[cnt] = sval[ka]
            out_prob[cnt] = scale * diag
            cnt += 1
        for lb in range(ka + 1, nd):
            if lb > ka + 1:
                pr = coef * (sgt[lb - 1] - sgt[lb]) if direct else pr * sgt[lb]
            if pr > 0.0:
                if cnt >= out_val.shape[0]:
                    return -1
                out_val[cnt] = (sval[ka] + sval[lb]) / 2.0
                out_prob[cnt] = scale * pr
                cnt += 1
            # the mass of the row's midpoints past this one, coef * (1 - F_lb)^(r-t)
            if direct:
                rest = coef * sgt[lb]
            else:
                rest = exp(log_coef + (r - t) * log1p(-sfl[lb])) if sfl[lb] < 1.0 else 0.0
            if rest <= eps * marg:
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


#: Radix digits of :func:`_radix_sort2` are this many bits wide, and its scratch counts one more than their values.
_RADIX_BITS = 11
_SORT_SCRATCH = (1 << _RADIX_BITS) + 1


@njit(cache=True, nogil=True)
def _radix_key(b):
    """A float64's bits as an unsigned key in the float's order: the sign flipped, or every bit for a negative."""
    return ~b if b >> np.uint64(63) else b | (np.uint64(1) << np.uint64(63))


@njit(cache=True, nogil=True)
def _radix_sort2(val, prob, lo, hi, val2, prob2, count):
    """Stable sort of ``val[lo:hi]`` carrying ``prob``, by digits of the float's bits from the lowest up (LSD radix).

    Each pass scatters the range by one digit into ``val2``/``prob2`` and back, a pass whose digit is the same for
    every value is skipped, and a short range is left to :func:`_sort2`. Linear in the length per digit, where the
    comparison sort's ``n log n`` dominated building a guide null of millions of atoms. ``count`` is the histogram.
    """
    n = hi - lo
    if n <= 4096:
        _sort2(val, prob, lo, hi, count)
        return
    mask = np.uint64((1 << _RADIX_BITS) - 1)
    src_b, dst_b = val.view(np.uint64), val2.view(np.uint64)
    src_p, dst_p = prob, prob2
    s_off, d_off = lo, 0
    moved = 0
    for shift in range(0, 64, _RADIX_BITS):
        count[:] = 0
        for i in range(n):
            count[np.int64((_radix_key(src_b[s_off + i]) >> np.uint64(shift)) & mask) + 1] += 1
        if count.max() == n:  # every value shares this digit
            continue
        for d in range(count.size - 1):
            count[d + 1] += count[d]
        for i in range(n):
            b = src_b[s_off + i]
            d = np.int64((_radix_key(b) >> np.uint64(shift)) & mask)
            dst_b[d_off + count[d]] = b
            dst_p[d_off + count[d]] = src_p[s_off + i]
            count[d] += 1
        src_b, dst_b = dst_b, src_b
        src_p, dst_p = dst_p, src_p
        s_off, d_off = d_off, s_off
        moved += 1
    if moved % 2 == 1:  # the sorted values sit in the scratch
        for i in range(n):
            val[lo + i] = val2[i]
            prob[lo + i] = prob2[i]


@njit(cache=True, nogil=True)
def _collapse(val_buf, prob_buf, lo, hi, val2, prob2, ints):
    """Sort ``[lo, hi)`` by value and merge equal values in place; returns the new end."""
    if hi - lo <= 0:
        return lo
    _radix_sort2(val_buf, prob_buf, lo, hi, val2, prob2, ints)
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
    pool_sorted_row, sub_nn_row, sub_start, s0, s1, m, eps, val_buf, prob_buf, val2, prob2, ints, sval, sfl, sgt
):
    """Build one guide null as merged, sorted atoms in ``val_buf``/``prob_buf``; returns their count.

    Mixes every control guide's exact median-atom distribution, weighted as in :func:`_pool_tails`, then
    sorts and merges equal values. Returns ``-1`` if the atoms do not fit the buffer.
    """
    total = 0
    for sub in range(s0, s1):
        total += sub_nn_row[sub]
    cnt = 0
    for sub in range(s0, s1):
        nn = sub_nn_row[sub]
        if nn == 0:
            continue
        cnt = _add_sub_atoms(
            pool_sorted_row, sub_start[sub], nn, m, nn / total, eps, val_buf, prob_buf, cnt, sval, sfl, sgt
        )
        if cnt < 0:
            return -1
    if cnt == 0:
        return 0
    return _collapse(val_buf, prob_buf, 0, cnt, val2, prob2, ints)


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
def _tie(k, g, aval, lo_c, le_end, gt_start, cut, vals, cum, obs, plow, phigh, pnan, fs, dp, stride):
    """The term of :func:`_boundary` for low value ``aval`` whose midpoint with a high atom equals ``obs``.

    That is ``P(#high == #low, the largest low guide at aval, the smallest high one at a midpoint of exactly obs)``:
    the term of ``aval`` with the cut taken at ``>= obs``, less the one at ``> obs`` that the sweep sums. Guide
    ``g`` found it; an earlier guide with an atom at ``aval`` has taken it already. The low weights come from
    each guide's atoms below and at ``aval``, and the cut, as :func:`_boundary` holds it, is the ``> obs`` one.
    """
    inv_low, inv_high, top, below, grown, wh, wh_ge = fs[0], fs[1], fs[2], fs[3], fs[5], fs[4], fs[6]
    for b in range(g):
        n_lt = _count_below(vals, lo_c[b], le_end[b], aval, False)
        if lo_c[b] + n_lt < le_end[b] and vals[lo_c[b] + n_lt] == aval:
            return 0.0
    for b in range(k):
        lo = lo_c[b]
        n_lt = _count_below(vals, lo, le_end[b], aval, False)
        n_le = _count_below(vals, lo, le_end[b], aval, True)
        start = cum[lo + n_lt - 1] if n_lt > 0 else 0.0
        below[b] = plow[b] * start * inv_low[b]
        grown[b] = plow[b] * ((cum[lo + n_le - 1] if n_le > 0 else 0.0) - start) * inv_low[b]
        q = cut[b]
        while q > gt_start[b] and (aval + vals[q - 1]) / 2.0 >= obs:
            q -= 1
        wh_ge[b] = wh[b]
        if q < cut[b]:
            wh_ge[b] = phigh[b] * (top[b] - (cum[q - 1] if q > lo else 0.0)) * inv_high[b]
    return _count_grown(k, below, grown, wh_ge, pnan, dp, stride) - _count_grown(k, below, grown, wh, pnan, dp, stride)


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
    over the guides (:func:`_count_grown`) that keeps every term positive. ``fs`` (7 rows), ``ints``
    (6 rows) and ``dp`` (4 rows of ``2 k + 5``) are scratch.

    Also returns ``P(#high == #low >= 1)`` over the guides with atoms on their side, the part of it whose
    midpoint equals ``obs`` (see :func:`_tie`), and whether no atom equals ``obs``: then the rest of that split
    has its midpoint below ``obs``.
    """
    inv_low, inv_high, top, wl, wh, grown = fs[0], fs[1], fs[2], fs[3], fs[4], fs[5]
    clean = True
    tie = 0.0
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
        clean &= cnt_le == _count_below(vals, lo, hi, obs, False)  # no atom at obs
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
    split = _count_at(k, -1, 0, wh, wl, pnan, dp, stride) - all_nan
    if split <= 0.0:
        return 0.0, 0.0, 0.0, clean
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
                if (vals[p] + vtop) / 2.0 == obs:  # vtop is the largest high atom that can tie this one
                    tie += _tie(
                        k, a, vals[p], lo_c, le_end, gt_start, cut, vals, cum, obs, plow, phigh, pnan, fs, dp, stride
                    )
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
    return total, split, tie, clean


@njit(cache=True, nogil=True)
def _build_into(
    row, nn_row, sub_start, pool_sub_start, pool_sub_stop, p, m, eps,
    val_buf, prob_buf, val2, prob2, ints, sval, sfl, sgt, vals, cum, rvals, rcum, lo, room,
):  # fmt: skip
    """Build the null of pool ``p`` at ``m`` draws and, if its merged atoms fit ``room``, write them at ``lo``.

    The store keeps the atom values, their running mass from the bottom (``cum``) and, negated and reversed,
    from the top (``rvals``/``rcum``). Returns the merged count, or ``-1`` if the atoms overflow the buffer,
    and the mass the pruning dropped (all of it on an overflow).
    """
    end = _build_construct(
        row, nn_row, sub_start, pool_sub_start[p], pool_sub_stop[p], m, eps,
        val_buf, prob_buf, val2, prob2, ints, sval, sfl, sgt,
    )  # fmt: skip
    if end < 0:
        return end, 1.0
    kept = 0.0
    for z in range(end):
        kept += prob_buf[z]
    if end <= room:
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
    return end, max(0.0, 1.0 - kept)


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
    construct_pool,
    construct_nn,
    null_base,
    null_index,
    null_pool,
    null_size,
    null_caps,
    gene_offsets,
    gene_observed,
    eps,
    max_lost,
    n_slots,
    n_gene_slots,
):
    """Two-sided p-value of every gene against its guides' control pools, exactly.

    A guide's null depends only on its pool and measured cell count, so per feature each distinct
    ``(null_pool, null_size)`` null is built once into a shared atom store, and the genes then only
    read it. Guide ``c``'s null in feature ``j`` is ``null_index[null_base[pool] + construct_nn[c, j]]``,
    ``-1`` where the guide measures nothing and so leaves the gene. The nulls are dealt to ``n_slots`` slots and the genes to
    ``n_gene_slots`` (see :func:`_deal`), each with its own scratch; more gene slots than threads let a
    thread that finishes early take another.

    A gene whose even-count boundary needs the shape of a guide null that lost more than ``max_lost`` of its
    mass, to pruning or to an atom buffer it overflowed, is left ``NaN``; the second array returned marks those.
    """
    n_genes = gene_offsets.size - 1
    n_features = gene_observed.shape[1]
    n_constructs = construct_nn.shape[0]
    n_nulls = null_pool.size
    out = np.empty((n_genes, n_features), dtype=np.float64)
    unresolved = np.zeros((n_genes, n_features), dtype=np.bool_)
    max_cap = 1
    for u in range(n_nulls):
        max_cap = max(max_cap, null_caps[u])
    max_k = max(1, np.diff(gene_offsets).max())
    max_width = max(1, (sub_stop - sub_start).max())
    val_t = np.empty((n_slots, max_cap), dtype=np.float64)
    prob_t = np.empty((n_slots, max_cap), dtype=np.float64)
    val2_t = np.empty((n_slots, max_cap), dtype=np.float64)  # the radix sort's other buffer
    prob2_t = np.empty((n_slots, max_cap), dtype=np.float64)
    sort_t = np.empty((n_slots, max(_SORT_SCRATCH, 128)), dtype=np.int64)
    sval_t = np.empty((n_slots, max_width), dtype=np.float64)
    sfl_t = np.empty((n_slots, max_width), dtype=np.float64)
    sgt_t = np.empty((n_slots, max_width), dtype=np.float64)
    # per gene: lo_c, hi_c, its guides' tails, and the _boundary scratch; rows written at every atom,
    # padded by a cache line so that slots on different cores share none
    row_k = max_k + 8
    lohi_t = np.empty((n_gene_slots, 2, row_k), dtype=np.int64)
    tails_t = np.empty((n_gene_slots, 6, row_k), dtype=np.float64)
    fs_t = np.empty((n_gene_slots, 7, row_k), dtype=np.float64)
    ints_t = np.empty((n_gene_slots, 6, row_k), dtype=np.int64)
    dp_t = np.empty((n_gene_slots, 4, 2 * max_k + 5 + 8), dtype=np.float64)
    used = np.empty(n_nulls, dtype=np.bool_)
    counts = np.empty(n_nulls, dtype=np.int64)
    lost = np.empty(n_nulls, dtype=np.float64)  # the mass each null's atoms lack
    offs = np.zeros(n_nulls + 1, dtype=np.int64)
    reserve = np.zeros(n_nulls, dtype=np.int64)  # the merged length each null's stretch of the store holds
    grow = np.empty(n_nulls, dtype=np.int64)
    null_order, null_start = _deal(null_caps.astype(np.float64), n_slots)
    gene_cost = np.empty(n_genes, dtype=np.float64)
    vals = np.empty(0, dtype=np.float64)  # the atom values of every used null, back to back
    cum = np.empty(0, dtype=np.float64)  # running mass from each list's start
    rvals = np.empty(0, dtype=np.float64)  # each list negated and reversed, for the < side
    rcum = np.empty(0, dtype=np.float64)
    for j in range(n_features):
        row = pool_sorted[j]
        nn_row = sub_nn[j]
        used[:] = False
        for c in range(n_constructs):
            u = null_index[null_base[construct_pool[c]] + construct_nn[c, j]]
            if u >= 0:
                used[u] = True
        # build each used null once into its reserved stretch of the store; a null longer than its reserve
        # (every null, at the first feature) is built again once the store has grown to hold it
        for u in range(n_nulls):
            offs[u + 1] = offs[u] + (reserve[u] if used[u] else 0)
        if offs[n_nulls] > vals.size:  # the store is kept across features, so its pages are touched once
            vals = np.empty(offs[n_nulls], dtype=np.float64)
            cum = np.empty(offs[n_nulls], dtype=np.float64)
            rvals = np.empty(offs[n_nulls], dtype=np.float64)
            rcum = np.empty(offs[n_nulls], dtype=np.float64)
        for t in prange(n_slots):
            for ui in range(null_start[t], null_start[t + 1]):
                u = null_order[ui]
                counts[u] = 0
                lost[u] = 0.0
                if used[u]:
                    counts[u], lost[u] = _build_into(
                        row, nn_row, sub_start, pool_sub_start, pool_sub_stop, null_pool[u], null_size[u], eps,
                        val_t[t], prob_t[t], val2_t[t], prob2_t[t], sort_t[t], sval_t[t], sfl_t[t], sgt_t[t],
                        vals, cum, rvals, rcum, offs[u], reserve[u],
                    )  # fmt: skip
        n_grow = 0
        for u in range(n_nulls):
            if counts[u] < 0:  # more atoms than the null's buffer holds: it keeps none
                counts[u] = 0
            elif counts[u] > reserve[u]:
                grow[n_grow] = u
                n_grow += 1
        if n_grow > 0:
            for i in range(n_grow):
                reserve[grow[i]] = counts[grow[i]] + counts[grow[i]] // 4  # room for the next features' nulls
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
                        row, nn_row, sub_start, pool_sub_start, pool_sub_stop, null_pool[u], null_size[u], eps,
                        val_t[t], prob_t[t], val2_t[t], prob2_t[t], sort_t[t], sval_t[t], sfl_t[t], sgt_t[t],
                        vals, cum, rvals, rcum, offs[u], reserve[u],
                    )  # fmt: skip
        # a gene costs about its guides' atom count, more when an even count sends it through the boundary sweep
        for g in range(n_genes):
            tot = 0
            k = 0
            for c in range(gene_offsets[g], gene_offsets[g + 1]):
                u = null_index[null_base[construct_pool[c]] + construct_nn[c, j]]
                if u >= 0:
                    tot += counts[u]
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
                worst = 0.0  # the most mass any measured guide null of the gene lacks atoms for
                for c in range(gene_offsets[g], gene_offsets[g + 1]):
                    u = null_index[null_base[construct_pool[c]] + construct_nn[c, j]]
                    if u < 0:  # its observed median is nan, so it is not in the gene's median either
                        continue
                    lo_c[k] = offs[u]
                    hi_c[k] = offs[u] + counts[u]
                    # exact tails from the control guides' order statistics, so the gene count that decides
                    # significance never depends on the atom pruning
                    p = null_pool[u]
                    gt_arr[k], lt_arr[k], le_arr[k], ge_arr[k], meas_arr[k] = _pool_tails(
                        row, nn_row, sub_start, pool_sub_start[p], pool_sub_stop[p], construct_nn[c, j], obs
                    )
                    nan_arr[k] = 1.0 - meas_arr[k]
                    if meas_arr[k] > 0.0:
                        worst = max(worst, lost[u])
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
                    if worst > max_lost:  # the boundary needs the shape of a null that kept too little of it
                        out[g, j] = np.nan
                        unresolved[g, j] = True
                        continue
                    over, split, tie, clean = _boundary(
                        k, lo_c, hi_c, vals, cum, obs, le_arr, gt_arr, nan_arr, fs, ints, dp
                    )
                    above += over
                    # With no atom at obs, the split's midpoint is below obs where it is neither above nor on it.
                    # Read off as the difference only while that keeps its digits, else swept as well.
                    if clean and split - over - tie >= _SWEEP_BOTH * split:
                        below += split - over - tie
                    else:
                        below += _boundary(k, lo_c, hi_c, rvals, rcum, -obs, ge_arr, lt_arr, nan_arr, fs, ints, dp)[0]
                out[g, j] = _two_sided(above, below, 1.0 - all_nan if any_measured else 0.0)
    return out, unresolved


@njit(parallel=True, cache=True, nogil=True)
def _missing_and_infinite(values):
    """Whether ``values`` holds a ``NaN``, and whether it holds an infinity, in one pass and no copy."""
    n_rows = values.shape[0]
    n_chunks = min(N_CHUNKS, max(n_rows, 1))
    flags = np.zeros((n_chunks, 2), dtype=np.bool_)
    for chunk in prange(n_chunks):
        for i in range(chunk, n_rows, n_chunks):
            for j in range(values.shape[1]):
                x = values[i, j]
                if np.isnan(x):
                    flags[chunk, 0] = True
                elif np.isinf(x):
                    flags[chunk, 1] = True
    return flags[:, 0].any(), flags[:, 1].any()


@njit(parallel=True, cache=True, nogil=True)
def _measured_counts(values, order, offsets):
    """Per group and feature, how many of the group's rows ``order[offsets[g]:offsets[g + 1]]`` are not ``NaN``."""
    n_groups = offsets.size - 1
    out = np.zeros((n_groups, values.shape[1]), dtype=np.int64)
    for g in prange(n_groups):
        for i in range(offsets[g], offsets[g + 1]):
            row = order[i]
            for j in range(values.shape[1]):
                out[g, j] += not np.isnan(values[row, j])
    return out


@njit(cache=True, nogil=True)
def _null_index(construct_pool, construct_nn, null_base, n_keys):
    """Number the distinct ``(pool, measured count)`` nulls the guides use, over every feature.

    Key ``null_base[pool] + count`` maps to its null, numbered by pool and then count, or to ``-1``
    where no guide uses it; a count of zero is never used, as a guide measuring nothing has no null.
    """
    index = np.full(n_keys, -1, dtype=np.int64)
    for c in range(construct_nn.shape[0]):
        base = null_base[construct_pool[c]]
        for j in range(construct_nn.shape[1]):
            if construct_nn[c, j] > 0:
                index[base + construct_nn[c, j]] = 0
    n = 0
    for key in range(n_keys):
        if index[key] == 0:
            index[key] = n
            n += 1
    return index


#: Hard limit on a guide null's per-thread atom buffer, so a huge screen cannot ask for unbounded memory;
#: the probability pruning keeps the real count far below this, and small problems never approach it.
_ATOM_CAP_MAX = 40_000_000


def _sub_atom_bound(nn: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Upper bound on the median atoms of ``r`` draws from ``nn`` distinct-or-not values, before merging.

    Past its far tails of :data:`_ATOM_EPS`, the central order statistic lies within the band
    ``|F - 1/2| < ~3/sqrt(r)``; that band holds at most ``nw = min(nn, 7*nn/sqrt(r))`` distinct values, so
    at most ``nw^2 + nw`` even-median atoms, never more than the ``nn(nn+1)/2 + nn`` pairs.
    """
    nw = np.minimum(nn, (7 * nn) // np.sqrt(np.maximum(r, 1)).astype(np.int64) + 4)
    return np.minimum(nn * (nn + 1) // 2 + nn, nw * nw + nw)


def _null_caps(sub_start, sub_stop, pool_sub_start, pool_sub_stop, null_pool, null_size):
    """Per-null atom-buffer size: an upper bound on the atoms a guide null generates before merging.

    A null of ``m`` draws adds :func:`_sub_atom_bound` atoms for each control guide of its pool, at most
    those of the guide's full cell count, as it measures no more cells than it has in any feature.
    """
    widths = (sub_stop - sub_start).astype(np.int64)
    caps = np.empty(null_pool.size, dtype=np.int64)
    for p in np.unique(null_pool):
        nulls = np.flatnonzero(null_pool == p)
        w = widths[pool_sub_start[p] : pool_sub_stop[p]]
        bound = _sub_atom_bound(w[None, :], null_size[nulls].astype(np.int64)[:, None]).sum(axis=1)
        caps[nulls] = np.minimum(bound + 16, _ATOM_CAP_MAX)
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
        ValueError: ``control_scope`` is unknown, a scope that needs ``condition`` or ``reference_condition`` lacks it, ``"within_perturbation"`` is used without ``condition`` in ``groupby``, the controls in ``reference_condition`` are empty, no guide has a null to draw from, or the matrix holds an infinite value.

    Notes:
        The p-value is two-sided, ``2 * min(p, 1 - p)`` with ``p`` the probability that a null median exceeds the observed one.
        It is exact rather than simulated, so it has no lower floor and can be zero when the observed median lies beyond every attainable null median.
        A null median equal to the observed one counts half, so a feature where most cells share one value, such as a count that is mostly zero, is not called because its median cannot move; Brieflow counts such ties as below.
        Medians skip missing values, and each feature is tested on the cells that measure it: a guide's null picks a control guide in proportion to the cells it measures in the feature and draws as many of those as the tested guide measures, and a guide that measures none leaves its gene's median and null.
        A feature whose observed median is missing, or that no control cell measures, is left ``NaN``.
        So is a gene whose median can fall between two of its guides when a guide null has more distinct medians than can be held, with a warning; this takes thousands of cells in every control guide against guides of a few.

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
    has_nan, has_inf = _missing_and_infinite(values)
    if has_inf:
        raise ValueError(
            "bootstrap_features: the matrix holds infinite values, whose medians are undefined; drop or fill them "
            "first, for example with mt.pp.feature_select"
        )
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
    pool_keys = np.unique(pool_of_row[pool_rows])
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
    if has_nan:
        construct_nn = _measured_counts(values, *_masked_offsets(construct_of_row, constructs.size, tested))
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

    genes, gene_starts = np.unique(construct_unit, return_index=True)
    gene_offsets = np.append(gene_starts, construct_unit.size).astype(np.int64)
    gene_of_construct = np.searchsorted(genes, construct_unit)
    gene_observed = grouped_stat(construct_observed, gene_of_construct, genes.size, MEDIAN)
    gene_baseline = grouped_stat(pool_median[construct_pool], gene_of_construct, genes.size, MEDIAN)

    guide_p = _guide_pvalues(
        pool_sorted,
        sub_nn,
        sub_start,
        pool_sub_start,
        pool_sub_stop,
        construct_pool,
        construct_nn,
        construct_observed,
    )
    # a guide's null depends only on its pool and measured cell count, so guides sharing both share it; a pool's
    # keys run over the counts up to its largest guide
    pool_largest = np.zeros(pool_keys.size, dtype=np.int64)
    np.maximum.at(pool_largest, construct_pool, construct_size)
    null_base = np.concatenate([[0], np.cumsum(pool_largest + 1)]).astype(np.int64)
    null_index = _null_index(construct_pool, construct_nn, null_base, int(null_base[-1]))
    keys = np.flatnonzero(null_index >= 0)
    null_pool = (np.searchsorted(null_base, keys, side="right") - 1).astype(np.int64)
    null_size = (keys - null_base[null_pool]).astype(np.int64)
    n_threads = numba.get_num_threads()
    # threads take gene slots one at a time as they free up, so one costly gene does not hold up the rest
    with numba.parallel_chunksize(1):
        gene_p, unresolved = _gene_pvalues(
            pool_sorted,
            sub_nn,
            sub_start,
            sub_stop,
            pool_sub_start,
            pool_sub_stop,
            construct_pool,
            construct_nn,
            null_base,
            null_index,
            null_pool,
            null_size,
            _null_caps(sub_start, sub_stop, pool_sub_start, pool_sub_stop, null_pool, null_size),
            gene_offsets,
            gene_observed,
            _ATOM_EPS,
            _MAX_LOST,
            n_threads,
            8 * n_threads,
        )
    if unresolved.any():
        logger.warning(
            "bootstrap_features: %d gene-feature p-value(s) are left NaN: a gene whose median can fall between two "
            "of its guides needs their nulls' distinct medians, and a guide null had too many to hold",
            int(unresolved.sum()),
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

    def sub_level(position: int) -> np.ndarray:
        return sub_keys.get_level_values(position).astype(str).to_numpy()[construct_sub]

    guides = pd.DataFrame(
        {
            "group": np.repeat([gene_labels[g] for g in gene_of_construct], n_features),
            "replicate": np.repeat(sub_level(len(by)), n_features),
            "n_cells": np.repeat(construct_size, n_features),
            "feature": np.tile(features, constructs.size),
            "effect": (construct_observed - pool_median[construct_pool]).ravel(),
            "pvalue": np.where(np.isfinite(construct_observed), guide_p, np.nan).ravel(),
        }
    )
    if condition is not None:
        guides.insert(2, condition, np.repeat(sub_level(sub_by.index(condition)), n_features))
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
