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
#: result still matches a brute-force enumeration. :func:`_null_caps` bounds a null's atoms by the same band.
_ATOM_EPS = 1e-8
#: A gene whose even-count boundary reads a guide null that lacks more than this of its mass, which only a
#: null of more atoms than :func:`_null_caps` allows does, is left ``NaN`` rather than given a p-value off by that much.
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
    # down from the largest low value: the terms shrink, and those below a value add at most P(max of the low
    # draws below it) times its own share of high draws, so the sum stops once that is lost to rounding
    j = lo + cle
    while j > lo:
        a = np.float64(v[j - 1])  # a float32 midpoint would round differently from the float64 observed median
        i = j - 1
        while i > lo and v[i - 1] == a:
            i -= 1
        below = ((i - lo) / cle) ** t
        pmax = ((j - lo) / cle) ** t - below  # P(max of t low draws == a)
        # first high value b with the actual midpoint (a+b)/2 > x
        left = int(lo + cle)
        right = int(lo + n)
        while left < right:
            mid = (left + right) // 2
            if (a + v[mid]) / 2.0 > x:
                right = mid
            else:
                left = mid + 1
        if left == lo + n:  # no high value passes, nor will one for a smaller low value
            break
        share = ((n - _count_below(v, lo, lo + n, v[left], False)) / nhigh) ** t  # count >= v[left]
        contrib += pmax * share
        if below * share <= 1e-17 * contrib:
            break
        j = i
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
        above = ((end - j) / nge) ** t
        pmin = ((end - i) / nge) ** t - above  # P(min of t high draws == b)
        # largest low value a with the actual midpoint (a+b)/2 < x
        left = int(lo)
        right = int(lo + hi)
        while left < right:
            mid = (left + right) // 2
            if (v[mid] + b) / 2.0 < x:
                left = mid + 1
            else:
                right = mid
        if left == lo:  # no low value passes, nor will one for a larger high value
            break
        share = ((left - lo) / nlow) ** t
        contrib += pmin * share
        if above * share <= 1e-17 * contrib:
            break
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
def _add_sub_atoms(v, lo, n, r, scale, eps, out_val, out_prob, cnt, limit, sval, sfl, sgt):
    """Append the atoms of the median of r draws from ``v[lo:lo+n]`` (sorted, no nan), and count them.

    Each atom is ``(value, scale*prob)``. Only the two far tails of the median's distribution are left out,
    each of mass at most ``eps``, and for even r the far end of each row of midpoints, at most ``eps`` of the
    row's lower value's mass; the atoms between are all kept, however small, as many of them can carry much
    mass between them, and ties make their sizes rise and fall. For even r the distinct values and their tail
    powers are precomputed into the scratch arrays ``sval``/``sfl``/``sgt`` so the off-diagonal inner loop
    is a plain product, O(distinct) powers rather than O(distinct^2). Only the atoms that fit ``out_val`` are
    written, so an empty one counts them. Returns the count, or ``-1`` once it passes ``limit``.
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
                if cnt >= limit:
                    return -1
                if cnt < out_val.shape[0]:
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
            if cnt >= limit:
                return -1
            if cnt < out_val.shape[0]:
                out_val[cnt] = sval[ka]
                out_prob[cnt] = scale * diag
            cnt += 1
        for lb in range(ka + 1, nd):
            if lb > ka + 1:
                pr = coef * (sgt[lb - 1] - sgt[lb]) if direct else pr * sgt[lb]
            if pr > 0.0:
                if cnt >= limit:
                    return -1
                if cnt < out_val.shape[0]:
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
def _construct_atoms(pool_sorted_row, sub_nn_row, sub_start, s0, s1, m, eps, val_buf, prob_buf, limit, sval, sfl, sgt):
    """Every control guide's exact median atoms for one guide null, weighted as in :func:`_pool_tails`, unmerged.

    Writes those that fit ``val_buf``/``prob_buf`` and returns their count, or ``-1`` past ``limit``.
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
            pool_sorted_row, sub_start[sub], nn, m, nn / total, eps, val_buf, prob_buf, cnt, limit, sval, sfl, sgt
        )
        if cnt < 0:
            return -1
    return cnt


@njit(cache=True, nogil=True)
def _build_construct(
    pool_sorted_row, sub_nn_row, sub_start, s0, s1, m, eps, val_buf, prob_buf, val2, prob2, ints, sval, sfl, sgt
):
    """Build one guide null as merged, sorted atoms in ``val_buf``/``prob_buf``; returns their count.

    Mixes every control guide's exact median-atom distribution (:func:`_construct_atoms`), then sorts and
    merges equal values. Returns ``-1`` if the atoms do not fit the buffer.
    """
    cnt = _construct_atoms(
        pool_sorted_row, sub_nn_row, sub_start, s0, s1, m, eps, val_buf, prob_buf, val_buf.shape[0], sval, sfl, sgt
    )
    if cnt <= 0:
        return cnt
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
def _any_positive(x, k):
    """Whether any of ``x[:k]`` is positive, without slicing ``x``."""
    for i in range(k):
        if x[i] > 0.0:
            return True
    return False


@njit(cache=True, nogil=True)
def _sides(k, lo_c, hi_c, vals, cum, obs, inclusive, plow, phigh, sp, inv_low, inv_high, top, wl, wh):
    """Split each guide's atoms at ``obs``: low are those at most ``obs`` (``inclusive``) or below it, high the rest.

    ``sp`` is the first high atom; ``inv_low``/``inv_high`` normalise the running mass of each side, and ``wl``/``wh``
    are the exact probabilities of being low or high, zero for a side with no atom, whose shape the atoms cannot give.
    """
    for a in range(k):
        lo = lo_c[a]
        hi = hi_c[a]
        sp[a] = lo + _count_below(vals, lo, hi, obs, inclusive)
        at = cum[sp[a] - 1] if sp[a] > lo else 0.0
        top[a] = cum[hi - 1] if hi > lo else 0.0
        inv_low[a] = 1.0 / at if at > 0.0 else 0.0
        inv_high[a] = 1.0 / (top[a] - at) if top[a] > at else 0.0
        wl[a] = plow[a] if inv_low[a] > 0.0 else 0.0
        wh[a] = phigh[a] if inv_high[a] > 0.0 else 0.0


@njit(cache=True, nogil=True, inline="always")
def _coef(k, skip, base, var, poly, deg):
    """The coefficient of ``t ** deg`` in the product over the guides other than ``skip`` of ``base + t * var``.

    That is the probability that exactly ``deg`` of them are on the ``var`` side and the rest on the ``base`` side.
    """
    poly[0] = 1.0
    for d in range(1, deg + 1):
        poly[d] = 0.0
    n = 0
    for c in range(k):
        if c == skip:
            continue
        n += 1
        for d in range(min(deg, n), 0, -1):
            poly[d] = poly[d] * base[c] + poly[d - 1] * var[c]
        poly[0] *= base[c]
    return poly[deg]


@njit(cache=True, nogil=True)
def _past(lo, sp, hi, vals, cum, obs, x, reach):
    """The running mass below the first high atom whose midpoint with ``x`` is above ``obs`` (at least it, if ``reach``)."""
    left, right = sp, hi
    while left < right:
        mid = (left + right) // 2
        m = (x + vals[mid]) / 2.0
        if m > obs or (reach and m == obs):
            right = mid
        else:
            left = mid + 1
    return left, (cum[left - 1] if left > lo else 0.0)


@njit(cache=True, nogil=True)
def _before(lo, sp, vals, cum, obs, y):
    """The running mass of the low atoms whose midpoint with ``y`` is below ``obs``."""
    left, right = lo, sp
    while left < right:
        mid = (left + right) // 2
        if (vals[mid] + y) / 2.0 < obs:
            left = mid + 1
        else:
            right = mid
    return left, (cum[left - 1] if left > lo else 0.0)


@njit(cache=True, nogil=True)
def _sweep_up(k, lo_c, hi_c, sp, sp_l, vals, cum, obs, sc, budget):
    """The parts of the even split above ``obs``, with low at ``<= obs``, and at or above ``obs``, with low below it.

    One ascending pass over the low atoms of every guide, merged, takes each in turn as the largest low value ``x``:
    its guide's step in ``F``, its probability of being low and at most ``x``, times the probability that half of the
    other guides are low and at most ``x`` and the rest high and past ``x``'s cut, the atoms whose midpoint with ``x``
    passes ``obs`` (:func:`_coef`), is its term. Atoms at one value are taken one guide after the other, so every term
    is positive, and the cuts only move down. The two splits share the atoms below ``obs`` and differ in how they
    weigh each side, ``sc`` (rows ``wl * inv_low`` and ``wh * inv_high`` of the ``<= obs`` split, then of the ``< obs``
    one). The pass starts at the first value at which either split's whole probability, every low guide at most that
    value and every high one past its cut, exceeds ``budget``, so either part it leaves out is at most ``budget``.
    """
    lw_u, hw_u, lw_l, hw_l = sc[0], sc[1], sc[2], sc[3]
    # the pass's own scratch: a view of the threads' shared scratch would touch its reference count at every call
    ip, iq, ir = np.empty(k, dtype=np.int64), np.empty(k, dtype=np.int64), np.empty(k, dtype=np.int64)
    f, h, fl_, hr, poly = np.empty(k), np.empty(k), np.empty(k), np.empty(k), np.empty(k + 1)
    half = k // 2
    x0 = np.inf
    for a in range(k):  # the start: the bound grows with the value, so each guide's first atom past it is searched
        left, right = lo_c[a], sp[a]
        while left < right:
            mid = (left + right) // 2
            x = vals[mid]
            for c in range(k):
                lo = lo_c[c]
                i = lo + _count_below(vals, lo, sp[c], x, True)
                low = cum[i - 1] if i > lo else 0.0
                top = cum[hi_c[c] - 1]
                f[c] = lw_u[c] * low
                h[c] = hw_u[c] * (top - _past(lo, sp[c], hi_c[c], vals, cum, obs, x, True)[1])
                fl_[c] = lw_l[c] * min(low, cum[sp_l[c] - 1] if sp_l[c] > lo else 0.0)
                hr[c] = hw_l[c] * (top - _past(lo, sp_l[c], hi_c[c], vals, cum, obs, x, True)[1])
            if max(_coef(k, -1, h, f, poly, half), _coef(k, -1, hr, fl_, poly, half)) > budget:
                right = mid
            else:
                left = mid + 1
        if left < sp[a]:
            x0 = min(x0, vals[left])
    if x0 == np.inf:
        return 0.0, 0.0
    for c in range(k):
        lo = lo_c[c]
        ip[c] = lo + _count_below(vals, lo, sp[c], x0, False)
        low = cum[ip[c] - 1] if ip[c] > lo else 0.0
        f[c], fl_[c] = lw_u[c] * low, lw_l[c] * low  # x0 is at most obs, so the atoms below it are low in both
        iq[c], m = _past(lo, sp[c], hi_c[c], vals, cum, obs, x0, False)
        h[c] = hw_u[c] * (cum[hi_c[c] - 1] - m)
        ir[c], m = _past(lo, sp_l[c], hi_c[c], vals, cum, obs, x0, True)
        hr[c] = hw_l[c] * (cum[hi_c[c] - 1] - m)
    if k == 4:
        return _sweep_up4(lo_c, hi_c, sp, sp_l, vals, cum, obs, sc, ip, iq, ir, f, fl_, h, hr)
    over = at_least = 0.0
    while True:
        x = np.inf
        for c in range(k):
            if ip[c] < sp[c] and vals[ip[c]] < x:
                x = vals[ip[c]]
        if x == np.inf:
            break
        for c in range(k):
            lo, s, top = lo_c[c], sp[c], cum[hi_c[c] - 1]
            q = iq[c]
            if q > s and (x + vals[q - 1]) / 2.0 > obs:
                while q > s and (x + vals[q - 1]) / 2.0 > obs:
                    q -= 1
                iq[c] = q
                h[c] = hw_u[c] * (top - (cum[q - 1] if q > lo else 0.0))
            r, s = ir[c], sp_l[c]  # a float midpoint with a value just below obs can round up to an atom at obs
            if r > s and (x + vals[r - 1]) / 2.0 >= obs:
                while r > s and (x + vals[r - 1]) / 2.0 >= obs:
                    r -= 1
                ir[c] = r
                hr[c] = hw_l[c] * (top - (cum[r - 1] if r > lo else 0.0))
        below_obs = x < obs
        for a in range(k):
            if ip[a] < sp[a] and vals[ip[a]] == x:
                low = cum[ip[a]]
                step = lw_u[a] * low - f[a]
                over += step * _coef(k, a, h, f, poly, half - 1)
                f[a] += step
                if below_obs:
                    step = lw_l[a] * low - fl_[a]
                    at_least += step * _coef(k, a, hr, fl_, poly, half - 1)
                    fl_[a] += step
                ip[a] += 1
    return over, at_least


@njit(cache=True, nogil=True, inline="always")
def _cuts(q, r, e, s, lo, vals, cum, x, obs, below, hw, hv, top, h, hr):
    """Move a guide's two high cuts for the low value ``x``, and return them with the masses past them.

    ``q`` is the first atom past ``e`` whose midpoint with ``x`` is above ``obs``, and, while ``x`` is below ``obs``,
    ``r`` the first past ``s`` whose midpoint is at least ``obs``. ``r`` is at most ``q``, and only atoms whose
    midpoint is ``obs`` itself lie between them. The masses are ``hw``, ``hv`` times the mass from each to the top.
    """
    moved = False
    while q > e and (x + vals[q - 1]) / 2.0 > obs:
        q -= 1
        moved = True
    if moved:
        h = hw * (top - (cum[q - 1] if q > lo else 0.0))
    if below:
        start = min(r, q)
        t = start
        while t > s and (x + vals[t - 1]) / 2.0 >= obs:
            t -= 1
        if t != r:
            r = t
            hr = hv * (top - (cum[t - 1] if t > lo else 0.0))
    return q, r, h, hr


@njit(cache=True, nogil=True)
def _sweep_up4(lo_c, hi_c, sp, sp_l, vals, cum, obs, sc, ip, iq, ir, f, fl_, h, hr):
    """The pass of :func:`_sweep_up` for four guides, from the state it set up, its state held in scalars.

    The probability that one of the other three guides is low and the other two high is written out.
    """
    lo0, lo1, lo2, lo3 = lo_c[0], lo_c[1], lo_c[2], lo_c[3]
    e0, e1, e2, e3 = sp[0], sp[1], sp[2], sp[3]
    s0, s1, s2, s3 = sp_l[0], sp_l[1], sp_l[2], sp_l[3]
    t0, t1, t2, t3 = cum[hi_c[0] - 1], cum[hi_c[1] - 1], cum[hi_c[2] - 1], cum[hi_c[3] - 1]
    lw0, lw1, lw2, lw3 = sc[0, 0], sc[0, 1], sc[0, 2], sc[0, 3]
    hw0, hw1, hw2, hw3 = sc[1, 0], sc[1, 1], sc[1, 2], sc[1, 3]
    lv0, lv1, lv2, lv3 = sc[2, 0], sc[2, 1], sc[2, 2], sc[2, 3]
    hv0, hv1, hv2, hv3 = sc[3, 0], sc[3, 1], sc[3, 2], sc[3, 3]
    p0, p1, p2, p3 = ip[0], ip[1], ip[2], ip[3]
    q0, q1, q2, q3 = iq[0], iq[1], iq[2], iq[3]
    r0, r1, r2, r3 = ir[0], ir[1], ir[2], ir[3]
    f0, f1, f2, f3 = f[0], f[1], f[2], f[3]  # F at <= obs
    g0, g1, g2, g3 = fl_[0], fl_[1], fl_[2], fl_[3]  # F at < obs
    h0, h1, h2, h3 = h[0], h[1], h[2], h[3]  # past the cut at > obs
    k0, k1, k2, k3 = hr[0], hr[1], hr[2], hr[3]  # past the cut at >= obs, high at >= obs
    x0 = vals[p0] if p0 < e0 else np.inf
    x1 = vals[p1] if p1 < e1 else np.inf
    x2 = vals[p2] if p2 < e2 else np.inf
    x3 = vals[p3] if p3 < e3 else np.inf
    over = at_least = 0.0
    while True:
        x = min(min(x0, x1), min(x2, x3))
        if x == np.inf:
            break
        below = x < obs
        q0, r0, h0, k0 = _cuts(q0, r0, e0, s0, lo0, vals, cum, x, obs, below, hw0, hv0, t0, h0, k0)
        q1, r1, h1, k1 = _cuts(q1, r1, e1, s1, lo1, vals, cum, x, obs, below, hw1, hv1, t1, h1, k1)
        q2, r2, h2, k2 = _cuts(q2, r2, e2, s2, lo2, vals, cum, x, obs, below, hw2, hv2, t2, h2, k2)
        q3, r3, h3, k3 = _cuts(q3, r3, e3, s3, lo3, vals, cum, x, obs, below, hw3, hv3, t3, h3, k3)
        if x0 == x:
            low = cum[p0]
            step = lw0 * low - f0
            over += step * (f1 * h2 * h3 + f2 * h1 * h3 + f3 * h1 * h2)
            f0 += step
            if below:
                step = lv0 * low - g0
                at_least += step * (g1 * k2 * k3 + g2 * k1 * k3 + g3 * k1 * k2)
                g0 += step
            p0 += 1
            x0 = vals[p0] if p0 < e0 else np.inf
        if x1 == x:
            low = cum[p1]
            step = lw1 * low - f1
            over += step * (f0 * h2 * h3 + f2 * h0 * h3 + f3 * h0 * h2)
            f1 += step
            if below:
                step = lv1 * low - g1
                at_least += step * (g0 * k2 * k3 + g2 * k0 * k3 + g3 * k0 * k2)
                g1 += step
            p1 += 1
            x1 = vals[p1] if p1 < e1 else np.inf
        if x2 == x:
            low = cum[p2]
            step = lw2 * low - f2
            over += step * (f0 * h1 * h3 + f1 * h0 * h3 + f3 * h0 * h1)
            f2 += step
            if below:
                step = lv2 * low - g2
                at_least += step * (g0 * k1 * k3 + g1 * k0 * k3 + g3 * k0 * k1)
                g2 += step
            p2 += 1
            x2 = vals[p2] if p2 < e2 else np.inf
        if x3 == x:
            low = cum[p3]
            step = lw3 * low - f3
            over += step * (f0 * h1 * h2 + f1 * h0 * h2 + f2 * h0 * h1)
            f3 += step
            if below:
                step = lv3 * low - g3
                at_least += step * (g0 * k1 * k2 + g1 * k0 * k2 + g2 * k0 * k1)
                g3 += step
            p3 += 1
            x3 = vals[p3] if p3 < e3 else np.inf
    return over, at_least


@njit(cache=True, nogil=True)
def _sweep_down(k, lo_c, hi_c, sp, vals, cum, obs, wl, wh, inv_low, inv_high, top, budget):
    """``P(#low == #high and the midpoint of the largest low and smallest high guide < obs)``.

    The mirror of :func:`_sweep_up`: descending over the high atoms, with each low guide's cut moving up.
    """
    ip, iq = np.empty(k, dtype=np.int64), np.empty(k, dtype=np.int64)
    g, l_, poly = np.empty(k), np.empty(k), np.empty(k + 1)
    half = k // 2
    y0 = -np.inf
    for a in range(k):  # the bound shrinks as the value grows
        left, right = sp[a], hi_c[a]
        while left < right:
            mid = (left + right) // 2
            y = vals[mid]
            for c in range(k):
                i = sp[c] + _count_below(vals, sp[c], hi_c[c], y, False)
                g[c] = wh[c] * (top[c] - (cum[i - 1] if i > lo_c[c] else 0.0)) * inv_high[c]
                l_[c] = wl[c] * _before(lo_c[c], sp[c], vals, cum, obs, y)[1] * inv_low[c]
            if _coef(k, -1, l_, g, poly, half) > budget:
                left = mid + 1
            else:
                right = mid
        if left > sp[a]:
            y0 = max(y0, vals[left - 1])
    if y0 == -np.inf:
        return 0.0
    for c in range(k):
        lo = lo_c[c]
        iq[c] = sp[c] + _count_below(vals, sp[c], hi_c[c], y0, True)
        g[c] = wh[c] * (top[c] - (cum[iq[c] - 1] if iq[c] > lo else 0.0)) * inv_high[c]
        ip[c], m = _before(lo, sp[c], vals, cum, obs, y0)
        l_[c] = wl[c] * m * inv_low[c]
    under = 0.0
    while True:
        y = -np.inf
        for c in range(k):
            if iq[c] > sp[c] and vals[iq[c] - 1] > y:
                y = vals[iq[c] - 1]
        if y == -np.inf:
            break
        for c in range(k):
            p = ip[c]
            if p < sp[c] and (vals[p] + y) / 2.0 < obs:
                while p < sp[c] and (vals[p] + y) / 2.0 < obs:
                    p += 1
                ip[c] = p
                l_[c] = wl[c] * (cum[p - 1] if p > lo_c[c] else 0.0) * inv_low[c]
        for a in range(k):
            if iq[a] > sp[a] and vals[iq[a] - 1] == y:
                iq[a] -= 1
                step = wh[a] * (top[a] - (cum[iq[a] - 1] if iq[a] > lo_c[a] else 0.0)) * inv_high[a] - g[a]
                under += step * _coef(k, a, l_, g, poly, half - 1)
                g[a] += step
    return under


@njit(cache=True, nogil=True)
def _sweep_pvalue(k, lo_c, hi_c, vals, cum, obs, gt, lt, le, ge, above, below, rtol, fl, ints):
    """The two-sided p-value of a gene of an even number ``k`` of measured guides, none ever nan.

    ``above``/``below`` are ``P(#high > #low)`` and its mirror; the rest is the split into as many low as high guides,
    where the gene's median is the midpoint of the largest low and the smallest high guide. The part of the split
    above ``obs`` takes low at ``<= obs`` and the part below it low at ``< obs``; :func:`_sweep_up` sums the first and
    the part of the second at or above ``obs``, so the part below is the split's mass less that one, or, where the
    difference would lose its digits, :func:`_sweep_down` sums it directly. Each sweep leaves out a far tail of at
    most ``rtol / 2 * min(above, below)`` from each part it sums, and ``2 * min(above, below)`` is at most the p-value,
    so the p-value moves by at most half ``rtol`` of itself; ``rtol = 0`` leaves out only tails that cannot occur.
    """
    sp_u, sp_l = ints[0], ints[1]
    inv_low_u, inv_high_u, top_u, wl_u, wh_u = fl[0], fl[1], fl[2], fl[3], fl[4]
    inv_low_l, inv_high_l, top_l, wl_l, wh_l = fl[5], fl[6], fl[7], fl[8], fl[9]
    _sides(k, lo_c, hi_c, vals, cum, obs, True, le, gt, sp_u, inv_low_u, inv_high_u, top_u, wl_u, wh_u)
    _sides(k, lo_c, hi_c, vals, cum, obs, False, lt, ge, sp_l, inv_low_l, inv_high_l, top_l, wl_l, wh_l)
    sc = np.empty((4, k))
    for a in range(k):
        sc[0, a], sc[1, a] = wl_u[a] * inv_low_u[a], wh_u[a] * inv_high_u[a]
        sc[2, a], sc[3, a] = wl_l[a] * inv_low_l[a], wh_l[a] * inv_high_l[a]
    budget = 0.5 * rtol * min(above, below)
    over, at_least = _sweep_up(k, lo_c, hi_c, sp_u, sp_l, vals, cum, obs, sc, budget)
    split = _coef(k, -1, wh_l, wl_l, np.empty(k + 1), k // 2)
    under = split - at_least
    if under < _SWEEP_BOTH * split:
        under = _sweep_down(k, lo_c, hi_c, sp_l, vals, cum, obs, wl_l, wh_l, inv_low_l, inv_high_l, top_l, budget)
    return _two_sided(above + over, below + under, 1.0)


@njit(cache=True, nogil=True)
def _build_into(row, nn_row, sub_start, s0, s1, m, eps, ints, sval, sfl, sgt, vals, cum, lo, hi):
    """Build a guide null of ``m`` draws in place, in the stretch ``[lo, hi)`` of the store that holds its atoms.

    The store keeps the atom values and their running mass from the bottom, ``cum``, which holds the masses until
    they are summed. Returns the merged count and the mass the pruning dropped.
    """
    n = hi - lo
    end = _build_construct(
        row, nn_row, sub_start, s0, s1, m, eps, vals[lo:hi], cum[lo:hi],
        np.empty(n, dtype=np.float64), np.empty(n, dtype=np.float64), ints, sval, sfl, sgt,
    )  # fmt: skip
    kept = 0.0
    for z in range(end):
        kept += cum[lo + z]
    running = 0.0
    for z in range(end):
        running += cum[lo + z]
        cum[lo + z] = running
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
    rtol,
):
    """Two-sided p-value of every gene against its guides' control pools, within ``rtol`` of the exact one.

    A guide's null depends only on its pool and measured cell count, so per feature each distinct
    ``(null_pool, null_size)`` null is built once into a shared atom store, and the genes then only
    read it. Guide ``c``'s null in feature ``j`` is ``null_index[null_base[pool] + construct_nn[c, j]]``,
    ``-1`` where the guide measures nothing and so leaves the gene. The nulls are dealt to ``n_slots`` slots and the genes to
    ``n_gene_slots`` (see :func:`_deal`), each with its own scratch; more gene slots than threads let a
    thread that finishes early take another.

    A gene whose even-count boundary needs the shape of a guide null that lost more than ``max_lost`` of its
    mass, to pruning or to more atoms than the largest of ``null_caps``, is left ``NaN``; the second array
    returned marks those. An even count's boundary is swept by :func:`_sweep_pvalue`.
    """
    n_genes = gene_offsets.size - 1
    n_features = gene_observed.shape[1]
    n_constructs = construct_nn.shape[0]
    n_nulls = null_pool.size
    out = np.empty((n_genes, n_features), dtype=np.float64)
    unresolved = np.zeros((n_genes, n_features), dtype=np.bool_)
    max_k = max(1, np.diff(gene_offsets).max())
    max_width = max(1, (sub_stop - sub_start).max())
    max_cap = 1  # a null of more atoms than this keeps none
    for u in range(n_nulls):
        max_cap = max(max_cap, null_caps[u])
    sort_t = np.empty((n_slots, max(_SORT_SCRATCH, 128)), dtype=np.int64)
    sval_t = np.empty((n_slots, max_width), dtype=np.float64)
    sfl_t = np.empty((n_slots, max_width), dtype=np.float64)
    sgt_t = np.empty((n_slots, max_width), dtype=np.float64)
    # per gene: lo_c, hi_c, its guides' tails, and the sweep's scratch, padded by a cache line so that slots on
    # different cores share none
    row_k = max_k + 8
    lohi_t = np.empty((n_gene_slots, 2, row_k), dtype=np.int64)
    tails_t = np.empty((n_gene_slots, 6, row_k), dtype=np.float64)
    dp_t = np.empty((n_gene_slots, 4, 2 * max_k + 5 + 8), dtype=np.float64)
    sf_t = np.empty((n_gene_slots, 10, row_k), dtype=np.float64)
    si_t = np.empty((n_gene_slots, 2, row_k), dtype=np.int64)
    used = np.empty(n_nulls, dtype=np.bool_)
    raw = np.empty(n_nulls, dtype=np.int64)  # each null's atoms before merging, -1 past max_cap
    counts = np.empty(n_nulls, dtype=np.int64)
    lost = np.empty(n_nulls, dtype=np.float64)  # the mass each null's atoms lack
    offs = np.zeros(n_nulls + 1, dtype=np.int64)
    cost = null_caps.astype(np.float64)  # a null's build cost, its atom count once known
    gene_cost = np.empty(n_genes, dtype=np.float64)
    no_atoms = np.empty(0, dtype=np.float64)
    vals = no_atoms  # the atom values of every used null, back to back
    cum = no_atoms  # running mass from each list's start
    for j in range(n_features):
        row = pool_sorted[j]
        nn_row = sub_nn[j]
        used[:] = False
        for c in range(n_constructs):
            u = null_index[null_base[construct_pool[c]] + construct_nn[c, j]]
            if u >= 0:
                used[u] = True
        # count each used null's atoms first, so that the store fits them exactly and each is built in place
        null_order, null_start = _deal(cost, n_slots)
        for t in prange(n_slots):
            for ui in range(null_start[t], null_start[t + 1]):
                u = null_order[ui]
                raw[u] = 0
                if used[u]:
                    p = null_pool[u]
                    raw[u] = _construct_atoms(
                        row, nn_row, sub_start, pool_sub_start[p], pool_sub_stop[p], null_size[u], eps,
                        no_atoms, no_atoms, max_cap, sval_t[t], sfl_t[t], sgt_t[t],
                    )  # fmt: skip
        for u in range(n_nulls):
            offs[u + 1] = offs[u] + max(raw[u], 0)
            cost[u] = raw[u]
        if offs[n_nulls] > vals.size:  # the store is kept across features, so its pages are touched once
            vals = cum = no_atoms  # the old one is freed before the new one is taken
            vals = np.empty(offs[n_nulls], dtype=np.float64)
            cum = np.empty(offs[n_nulls], dtype=np.float64)
        null_order, null_start = _deal(cost, n_slots)
        for t in prange(n_slots):
            for ui in range(null_start[t], null_start[t + 1]):
                u = null_order[ui]
                counts[u] = 0
                lost[u] = 0.0
                if used[u] and raw[u] < 0:  # more atoms than max_cap: it keeps none
                    lost[u] = 1.0
                elif used[u]:
                    p = null_pool[u]
                    counts[u], lost[u] = _build_into(
                        row, nn_row, sub_start, pool_sub_start[p], pool_sub_stop[p], null_size[u], eps,
                        sort_t[t], sval_t[t], sfl_t[t], sgt_t[t], vals, cum, offs[u], offs[u + 1],
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
            dp, sf, si = dp_t[t], sf_t[t], si_t[t]
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
                    # a guide no control cell measures is nan for certain: drop it, so the rest are never nan
                    km = 0
                    for ci in range(k):
                        if meas_arr[ci] > 0.0:
                            lo_c[km], hi_c[km] = lo_c[ci], hi_c[ci]
                            gt_arr[km], lt_arr[km], le_arr[km], ge_arr[km] = (
                                gt_arr[ci],
                                lt_arr[ci],
                                le_arr[ci],
                                ge_arr[ci],
                            )
                            nan_arr[km], meas_arr[km] = 0.0, 1.0
                            km += 1
                    k = km
                    if k % 2 == 1 or k == 0:  # an odd count never splits evenly
                        out[g, j] = _two_sided(above, below, 1.0 - all_nan if any_measured else 0.0)
                        continue
                    out[g, j] = _sweep_pvalue(
                        k, lo_c, hi_c, vals, cum, obs, gt_arr, lt_arr, le_arr, ge_arr, above, below, rtol, sf, si
                    )
                    continue
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


#: Hard limit on a guide null's atoms, so a huge screen cannot ask for unbounded memory;
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
    """Per-null upper bound on the atoms a guide null generates before merging; past the largest, a null keeps none.

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
    *,
    rtol: float = 1e-5,
) -> AnnData | None:
    """Test each feature of each gene against the null of resampling the control guides' cells.

    Each guide (each ``replicate`` within a ``groupby`` group) is summarised by the median of its cells, per feature, and each gene by the median of its guides' medians.
    The guide's null draws one control guide with probability proportional to its cell count, draws as many of that guide's cells as the tested guide has, with replacement, and takes their median; the gene's null is the median across its guides' such nulls.
    Because each draw comes from one control guide, the null carries the spread between control guides as well as the sampling error of a median.

    The null is not simulated but computed exactly: the median of a resample is an order statistic (a binomial tail in the guide's empirical distribution, and for an even cell count the mean of two consecutive order statistics), the guide null is the cell-count-weighted mixture of its control guides' such distributions, and the gene null is the order-statistic distribution of its independent guide nulls, summed over its guides by a dynamic program so that a gene of many guides stays cheap.
    The p-values are therefore the ones an infinite bootstrap converges to, with no simulation noise and no lower limit; a gene's, by default, to within ``rtol`` of itself.

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
        rtol: How close each gene's p-value stays to the exact one, relative to it: within ``rtol`` times the p-value.
            The gene null's far tails that cannot move the p-value by more than that are not summed.
            ``0`` sums every tail, so the p-values are exact.
            Guides' p-values are always exact.

    Returns:
        ``None``, or the modified copy.
        Writes ``uns["mantispy"][key_added]``, a tidy frame of ``group``, ``feature``, ``effect`` (the gene's median minus the median of its guides' control pools, in the units of ``X``), ``pvalue`` and ``qvalue`` (Benjamini-Hochberg over every gene and feature at once).
        Also writes ``uns["mantispy"][key_added + "_guides"]``, the same per guide, with ``replicate`` and ``n_cells`` columns and its own q-values.

    Raises:
        ValueError: ``rtol`` is negative or not finite, ``control_scope`` is unknown, a scope that needs ``condition`` or ``reference_condition`` lacks it, ``"within_perturbation"`` is used without ``condition`` in ``groupby``, the controls in ``reference_condition`` are empty, no guide has a null to draw from, or the matrix holds an infinite value.

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
    if not (np.isfinite(rtol) and rtol >= 0):
        raise ValueError(f"rtol must be a finite number at least 0, got {rtol!r}")
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
            float(rtol),
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
