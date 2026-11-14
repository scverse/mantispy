"""Exact local outlier factor through a parallel numba KD-tree.

Follows the arithmetic of sklearn's :class:`~sklearn.neighbors.LocalOutlierFactor` ``negative_outlier_factor_``:
squared distances are summed in float64 in feature order and square-rooted at the end, as sklearn's trees do,
and the per-row means replay numpy's pairwise summation in the input precision. The scores are therefore
bit-identical wherever sklearn's compiled sum of squares is not contracted into fused multiply-adds.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
from numba import njit

if TYPE_CHECKING:
    from builtins import range as prange
else:
    from numba import prange

#: Points per leaf. Smaller than sklearn's 40, because in eight correlated dimensions the cost is in scanning leaves.
LEAF_SIZE = 8
#: Fixed number of work chunks, so results and scratch layout never depend on the thread count.
_N_CHUNKS = 4096
#: Traversal stack depth per chunk; a tree needs one more slot than it has levels.
_STACK = 128


@njit(cache=True, nogil=True)
def _select(coords: np.ndarray, idx: np.ndarray, lo: int, hi: int, nth: int, dim: int) -> None:
    """Partition ``idx[lo:hi + 1]`` so ``idx[nth]`` holds its order statistic along ``dim`` (Hoare quickselect)."""
    while lo < hi:
        a = coords[idx[lo], dim]
        b = coords[idx[(lo + hi) // 2], dim]
        c = coords[idx[hi], dim]
        pivot = max(min(a, b), min(max(a, b), c))
        i, j = lo, hi
        while i <= j:
            while coords[idx[i], dim] < pivot:
                i += 1
            while coords[idx[j], dim] > pivot:
                j -= 1
            if i <= j:
                idx[i], idx[j] = idx[j], idx[i]
                i += 1
                j -= 1
        if nth <= j:
            hi = j
        elif nth >= i:
            lo = i
        else:
            return


@njit(parallel=True, cache=True, nogil=True)
def _build(coords: np.ndarray, n_levels: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Balanced KD-tree in sklearn's implicit layout: node ``i`` has children ``2i + 1`` and ``2i + 2``.

    Each node splits its points at the median of the dimension they vary most in; variance rather than sklearn's
    range keeps the boxes tight on correlated features such as one stain measured in two compartments.
    Returns the tree-order permutation, each node's start and end in it, and each node's bounding box.
    """
    n, d = coords.shape
    n_nodes = 2**n_levels - 1
    perm = np.arange(n, dtype=np.int64)
    start = np.empty(n_nodes, dtype=np.int64)
    end = np.empty(n_nodes, dtype=np.int64)
    lower = np.empty((n_nodes, d), dtype=np.float64)
    upper = np.empty((n_nodes, d), dtype=np.float64)
    start[0] = 0
    end[0] = n
    for level in range(n_levels):
        first = 2**level - 1
        for offset in prange(2**level):
            node = first + offset
            s, e = start[node], end[node]
            widest = 0
            best = -1.0
            for f in range(d):
                low, high, total = np.inf, -np.inf, 0.0
                for p in range(s, e):
                    v = coords[perm[p], f]
                    low = min(low, v)
                    high = max(high, v)
                    total += v
                lower[node, f] = low
                upper[node, f] = high
                if level < n_levels - 1:
                    mean = total / (e - s)
                    spread = 0.0
                    for p in range(s, e):
                        dev = coords[perm[p], f] - mean
                        spread += dev * dev
                    if spread > best:
                        best = spread
                        widest = f
            if level == n_levels - 1:
                continue
            mid = s + (e - s) // 2
            if e - s > 1:
                _select(coords, perm, s, e - 1, mid, widest)
            start[2 * node + 1] = s
            end[2 * node + 1] = mid
            start[2 * node + 2] = mid
            end[2 * node + 2] = e
    return perm, start, end, lower, upper


@njit(cache=True, nogil=True)
def _box_rdist(lower: np.ndarray, upper: np.ndarray, node: int, x: np.ndarray) -> float:
    """Squared distance from ``x`` to the node's bounding box; never above that to any point inside it."""
    total = 0.0
    for f in range(x.size):
        gap = max(lower[node, f] - x[f], x[f] - upper[node, f], 0.0)
        total += gap * gap
    return total


@njit(cache=True, nogil=True)
def _scan_leaf(
    points_t: np.ndarray,
    perm: np.ndarray,
    s: int,
    e: int,
    p: int,
    x: np.ndarray,
    acc: np.ndarray,
    heap_d: np.ndarray,
    heap_i: np.ndarray,
) -> None:
    """Offer every point of a leaf except ``p`` to the sorted candidate list, ordered by distance then row."""
    count = e - s
    acc[:count] = 0.0
    # Across candidates, not features: each lane sums one candidate's squares in feature order, so no reassociation.
    for f in range(x.size):
        xf = x[f]
        column = points_t[f]
        for t in range(count):
            diff = xf - column[s + t]
            acc[t] += diff * diff
    m = heap_d.size
    for t in range(count):
        q = s + t
        dist = acc[t]
        worst = heap_d[m - 1]
        if q == p or dist > worst or (dist == worst and (heap_i[m - 1] < 0 or perm[q] > perm[heap_i[m - 1]])):
            continue
        slot = m - 1
        while slot > 0 and (heap_d[slot - 1] > dist or (heap_d[slot - 1] == dist and perm[heap_i[slot - 1]] > perm[q])):
            heap_d[slot] = heap_d[slot - 1]
            heap_i[slot] = heap_i[slot - 1]
            slot -= 1
        heap_d[slot] = dist
        heap_i[slot] = q


@njit(cache=True, nogil=True)
def _pairwise_mean(values: np.ndarray, n: int, out: np.ndarray) -> None:
    """``out[0] = values[:n].mean()`` exactly as numpy computes it, pairwise in the array's precision (``n <= 128``)."""
    if n < 8:
        total = values[0]
        for i in range(1, n):
            total += values[i]
    else:
        r0, r1, r2, r3 = values[0], values[1], values[2], values[3]
        r4, r5, r6, r7 = values[4], values[5], values[6], values[7]
        i = 8
        while i < n - n % 8:
            r0 += values[i]
            r1 += values[i + 1]
            r2 += values[i + 2]
            r3 += values[i + 3]
            r4 += values[i + 4]
            r5 += values[i + 5]
            r6 += values[i + 6]
            r7 += values[i + 7]
            i += 8
        total = ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7))
        while i < n:
            total += values[i]
            i += 1
    out[0] = total
    out[0] = out[0] / n


@njit(parallel=True, cache=True, nogil=True)
def _knn(
    coords: np.ndarray, n_levels: int, k: int, m: int, n_chunks: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]:
    """The ``k`` nearest other rows of every row, ordered by distance then row, all in tree order.

    Returns the tree-order permutation, the rows in tree order, each row's neighbours as tree positions, the
    squared distance to each row's k-th neighbour, and whether any row has a tie straddling its k-th neighbour,
    looking ``m - k`` neighbours further.
    """
    n, d = coords.shape
    perm, start, end, lower, upper = _build(coords, n_levels)
    points = np.empty((n, d), dtype=np.float64)
    points_t = np.empty((d, n), dtype=np.float64)
    for p in prange(n):
        for f in range(d):
            v = coords[perm[p], f]
            points[p, f] = v
            points_t[f, p] = v

    first_leaf = 2 ** (n_levels - 1) - 1
    n_leaves = 2 ** (n_levels - 1)
    widest_leaf = 0
    for leaf in range(first_leaf, first_leaf + n_leaves):
        widest_leaf = max(widest_leaf, end[leaf] - start[leaf])

    neighbors = np.empty((n, k), dtype=np.int32)
    k_rdist = np.empty(n, dtype=np.float64)
    tied = np.zeros(n_chunks, dtype=np.bool_)
    heap_d = np.empty((n_chunks, m), dtype=np.float64)
    heap_i = np.empty((n_chunks, m), dtype=np.int64)
    stack_node = np.empty((n_chunks, _STACK), dtype=np.int64)
    stack_rdist = np.empty((n_chunks, _STACK), dtype=np.float64)
    acc = np.empty((n_chunks, widest_leaf), dtype=np.float64)
    x = np.empty((n_chunks, d), dtype=np.float64)

    for chunk in prange(n_chunks):
        hd, hi, xc, ac = heap_d[chunk], heap_i[chunk], x[chunk], acc[chunk]
        sn, sr = stack_node[chunk], stack_rdist[chunk]
        for leaf in range(first_leaf + chunk * n_leaves // n_chunks, first_leaf + (chunk + 1) * n_leaves // n_chunks):
            for p in range(start[leaf], end[leaf]):
                xc[:] = points[p]
                hd[:] = np.inf
                hi[:] = -1
                _scan_leaf(points_t, perm, start[leaf], end[leaf], p, xc, ac, hd, hi)
                top = 0
                sn[0] = 0
                sr[0] = 0.0
                while top >= 0:
                    node = sn[top]
                    rdist = sr[top]
                    top -= 1
                    if rdist > hd[m - 1]:
                        continue
                    if node >= first_leaf:
                        if node != leaf:
                            _scan_leaf(points_t, perm, start[node], end[node], p, xc, ac, hd, hi)
                        continue
                    left, right = 2 * node + 1, 2 * node + 2
                    r_left = _box_rdist(lower, upper, left, xc)
                    r_right = _box_rdist(lower, upper, right, xc)
                    near, far, r_near, r_far = left, right, r_left, r_right
                    if r_right < r_left:
                        near, far, r_near, r_far = right, left, r_right, r_left
                    if r_far <= hd[m - 1]:
                        top += 1
                        sn[top] = far
                        sr[top] = r_far
                    if r_near <= hd[m - 1]:
                        top += 1
                        sn[top] = near
                        sr[top] = r_near
                # A tie straddling the k-th neighbour leaves the set ambiguous; compared after the sqrt, as before.
                if m > k and math.sqrt(hd[k - 1]) == math.sqrt(hd[k]):
                    tied[chunk] = True
                for j in range(k):
                    neighbors[p, j] = hi[j]
                k_rdist[p] = hd[k - 1]
    return perm, points, neighbors, k_rdist, bool(tied.any())


@njit(parallel=True, cache=True, nogil=True)
def _lof_kernel(coords: np.ndarray, n_levels: int, k: int, m: int, n_chunks: int, nof: np.ndarray) -> bool:
    """Fill ``nof`` (in ``nof``'s precision) with LOF's ``negative_outlier_factor_``; ``False`` on a k-th-distance tie."""
    n, d = coords.shape
    perm, points, neighbors, k_rdist, tied = _knn(coords, n_levels, k, m, n_chunks)
    if tied:
        return False
    # sklearn casts neighbour distances to the input precision before using them.
    kdist = np.empty(n, dtype=nof.dtype)
    for p in prange(n):
        kdist[p] = math.sqrt(k_rdist[p])
    reach = np.empty((n_chunks, k), dtype=nof.dtype)
    mean = np.empty((n_chunks, 1), dtype=nof.dtype)

    # Local reachability density, recomputing each neighbour distance in the same order as the search did.
    lrd = np.empty(n, dtype=nof.dtype)
    for chunk in prange(n_chunks):
        rc, mc = reach[chunk], mean[chunk]
        for p in range(chunk * n // n_chunks, (chunk + 1) * n // n_chunks):
            for j in range(k):
                q = neighbors[p, j]
                total = 0.0
                for f in range(d):
                    diff = points[p, f] - points[q, f]
                    total += diff * diff
                rc[j] = math.sqrt(total)
                rc[j] = max(rc[j], kdist[q])
            _pairwise_mean(rc, k, mc)
            mc[0] = mc[0] + mc.dtype.type(1e-10)
            lrd[p] = 1.0 / mc[0]

    for chunk in prange(n_chunks):
        rc, mc = reach[chunk], mean[chunk]
        for p in range(chunk * n // n_chunks, (chunk + 1) * n // n_chunks):
            for j in range(k):
                rc[j] = lrd[neighbors[p, j]] / lrd[p]
            _pairwise_mean(rc, k, mc)
            nof[perm[p]] = -mc[0]
    return True


def tree_levels(n: int) -> int:
    """Levels of a balanced tree whose leaves hold :data:`LEAF_SIZE` to twice that many rows, as sklearn sizes its KDTree."""
    return int(math.log2(max(1, (n - 1) // LEAF_SIZE))) + 1


def n_chunks(n_levels: int) -> int:
    """Work chunks for a tree of ``n_levels``: fixed, never derived from the thread count."""
    return min(_N_CHUNKS, 2 ** (n_levels - 1))


def lof_negative_outlier_factor(fitted: np.ndarray, n_neighbors: int) -> np.ndarray | None:
    """Exact LOF ``negative_outlier_factor_`` of the rows of ``fitted``, or ``None`` when sklearn must decide.

    Returns ``None`` when equidistant neighbours straddle the k-th position, where the neighbour set depends
    on how a search breaks ties.
    """
    n = fitted.shape[0]
    coords = np.ascontiguousarray(fitted, dtype=np.float64)
    k = max(1, min(n_neighbors, n - 1))
    # One neighbour past what LOF needs, to spot a tie at the k-th distance.
    m = min(k + 1, n - 1)
    n_levels = tree_levels(n)
    nof = np.empty(n, dtype=np.float32 if fitted.dtype == np.float32 else np.float64)
    if not _lof_kernel(coords, n_levels, k, m, n_chunks(n_levels), nof):
        return None
    return nof
