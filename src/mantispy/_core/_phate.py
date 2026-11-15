"""PHATE kernels: the alpha-decay affinity, the diffusion potential and metric MDS.

Written from Moon et al., Nat Biotechnol 2019 (PHATE).
The neighbour search, the kernel rows, the diffusion powering and the SMACOF iterations are parallel numba kernels over
a fixed number of row chunks, so results never depend on the thread count, and scratch is allocated before each
parallel loop rather than inside it.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import scipy.sparse as sp
from numba import njit

from mantispy._core._numba import N_CHUNKS

if TYPE_CHECKING:
    from builtins import range as prange
else:
    from numba import prange

#: Affinities below this are dropped from the kernel.
THRESHOLD = 1e-4
#: Added inside the log of the potential, so a transition probability of zero stays finite.
LOG_OFFSET = 1e-7
#: Entries of one block of inner products in the neighbour search and k-means, which bounds their scratch memory.
_BLOCK_ENTRIES = 1 << 24
#: How much faster a dense product runs per operation than the sparse kernel; it sets where powering switches.
_DENSE_PRODUCT_SPEEDUP = 10
#: Diffusion coordinates the landmarks are clustered on.
_N_SVD = 100


@njit(parallel=True, cache=True, nogil=True)
def _select_neighbours(
    gram: np.ndarray, chosen: np.ndarray, half_norms: np.ndarray, first: int, out_idx: np.ndarray, out_sq: np.ndarray
) -> None:
    """The ``k`` nearest rows of each of ``chosen`` into rows ``first ..`` of the outputs, ties broken by row.

    ``gram`` holds the inner products of the chosen rows with every row. A squared distance is
    ``|a|^2 + |b|^2 - 2 a.b``, so a candidate is ranked by ``|b|^2 / 2 - a.b`` alone and the full distance is formed
    only for the few that enter the list; each row's own column goes first at distance zero.
    """
    n_rows, n = gram.shape
    k = out_idx.shape[1]
    n_chunks = min(N_CHUNKS, n_rows)
    for chunk in prange(n_chunks):
        for r in range(chunk * n_rows // n_chunks, (chunk + 1) * n_rows // n_chunks):
            best_d = out_sq[first + r]
            best_i = out_idx[first + r]
            best_d[:] = np.inf
            best_i[:] = n
            own = chosen[r]
            best_d[0] = -np.inf
            best_i[0] = own
            base = 2.0 * half_norms[own]
            row = gram[r]
            if k < 2:
                best_d[0] = 0.0
                continue
            # Rank key of the current k-th candidate; the k-th is replaced only by a strictly nearer row.
            limit = np.inf
            for j in range(n):
                key = half_norms[j] - np.float64(row[j])
                if key >= limit or j == own:
                    continue
                d = max(base + 2.0 * key, 0.0)
                slot = k - 1
                while slot > 1 and best_d[slot - 1] > d:
                    best_d[slot] = best_d[slot - 1]
                    best_i[slot] = best_i[slot - 1]
                    slot -= 1
                best_d[slot] = d
                best_i[slot] = j
                if best_d[k - 1] < np.inf:
                    limit = (best_d[k - 1] - base) / 2.0
            best_d[0] = 0.0


@njit(parallel=True, cache=True, nogil=True)
def _refine(
    x: np.ndarray,
    rows: np.ndarray,
    candidates: np.ndarray,
    screened: np.ndarray,
    bound: np.ndarray,
    out_idx: np.ndarray,
    out_sq: np.ndarray,
    exact: np.ndarray,
) -> None:
    """Exact squared distances to each row's screened candidates, the ``k`` nearest kept, ties broken by row.

    A row is marked exact when no row outside its candidates can be nearer than its ``k``-th: everything left out
    was screened at least as far as the last candidate, and the screening is wrong by at most ``bound``.
    """
    n_rows, width = candidates.shape
    k = out_idx.shape[1]
    d = x.shape[1]
    n_chunks = min(N_CHUNKS, n_rows)
    order_d = np.empty((n_chunks, width), dtype=np.float64)
    order_i = np.empty((n_chunks, width), dtype=np.int64)
    for chunk in prange(n_chunks):
        best_d, best_i = order_d[chunk], order_i[chunk]
        for r in range(chunk * n_rows // n_chunks, (chunk + 1) * n_rows // n_chunks):
            own = rows[r]
            filled = 0
            for c in range(width):
                j = candidates[r, c]
                if j == own:
                    dist = -1.0  # the row itself always comes first
                else:
                    dist = 0.0
                    for f in range(d):
                        diff = x[own, f] - x[j, f]
                        dist += diff * diff
                slot = filled
                while slot > 0 and (best_d[slot - 1] > dist or (best_d[slot - 1] == dist and best_i[slot - 1] > j)):
                    best_d[slot] = best_d[slot - 1]
                    best_i[slot] = best_i[slot - 1]
                    slot -= 1
                best_d[slot] = dist
                best_i[slot] = j
                filled += 1
            for c in range(k):
                out_idx[r, c] = best_i[c]
                out_sq[r, c] = max(best_d[c], 0.0)
            exact[r] = screened[r, width - 1] - bound[r] > out_sq[r, k - 1]


def _screen(x: np.ndarray, rows: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """The ``k`` nearest rows of each of ``rows`` by inner products in ``x``'s own precision, one block at a time."""
    n = x.shape[0]
    half_norms = np.einsum("ij,ij->i", x, x, dtype=np.float64) / 2.0
    idx = np.empty((rows.size, k), dtype=np.int64)
    sq = np.empty((rows.size, k), dtype=np.float64)
    block = max(1, _BLOCK_ENTRIES // max(n, 1))
    gram = np.empty((min(block, rows.size), n), dtype=x.dtype)
    for start in range(0, rows.size, block):
        chosen = rows[start : start + block]
        out = gram[: chosen.size]
        np.matmul(x[chosen], x.T, out=out)
        _select_neighbours(out, chosen, half_norms, start, idx, sq)
    return idx, sq


def nearest_neighbours(x: np.ndarray, k: int, rows: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Exact ``k`` nearest neighbours of ``rows`` (all rows by default) among the rows of ``x``, self included first.

    Returns ``(indices, distances)``, each ``(len(rows), k)``, distances Euclidean and ascending.
    Candidates are screened in single precision, twice as many as asked for, and their distances recomputed exactly;
    a row whose screening cannot rule out a nearer row outside its candidates is searched again in double precision.
    """
    n = x.shape[0]
    rows = np.arange(n, dtype=np.int64) if rows is None else rows.astype(np.int64)
    centred = np.ascontiguousarray(x - x.mean(axis=0), dtype=np.float64)
    width = min(n, 2 * k)
    idx = np.empty((rows.size, k), dtype=np.int64)
    sq = np.empty((rows.size, k), dtype=np.float64)
    exact = np.zeros(rows.size, dtype=np.bool_)
    norms = np.einsum("ij,ij->i", centred, centred)
    if width < n:
        candidates, screened = _screen(centred.astype(np.float32), rows, width)
        # Rounding the rows to single precision and the inner products in it bound the error of a screened distance.
        unit = (2.0 * (centred.shape[1] + 2) + 2.0) * 2.0**-23
        bound = unit * (norms[rows] + norms.max())
        _refine(centred, rows, candidates, screened, bound, idx, sq, exact)
    redo = np.flatnonzero(~exact)
    if redo.size:
        candidates, screened = _screen(centred, rows[redo], k)
        sub_idx = np.empty((redo.size, k), dtype=np.int64)
        sub_sq = np.empty((redo.size, k), dtype=np.float64)
        # A double-precision search is exact; the refine only recomputes its distances from differences.
        _refine(
            centred,
            rows[redo],
            candidates,
            np.full_like(screened, np.inf),
            np.zeros(redo.size),
            sub_idx,
            sub_sq,
            np.empty(redo.size, dtype=np.bool_),
        )
        idx[redo], sq[redo] = sub_idx, sub_sq
    return idx, np.sqrt(sq)


def knn_with_tail(x: np.ndarray, knn: int, decay: float | None) -> tuple[np.ndarray, np.ndarray]:
    """Enough neighbours per row that the alpha-decay kernel is exact, ragged rows padded with ``-1`` indices.

    ``idx`` and ``dist`` hold the row itself first. A row's kernel reaches past its ``knn``-th neighbour, out to where the affinity falls below :data:`THRESHOLD`.
    The search starts at a small multiple of ``knn`` and doubles the width for the rows whose last neighbour still
    lies inside that radius, so no affinity above the threshold is cut off.
    """
    n = x.shape[0]
    if decay is None:
        return nearest_neighbours(x, knn + 1)
    reach = (-math.log(THRESHOLD)) ** (1.0 / decay)
    width = min(n, 4 * (knn + 1))
    idx, dist = nearest_neighbours(x, width)
    open_rows = np.flatnonzero(dist[:, -1] <= reach * dist[:, knn]) if width < n else np.empty(0, int)
    while open_rows.size:
        width = min(n, 2 * width)
        wider_idx, wider_dist = nearest_neighbours(x, width, open_rows)
        pad = width - idx.shape[1]
        idx = np.pad(idx, ((0, 0), (0, pad)), constant_values=-1)
        dist = np.pad(dist, ((0, 0), (0, pad)), constant_values=np.inf)
        idx[open_rows], dist[open_rows] = wider_idx, wider_dist
        if width == n:
            break
        still = wider_dist[:, -1] <= reach * wider_dist[:, knn]
        open_rows = open_rows[still]
    return idx, dist


@njit(parallel=True, cache=True, nogil=True)
def _kernel_rows(
    idx: np.ndarray, dist: np.ndarray, knn: int, decay: float, use_decay: bool, counts: np.ndarray
) -> None:
    """How many affinities of each row clear the threshold."""
    n, width = idx.shape
    n_chunks = min(N_CHUNKS, n)
    for chunk in prange(n_chunks):
        for i in range(chunk * n // n_chunks, (chunk + 1) * n // n_chunks):
            eps = dist[i, knn]
            c = 0
            for j in range(width):
                if idx[i, j] < 0:
                    break
                if use_decay:
                    a = _affinity(dist[i, j], eps, decay)
                    if a >= THRESHOLD:
                        c += 1
                elif j <= knn:
                    c += 1
            counts[i] = c


@njit(cache=True, nogil=True)
def _affinity(d: float, eps: float, decay: float) -> float:
    """``exp(-(d / eps) ** decay)``, with a zero bandwidth (duplicate rows) giving one at distance zero."""
    if eps == 0.0:
        return 1.0 if d == 0.0 else 0.0
    return math.exp(-((d / eps) ** decay))


@njit(parallel=True, cache=True, nogil=True)
def _kernel_fill(
    idx: np.ndarray,
    dist: np.ndarray,
    knn: int,
    decay: float,
    use_decay: bool,
    indptr: np.ndarray,
    indices: np.ndarray,
    values: np.ndarray,
    distances: np.ndarray,
) -> None:
    """Write each row's affinities above the threshold, and the distances they came from, into CSR arrays."""
    n, width = idx.shape
    n_chunks = min(N_CHUNKS, n)
    for chunk in prange(n_chunks):
        for i in range(chunk * n // n_chunks, (chunk + 1) * n // n_chunks):
            eps = dist[i, knn]
            pos = indptr[i]
            for j in range(width):
                if idx[i, j] < 0:
                    break
                if use_decay:
                    a = _affinity(dist[i, j], eps, decay)
                    if a < THRESHOLD:
                        continue
                elif j <= knn:
                    a = 1.0
                else:
                    continue
                indices[pos] = idx[i, j]
                values[pos] = a
                distances[pos] = dist[i, j]
                pos += 1


def affinity(idx: np.ndarray, dist: np.ndarray, knn: int, decay: float | None) -> tuple[sp.csr_matrix, sp.csr_matrix]:
    """Symmetric alpha-decay affinity ``K`` and the neighbour distances it was built from, both sparse.

    Row ``i`` has bandwidth ``eps_i``, its distance to its ``knn``-th nearest other row, and the directed affinity
    ``exp(-(d_ij / eps_i) ** decay)``; ``K`` is the mean of that and its transpose, so the diagonal is one.
    With ``decay=None`` the directed affinity is one on the row and its ``knn`` neighbours and zero elsewhere.
    """
    n = idx.shape[0]
    use_decay = decay is not None
    alpha = 0.0 if decay is None else float(decay)
    counts = np.empty(n, dtype=np.int64)
    _kernel_rows(idx, dist, knn, alpha, use_decay, counts)
    indptr = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(counts, out=indptr[1:])
    nnz = int(indptr[-1])
    indices = np.empty(nnz, dtype=np.int64)
    values = np.empty(nnz, dtype=np.float64)
    lengths = np.empty(nnz, dtype=np.float64)
    _kernel_fill(idx, dist, knn, alpha, use_decay, indptr, indices, values, lengths)
    directed = sp.csr_matrix((values, indices, indptr), shape=(n, n))
    kernel = ((directed + directed.T) * 0.5).tocsr()
    kernel.sort_indices()
    distances = sp.csr_matrix((lengths, indices, indptr), shape=(n, n))
    distances = distances.maximum(distances.T).tocsr()
    distances.setdiag(0)
    distances.eliminate_zeros()
    return kernel, distances


@njit(parallel=True, cache=True, nogil=True)
def _sparse_times_dense(
    indptr: np.ndarray, indices: np.ndarray, data: np.ndarray, q: np.ndarray, out: np.ndarray
) -> None:
    """``out = A @ q`` for CSR ``A``, rows shared out over the chunks."""
    n, m = out.shape
    n_chunks = min(N_CHUNKS, n)
    for chunk in prange(n_chunks):
        for i in range(chunk * n // n_chunks, (chunk + 1) * n // n_chunks):
            row = out[i]
            row[:] = 0.0
            for p in range(indptr[i], indptr[i + 1]):
                w = data[p]
                src = q[indices[p]]
                for c in range(m):
                    row[c] += w * src[c]


@njit(parallel=True, cache=True, nogil=True)
def _interpolate(indptr: np.ndarray, indices: np.ndarray, data: np.ndarray, q: np.ndarray, out: np.ndarray) -> None:
    """``out = A @ q`` for CSR ``A`` into ``out``'s own precision, each row summed in float64 first."""
    n, m = out.shape
    n_chunks = min(N_CHUNKS, n)
    rows = np.zeros((n_chunks, m), dtype=np.float64)
    for chunk in prange(n_chunks):
        acc = rows[chunk]
        for i in range(chunk * n // n_chunks, (chunk + 1) * n // n_chunks):
            acc[:] = 0.0
            for p in range(indptr[i], indptr[i + 1]):
                w = data[p]
                src = q[indices[p]]
                for c in range(m):
                    acc[c] += w * src[c]
            for c in range(m):
                out[i, c] = acc[c]


def interpolate(transitions: sp.csr_matrix, values: np.ndarray, dtype: type = np.float64) -> np.ndarray:
    """Each row's transition-weighted mean of the landmarks' ``values``, written straight into ``dtype``."""
    out: np.ndarray = np.empty((transitions.shape[0], values.shape[1]), dtype=dtype)
    _interpolate(
        transitions.indptr.astype(np.int64),
        transitions.indices.astype(np.int64),
        transitions.data.astype(np.float64),
        np.ascontiguousarray(values, dtype=np.float64),
        out,
    )
    return out


@njit(parallel=True, cache=True, nogil=True)
def _potential_inplace(x: np.ndarray, gamma: float) -> None:
    """Turn transition probabilities into the potential: ``-log(x + offset)`` for ``gamma=1``, a power otherwise."""
    n, m = x.shape
    n_chunks = min(N_CHUNKS, n)
    for chunk in prange(n_chunks):
        for i in range(chunk * n // n_chunks, (chunk + 1) * n // n_chunks):
            for c in range(m):
                if gamma == 1.0:
                    x[i, c] = -math.log(x[i, c] + LOG_OFFSET)
                else:
                    x[i, c] = 2.0 / (1.0 - gamma) * max(x[i, c], 0.0) ** ((1.0 - gamma) / 2.0)


def diffusion_power(operator: sp.csr_matrix, t: int) -> np.ndarray:
    """Dense ``operator ** t`` for a square row-stochastic sparse ``operator``.

    A sparse operator is powered by ``t - 1`` sparse-times-dense products, each costing its non-zeros times its width.
    A denser one, as the landmark operator usually is, goes through dense products by repeated squaring instead.
    """
    operator = sp.csr_matrix(operator)
    m = operator.shape[0]
    if t == 1:
        return operator.toarray()
    if operator.nnz * (t - 1) * _DENSE_PRODUCT_SPEEDUP > m * m * 2 * math.log2(t):
        return np.linalg.matrix_power(operator.toarray(), t)
    current = operator.toarray()
    following = np.empty_like(current)
    indptr = operator.indptr.astype(np.int64)
    indices = operator.indices.astype(np.int64)
    data = operator.data.astype(np.float64)
    for _ in range(t - 1):
        _sparse_times_dense(indptr, indices, data, current, following)
        current, following = following, current
    return current


def potential(transitions: np.ndarray, gamma: float) -> np.ndarray:
    """The diffusion potential of ``transitions``, in place."""
    _potential_inplace(transitions, float(gamma))
    return transitions


def von_neumann_entropy(spectrum: np.ndarray, t_max: int) -> np.ndarray:
    """Entropy of the normalised ``|spectrum| ** t`` for ``t`` in ``1 .. t_max``."""
    magnitude = np.abs(np.asarray(spectrum, dtype=np.float64))
    magnitude = magnitude[magnitude > 0]
    out = np.empty(t_max)
    for t in range(1, t_max + 1):
        p = magnitude**t
        p = p / p.sum()
        p = p[p > 0]
        out[t - 1] = -np.sum(p * np.log(p))
    return out


def knee(y: np.ndarray) -> int:
    """How many leading points of ``y`` lie before its knee, by the two-line L-method (Salvador and Chan, 2004).

    For each split the first ``c`` points and the rest are each fitted with a line, and the ``c`` whose
    length-weighted root-mean-square error is smallest wins.
    """
    y = np.asarray(y, dtype=np.float64)
    n = y.size
    if n < 4:
        return n
    x = np.arange(n, dtype=np.float64)

    def line_error(cum: dict[str, np.ndarray], lo: int, hi: int) -> float:
        # Residual sum of squares of a least-squares line over points lo..hi-1, from running sums.
        count = hi - lo
        sx, sy = cum["x"][hi] - cum["x"][lo], cum["y"][hi] - cum["y"][lo]
        sxx, sxy, syy = cum["xx"][hi] - cum["xx"][lo], cum["xy"][hi] - cum["xy"][lo], cum["yy"][hi] - cum["yy"][lo]
        vx = sxx - sx * sx / count
        vy = syy - sy * sy / count
        cxy = sxy - sx * sy / count
        rss = vy - (cxy * cxy / vx if vx > 0 else 0.0)
        return math.sqrt(max(rss, 0.0) / count)

    cum = {
        key: np.concatenate([[0.0], np.cumsum(values)])
        for key, values in {"x": x, "y": y, "xx": x * x, "xy": x * y, "yy": y * y}.items()
    }
    best, best_c = np.inf, 2
    for c in range(2, n - 1):
        error = (c - 1) / (n - 1) * line_error(cum, 0, c) + (n - c) / (n - 1) * line_error(cum, c, n)
        if error < best:
            best, best_c = error, c
    return best_c


def symmetric_spectrum(kernel: sp.csr_matrix, degree: np.ndarray) -> np.ndarray:
    """Eigenvalues of ``D^-1 kernel`` for a symmetric sparse ``kernel``, read off its symmetric conjugate."""
    from scipy.linalg import eigvalsh

    scale = 1.0 / np.sqrt(degree)
    dense = kernel.toarray()
    return eigvalsh(dense * scale[:, None] * scale[None, :], overwrite_a=True, check_finite=False)


def classic_mds(distances: np.ndarray, n_components: int) -> np.ndarray:
    """Classical (Torgerson) MDS of a square distance matrix."""
    from scipy.linalg import eigh
    from scipy.sparse.linalg import eigsh

    m = distances.shape[0]
    b = -0.5 * distances**2
    b -= b.mean(axis=0, keepdims=True)
    b -= b.mean(axis=1, keepdims=True)
    if m > n_components + 1:
        # Only the leading few eigenvectors are needed; a fixed start vector keeps Lanczos deterministic.
        values, vectors = eigsh(b, k=n_components, which="LA", v0=np.ones(m))
    else:
        values, vectors = eigh(b, subset_by_index=(m - n_components, m - 1))
    order = np.argsort(values)[::-1]
    values, vectors = np.clip(values[order], 0.0, None), vectors[:, order]
    # Fix each axis' sign so the embedding does not flip between platforms.
    signs = np.sign(vectors[np.argmax(np.abs(vectors), axis=0), np.arange(n_components)])
    return vectors * signs * np.sqrt(values)


@njit(parallel=True, cache=True, nogil=True)
def _guttman(distances: np.ndarray, x: np.ndarray, out: np.ndarray, stress: np.ndarray) -> None:
    """One SMACOF Guttman transform with unit weights, and each row's share of the stress at ``x``."""
    m, p = x.shape
    n_chunks = min(N_CHUNKS, m)
    for chunk in prange(n_chunks):
        for i in range(chunk * m // n_chunks, (chunk + 1) * m // n_chunks):
            for c in range(p):
                out[i, c] = 0.0
            row_stress = 0.0
            for j in range(m):
                if j == i:
                    continue
                gap = 0.0
                for c in range(p):
                    diff = x[i, c] - x[j, c]
                    gap += diff * diff
                gap = math.sqrt(gap)
                target = distances[i, j]
                row_stress += (gap - target) ** 2
                if gap > 0.0:
                    ratio = target / gap
                    for c in range(p):
                        out[i, c] += ratio * (x[i, c] - x[j, c])
            for c in range(p):
                out[i, c] /= m
            stress[i] = row_stress


def smacof(distances: np.ndarray, init: np.ndarray, max_iter: int = 300, tol: float = 1e-6) -> np.ndarray:
    """Metric MDS by SMACOF (de Leeuw), started from ``init``, until the stress stops falling by ``tol`` relative."""
    distances = np.ascontiguousarray(distances, dtype=np.float64)
    current = np.ascontiguousarray(init, dtype=np.float64).copy()
    following = np.empty_like(current)
    stress = np.empty(current.shape[0])
    previous = np.inf
    for _ in range(max_iter):
        _guttman(distances, current, following, stress)
        value = float(stress.sum()) / 2.0
        current, following = following, current
        if np.isfinite(previous) and previous - value <= tol * previous:
            break
        previous = value
    return current


@njit(parallel=True, cache=True, nogil=True)
def _nearest_centre(gram: np.ndarray, sizes: np.ndarray, labels: np.ndarray) -> None:
    """Each row's nearest centre, from its inner products with the centres and their squared norms."""
    n_rows, m = gram.shape
    n_chunks = min(N_CHUNKS, n_rows)
    for chunk in prange(n_chunks):
        for r in range(chunk * n_rows // n_chunks, (chunk + 1) * n_rows // n_chunks):
            best, best_c = np.inf, 0
            for c in range(m):
                d = sizes[c] - 2.0 * gram[r, c]
                if d < best:
                    best, best_c = d, c
            labels[r] = best_c


@njit(parallel=True, cache=True, nogil=True)
def _recentre(x: np.ndarray, order: np.ndarray, offsets: np.ndarray, centres: np.ndarray) -> None:
    """Each centre to the mean of its rows, which ``order`` lists grouped by centre; an empty centre stays put."""
    m, d = centres.shape
    for c in prange(m):
        start, stop = offsets[c], offsets[c + 1]
        if stop == start:
            continue
        for f in range(d):
            total = 0.0
            for p in range(start, stop):
                total += x[order[p], f]
            centres[c, f] = total / (stop - start)


def kmeans(x: np.ndarray, n_clusters: int, seed: int, max_iter: int = 20) -> np.ndarray:
    """Lloyd's k-means from k-means++ seeds, until almost no row changes centre or ``max_iter`` rounds.

    Each assignment takes the distances to the centres from one matrix product per block of rows.
    """
    from sklearn.cluster import kmeans_plusplus

    from mantispy._core._numba import group_offsets

    x = np.ascontiguousarray(x, dtype=np.float64)
    n = x.shape[0]
    # k-means++ seeds on a subsample: spread-out starts at a fraction of the cost of seeding on every row.
    sample = np.sort(np.random.default_rng(seed).choice(n, size=min(n, 3 * n_clusters), replace=False))
    centres = np.ascontiguousarray(kmeans_plusplus(x[sample], n_clusters, random_state=seed)[0])
    labels = np.full(n, -1, dtype=np.int64)
    block = max(1, _BLOCK_ENTRIES // n_clusters)
    gram = np.empty((min(block, n), n_clusters), dtype=np.float64)
    for _ in range(max_iter):
        previous = labels.copy()
        sizes = np.einsum("ij,ij->i", centres, centres)
        for lo in range(0, n, block):
            out = gram[: min(block, n - lo)]
            np.matmul(x[lo : lo + block], centres.T, out=out)
            _nearest_centre(out, sizes, labels[lo : lo + block])
        # Landmarks need a good partition, not a converged one: stop once almost no row moves.
        if np.count_nonzero(labels != previous) <= n // 1000:
            break
        order, offsets = group_offsets(labels, n_clusters)
        _recentre(x, order, offsets, centres)
    return labels


def _orthonormal(block: np.ndarray) -> np.ndarray:
    """An orthonormal basis of a tall block's columns, by Cholesky QR applied twice, which costs two matrix products."""
    from scipy.linalg import cholesky, solve_triangular

    for _ in range(2):
        try:
            factor = cholesky(block.T @ block, lower=False)
        except np.linalg.LinAlgError:
            return np.ascontiguousarray(np.linalg.qr(block)[0])
        block = solve_triangular(factor, block.T, trans="T", lower=False).T
    return np.ascontiguousarray(block)


def _leading_eigenpairs(matrix: sp.csr_matrix, k: int, seed: int, n_iter: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Leading ``k`` eigenpairs by magnitude of a symmetric sparse ``matrix``, by randomized subspace iteration.

    Each round multiplies the subspace by the matrix in parallel and re-orthonormalises it; the eigenpairs then come
    from the small projected matrix (Halko, Martinsson and Tropp, 2011).
    """
    n = matrix.shape[0]
    width = min(n, k + 10)
    indptr = matrix.indptr.astype(np.int64)
    indices = matrix.indices.astype(np.int64)
    data = matrix.data.astype(np.float64)
    basis = _orthonormal(np.random.default_rng(seed).standard_normal((n, width)))
    product = np.empty_like(basis)
    for _ in range(n_iter):
        _sparse_times_dense(indptr, indices, data, basis, product)
        basis = _orthonormal(product)
    _sparse_times_dense(indptr, indices, data, basis, product)
    values, vectors = np.linalg.eigh(basis.T @ product)
    order = np.argsort(np.abs(values))[::-1][:k]
    return values[order], basis @ vectors[:, order]


def landmarks(kernel: sp.csr_matrix, degree: np.ndarray, n_landmark: int, seed: int) -> np.ndarray:
    """Group the rows into at most ``n_landmark`` landmarks by k-means on their diffusion coordinates.

    The coordinates are the leading eigenvectors of the operator's symmetric conjugate, scaled by their eigenvalues.
    Returns each row's landmark, numbered ``0 .. m - 1`` over the landmarks that kept a row.
    """
    n = kernel.shape[0]
    scale = 1.0 / np.sqrt(degree)
    conjugate = (sp.diags(scale) @ kernel @ sp.diags(scale)).tocsr()
    values, vectors = _leading_eigenpairs(conjugate, min(_N_SVD, n - 1), seed)
    labels = kmeans(scale[:, None] * vectors * np.abs(values), n_landmark, seed)
    return np.unique(labels, return_inverse=True)[1].astype(np.int64)
