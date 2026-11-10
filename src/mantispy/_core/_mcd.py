"""Minimum covariance determinant, ported from R robustbase's ``covMcd`` defaults.

The search follows robustbase's FAST-MCD (``rffastmcd.f``): ``nsamp`` random (p + 1)-subsets, or every subset when ``n`` is small,
two concentration steps each, the ten best refined further, and for ``n >= 600`` a first pass on up to five random
sub-datasets of 300 rows and a second on their union.
The raw estimate is the mean and covariance of the best ``h`` rows, scaled by the consistency and small-sample factors;
the returned estimate reweights it, keeping the rows within the 0.975 chi-square quantile.
Only the random draws differ from robustbase, so the two agree wherever the search reaches the same subset.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit
from scipy.special import gammainc
from scipy.stats import chi2

N_SAMP = 500
N_MINI = 300
K_MINI = 5
N_BEST = 10

# robustbase's simulated small-sample factors for alpha = 1/2 (MCDcnp2s in R/covMcd.R), used where 1 <= p <= 9 and
# n.min[p] <= n <= n.max[p]; entry [p - 1][n - n.min[p]].
_SIM_N_MIN = (1, 4, 6, 8, 10, 12, 14, 16, 18)
_SIM_N_MAX = (2, 6, 10, 13, 16, 18, 20, 20, 20)
# fmt: off
_SIM_RAW = (
    (1.0, 3.075819),
    (3.15342, 1.799889, 2.258497),
    (2.12777, 2.718898, 1.993509, 2.220433, 1.820585),
    (2.950549, 2.145387, 2.382224, 1.927077, 2.032489, 1.8371),
    (2.229573, 2.494447, 2.016117, 2.190061, 1.877996, 1.978964, 1.767284),
    (2.580264, 2.109121, 2.240741, 1.944719, 2.043397, 1.821808, 1.89725),
    (2.165233, 2.320281, 2.007836, 2.086471, 1.884052, 1.950563, 1.76926),
    (2.340443, 2.048262, 2.161097, 1.926082, 1.995422),
    (2.093386, 2.185793, 1.948989),
)
_SIM_REW = (
    (1.0, 0.984724),
    (2.216302, 1.418526, 1.635601),
    (1.73511, 2.06681, 1.545905, 1.659655, 1.456835),
    (2.388892, 1.847626, 1.96998, 1.630723, 1.701272, 1.521008),
    (1.968153, 2.168931, 1.784373, 1.894409, 1.667912, 1.693007, 1.545176),
    (2.275891, 1.866587, 2.014249, 1.750567, 1.829363, 1.650019, 1.689043),
    (1.95771, 2.074066, 1.847385, 1.905408, 1.71393, 1.768425, 1.63908),
    (2.164121, 1.871024, 1.979485, 1.782417, 1.84489),
    (1.938822, 2.050409, 1.834863),
)
# fmt: on
# robustbase's replow(): the largest n searched over every (p + 1)-subset, for p + 1 <= 6.
_EXHAUSTIVE_N = (500, 50, 22, 17, 15, 14)


def h_alpha_n(n: int, p: int, alpha: float = 0.5) -> int:
    """Size of the MCD subset, robustbase's ``h.alpha.n``."""
    n2 = (n + p + 1) // 2
    return int(math.floor(2 * n2 - n + 2 * (n - n2) * alpha))


def consistency_factor(p: int, alpha: float | np.ndarray) -> np.ndarray:
    """Robustbase's ``.MCDcons``: rescales the covariance of the central ``alpha`` fraction to the full Gaussian."""
    return np.asarray(alpha / gammainc(p / 2 + 1, chi2.ppf(alpha, p) / 2))


def _power_fit(p: int, rows: tuple[tuple[float, float, float], tuple[float, float, float]]) -> tuple[float, float]:
    coefficients = np.array(rows)
    y = np.log(-coefficients[:, 0] / p ** coefficients[:, 1])
    design = np.column_stack([np.ones(2), -np.log(coefficients[:, 2] * p**2)])
    a, b = np.linalg.solve(design, y)
    return float(a), float(b)


def small_sample_factor(p: int, n: int, *, reweighted: bool) -> float:
    """Robustbase's ``.MCDcnp2`` (raw) or ``.MCDcnp2.rew`` (reweighted) at alpha = 1/2."""
    if p <= len(_SIM_N_MIN) and _SIM_N_MIN[p - 1] <= n <= _SIM_N_MAX[p - 1]:
        return float((_SIM_REW if reweighted else _SIM_RAW)[p - 1][n - _SIM_N_MIN[p - 1]])
    if p > 2:
        rows = (
            ((-1.02842572724793, 1.67659883081926, 2), (-0.26800273450853, 1.35968562893582, 3))
            if reweighted
            else ((-1.42764571687802, 1.26263336932151, 2), (-1.06141115981725, 1.28907991440387, 3))
        )
        a, b = _power_fit(p, rows)
    elif p == 2:
        a, b = (3.11101712909049, 1.91401056721863) if reweighted else (0.673292623522027, 0.691365864961895)
    else:
        a, b = (1.11098143415027, 1.5182890270453) if reweighted else (0.262024211897096, 0.604756680630497)
    return 1 / (1 - math.exp(a) / n**b)


def mcd_constants(n: np.ndarray, p: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per sample size: ``h``, the raw and reweighted correction factors, and whether the search is exhaustive.

    Computed outside numba because they need scipy's chi-square and gamma functions; a permutation null looks them up by size.
    """
    n = np.asarray(n, dtype=np.int64)
    size = np.maximum(n, p + 2)
    h = np.array([h_alpha_n(int(k), p) for k in size])
    raw = consistency_factor(p, h / size) * np.array([small_sample_factor(p, int(k), reweighted=False) for k in size])
    rew = consistency_factor(p, 0.975) * np.array([small_sample_factor(p, int(k), reweighted=True) for k in size])
    exhaustive = (p + 1 <= len(_EXHAUSTIVE_N)) & (size <= (_EXHAUSTIVE_N[p] if p < len(_EXHAUSTIVE_N) else 0))
    return h, raw, rew, exhaustive


@njit(cache=True, nogil=True)
def _moments(x: np.ndarray, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean and covariance (divisor ``k - 1``) of the given rows, in loops that allocate nothing per row."""
    k, p = rows.size, x.shape[1]
    centre = np.zeros(p)
    for i in rows:
        for a in range(p):
            centre[a] += x[i, a]
    centre /= k
    scatter = np.zeros((p, p))
    for i in rows:
        for a in range(p):
            da = x[i, a] - centre[a]
            for b in range(a + 1):
                scatter[a, b] += da * (x[i, b] - centre[b])
    for a in range(p):
        for b in range(a):
            scatter[b, a] = scatter[a, b]
    return centre, scatter / (k - 1)


@njit(cache=True, nogil=True)
def _logdet(cov: np.ndarray) -> float:
    eigenvalues = np.linalg.eigvalsh(cov)
    if eigenvalues[0] <= 1e-12 * max(eigenvalues[-1], 1e-300):
        return -np.inf
    return float(np.sum(np.log(eigenvalues)))


@njit(cache=True, nogil=True)
def _distances(x: np.ndarray, centre: np.ndarray, cov: np.ndarray) -> np.ndarray:
    inverse = np.linalg.inv(cov)
    n, p = x.shape
    out = np.empty(n)
    d = np.empty(p)
    for i in range(n):
        for a in range(p):
            d[a] = x[i, a] - centre[a]
        total = 0.0
        for a in range(p):
            inner = 0.0
            for b in range(p):
                inner += inverse[a, b] * d[b]
            total += d[a] * inner
        out[i] = total
    return out


@njit(cache=True, nogil=True)
def _smallest(values: np.ndarray, h: int) -> np.ndarray:
    """Indices of the ``h`` smallest values in linear time, ties at the cut taken in row order."""
    cut = np.partition(values, h - 1)[h - 1]
    rows = np.empty(h, dtype=np.int64)
    count = 0
    for i in range(values.size):
        if values[i] < cut:
            rows[count] = i
            count += 1
    for i in range(values.size):
        if count == h:
            break
        if values[i] == cut:
            rows[count] = i
            count += 1
    return rows


@njit(cache=True, nogil=True)
def _concentrate(x: np.ndarray, centre: np.ndarray, cov: np.ndarray, h: int, steps: int):
    """Up to ``steps`` concentration steps from ``(centre, cov)``; the best (logdet, centre, cov) seen."""
    best_det, best_centre, best_cov = np.inf, centre, cov
    previous = np.nan
    for step in range(steps):
        rows = _smallest(_distances(x, centre, cov), h)
        centre, cov = _moments(x, rows)
        det = _logdet(cov)
        if det == -np.inf:
            return det, centre, cov
        if det < best_det:
            best_det, best_centre, best_cov = det, centre, cov
        if step >= 1 and det == previous:
            break
        previous = det
    return best_det, best_centre, best_cov


@njit(cache=True, nogil=True)
def _start(x: np.ndarray, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean and covariance of a (p + 1)-subset, adding random rows while it is singular."""
    n = x.shape[0]
    chosen = np.zeros(n, dtype=np.bool_)
    chosen[rows] = True
    k = rows.size
    while True:
        centre, cov = _moments(x, np.flatnonzero(chosen))
        if _logdet(cov) > -np.inf or k >= n:
            return centre, cov
        extra = np.random.randint(n)
        while chosen[extra]:
            extra = np.random.randint(n)
        chosen[extra] = True
        k += 1


@njit(cache=True, nogil=True)
def _keep(dets, centres, covs, det, centre, cov) -> None:
    """Insert a solution into the ten-best store unless it is worse than all of them or already there."""
    if det >= dets[-1]:
        return
    for j in range(dets.size):
        if det == dets[j] and np.all(centres[j] == centre):
            return
    j = dets.size - 1
    while j > 0 and dets[j - 1] > det:
        dets[j] = dets[j - 1]
        centres[j] = centres[j - 1]
        covs[j] = covs[j - 1]
        j -= 1
    dets[j] = det
    centres[j] = centre
    covs[j] = cov


@njit(cache=True, nogil=True)
def _random_subset(n: int, k: int) -> np.ndarray:
    pool = np.arange(n)
    for i in range(k):
        j = i + np.random.randint(n - i)
        pool[i], pool[j] = pool[j], pool[i]
    return pool[:k].copy()


@njit(cache=True, nogil=True)
def _next_combination(rows: np.ndarray, n: int) -> bool:
    k = rows.size
    i = k - 1
    while i >= 0 and rows[i] == n - k + i:
        i -= 1
    if i < 0:
        return False
    rows[i] += 1
    for j in range(i + 1, k):
        rows[j] = rows[j - 1] + 1
    return True


@njit(cache=True, nogil=True)
def _final_steps(n: int, p: int) -> int:
    """Robustbase's concentration-step budget for the last stage, which shrinks with ``n * p``."""
    size = n * p
    if size <= 100000:
        return 100
    if size > 1000000:
        return 1
    return 11 - (size - 1) // 100000


@njit(cache=True, nogil=True)
def _search(x: np.ndarray, h: int, exhaustive: bool, n_samp: int):
    """Robustbase's FAST-MCD search: the logdet, mean and covariance of the best ``h`` rows found."""
    n, p = x.shape
    fraction = h / n
    dets = np.full(N_BEST, np.inf)
    centres = np.zeros((N_BEST, p))
    covs = np.zeros((N_BEST, p, p))
    if n < 2 * N_MINI:
        rows = np.arange(p + 1)
        for trial in range(n_samp if not exhaustive else 2**62):
            if exhaustive and trial > 0 and not _next_combination(rows, n):
                break
            centre, cov = _start(x, rows if exhaustive else _random_subset(n, p + 1))
            det, centre, cov = _concentrate(x, centre, cov, h, 2)
            if det == -np.inf:
                return det, centre, cov
            _keep(dets, centres, covs, det, centre, cov)
    else:
        groups = min(n // N_MINI, K_MINI)
        used = n if groups < K_MINI else K_MINI * N_MINI
        merged = x[_random_subset(n, used)]
        sizes = np.full(groups, used // groups)
        sizes[groups - used % groups :] += 1
        half_sub, half_merged = int(sizes[0] * fraction), int(used * fraction)
        start = 0
        for g in range(groups):
            sub = merged[start : start + sizes[g]]
            start += sizes[g]
            sub_dets = np.full(N_BEST, np.inf)
            sub_centres = np.zeros((N_BEST, p))
            sub_covs = np.zeros((N_BEST, p, p))
            for _ in range(n_samp // groups):
                centre, cov = _start(sub, _random_subset(sub.shape[0], p + 1))
                det, centre, cov = _concentrate(sub, centre, cov, half_sub, 2)
                if det > -np.inf:
                    _keep(sub_dets, sub_centres, sub_covs, det, centre, cov)
            for j in range(N_BEST):
                if sub_dets[j] < np.inf:
                    det, centre, cov = _concentrate(merged, sub_centres[j], sub_covs[j], half_merged, 2)
                    if det > -np.inf:
                        _keep(dets, centres, covs, det, centre, cov)
    best_det, best_centre, best_cov = np.inf, centres[0].copy(), covs[0].copy()
    steps = _final_steps(n, p)
    for j in range(1 if n > 5000 else N_BEST):
        if dets[j] < np.inf:
            det, centre, cov = _concentrate(x, centres[j], covs[j], h, steps)
            if det < best_det:
                best_det, best_centre, best_cov = det, centre, cov
    return best_det, best_centre, best_cov


@njit(cache=True, nogil=True)
def _univariate(x: np.ndarray, h: int) -> tuple[float, float]:
    """Robustbase's ``rfmcduni``: the window of ``h`` sorted values with the smallest spread, its mean and root mean square."""
    w = np.sort(x)
    windows = w.size - h + 1
    spread = np.empty(windows)
    sums = np.empty(windows)
    for j in range(windows):
        s = w[j : j + h]
        sums[j] = s.sum()
        spread[j] = (s * s).sum() - sums[j] ** 2 / h
    smallest = spread.min()
    ties = np.flatnonzero(spread == smallest)
    return sums[ties[(ties.size + 1) // 2 - 1]] / h, np.sqrt(smallest / h)


@njit(cache=True, nogil=True)
def mcd(
    x: np.ndarray, h: int, raw_factor: float, rew_factor: float, cutoff: float, exhaustive: bool, n_samp: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Reweighted MCD of ``x``, given the constants :func:`mcd_constants` returns for its size; ``NaN`` on an exact fit."""
    np.random.seed(seed)
    n, p = x.shape
    nan = (np.full(p, np.nan), np.full((p, p), np.nan))
    if p == 1:
        centre, scale = _univariate(x[:, 0], h)
        scale *= np.sqrt(raw_factor)
        if scale < 1e-7:
            return nan
        keep = ((x[:, 0] - centre) / scale) ** 2 < cutoff
    else:
        # Search on columns centred on their median and scaled by the h-th smallest absolute deviation, as robustbase does, so
        # that the singularity check does not depend on the units of each column.
        median = np.empty(p)
        mad = np.empty(p)
        for j in range(p):
            median[j] = np.median(x[:, j])
            mad[j] = np.partition(np.abs(x[:, j] - median[j]), h - 1)[h - 1]
            if mad[j] <= 0:
                return nan
        det, centre_v, cov = _search((x - median) / mad, h, exhaustive, n_samp)
        if det == -np.inf:
            return nan
        centre_v = centre_v * mad + median
        cov = cov * np.outer(mad, mad)
        keep = _distances(x, centre_v, cov * raw_factor) < cutoff
    rows = np.flatnonzero(keep)
    out_centre, out_cov = _moments(x, rows)
    if rows.size != n:
        out_cov *= rew_factor
    return out_centre, out_cov


def cov_mcd(x: np.ndarray, *, n_samp: int = N_SAMP, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Reweighted MCD location and scatter with robustbase ``covMcd``'s defaults (alpha = 1/2).

    Args:
        x: Observations in rows, without missing values.
        n_samp: Random (p + 1)-subsets the search starts from.
        seed: Seed for the random subsets.

    Returns:
        The location and the covariance, or arrays of ``NaN`` when more than ``h`` rows lie on a hyperplane (an exact fit).

    Raises:
        ValueError: There are no more rows than ``p + 1``.
    """
    x = np.ascontiguousarray(x, dtype=np.float64)
    n, p = x.shape
    if n <= p + 1:
        raise ValueError(f"the MCD needs more than p + 1 = {p + 1} rows, and there are {n}")
    h, raw, rew, exhaustive = mcd_constants(np.array([n]), p)
    return mcd(x, int(h[0]), float(raw[0]), float(rew[0]), float(chi2.ppf(0.975, p)), bool(exhaustive[0]), n_samp, seed)
