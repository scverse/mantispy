"""liblinear's L2-regularized logistic regression, the solver behind ``LogisticRegression(solver="liblinear")``.

scikit-learn fits ``penalty="l2", dual=False`` with liblinear's solver type 0, ``L2R_LR``: a trust-region Newton method
(TRON) whose inner steps are conjugate gradients over Hessian-vector products, minimizing

    0.5 * ||w||^2 + sum_i C_i * log(1 + exp(-y_i * w @ x_i)),

with ``y_i`` in ``{-1, +1}`` and the bias appended to ``x_i`` as a constant column of ``intercept_scaling``, so the
intercept is regularized like every other weight. This is a port of ``tron.cpp`` and ``l2r_lr_fun`` from the copy
scikit-learn vendors, step for step: the same start at zero, trust-region updates, conjugate-gradient cut-off and
stopping rule ``||g|| <= eps * max(min(n_pos, n_neg), 1) / n * ||g_0||``, and the rows in liblinear's order, the
negative class first. It lands on the same iterate liblinear does, so the fit agrees with it to rounding, not merely to
the solver tolerance; a solver run to the exact optimum differs from liblinear by as much as that tolerance allows.
"""

from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True, nogil=True)
def _dot(a: np.ndarray, b: np.ndarray) -> float:
    total = 0.0
    for j in range(a.size):
        total += a[j] * b[j]
    return total


@njit(cache=True, nogil=True)
def _axpy(alpha: float, x: np.ndarray, y: np.ndarray) -> None:
    """``y += alpha * x`` in place."""
    for j in range(y.size):
        y[j] += alpha * x[j]


@njit(cache=True, nogil=True)
def _loss(x: np.ndarray, y: np.ndarray, c: np.ndarray, w: np.ndarray, z: np.ndarray) -> float:
    """The objective at ``w``, leaving the margins ``x @ w`` in ``z`` for :func:`_gradient`."""
    f = 0.5 * _dot(w, w)
    for i in range(x.shape[0]):
        z[i] = _dot(x[i], w)
        yz = y[i] * z[i]
        if yz >= 0:
            f += c[i] * np.log(1 + np.exp(-yz))
        else:
            f += c[i] * (-yz + np.log(1 + np.exp(yz)))
    return f


@njit(cache=True, nogil=True)
def _gradient(
    x: np.ndarray, y: np.ndarray, c: np.ndarray, w: np.ndarray, z: np.ndarray, curvature: np.ndarray, g: np.ndarray
) -> None:
    """The gradient at ``w`` into ``g``, from the margins :func:`_loss` left in ``z``, and each row's Hessian weight."""
    g[:] = w
    for i in range(x.shape[0]):
        sigma = 1 / (1 + np.exp(-y[i] * z[i]))
        curvature[i] = c[i] * sigma * (1 - sigma)
        _axpy(c[i] * (sigma - 1) * y[i], x[i], g)


@njit(cache=True, nogil=True)
def _hessian_times(x: np.ndarray, curvature: np.ndarray, s: np.ndarray, out: np.ndarray) -> None:
    """``(I + X' diag(curvature) X) s`` into ``out``."""
    out[:] = s
    for i in range(x.shape[0]):
        _axpy(curvature[i] * _dot(x[i], s), x[i], out)


@njit(cache=True, nogil=True)
def _conjugate_gradient(
    x: np.ndarray,
    curvature: np.ndarray,
    delta: float,
    g: np.ndarray,
    s: np.ndarray,
    r: np.ndarray,
    d: np.ndarray,
    hd: np.ndarray,
) -> None:
    """``TRON::trcg``: an approximate Newton step ``s`` within radius ``delta``, and its residual ``r``."""
    s[:] = 0.0
    r[:] = -g
    d[:] = r
    cgtol = 0.1 * np.sqrt(_dot(g, g))
    rtr = _dot(r, r)
    while np.sqrt(_dot(r, r)) > cgtol:
        _hessian_times(x, curvature, d, hd)
        alpha = rtr / _dot(d, hd)
        _axpy(alpha, d, s)
        if np.sqrt(_dot(s, s)) > delta:
            # Step back and walk along d to the trust-region boundary.
            _axpy(-alpha, d, s)
            std, sts, dtd = _dot(s, d), _dot(s, s), _dot(d, d)
            dsq = delta * delta
            rad = np.sqrt(std * std + dtd * (dsq - sts))
            alpha = (dsq - sts) / (std + rad) if std >= 0 else (rad - std) / dtd
            _axpy(alpha, d, s)
            _axpy(-alpha, hd, r)
            return
        _axpy(-alpha, hd, r)
        rnew = _dot(r, r)
        beta = rnew / rtr
        for j in range(d.size):
            d[j] = beta * d[j] + r[j]
        rtr = rnew


@njit(cache=True, nogil=True)
def fit_l2_logistic(x: np.ndarray, y: np.ndarray, c: np.ndarray, tol: float, max_iter: int) -> np.ndarray:
    """Weights minimizing liblinear's ``L2R_LR`` objective, by ``TRON::tron``.

    Args:
        x: ``(n, m)`` float64 rows, the bias column included, negative-class rows first as liblinear orders them.
        y: ``(n,)`` float64 labels in ``{-1, +1}``.
        c: ``(n,)`` per-row cost, ``C`` times the row's class weight and sample weight.
        tol: scikit-learn's ``tol``, liblinear's ``eps``.
        max_iter: Cap on the accepted trust-region steps.

    Returns:
        ``(m,)`` weights, the bias weight last when ``x`` carries it last.
    """
    n, m = x.shape
    n_pos = 0
    for i in range(n):
        n_pos += y[i] > 0
    eps = tol * max(min(n_pos, n - n_pos), 1) / n

    w = np.zeros(m)
    w_new, g, s, r, d, hd = np.empty(m), np.empty(m), np.empty(m), np.empty(m), np.empty(m), np.empty(m)
    z, curvature = np.empty(n), np.empty(n)
    f = _loss(x, y, c, w, z)
    _gradient(x, y, c, w, z, curvature, g)
    delta = np.sqrt(_dot(g, g))
    gnorm0 = delta
    iteration = 1
    while iteration <= max_iter and np.sqrt(_dot(g, g)) > eps * gnorm0:
        _conjugate_gradient(x, curvature, delta, g, s, r, d, hd)
        w_new[:] = w
        _axpy(1.0, s, w_new)
        gs = _dot(g, s)
        predicted = -0.5 * (gs - _dot(s, r))
        f_new = _loss(x, y, c, w_new, z)
        actual = f - f_new

        snorm = np.sqrt(_dot(s, s))
        if iteration == 1:
            delta = min(delta, snorm)
        alpha = 4.0 if f_new - f - gs <= 0 else max(0.25, -0.5 * (gs / (f_new - f - gs)))
        if actual < 1e-4 * predicted:
            delta = min(max(alpha, 0.25) * snorm, 0.5 * delta)
        elif actual < 0.25 * predicted:
            delta = max(0.25 * delta, min(alpha * snorm, 0.5 * delta))
        elif actual < 0.75 * predicted:
            delta = max(0.25 * delta, min(alpha * snorm, 4.0 * delta))
        else:
            delta = max(delta, min(alpha * snorm, 4.0 * delta))

        if actual > 1e-4 * predicted:
            iteration += 1
            w[:] = w_new
            f = f_new
            _gradient(x, y, c, w, z, curvature, g)
        if f < -1.0e32 or (abs(actual) <= 0 and predicted <= 0):
            break
        if abs(actual) <= 1.0e-12 * abs(f) and abs(predicted) <= 1.0e-12 * abs(f):
            break
    return w
