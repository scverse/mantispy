"""Circular-covariate helpers: correlation, cycle orientation, neighbour alignment."""

import numpy as np

from mantispy.tl._cyclic import (
    _wrap,
    circular_correlation,
    circular_neighbor_error,
    orient_cycle,
)


def test_circular_correlation_is_wrap_invariant_and_detects_a_relationship():
    rng = np.random.default_rng(0)
    theta = rng.uniform(-np.pi, np.pi, size=500)
    phi = _wrap(theta + 0.1 * rng.normal(size=500))
    r = circular_correlation(theta, phi)
    assert r > 0.8
    assert abs(r - circular_correlation(theta + 2 * np.pi, phi)) < 1e-9
    assert abs(circular_correlation(theta, rng.uniform(-np.pi, np.pi, size=500))) < 0.2


def test_orient_cycle_puts_the_peak_at_zero_and_orders_direction():
    theta = np.linspace(-np.pi, np.pi, 361, endpoint=False)
    peak = np.cos(theta - 1.0)  # S-score peak at angle 1.0
    direction = np.sin(theta - 1.0)  # rises just after the peak
    oriented = orient_cycle(theta, peak, direction)
    assert abs(oriented[np.argmax(peak)]) < 0.05
    mask = (oriented > 0) & (oriented < 1.0)
    assert np.corrcoef(oriented[mask], direction[mask])[0, 1] > 0


def test_circular_neighbor_error_is_small_for_adjacent_neighbours():
    theta = _wrap(np.linspace(0, 2 * np.pi, 200, endpoint=False))
    idx = np.arange(200)
    adjacent = np.c_[(idx - 1) % 200, (idx + 1) % 200]
    random = np.random.default_rng(0).integers(0, 200, size=(200, 2))
    assert circular_neighbor_error(theta, adjacent) < 0.2
    assert circular_neighbor_error(theta, adjacent) < circular_neighbor_error(theta, random)
