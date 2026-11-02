"""Working with a cyclic covariate such as a cell-cycle clock, an angle rather than a scalar.

A clock is a position on a circle, so its statistics are circular: correlation, mean and shift all use
sine and cosine, never a linear regression on radians, which folds at the wrap from ``+pi`` to ``-pi``.
"""

from __future__ import annotations

import numpy as np

_TWO_PI = 2.0 * np.pi


def _wrap(angle: np.ndarray) -> np.ndarray:
    """Wrap an angle, or an array of angles, into ``[-pi, pi)``."""
    return (np.asarray(angle, dtype=float) + np.pi) % _TWO_PI - np.pi


def _circular_mean(angle: np.ndarray) -> float:
    """The mean direction of a set of angles, ignoring NaN."""
    angle = np.asarray(angle, dtype=float)
    return float(np.arctan2(np.nanmean(np.sin(angle)), np.nanmean(np.cos(angle))))


def circular_correlation(theta: np.ndarray, phi: np.ndarray) -> float:
    """Circular correlation of two angles, the Jammalamadaka-SenGupta coefficient.

    Args:
        theta: Angles in radians.
        phi: Angles in radians, paired with ``theta``.

    Returns:
        The coefficient in ``[-1, 1]``, invariant to a shift of either input by a whole turn.
        ``0.0`` when either angle has no spread, or when no pair is finite.
    """
    theta = np.asarray(theta, dtype=float)
    phi = np.asarray(phi, dtype=float)
    ok = np.isfinite(theta) & np.isfinite(phi)
    theta, phi = theta[ok], phi[ok]
    if theta.size == 0:
        return 0.0
    centred_theta = np.sin(theta - _circular_mean(theta))
    centred_phi = np.sin(phi - _circular_mean(phi))
    denominator = np.sqrt(np.sum(centred_theta**2) * np.sum(centred_phi**2))
    return 0.0 if denominator == 0.0 else float(np.sum(centred_theta * centred_phi) / denominator)


def orient_cycle(theta: np.ndarray, peak_score: np.ndarray, direction_score: np.ndarray) -> np.ndarray:
    """Rephase an arbitrary cycle angle so it reads the same way every time.

    The raw angle of an embedding has an arbitrary zero and an arbitrary sense. This rotates it so the
    peak of ``peak_score`` (the S score, say) sits at 0, and reflects it if needed so ``direction_score``
    (the G2/M score) rises as the angle increases away from 0.

    Args:
        theta: The angles to rephase, in radians.
        peak_score: A per-cell score whose peak marks where 0 should be.
        direction_score: A per-cell score that should increase just after 0.

    Returns:
        The rephased angles in ``[-pi, pi)``.
    """
    theta = np.asarray(theta, dtype=float)
    peak = np.asarray(peak_score, dtype=float)
    direction = np.asarray(direction_score, dtype=float)
    peak_phase = np.arctan2(np.sum(np.sin(theta) * peak), np.sum(np.cos(theta) * peak))
    rotated = _wrap(theta - peak_phase)
    direction_phase = np.arctan2(np.sum(np.sin(rotated) * direction), np.sum(np.cos(rotated) * direction))
    if direction_phase < 0:
        rotated = _wrap(-rotated)
    return rotated


def circular_neighbor_error(theta: np.ndarray, neighbors: np.ndarray) -> float:
    """Median angular distance between each angle and the mean angle of its neighbours.

    A low value means the embedding that produced ``neighbors`` places cells of a similar cycle position
    together, so it is a score for how well an embedding aligns with the cycle.

    Args:
        theta: The cycle angle of each cell, in radians.
        neighbors: Integer indices of each cell's nearest neighbours, shape ``(n_cells, k)``.

    Returns:
        The median over cells of the absolute angular distance to the neighbour mean direction.
    """
    theta = np.asarray(theta, dtype=float)
    neighbour_angles = theta[np.asarray(neighbors)]
    mean_direction = np.arctan2(
        np.nanmean(np.sin(neighbour_angles), axis=1), np.nanmean(np.cos(neighbour_angles), axis=1)
    )
    return float(np.nanmedian(np.abs(_wrap(theta - mean_direction))))
