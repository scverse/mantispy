"""Working with a cyclic covariate such as a cell-cycle clock, an angle rather than a scalar.

A clock is a position on a circle, so its statistics are circular: correlation, mean and shift all use
sine and cosine, never a linear regression on radians, which folds at the wrap from ``+pi`` to ``-pi``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from mantispy._core._reduce import get_matrix
from mantispy._core.frames import as_frame

if TYPE_CHECKING:
    from anndata import AnnData

_TWO_PI = 2.0 * np.pi


def _wrap(angle: np.ndarray | float) -> np.ndarray:
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


def _reference_mask(obs: pd.DataFrame, by: str, reference: object) -> np.ndarray:
    """Rows that are the reference: a value of the ``by`` column, a boolean ``obs`` column, or a mask.

    Deliberately not :func:`mantispy._core.masks.reference_mask`: here the reference is usually one value
    of ``by`` (``"NTC"``) compared against another, which that control-class contract does not express.
    """
    if isinstance(reference, str):
        if reference in set(obs[by]):
            return (obs[by] == reference).to_numpy()
        if reference in obs.columns:
            return obs[reference].to_numpy().astype(bool)
        raise ValueError(f"reference {reference!r} is neither a value of {by!r} nor a column of obs")
    return np.asarray(reference, dtype=bool)


def _group_masks(obs: pd.DataFrame, by: str | None, n: int) -> list[tuple[object, np.ndarray]]:
    """One (label, boolean mask) per group, or a single all-rows group when ``by`` is None."""
    if by is None:
        return [(None, np.ones(n, dtype=bool))]
    values = obs[by].to_numpy()
    return [(value, values == value) for value in pd.unique(values)]


def _paired_pvalue(shifts: np.ndarray) -> float:
    """Wilcoxon signed-rank p-value of the per-block shifts against zero; NaN when it cannot be computed."""
    from scipy.stats import wilcoxon

    if np.count_nonzero(np.abs(shifts) > 1e-12) < 1:
        return float("nan")
    try:
        return float(wilcoxon(shifts).pvalue)
    except ValueError:
        return float("nan")


def cyclic_shift(
    adata: AnnData,
    angle_key: str,
    by: str,
    reference: object,
    block: str | None = None,
) -> pd.DataFrame:
    """Circular-mean shift of each group's cycle angle against a reference, within and across blocks.

    For each group of ``by``, the angle is compared with the ``reference`` group inside each ``block``
    (a well), and the per-block shifts are combined across blocks with a paired test. A knockout that
    pushes cells around the cycle shows up as a non-zero shift; the reference against itself is zero.

    Args:
        adata: Object carrying the cycle angle in ``obs[angle_key]``.
        angle_key: ``obs`` column holding the angle in radians.
        by: ``obs`` column naming the groups to score, one row of the result each.
        reference: The reference group: a value of ``by``, a boolean ``obs`` column name, or a mask.
        block: ``obs`` column the comparison is paired within, normally the well; ``None`` pools all rows.

    Returns:
        A frame with one row per group: ``group``, the circular-mean ``shift`` in radians, the paired
        ``pvalue`` over blocks, and ``n_blocks`` that contributed both the group and the reference.
    """
    obs = as_frame(adata.obs)
    theta = obs[angle_key].to_numpy(dtype=float)
    reference_rows = _reference_mask(obs, by, reference)
    blocks = obs[block].to_numpy() if block is not None else np.zeros(adata.n_obs, dtype=int)

    rows = []
    for group, group_rows in _group_masks(obs, by, adata.n_obs):
        shifts = []
        for value in pd.unique(blocks):
            in_block = blocks == value
            group_angles = theta[group_rows & in_block]
            reference_angles = theta[reference_rows & in_block]
            group_angles = group_angles[np.isfinite(group_angles)]
            reference_angles = reference_angles[np.isfinite(reference_angles)]
            if group_angles.size == 0 or reference_angles.size == 0:
                continue
            shifts.append(float(_wrap(_circular_mean(group_angles) - _circular_mean(reference_angles))))
        if not shifts:
            rows.append((group, float("nan"), float("nan"), 0))
            continue
        per_block = np.asarray(shifts)
        rows.append((group, _circular_mean(per_block), _paired_pvalue(per_block), per_block.size))
    return pd.DataFrame(rows, columns=["group", "shift", "pvalue", "n_blocks"])


def _fit_residual(design: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-column residual sum of squares of a least-squares fit, and the coefficients, NaN-aware.

    A column's missing cells are imputed with its mean for the fit and ignored when scoring, so a feature
    with a few missing values drops only those cells rather than poisoning the whole column.
    """
    column_mean = np.nanmean(values, axis=0)
    filled = np.nan_to_num(np.where(np.isfinite(values), values, column_mean), nan=0.0)
    coefficients, *_ = np.linalg.lstsq(design, filled, rcond=None)
    residual = values - design @ coefficients
    return np.nansum(residual**2, axis=0), coefficients


def feature_cycle_association(
    adata: AnnData,
    angle_key: str,
    covariates: Sequence[str] = (),
    by: str | None = None,
    layer: str | None = None,
) -> pd.DataFrame:
    """How strongly each feature follows the cycle, as a sinusoid fitted within groups.

    Each feature is regressed on ``[sin(angle), cos(angle)]``, giving the share of its variance the cycle
    explains and the angle at which it peaks. Regressing nuisance ``covariates`` out as well gives the
    share the cycle explains beyond them, which separates a real cycle feature from one that only tracks
    cell size or sequencing depth.

    Args:
        adata: Object with the features in ``X`` (or ``layer``) and the angle in ``obs[angle_key]``.
        angle_key: ``obs`` column holding the angle in radians.
        covariates: Numeric ``obs`` columns to regress out before scoring the cycle's extra contribution.
        by: Fit within each group of this ``obs`` column and average, or ``None`` to fit over all rows.
        layer: Read this layer instead of ``X``.

    Returns:
        A frame with one row per feature: ``feature``, ``r2``, ``r2_adjusted`` (after the covariates),
        and ``peak_angle`` in radians.
    """
    obs = as_frame(adata.obs)
    theta = obs[angle_key].to_numpy(dtype=float)
    matrix = np.asarray(get_matrix(adata, layer), dtype=float)
    nuisance = (
        np.column_stack([obs[column].to_numpy(dtype=float) for column in covariates]) if len(covariates) else None
    )

    r2_groups, r2_adjusted_groups, peak_groups = [], [], []
    for _, rows in _group_masks(obs, by, adata.n_obs):
        angle = theta[rows]
        values = matrix[rows]
        block = nuisance[rows] if nuisance is not None else None
        # A cell with no angle cannot be placed on the cycle, so it is dropped for the whole group.
        finite = np.isfinite(angle)
        angle, values = angle[finite], values[finite]
        if block is not None:
            block = block[finite]
        sin, cos, ones = np.sin(angle), np.cos(angle), np.ones(angle.size)
        cycle_design = np.column_stack([ones, sin, cos])
        if angle.size <= cycle_design.shape[1]:
            continue
        total = np.nansum((values - np.nanmean(values, axis=0)) ** 2, axis=0)
        safe = total > 0
        residual, coefficients = _fit_residual(cycle_design, values)
        r2 = np.where(safe, 1.0 - residual / np.where(safe, total, 1.0), 0.0)
        peak = np.arctan2(coefficients[1], coefficients[2])
        if block is None:
            r2_adjusted = r2
        else:
            reduced, _ = _fit_residual(np.column_stack([ones, block]), values)
            full, _ = _fit_residual(np.column_stack([ones, block, sin, cos]), values)
            r2_adjusted = np.where(safe, (reduced - full) / np.where(safe, total, 1.0), 0.0)
        r2_groups.append(r2)
        r2_adjusted_groups.append(r2_adjusted)
        peak_groups.append(peak)

    peaks = np.array(peak_groups)
    return pd.DataFrame(
        {
            "feature": list(adata.var_names),
            "r2": np.mean(r2_groups, axis=0),
            "r2_adjusted": np.mean(r2_adjusted_groups, axis=0),
            "peak_angle": np.arctan2(np.mean(np.sin(peaks), axis=0), np.mean(np.cos(peaks), axis=0)),
        }
    )
