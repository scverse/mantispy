"""Working with a circular covariate such as a cell-cycle clock, an angle rather than a scalar.

A clock is a position on a circle, so its statistics are circular: correlation, mean and shift all use
sine and cosine, never a linear regression on the raw angle, which folds at the wrap.

Every function takes ``period``, the length of one full turn in the units the angle is stored in:
``2 * np.pi`` for radians (the default), ``360`` for degrees, ``1`` for turns. Angles that come back are in
the same units.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from mantispy._core._reduce import get_matrix
from mantispy._core._stats import benjamini_hochberg
from mantispy._core.frames import as_frame
from mantispy._core.mutation import inplace_or_copy

if TYPE_CHECKING:
    from anndata import AnnData

_TWO_PI = 2.0 * np.pi
#: Below this many blocks a signed-rank test across blocks cannot reach p < 0.05 (its floor is 2 / 2**n).
_MIN_BLOCKS = 6
#: Angles on which a fitted curve is evaluated to find its peak and to compare two fits.
_GRID = np.linspace(-np.pi, np.pi, 360, endpoint=False)


def _radians(angle: np.ndarray | float, period: float) -> np.ndarray:
    """An angle in units of ``period`` converted to radians."""
    return np.asarray(angle, dtype=float) * (_TWO_PI / period)


def _units(radians: np.ndarray | float, period: float) -> np.ndarray:
    """An angle in radians converted back to units of ``period``."""
    return np.asarray(radians, dtype=float) * (period / _TWO_PI)


def _wrap(angle: np.ndarray | float) -> np.ndarray:
    """Wrap an angle in radians, or an array of them, into ``[-pi, pi)``."""
    return (np.asarray(angle, dtype=float) + np.pi) % _TWO_PI - np.pi


def _circular_mean(angle: np.ndarray) -> float:
    """The mean direction of a set of angles in radians, ignoring NaN."""
    angle = np.asarray(angle, dtype=float)
    return float(np.arctan2(np.nanmean(np.sin(angle)), np.nanmean(np.cos(angle))))


def circular_correlation(theta: np.ndarray, phi: np.ndarray, period: float = _TWO_PI) -> float:
    """Circular correlation of two angles, the Jammalamadaka-SenGupta coefficient.

    Args:
        theta: Angles in units of ``period``.
        phi: Angles in units of ``period``, paired with ``theta``.
        period: The length of one full turn: ``2 * np.pi`` for radians, ``360`` for degrees, ``1`` for turns.

    Returns:
        The coefficient in ``[-1, 1]``, invariant to a shift of either input by a whole turn.
        ``0.0`` when either angle has no spread, or when no pair is finite.
    """
    theta = _radians(theta, period)
    phi = _radians(phi, period)
    ok = np.isfinite(theta) & np.isfinite(phi)
    theta, phi = theta[ok], phi[ok]
    if theta.size == 0:
        return 0.0
    centred_theta = np.sin(theta - _circular_mean(theta))
    centred_phi = np.sin(phi - _circular_mean(phi))
    denominator = np.sqrt(np.sum(centred_theta**2) * np.sum(centred_phi**2))
    return 0.0 if denominator == 0.0 else float(np.sum(centred_theta * centred_phi) / denominator)


def _score_phase(theta: np.ndarray, score: np.ndarray) -> float:
    """The angle at which a per-cell score peaks, from its first circular harmonic over the finite cells.

    The score is mean-centred first. Without that, a score with a nonzero mean adds the circular mean of
    where the cells sit, which pulls the answer toward the most crowded arc of the cycle.
    """
    ok = np.isfinite(theta) & np.isfinite(score)
    centred = score[ok] - score[ok].mean()
    return float(np.arctan2(np.sum(np.sin(theta[ok]) * centred), np.sum(np.cos(theta[ok]) * centred)))


def orient_cycle(
    theta: np.ndarray, peak_score: np.ndarray, direction_score: np.ndarray, period: float = _TWO_PI
) -> np.ndarray:
    """Rephase an arbitrary cycle angle so it reads the same way every time.

    The raw angle of an embedding has an arbitrary zero and an arbitrary sense. This rotates it so the
    peak of ``peak_score`` (the S score, say) sits at 0, and reflects it if needed so ``direction_score``
    (the G2/M score) rises as the angle increases away from 0. Both scores are mean-centred, so an offset
    in a score does not move the zero point, and cells with a missing angle or score are left out of the fit.

    Args:
        theta: The angles to rephase, in units of ``period``.
        peak_score: A per-cell score whose peak marks where 0 should be.
        direction_score: A per-cell score that should increase just after 0.
        period: The length of one full turn: ``2 * np.pi`` for radians, ``360`` for degrees, ``1`` for turns.

    Returns:
        The rephased angles in ``[-period / 2, period / 2)``, NaN where ``theta`` is NaN.
    """
    theta = _radians(theta, period)
    peak = np.asarray(peak_score, dtype=float)
    direction = np.asarray(direction_score, dtype=float)
    rotated = _wrap(theta - _score_phase(theta, peak))
    if _score_phase(rotated, direction) < 0:
        rotated = _wrap(-rotated)
    return _units(rotated, period)


def circular_neighbor_error(theta: np.ndarray, neighbors: np.ndarray, period: float = _TWO_PI) -> float:
    """Median angular distance between each angle and the mean angle of its neighbours.

    A low value means the embedding that produced ``neighbors`` places cells of a similar cycle position
    together, so it is a score for how well an embedding aligns with the cycle.

    Args:
        theta: The cycle angle of each cell, in units of ``period``.
        neighbors: Integer indices of each cell's nearest neighbours, shape ``(n_cells, k)``.
        period: The length of one full turn: ``2 * np.pi`` for radians, ``360`` for degrees, ``1`` for turns.

    Returns:
        The median over cells of the absolute angular distance to the neighbour mean direction, in units of ``period``.
    """
    theta = _radians(theta, period)
    neighbour_angles = theta[np.asarray(neighbors)]
    mean_direction = np.arctan2(
        np.nanmean(np.sin(neighbour_angles), axis=1), np.nanmean(np.cos(neighbour_angles), axis=1)
    )
    return float(_units(np.nanmedian(np.abs(_wrap(theta - mean_direction))), period))


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


def _unit_sums(theta: np.ndarray, units: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-unit sums of sine and cosine of the finite angles, with the unit labels they belong to."""
    ok = np.isfinite(theta)
    labels, codes = np.unique(units[ok], return_inverse=True)
    sin = np.bincount(codes, weights=np.sin(theta[ok]), minlength=labels.size)
    cos = np.bincount(codes, weights=np.cos(theta[ok]), minlength=labels.size)
    return labels, sin, cos


def _permutation_pvalue(
    theta: np.ndarray,
    group_rows: np.ndarray,
    reference_rows: np.ndarray,
    blocks: np.ndarray,
    units: np.ndarray,
    n_permutations: int,
    rng: np.random.Generator,
) -> float:
    """Two-sided p-value of the mean shift, permuting group and reference labels between units within each block.

    The unit (a guide, say) is what moves together, so a guide with many cells counts once. Each block keeps
    its number of group and reference units, so the null compares like with like inside every well.
    """
    per_block = []
    for value in pd.unique(blocks):
        in_block = blocks == value
        rows = in_block & (group_rows | reference_rows)
        labels, sin, cos = _unit_sums(theta[rows], units[rows])
        is_group = np.isin(labels, units[in_block & group_rows])
        if is_group.all() or not is_group.any():
            continue
        per_block.append((sin, cos, is_group))
    if not per_block:
        return float("nan")

    def statistic(assignments: list[np.ndarray]) -> float:
        shifts = [
            np.arctan2(sin[pick].sum(), cos[pick].sum()) - np.arctan2(sin[~pick].sum(), cos[~pick].sum())
            for (sin, cos, _), pick in zip(per_block, assignments, strict=True)
        ]
        return abs(_circular_mean(_wrap(np.asarray(shifts))))

    observed = statistic([is_group for _, _, is_group in per_block])
    exceed = sum(
        statistic([rng.permutation(is_group) for _, _, is_group in per_block]) >= observed - 1e-12
        for _ in range(n_permutations)
    )
    return (exceed + 1) / (n_permutations + 1)


@inplace_or_copy()
def circular_shift(
    adata: AnnData,
    angle_key: str,
    by: str,
    reference: object,
    block: str | None = None,
    unit: str | None = None,
    n_permutations: int = 0,
    period: float = _TWO_PI,
    seed: int = 0,
    key_added: str = "circular_shift",
    copy: bool = False,
) -> AnnData | None:
    """Circular-mean shift of each group's angle against a reference, within and across blocks.

    For each group of ``by``, the angle is compared with the ``reference`` group inside each ``block``
    (a well), and the per-block shifts are combined across blocks. A knockout that pushes cells around the
    cycle shows up as a non-zero shift.

    Without ``n_permutations`` the per-block shifts are tested against zero with a Wilcoxon signed-rank
    test. Its smallest attainable p-value is ``2 / 2**n_blocks``, 0.25 with three wells, so it warns below
    six blocks. With ``n_permutations`` the p-value instead comes from shuffling group and reference labels
    between ``unit`` (the guide) within each block, which works with few blocks and treats the guide, not
    the cell, as the replicate.

    The test is on the circular mean. A knockout that moves cells out of one arc and into the opposite one
    can leave the mean where it was, so a shift near zero does not rule out a change in where cells sit.

    Args:
        adata: Object carrying the angle in ``obs[angle_key]``.
        angle_key: ``obs`` column holding the angle, in units of ``period``.
        by: ``obs`` column naming the groups to score, one row of the result each.
        reference: The reference group: a value of ``by``, a boolean ``obs`` column name, or a mask.
            The reference is not scored against itself, so it has no row.
        block: ``obs`` column the comparison is paired within, normally the well; ``None`` pools all rows.
        unit: ``obs`` column naming the replicate unit the permutation shuffles, normally the guide.
            ``None`` shuffles single cells. Each unit must belong to one group.
        n_permutations: Permutations for the within-block null; ``0`` uses the signed-rank test across blocks.
        period: The length of one full turn: ``2 * np.pi`` for radians, ``360`` for degrees, ``1`` for turns.
        seed: Seed for the permutations.
        key_added: Key the table is written under in ``uns["mantispy"]``.
        copy: Return a modified copy instead of mutating in place.

    Returns:
        ``None``, or the modified copy.
        Writes ``uns["mantispy"][key_added]`` with one row per group: ``group``, the circular-mean ``shift``
        in units of ``period``, ``n_blocks`` that held both the group and the reference, ``pvalue`` and the
        Benjamini-Hochberg ``qvalue`` over the groups.

    Raises:
        ValueError: ``reference`` names neither a value of ``by`` nor an ``obs`` column, or a ``unit`` spans
            more than one group.
    """
    obs = as_frame(adata.obs)
    theta = _radians(obs[angle_key].to_numpy(dtype=float), period)
    reference_rows = _reference_mask(obs, by, reference)
    blocks = obs[block].to_numpy() if block is not None else np.zeros(adata.n_obs, dtype=int)
    units = obs[unit].astype(str).to_numpy() if unit is not None else np.arange(adata.n_obs).astype(str)
    if unit is not None:
        spans = obs.groupby(unit, observed=True)[by].nunique()
        if (spans > 1).any():
            raise ValueError(f"{int((spans > 1).sum())} values of {unit!r} span more than one group of {by!r}")
    rng = np.random.default_rng(seed)

    rows, few = [], []
    for group, group_rows in _group_masks(obs, by, adata.n_obs):
        group_rows = group_rows & ~reference_rows
        if not group_rows.any():
            continue
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
            rows.append((group, np.nan, 0, np.nan))
            continue
        per_block = np.asarray(shifts)
        if n_permutations:
            pvalue = _permutation_pvalue(theta, group_rows, reference_rows, blocks, units, n_permutations, rng)
        else:
            pvalue = _paired_pvalue(per_block)
            if per_block.size < _MIN_BLOCKS:
                few.append(group)
        rows.append((group, float(_units(_circular_mean(per_block), period)), per_block.size, pvalue))

    if few:
        warnings.warn(
            f"circular_shift: {len(few)} group(s) span fewer than {_MIN_BLOCKS} blocks, where the signed-rank "
            "test across blocks cannot reach p < 0.05 (its floor is 0.25 with three blocks). Pass "
            f"n_permutations= (and unit=, the guide) for a within-block permutation test. Groups: {few[:5]}",
            UserWarning,
            stacklevel=3,
        )
    table = pd.DataFrame(rows, columns=["group", "shift", "n_blocks", "pvalue"])
    table["qvalue"] = benjamini_hochberg(table["pvalue"].to_numpy())
    adata.uns.setdefault("mantispy", {})[key_added] = table
    return None


def harmonic_design(angle: np.ndarray, n_harmonics: int) -> np.ndarray:
    """Columns ``sin(k * angle), cos(k * angle)`` for ``k = 1 .. n_harmonics``, for an angle in radians."""
    return np.column_stack([f(k * angle) for k in range(1, n_harmonics + 1) for f in (np.sin, np.cos)])


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


def _cycle_fit(
    angle: np.ndarray, values: np.ndarray, nuisance: np.ndarray | None, n_harmonics: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """R^2, R^2 beyond the nuisance covariates, and the fitted curve on ``_GRID``, per feature."""
    ones = np.ones(angle.size)
    harmonics = harmonic_design(angle, n_harmonics)
    total = np.nansum((values - np.nanmean(values, axis=0)) ** 2, axis=0)
    safe = total > 0
    residual, coefficients = _fit_residual(np.column_stack([ones, harmonics]), values)
    r2 = np.where(safe, 1.0 - residual / np.where(safe, total, 1.0), 0.0)
    curve = harmonic_design(_GRID, n_harmonics) @ coefficients[1:]
    if nuisance is None:
        return r2, r2, curve
    reduced, _ = _fit_residual(np.column_stack([ones, nuisance]), values)
    full, _ = _fit_residual(np.column_stack([ones, nuisance, harmonics]), values)
    return r2, np.where(safe, (reduced - full) / np.where(safe, total, 1.0), 0.0), curve


def _curve_correlation(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Pearson correlation, per column, of two fitted curves on the same grid; NaN for a flat curve."""
    first = first - first.mean(axis=0)
    second = second - second.mean(axis=0)
    denominator = np.sqrt((first**2).sum(axis=0) * (second**2).sum(axis=0))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(denominator > 0, (first * second).sum(axis=0) / denominator, np.nan)


@inplace_or_copy()
def feature_cycle_association(
    adata: AnnData,
    angle_key: str,
    covariates: Sequence[str] = (),
    by: str | None = None,
    layer: str | None = None,
    n_harmonics: int = 1,
    period: float = _TWO_PI,
    seed: int = 0,
    key_added: str = "feature_cycle_association",
    copy: bool = False,
) -> AnnData | None:
    """How strongly each feature follows the cycle, as a periodic curve fitted within groups.

    Each feature is regressed on ``sin(k * angle)`` and ``cos(k * angle)`` for ``k`` up to ``n_harmonics``,
    giving the share of its variance the cycle explains and the angle at which the fitted curve peaks. One
    harmonic is a single smooth wave. DNA content rises through S and drops sharply at division, which one
    wave cannot follow, so a feature with a sharp step needs two or three. Regressing nuisance
    ``covariates`` out as well gives the share the cycle explains beyond them, which separates a real cycle
    feature from one that only tracks cell size or sequencing depth.

    The cells of each group are also split into two random halves and fitted separately. The correlation
    of the two fitted curves says whether the shape is reproducible or fitted to noise.

    Args:
        adata: Object with the features in ``X`` (or ``layer``) and the angle in ``obs[angle_key]``.
        angle_key: ``obs`` column holding the angle, in units of ``period``.
        covariates: Numeric ``obs`` columns to regress out before scoring the cycle's extra contribution.
        by: Fit within each group of this ``obs`` column and average, or ``None`` to fit over all rows.
        layer: Read this layer instead of ``X``.
        n_harmonics: Harmonics in the fitted curve.
        period: The length of one full turn: ``2 * np.pi`` for radians, ``360`` for degrees, ``1`` for turns.
        seed: Seed for the split into halves.
        key_added: Key the table is written under in ``uns["mantispy"]``.
        copy: Return a modified copy instead of mutating in place.

    Returns:
        ``None``, or the modified copy.
        Writes ``uns["mantispy"][key_added]`` with one row per feature: ``feature``, ``r2``, ``r2_adjusted``
        (after the covariates), ``peak_angle`` in units of ``period``, and ``split_half_r``, the correlation
        of the curves fitted to two random halves of the cells (NaN for a flat curve).

    Raises:
        ValueError: ``n_harmonics`` is below one, or no group has enough cells with an angle to fit.
    """
    if n_harmonics < 1:
        raise ValueError(f"n_harmonics must be at least 1, got {n_harmonics}")
    obs = as_frame(adata.obs)
    theta = _radians(obs[angle_key].to_numpy(dtype=float), period)
    matrix = np.asarray(get_matrix(adata, layer), dtype=float)
    nuisance = (
        np.column_stack([obs[column].to_numpy(dtype=float) for column in covariates]) if len(covariates) else None
    )
    rng = np.random.default_rng(seed)
    # Each half has to hold more cells than the full model has parameters.
    minimum = 2 * (1 + 2 * n_harmonics + (0 if nuisance is None else nuisance.shape[1]))

    r2_groups, adjusted_groups, curves, reproducibility = [], [], [], []
    for _, rows in _group_masks(obs, by, adata.n_obs):
        # A cell with no angle cannot be placed on the cycle, so it is dropped for the whole group.
        finite = rows & np.isfinite(theta)
        angle, values = theta[finite], matrix[finite]
        block = nuisance[finite] if nuisance is not None else None
        if angle.size <= minimum:
            continue
        r2, adjusted, curve = _cycle_fit(angle, values, block, n_harmonics)
        half = rng.permutation(angle.size) < angle.size // 2
        first, second = (
            _cycle_fit(angle[h], values[h], None if block is None else block[h], n_harmonics)[2] for h in (half, ~half)
        )
        r2_groups.append(r2)
        adjusted_groups.append(adjusted)
        curves.append(curve)
        reproducibility.append(_curve_correlation(first, second))
    if not curves:
        raise ValueError(f"no group has more than {minimum} cells with a finite {angle_key!r} to fit")

    # The peak of the mean curve, so a feature that peaks at the same place in every group keeps it.
    peak = _GRID[np.argmax(np.mean(curves, axis=0), axis=0)]
    with warnings.catch_warnings():
        # A feature flat in every group has no reproducibility to report; NaN says so.
        warnings.simplefilter("ignore", RuntimeWarning)
        split_half = np.nanmean(reproducibility, axis=0)
    table = pd.DataFrame(
        {
            "feature": list(adata.var_names),
            "r2": np.mean(r2_groups, axis=0),
            "r2_adjusted": np.mean(adjusted_groups, axis=0),
            "peak_angle": _units(peak, period),
            "split_half_r": split_half,
        }
    )
    adata.uns.setdefault("mantispy", {})[key_added] = table
    return None
