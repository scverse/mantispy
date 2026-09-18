"""Dose response per compound: a monotonic trend test and a four-parameter logistic fit."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from anndata import AnnData

from mantispy._core._stats import benjamini_hochberg
from mantispy._core.frames import as_frame
from mantispy._core.logging import get_logger
from mantispy._core.mutation import inplace_or_copy

#: Column order of the output table, so an empty result still carries its columns.
_COLUMNS = ("compound", "n_doses", "spearman", "pvalue", "ec50", "hill_slope", "bottom", "top", "r_squared", "fit_ok")


def four_parameter_logistic(
    log_dose: np.ndarray, bottom: float, top: float, log_ec50: float, hill: float
) -> np.ndarray:
    """The standard sigmoid of dose-response pharmacology, in log10 dose.

    Args:
        log_dose: Base-10 logarithm of the dose.
        bottom: Response the curve approaches at low dose.
        top: Response the curve approaches at high dose.
        log_ec50: Base-10 logarithm of the dose halfway between ``bottom`` and ``top``.
        hill: Slope at the inflection point, negative for a decreasing curve.

    Returns:
        The response at each dose, the same shape as ``log_dose``.
    """
    return bottom + (top - bottom) / (1.0 + 10.0 ** ((log_ec50 - log_dose) * hill))


def _fit_curve(
    log_dose: np.ndarray, response: np.ndarray, min_r_squared: float
) -> tuple[float, float, float, float, float, bool]:
    """Fit the logistic and return ``(ec50, hill_slope, bottom, top, r_squared, fit_ok)``.

    A failed fit returns NaNs and ``fit_ok=False`` instead of raising.
    ``fit_ok`` also requires ``r_squared >= min_r_squared`` and an EC50 inside the tested doses, because four parameters converge on almost any five or six points.
    On pure noise at six doses the optimizer succeeds 59 times in 60; with these checks 12 in 200 fits pass, and real curves still do.
    """
    from scipy.optimize import OptimizeWarning, curve_fit

    guess = [float(response.min()), float(response.max()), float(np.median(log_dose)), 1.0]
    failed = (np.nan, np.nan, np.nan, np.nan, np.nan, False)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", OptimizeWarning)
            parameters, _ = curve_fit(four_parameter_logistic, log_dose, response, p0=guess, maxfev=5000)
    except (RuntimeError, ValueError, TypeError):
        return failed
    if not np.isfinite(parameters).all():
        return failed

    residual = response - four_parameter_logistic(log_dose, *parameters)
    total = float(np.sum((response - response.mean()) ** 2))
    r_squared = float(1.0 - np.sum(residual**2) / total) if total > 0 else np.nan
    ec50 = float(10.0 ** parameters[2])
    interpolated = bool(10.0 ** log_dose.min() <= ec50 <= 10.0 ** log_dose.max())
    fit_ok = bool(np.isfinite(r_squared) and r_squared >= min_r_squared and interpolated)
    return ec50, float(parameters[3]), float(parameters[0]), float(parameters[1]), r_squared, fit_ok


@inplace_or_copy()
def dose_response(
    adata: AnnData,
    compound_key: str = "Metadata_Compound",
    dose_key: str = "Metadata_Concentration",
    response: str = "hits_distance",
    min_doses: int = 4,
    min_r_squared: float = 0.8,
    key_added: str = "dose_response",
    copy: bool = False,
) -> AnnData | None:
    """Test whether each compound's response grows with concentration.

    Each compound gets two results.
    The Spearman correlation between dose and response tests for a monotonic trend, makes no assumption about shape and works with three doses.
    A four-parameter logistic fit adds an EC50 and a Hill slope, and is only attempted with at least ``min_doses`` distinct doses.

    ``fit_ok`` is True only when the fit succeeded, the curve explains the data (``r_squared >= min_r_squared``) and the EC50 lies inside the tested dose range.
    Four parameters converge on almost any five or six points.
    On pure noise the optimizer succeeds 59 times in 60, and these checks reduce that to 12 in 200 without losing real curves.
    Read ``spearman`` and its q-value first.

    Args:
        adata: Object carrying a compound, a dose and a per-row response.
        compound_key: ``obs`` column holding the compound identity.
        dose_key: ``obs`` column holding the concentration. Doses must be positive; rows with a zero dose, such as vehicle, are dropped, since the fit is in log dose.
        response: ``obs`` column holding the per-row response, normally the distance written by :func:`~mantispy.tl.hit_calling`.
        min_doses: Distinct doses below which the curve is skipped and only the trend is reported.
        min_r_squared: Coefficient of determination a fit needs before it is marked ok.
        key_added: Name for the output table.
        copy: Return a modified copy instead of mutating in place.

    Returns:
        ``None``, or the modified copy.
        Writes ``uns["mantispy"][key_added]`` with ``compound``, ``n_doses``, ``spearman``, ``pvalue``, ``qvalue``, ``ec50``, ``hill_slope``, ``bottom``, ``top``, ``r_squared`` and ``fit_ok``.
        ``bottom`` and ``top`` are the fitted asymptotes, so for a decreasing response, such as an inhibitor's, ``bottom`` is greater than ``top``.

    Raises:
        KeyError: ``obs`` has no ``compound_key``, no ``dose_key``, or no ``response`` column to use as the response.

    Notes:
        Compounds with fewer than two usable doses are left out of the table.

        An EC50 outside the tested doses is an extrapolation, usually from a curve that has not plateaued within the tested range, and its row has ``fit_ok=False``.
        Compare ``ec50`` against the dose range before quoting it.
    """
    from scipy.stats import ConstantInputWarning, spearmanr

    obs = as_frame(adata.obs)
    for column in (compound_key, dose_key):
        if column not in obs:
            raise KeyError(f"obs has no column {column!r}")
    if response not in obs:
        raise KeyError(
            f"obs has no column {response!r} to use as the response; run mt.tl.hit_calling first, which "
            "writes obs['hits_distance'], or name another column"
        )

    records = []
    for compound, block in obs.groupby(compound_key, observed=True):
        doses = block[dose_key].to_numpy(dtype=float)
        values = block[response].to_numpy(dtype=float)
        usable = np.isfinite(doses) & np.isfinite(values) & (doses > 0)
        doses, values = doses[usable], values[usable]
        n_doses = len(np.unique(doses))

        if n_doses < 2:
            get_logger().debug("dose_response skipped %s: %d usable dose(s)", compound, n_doses)
            continue

        with warnings.catch_warnings():
            # A constant response means no trend; spearman returns NaN, which the table keeps.
            warnings.simplefilter("ignore", ConstantInputWarning)
            correlation, pvalue = spearmanr(doses, values)
        ec50, hill, bottom, top, r_squared, fit_ok = (np.nan, np.nan, np.nan, np.nan, np.nan, False)
        if n_doses >= min_doses:
            ec50, hill, bottom, top, r_squared, fit_ok = _fit_curve(np.log10(doses), values, min_r_squared)

        records.append(
            {
                "compound": str(compound),
                "n_doses": int(n_doses),
                "spearman": float(correlation),
                "pvalue": float(pvalue),
                "ec50": ec50,
                "hill_slope": hill,
                "bottom": bottom,
                "top": top,
                "r_squared": r_squared,
                "fit_ok": fit_ok,
            }
        )

    table = pd.DataFrame(records, columns=list(_COLUMNS))
    table["qvalue"] = benjamini_hochberg(table["pvalue"].to_numpy()) if len(table) else []
    adata.uns.setdefault("mantispy", {})[key_added] = table
    get_logger().info("dose_response fitted %d of %d compound(s)", int(table["fit_ok"].sum()), len(table))
    return None
