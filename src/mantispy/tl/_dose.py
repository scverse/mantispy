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
_COLUMNS = (
    "compound",
    "n_doses",
    "spearman",
    "pvalue",
    "ec50",
    "hill_slope",
    "bottom",
    "top",
    "r_squared",
    "fit_ok",
    "hitcall",
    "hitcall_model",
)


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
    # The exponent overflows while an optimizer explores steep slopes far from the data. At 1e300 the term is
    # already infinite next to the numerator, so clipping changes the value by nothing and drops the warning.
    exponent = np.clip((log_ec50 - log_dose) * hill, -300.0, 300.0)
    return bottom + (top - bottom) / (1.0 + 10.0**exponent)


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


#: Degrees of freedom of the Student-t error model, as in tcplfit2's ``errfun="dt4"``.
_ERROR_DF = 4

#: Multiple of the baseline MAD that sets the cutoff when one is not given, as in the ToxCast pipeline's ``3 * bmad``.
_CUTOFF_MADS = 3.0


def _log_likelihood(residuals: np.ndarray, log_scale: float) -> float:
    """Student-t log-likelihood of the residuals at scale ``exp(log_scale)``, tcplfit2's ``tcplObj``."""
    from scipy.stats import t

    return float(np.sum(t.logpdf(residuals / np.exp(log_scale), _ERROR_DF) - log_scale))


def _linear_in_log_dose(log_dose: np.ndarray, slope: float) -> np.ndarray:
    """A line through the baseline at the lowest dose tested, for curves still climbing at the top dose.

    tcplfit2's ``poly1`` runs through the origin of baseline-corrected response, so its top is a rescaling of the
    one slope. Anchoring at the lowest tested dose is the same shape in log space, and keeps the re-parameterization
    in :func:`_with_top_at` to that one number.
    """
    return slope * (log_dose - log_dose.min())


#: Free parameters of each model, before the error scale.
_MODEL_PARAMETERS = {"logistic": 4, "linear": 1}


def _predict(name: str, log_dose: np.ndarray, parameters: np.ndarray) -> np.ndarray:
    if name == "logistic":
        return four_parameter_logistic(log_dose, *parameters[:4])
    return _linear_in_log_dose(log_dose, float(parameters[0]))


def _span(log_dose: np.ndarray) -> float:
    return float(log_dose.max() - log_dose.min())


def _top_of(name: str, log_dose: np.ndarray, parameters: np.ndarray) -> float:
    """The response the model reaches: the fitted asymptote, or the value at the highest dose tested."""
    if name == "logistic":
        return float(parameters[1])
    return float(parameters[0]) * _span(log_dose)


def _with_top_at(name: str, log_dose: np.ndarray, parameters: np.ndarray, target: float) -> np.ndarray:
    """The same curve re-parameterized to reach exactly ``target``, as tcplfit2's ``toplikelihood`` does."""
    moved = parameters.copy()
    if name == "logistic":
        moved[1] = target
    else:
        span = _span(log_dose)
        moved[0] = target / span if span > 0 else moved[0]
    return moved


def _fit_maximum_likelihood(
    name: str, log_dose: np.ndarray, response: np.ndarray, start: np.ndarray
) -> tuple[np.ndarray, float] | None:
    """Fit a model and the error scale together by maximum likelihood under the t error model."""
    from scipy.optimize import minimize

    n_parameters = _MODEL_PARAMETERS[name]

    def negative(parameters: np.ndarray) -> float:
        value = -_log_likelihood(response - _predict(name, log_dose, parameters), parameters[n_parameters])
        return float(value) if np.isfinite(value) else 1e18

    guess = np.append(start, np.log(max(float(np.std(response)), 1e-8)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        fitted = minimize(negative, guess, method="Nelder-Mead", options={"maxiter": 4000, "xatol": 1e-6})
    if not fitted.success or not np.isfinite(fitted.x).all():
        return None
    return fitted.x, -float(fitted.fun)


def _winning_model(log_dose: np.ndarray, response: np.ndarray, start: np.ndarray) -> tuple[str, np.ndarray, float]:
    """Fit both models and keep the one with the lower AIC, as tcplfit2 keeps the best of its ten."""
    slope = float(np.polyfit(log_dose, response, 1)[0]) if len(np.unique(log_dose)) > 1 else 0.0
    starts = {"logistic": start, "linear": np.array([slope])}
    best: tuple[str, np.ndarray, float] | None = None
    for name, guess in starts.items():
        if name == "logistic" and not np.isfinite(guess).all():
            continue
        fitted = _fit_maximum_likelihood(name, log_dose, response, guess)
        if fitted is None:
            continue
        n_parameters = _MODEL_PARAMETERS[name] + 1
        aic = 2.0 * n_parameters - 2.0 * fitted[1]
        if best is None or aic < best[2]:
            best = (name, fitted[0], aic)
    return best if best is not None else ("none", np.array([]), np.inf)


def _hitcall(
    name: str, log_dose: np.ndarray, response: np.ndarray, parameters: np.ndarray, aic: float, cutoff: float
) -> float:
    """Continuous hit call: the three weights of Feshuk et al. 2023, multiplied.

    The weights are the confidence that the curve beats a constant fit, that at least one concentration's median
    response passes the cutoff, and that the fitted asymptote passes it. Ported from tcplfit2's ``hitcontinner``
    and ``toplikelihood``.

    mantispy picks between a logistic and a line in log dose; tcplfit2 picks among ten models, so the Akaike
    weight behind the first term is taken over a smaller set and a number from here will not match one from there.
    """
    from scipy.special import expit
    from scipy.stats import chi2, t

    top = _top_of(name, log_dose, parameters)
    log_scale = float(parameters[_MODEL_PARAMETERS[name]])

    # P1: the Akaike weight of the winning curve against the constant model, which fits the error scale only.
    constant = _fit_maximum_likelihood_constant(response)
    if constant is None:
        return float("nan")
    aic_constant = 2.0 * 1.0 - 2.0 * constant
    p1 = 1.0 - float(expit((aic - aic_constant) / 2.0))

    # P2: one minus the odds of every concentration's median response falling short of the cutoff.
    # Each factor is the chance that a response this far out still came from a truth below the cutoff, which for a
    # response above it is the upper tail: tcplfit2 writes this as pt(..., lower.tail = top < 0).
    medians = pd.Series(response).groupby(log_dose).median().to_numpy()
    standardized = (medians - np.sign(top) * cutoff) / np.exp(log_scale)
    below = t.sf(standardized, _ERROR_DF) if top >= 0 else t.cdf(standardized, _ERROR_DF)
    p2 = 1.0 - float(np.prod(below))

    # P3: a likelihood profile on the asymptote. The curve is re-parameterized to put its top exactly on the
    # cutoff, which for an asymptote is the assignment itself, and the drop in log-likelihood is read as a
    # chi-square on one degree of freedom.
    at_cutoff = _with_top_at(name, log_dose, parameters, float(np.sign(top) * cutoff))
    profile = _log_likelihood(response - _predict(name, log_dose, at_cutoff), log_scale)
    mll = float(len(parameters)) - aic / 2.0
    tail = float(chi2.cdf(2.0 * (mll - profile), 1))
    p3 = (1.0 + tail) / 2.0 if abs(top) >= abs(cutoff) else (1.0 - tail) / 2.0

    return float(np.clip(p1 * p2 * p3, 0.0, 1.0))


def _fit_maximum_likelihood_constant(response: np.ndarray) -> float | None:
    """Log-likelihood of the constant model, which is zero response once the baseline is subtracted."""
    from scipy.optimize import minimize_scalar

    fitted = minimize_scalar(
        lambda log_scale: -_log_likelihood(response, log_scale),
        bounds=(np.log(1e-8), np.log(max(float(np.std(response)) * 100.0, 1e-6))),
        method="bounded",
    )
    return -float(fitted.fun) if fitted.success else None


def _baseline_and_cutoff(
    adata: AnnData, response: str, reference: str | None, cutoff: float | None
) -> tuple[float, float]:
    """Level the controls sit at, and the response a curve has to clear to be called active.

    Both come from the control rows, as the ToxCast pipeline's ``bmed`` and ``3 * bmad`` do.
    Without controls there is no scale to call activity on, so the hit call is left out rather than guessed at.
    """
    from mantispy._core._stats import MAD_TO_SIGMA
    from mantispy._core.masks import reference_mask

    values = as_frame(adata.obs)[response].to_numpy(dtype=float)
    if reference is None or (reference == "negcon" and "Metadata_Control" not in adata.obs):
        if cutoff is None:
            get_logger().info("dose_response: no controls to set a cutoff from, so no hit call")
            return 0.0, np.nan
        return 0.0, float(cutoff)

    control = values[reference_mask(adata, reference)]
    control = control[np.isfinite(control)]
    if control.size < 2:
        if cutoff is None:
            get_logger().info("dose_response: %d control row(s), too few for a cutoff, so no hit call", control.size)
            return 0.0, np.nan
        return 0.0, float(cutoff)

    baseline = float(np.median(control))
    if cutoff is not None:
        return baseline, float(cutoff)
    spread = MAD_TO_SIGMA * float(np.median(np.abs(control - baseline)))
    return baseline, _CUTOFF_MADS * spread if spread > 0 else np.nan


@inplace_or_copy()
def dose_response(
    adata: AnnData,
    compound_key: str = "Metadata_Compound",
    dose_key: str = "Metadata_Concentration",
    response: str = "hits_row_distance",
    min_doses: int = 4,
    min_r_squared: float = 0.8,
    reference: str | None = "negcon",
    cutoff: float | None = None,
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

    ``hitcall`` grades the curve on the scale the controls set, rather than only asking whether the optimizer converged.
    It is the product of the three weights of the ToxCast pipeline (Feshuk et al. 2023): the confidence that the
    curve beats a constant fit, that at least one concentration's median response clears the cutoff, and that the
    fitted asymptote clears it. The EPA reads ``hitcall >= 0.9`` as active. It needs controls to set a baseline and
    a cutoff, or an explicit ``cutoff``, and is ``NaN`` without them.

    The arithmetic is ported from ``tcplfit2``'s ``hitcontinner`` and ``toplikelihood``, including its Student-t
    error model on four degrees of freedom. The model set is where the numbers will differ: ``tcplfit2`` picks a
    winner among ten models, mantispy between two, the logistic and a line in log dose anchored at the lowest dose.
    The line is there for the same reason tcplfit2 carries ``poly1``: most real curves are still climbing at the top
    concentration, and a logistic with no plateau to find reports no EC50 at all. ``hitcall_model`` names the winner.

    Args:
        adata: Object carrying a compound, a dose and a per-row response.
        compound_key: ``obs`` column holding the compound identity.
        dose_key: ``obs`` column holding the concentration. Doses must be positive; rows with a zero dose, such as vehicle, are dropped, since the fit is in log dose.
        response: ``obs`` column holding the per-row response, normally ``hits_row_distance`` from :func:`~mantispy.tl.hit_calling`. Its group-level sibling ``hits_distance`` is one number repeated over each group's rows, so the controls show no spread and no cutoff can be read from them.
        min_doses: Distinct doses below which the curve is skipped and only the trend is reported.
        min_r_squared: Coefficient of determination a fit needs before it is marked ok.
        reference: Rows that set the baseline the response is read against and the spread the cutoff comes from. ``None`` leaves the hit call out unless ``cutoff`` is given.
        cutoff: Response a curve has to clear to count as active. The default takes three times the controls' MAD, the ToxCast pipeline's ``3 * bmad``.
        key_added: Name for the output table.
        copy: Return a modified copy instead of mutating in place.

    Returns:
        ``None``, or the modified copy.
        Writes ``uns["mantispy"][key_added]`` with ``compound``, ``n_doses``, ``spearman``, ``pvalue``, ``qvalue``, ``ec50``, ``hill_slope``, ``bottom``, ``top``, ``r_squared``, ``fit_ok``, ``hitcall`` and ``hitcall_model``.
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
            "writes obs['hits_row_distance'], or name another column"
        )

    baseline, activity_cutoff = _baseline_and_cutoff(adata, response, reference, cutoff)

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
        hitcall, hitcall_model = np.nan, ""
        if n_doses >= min_doses:
            log_dose = np.log10(doses)
            ec50, hill, bottom, top, r_squared, fit_ok = _fit_curve(log_dose, values, min_r_squared)
            if np.isfinite(activity_cutoff):
                # The hit call reads the response against the baseline the controls sit at, as tcpl's bmed does.
                centred = values - baseline
                start = np.array([bottom - baseline, top - baseline, np.log10(ec50), hill])
                name, parameters, aic = _winning_model(log_dose, centred, start)
                if name != "none":
                    hitcall = _hitcall(name, log_dose, centred, parameters, aic, activity_cutoff)
                    hitcall_model = name

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
                "hitcall": hitcall,
                "hitcall_model": hitcall_model,
            }
        )

    table = pd.DataFrame(records, columns=list(_COLUMNS))
    table["qvalue"] = benjamini_hochberg(table["pvalue"].to_numpy()) if len(table) else []
    adata.uns.setdefault("mantispy", {})[key_added] = table
    get_logger().info("dose_response fitted %d of %d compound(s)", int(table["fit_ok"].sum()), len(table))
    return None
