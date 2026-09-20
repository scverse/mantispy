"""Dose response per compound: a monotonic trend test, a four-parameter logistic fit, and the response per feature."""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from math import lgamma, log, pi

import anndata as ad
import numpy as np
import pandas as pd
from anndata import AnnData

from mantispy._core._numba import group_offsets
from mantispy._core._reduce import get_matrix
from mantispy._core._stats import MAD_TO_SIGMA, benjamini_hochberg
from mantispy._core.frames import as_frame
from mantispy._core.logging import get_logger, report_drop
from mantispy._core.masks import held_out_reference, reference_mask
from mantispy._core.mutation import inplace_or_copy
from mantispy._core.provenance import record_params
from mantispy._core.schema import stamp

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
    from scipy.special import expit

    # The same sigmoid written through expit, which is overflow-safe: an optimizer exploring steep slopes far from
    # the data drives 10**((log_ec50 - log_dose) * hill) past the float range, and the naive form warns there.
    return bottom + (top - bottom) * expit(log(10.0) * (log_dose - log_ec50) * hill)


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

#: Normalizing constant of the Student-t log-density at ``_ERROR_DF``. It does not depend on the data, and
#: ``scipy.stats.t.logpdf`` spends ten times the arithmetic re-deriving it on every one of the thousand objective
#: evaluations a single curve costs.
_T_LOG_CONSTANT = lgamma((_ERROR_DF + 1) / 2) - lgamma(_ERROR_DF / 2) - 0.5 * log(_ERROR_DF * pi)


def _log_likelihood(residuals: np.ndarray, log_scale: float) -> float:
    """Student-t log-likelihood of the residuals at scale ``exp(log_scale)``, tcplfit2's ``tcplObj``."""
    scaled = residuals * np.exp(-log_scale)
    density = _T_LOG_CONSTANT - 0.5 * (_ERROR_DF + 1) * np.log1p(scaled * scaled / _ERROR_DF)
    return float(np.sum(density) - log_scale * residuals.size)


@dataclass(frozen=True)
class _Fit:
    """One fitted model: its name, its parameters with the error scale last, and its maximised log-likelihood."""

    name: str
    parameters: np.ndarray
    log_likelihood: float

    @property
    def log_scale(self) -> float:
        """The error scale, which every model carries as its last parameter."""
        return float(self.parameters[-1])

    @property
    def aic(self) -> float:
        return 2.0 * len(self.parameters) - 2.0 * self.log_likelihood

    def predict(self, log_dose: np.ndarray) -> np.ndarray:
        if self.name == "logistic":
            return four_parameter_logistic(log_dose, *self.parameters[:-1])
        # tcplfit2's poly1 runs through the origin of baseline-corrected response, so its top is a rescaling of
        # the one slope. Anchoring at the lowest tested dose is the same shape in log space.
        return self.parameters[0] * (log_dose - log_dose.min())

    def _top_axis(self, log_dose: np.ndarray) -> tuple[int, float]:
        """Which parameter carries the top, and what it is multiplied by to give it."""
        if self.name == "logistic":
            return 1, 1.0
        return 0, float(log_dose.max() - log_dose.min())

    def top(self, log_dose: np.ndarray) -> float:
        """The response the model reaches: the fitted asymptote, or the value at the highest dose tested."""
        index, scale = self._top_axis(log_dose)
        return float(self.parameters[index]) * scale

    def with_top_at(self, log_dose: np.ndarray, target: float) -> np.ndarray:
        """The same curve re-parameterized to reach exactly ``target``, the starting point of the profile fit."""
        index, scale = self._top_axis(log_dose)
        moved = self.parameters.copy()
        if scale > 0:
            moved[index] = target / scale
        return moved


def _fit_maximum_likelihood(name: str, log_dose: np.ndarray, response: np.ndarray, start: np.ndarray) -> _Fit | None:
    """Fit a model and the error scale together by maximum likelihood under the t error model."""
    from scipy.optimize import minimize

    def negative(parameters: np.ndarray) -> float:
        candidate = _Fit(name, parameters, 0.0)
        value = -_log_likelihood(response - candidate.predict(log_dose), candidate.log_scale)
        return float(value) if np.isfinite(value) else 1e18

    guess = np.append(start, np.log(max(float(np.std(response)), 1e-8)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        fitted = minimize(negative, guess, method="Nelder-Mead", options={"maxiter": 4000, "xatol": 1e-6})
    if not fitted.success or not np.isfinite(fitted.x).all():
        return None
    return _Fit(name, fitted.x, -float(fitted.fun))


def _profile_at_top(fit: _Fit, log_dose: np.ndarray, response: np.ndarray, target: float) -> float | None:
    """Largest log-likelihood the model reaches with its top held at ``target``, tcplfit2's ``toplikelihood``.

    Every other parameter, the error scale included, is fitted again around the pinned top. Leaving them where the
    unconstrained fit put them would understate this likelihood, and so overstate the confidence read from the drop.
    """
    from scipy.optimize import minimize

    index, _ = fit._top_axis(log_dose)
    pinned = fit.with_top_at(log_dose, target)

    def negative(free: np.ndarray) -> float:
        candidate = _Fit(fit.name, np.insert(free, index, pinned[index]), 0.0)
        value = -_log_likelihood(response - candidate.predict(log_dose), candidate.log_scale)
        return float(value) if np.isfinite(value) else 1e18

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        start = np.delete(pinned, index)
        fitted = minimize(negative, start, method="Nelder-Mead", options={"maxiter": 4000, "xatol": 1e-6})
    return -float(fitted.fun) if np.isfinite(fitted.fun) else None


def _winning_model(log_dose: np.ndarray, response: np.ndarray) -> _Fit | None:
    """Fit both models and keep the one with the lower AIC, as tcplfit2 keeps the best of its ten.

    Each model starts from its own heuristic rather than from the least-squares fit in the table. Seeding the
    logistic from that fit tied the two together: where it failed to converge the start was NaN, the logistic was
    skipped, and the model set silently collapsed to the line because a different estimator had given up.
    """
    slope = float(np.polyfit(log_dose, response, 1)[0]) if len(np.unique(log_dose)) > 1 else 0.0
    starts = {
        "logistic": np.array([float(response.min()), float(response.max()), float(np.median(log_dose)), 1.0]),
        "linear": np.array([slope]),
    }
    fits = [
        fitted
        for name, guess in starts.items()
        if np.isfinite(guess).all() and (fitted := _fit_maximum_likelihood(name, log_dose, response, guess))
    ]
    return min(fits, key=lambda fit: fit.aic) if fits else None


def _hitcall(fit: _Fit, log_dose: np.ndarray, response: np.ndarray, cutoff: float) -> float:
    """Continuous hit call: the three weights of Feshuk et al. 2023, multiplied.

    The weights are the confidence that the curve beats a constant fit, that at least one concentration's median
    response passes the cutoff, and that the fitted asymptote passes it. Ported from tcplfit2's ``hitcontinner``
    and ``toplikelihood``.

    mantispy picks between a logistic and a line in log dose; tcplfit2 picks among ten models, so the Akaike
    weight behind the first term is taken over a smaller set and a number from here will not match one from there.
    """
    from scipy.special import expit
    from scipy.stats import chi2, t

    top = fit.top(log_dose)
    log_scale = fit.log_scale

    # P1: the Akaike weight of the winning curve against the constant model, which fits the error scale only.
    constant = _fit_maximum_likelihood_constant(response)
    if constant is None:
        return float("nan")
    aic_constant = 2.0 * 1.0 - 2.0 * constant
    p1 = 1.0 - float(expit((fit.aic - aic_constant) / 2.0))

    # P2: one minus the odds of every concentration's median response falling short of the cutoff.
    # Each factor is the chance that a response this far out still came from a truth below the cutoff, which for a
    # response above it is the upper tail: tcplfit2 writes this as pt(..., lower.tail = top < 0).
    medians = pd.Series(response).groupby(log_dose).median().to_numpy()
    standardized = (medians - np.sign(top) * cutoff) / np.exp(log_scale)
    below = t.sf(standardized, _ERROR_DF) if top >= 0 else t.cdf(standardized, _ERROR_DF)
    p2 = 1.0 - float(np.prod(below))

    # P3: a likelihood profile on the asymptote. The top is pinned to the cutoff and the rest of the curve is
    # fitted again around it, and the drop in log-likelihood is read as a chi-square on one degree of freedom.
    profile = _profile_at_top(fit, log_dose, response, float(np.sign(top) * cutoff))
    if profile is None:
        return float("nan")
    tail = float(chi2.cdf(2.0 * (fit.log_likelihood - profile), 1))
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
    control = np.empty(0)
    # reference_mask raises when negcon is asked for and the column is absent. Here that is not fatal: the curve
    # and the trend need no controls, so only the hit call is left out.
    if reference is not None and not (reference == "negcon" and "Metadata_Control" not in adata.obs):
        values = as_frame(adata.obs)[response].to_numpy(dtype=float)
        control = values[held_out_reference(adata, reference_mask(adata, reference), response)]
        control = control[np.isfinite(control)]

    if control.size < 2:
        if cutoff is None:
            get_logger().info(
                "dose_response: %d usable control row(s), so no hit call. Mark the controls with "
                "mt.pp.annotate_controls, or pass an explicit cutoff.",
                control.size,
            )
        return 0.0, float(cutoff) if cutoff is not None else np.nan

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
        cutoff: Response a curve has to clear to count as active. The default takes three times the controls' MAD, the ToxCast pipeline's ``3 * bmad``, over the controls :func:`~mantispy.tl.hit_calling` held out of its own fit rather than over all of them.
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

        ``dose_key`` is read as given. A plate map that records one concentration to several precisions splits
        a ladder, which is a property of how the dataset was written rather than of the fit; the loader resolves
        it, as :func:`~mantispy.ds.oasis_pilot` does with ``Metadata_ConcentrationNominal``.

        :func:`dose_features` asks the same question of every feature rather than of one response column, and
        :func:`dose_direction` asks whether the phenotype stays the same one as the concentration rises.

        An EC50 outside the tested doses is an extrapolation, usually from a curve that has not plateaued within the tested range, and its row has ``fit_ok=False``.
        Compare ``ec50`` against the dose range before quoting it.
    """
    from scipy.stats import ConstantInputWarning, spearmanr

    obs = as_frame(adata.obs)
    _require_columns(obs, compound_key, dose_key)
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
                if (fit := _winning_model(log_dose, centred)) is not None:
                    hitcall, hitcall_model = _hitcall(fit, log_dose, centred, activity_cutoff), fit.name

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


#: Column order of the per-feature table.
_FEATURE_COLUMNS = (
    "compound",
    "feature",
    "n_doses",
    "bmd",
    "max_z",
    "direction",
    "spearman",
    "pvalue",
)

#: Column order of the per-dose table.
_DIRECTION_COLUMNS = (
    "compound",
    "dose",
    "n_wells",
    "amplitude",
    "amplitude_null",
    "step_amplitude",
    "split_half_cosine",
    "cosine_to_top",
    "viability",
    "phase",
)

#: What a concentration is doing, in the order they normally appear along a ladder.
DOSE_PHASES = ("silent", "responding", "saturated", "cytotoxic")


def _control_scale(adata: AnnData, reference: str | None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """The controls' centre and spread per feature, which features have one, and which rows the controls are.

    The centre and the spread are the ToxCast pipeline's ``bmed`` and ``bmad``. A feature whose controls show no
    spread has no scale to read a response against and is left out, so the two arrays come back already narrowed
    to ``keep``.
    """
    rows = reference_mask(adata, reference)
    if int(rows.sum()) < 2:
        raise ValueError(
            f"a per-feature dose analysis needs at least two control rows to set the scale, got {int(rows.sum())}. "
            "Mark the controls with mt.pp.annotate_controls, or name another reference. Subsetting to a "
            "dose_direction phase drops them, since a control is in no phase; keep them alongside the window."
        )
    control = get_matrix(adata, rows=np.flatnonzero(rows)).astype(np.float64)
    baseline = np.nanmedian(control, axis=0)
    # In place: two more copies of the control block would be the largest allocation in the function.
    control -= baseline
    np.abs(control, out=control)
    spread = MAD_TO_SIGMA * np.nanmedian(control, axis=0)
    keep = np.flatnonzero((spread > 0) & np.isfinite(spread))
    report_drop(
        "feature(s) with no spread among the controls",
        adata.n_vars - keep.size,
        adata.n_vars,
        remedy="run mt.pp.normalize and drop the features flagged by var['degenerate_scale']",
    )
    return baseline[keep], spread[keep], keep, rows


def _z_rows(adata: AnnData, rows: np.ndarray, baseline: np.ndarray, spread: np.ndarray, keep: np.ndarray) -> np.ndarray:
    """Those rows' response in MADs of the controls, over the features the controls give a scale for."""
    with np.errstate(invalid="ignore"):
        return (get_matrix(adata, rows=rows)[:, keep].astype(np.float64) - baseline) / spread


def _spearman_against(values: np.ndarray, block: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Spearman of ``values`` against every column of ``block``, with the two-sided p-value scipy reports.

    ``scipy.stats.spearmanr`` builds the whole square of correlations between the features to return one row of it,
    which on a full feature set is thousands of times the work and the memory this needs.
    """
    from scipy.stats import t

    from mantispy._core._corr import _rank_columns

    ranks = _rank_columns(np.column_stack([values, block]))
    centred = ranks - ranks.mean(axis=0)
    norms = np.sqrt(np.sum(centred**2, axis=0))
    with np.errstate(invalid="ignore", divide="ignore"):
        rho = np.clip(centred.T @ centred[:, 0] / (norms[0] * norms), -1.0, 1.0)[1:]
        statistic = rho * np.sqrt((ranks.shape[0] - 2) / (1.0 - rho**2))
    pvalue = 2.0 * t.sf(np.abs(statistic), ranks.shape[0] - 2)
    return rho, np.where(np.isfinite(rho), pvalue, np.nan)


def _nanmedian(block: np.ndarray) -> np.ndarray:
    """Median down the rows, skipping missing values, for the short and wide blocks this module works on.

    ``np.nanmedian`` routes anything under 600 rows through a masked array and ``np.ma.median``, which is pure
    Python; a dose group is four to eight rows, so every median here takes that path and spends about ninety-nine
    percent of its time on the wrapper. Sorting puts the missing values last, so the median is the middle of
    however many were measured.
    """
    ordered = np.sort(block, axis=0)
    measured = block.shape[0] - np.isnan(block).sum(axis=0)
    columns = np.arange(block.shape[1])
    low = ordered[np.maximum((measured - 1) // 2, 0), columns]
    high = ordered[np.maximum(measured // 2, 0), columns]
    return np.where(measured > 0, 0.5 * (low + high), np.nan)


def _groups(codes: np.ndarray, n_groups: int) -> list[np.ndarray]:
    """Row indices of each group, taken once rather than by scanning the codes per group."""
    order, offsets = group_offsets(np.ascontiguousarray(codes, dtype=np.int32), n_groups)
    return [order[offsets[index] : offsets[index + 1]] for index in range(n_groups)]


def _dose_medians(block: np.ndarray, doses: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The sorted doses, and each one's median profile over its replicate rows."""
    order, codes = np.unique(doses, return_inverse=True)
    return order, np.stack([_nanmedian(block[rows]) for rows in _groups(codes, order.size)])


def _benchmark_dose(doses: np.ndarray, z: np.ndarray, cutoff: float) -> np.ndarray:
    """Lowest dose at which each feature reaches ``cutoff``, interpolated between the doses either side of it.

    A feature that never reaches the cutoff has no benchmark dose and comes back NaN. One already past it at the
    lowest dose tested gets that dose, which is a bound rather than an estimate: clamping ``previous`` to the
    first dose leaves it no span to interpolate across, so the crossing falls on the dose itself.
    """
    over = np.abs(z) >= cutoff
    first = np.argmax(over, axis=0)
    previous = np.maximum(first - 1, 0)
    columns = np.arange(z.shape[1])
    log_dose = np.log10(doses)
    low, high = np.abs(z[previous, columns]), np.abs(z[first, columns])
    span = high - low
    fraction = np.divide(cutoff - low, span, out=np.zeros_like(span), where=span > 0)
    crossing = log_dose[previous] + fraction * (log_dose[first] - log_dose[previous])
    return np.where(over.any(axis=0), 10.0**crossing, np.nan)


def _usable_doses(
    obs: pd.DataFrame, compound_key: str, dose_key: str, control: np.ndarray
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Each compound's treated rows and their doses, leaving out the controls and the zero doses."""
    doses = obs[dose_key].to_numpy(dtype=float)
    usable = np.isfinite(doses) & (doses > 0) & ~control
    selected = np.flatnonzero(usable)
    codes, keys = pd.factorize(obs[compound_key].to_numpy()[usable], sort=True)
    blocks = {}
    for key, within in zip(keys, _groups(codes, len(keys)), strict=True):
        rows = selected[within]
        blocks[key] = (rows, doses[rows])
    return blocks


def _require_columns(obs: pd.DataFrame, *columns: str) -> None:
    for column in columns:
        if column not in obs:
            raise KeyError(f"obs has no column {column!r}")


@inplace_or_copy()
def dose_features(
    adata: AnnData,
    compound_key: str = "Metadata_Compound",
    dose_key: str = "Metadata_Concentration",
    reference: str | None = "negcon",
    cutoff_mads: float = _CUTOFF_MADS,
    min_doses: int = 4,
    key_added: str = "dose_features",
    copy: bool = False,
) -> AnnData | None:
    """Which features respond to a compound's concentration, and at what concentration each one starts.

    :func:`dose_response` grades one number per well against the dose. This grades every feature, which answers a
    different question: not whether the compound did something, but what it did and in what order.

    Each feature gets a benchmark dose, the lowest concentration at which its median response reaches
    ``cutoff_mads`` times the spread the controls show on it. This is the ToxCast pipeline's ``3 * bmad`` read as a
    dose rather than as a yes or no, and it is defined for a feature that is still climbing at the top
    concentration, which an EC50 is not. Sorting the table by ``bmd`` gives the order the phenotype arrives in.

    Args:
        adata: Object carrying a compound and a dose per row, at well resolution.
        compound_key: ``obs`` column holding the compound identity.
        dose_key: ``obs`` column holding the concentration. Rows with a zero or missing dose are left out, since the doses are read in log space.
        reference: Rows that set each feature's baseline and spread. ``"negcon"`` reads ``Metadata_Control``.
        cutoff_mads: Multiples of the controls' MAD a feature has to reach to get a benchmark dose. The ToxCast pipeline uses three.
        min_doses: Distinct doses below which a compound is left out of the table.
        key_added: Name for the output table.
        copy: Return a modified copy instead of mutating in place.

    Returns:
        ``None``, or the modified copy.
        Writes ``uns["mantispy"][key_added]``, one row per compound and feature, with ``compound``, ``feature``,
        ``n_doses``, ``bmd``, ``max_z``, ``direction``, ``spearman``, ``pvalue`` and ``qvalue``.
        ``bmd`` is NaN for a feature that never reaches the cutoff, which is the table's activity call.
        ``max_z`` is the largest response any concentration reached, in MADs of the controls, and ``direction`` is
        its sign. ``qvalue`` corrects the Spearman p-values within each compound, which is the experiment.

    Raises:
        KeyError: ``obs`` has no ``compound_key`` or no ``dose_key``.
        ValueError: Fewer than two reference rows, so there is no scale to read a response against.

    Notes:
        A benchmark dose at the lowest concentration tested is a bound, not an estimate: the feature was already
        past the cutoff before the series began.

        The Spearman correlation runs over the compound's wells, not over the per-dose medians, so it uses the
        replicates. A feature with a missing value in any of those wells has no Spearman and no q-value; its
        benchmark dose is still read, because the medians skip missing wells.

        Features whose controls show no spread are left out entirely. Run :func:`mantispy.pp.normalize` and drop
        ``var["degenerate_scale"]`` first and there will be none.
    """
    obs = as_frame(adata.obs)
    _require_columns(obs, compound_key, dose_key)
    baseline, spread, keep, control = _control_scale(adata, reference)
    names = np.asarray(adata.var_names)[keep]

    frames = []
    for compound, (rows, doses) in _usable_doses(obs, compound_key, dose_key, control).items():
        n_doses = len(np.unique(doses))
        if n_doses < min_doses:
            get_logger().debug("dose_features skipped %s: %d usable dose(s)", compound, n_doses)
            continue
        block = _z_rows(adata, rows, baseline, spread, keep)
        order, z = _dose_medians(block, doses)
        rho, pvalue = _spearman_against(np.log10(doses), block)
        # The concentration each feature reached furthest at, and which way it went.
        peak = z[np.argmax(np.abs(np.nan_to_num(z, nan=0.0)), axis=0), np.arange(z.shape[1])]
        frame = pd.DataFrame(
            {
                "compound": str(compound),
                "feature": names,
                "n_doses": n_doses,
                "bmd": _benchmark_dose(order, z, cutoff_mads),
                "max_z": np.abs(peak),
                "direction": np.sign(peak),
                "spearman": rho,
                "pvalue": pvalue,
            }
        )
        frame["qvalue"] = benjamini_hochberg(frame["pvalue"].to_numpy())
        frames.append(frame)

    columns = [*_FEATURE_COLUMNS, "qvalue"]
    table = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=columns)
    adata.uns.setdefault("mantispy", {})[key_added] = table[columns]
    get_logger().info(
        "dose_features: %d of %d compound-feature pairs reach %.1f MADs",
        int(table["bmd"].notna().sum()),
        len(table),
        cutoff_mads,
    )
    return None


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    """Cosine between two response directions, reading a missing or infinite feature as no movement."""
    left = np.nan_to_num(left, nan=0.0, posinf=0.0, neginf=0.0)
    right = np.nan_to_num(right, nan=0.0, posinf=0.0, neginf=0.0)
    norms = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(left @ right / norms) if norms > 0 else np.nan


def _amplitude(profile: np.ndarray) -> float:
    """How far a profile sits from the controls: its root-mean-square response, in MADs of the controls."""
    measured = profile[np.isfinite(profile)]
    return float(np.sqrt(np.mean(measured**2))) if measured.size else np.nan


def _split_half_cosine(block: np.ndarray, labels: np.ndarray) -> float:
    """Cosine between the two halves' median profiles, split by alternate levels of ``labels``.

    Halving by plate asks whether the direction survives a different plate, which is the harder and more useful
    question. A column with one level falls back to splitting by position, and one well at a concentration has no
    halves to compare, so it comes back NaN.
    """
    left = np.isin(labels, pd.unique(labels)[::2])
    if left.all():
        left = np.arange(labels.size) % 2 == 0
    if left.all() or not left.any():
        return np.nan
    return _cosine(_nanmedian(block[left]), _nanmedian(block[~left]))


#: Control groups drawn per layout to estimate the amplitude floor. The median of this many is stable to about 5%.
_NULL_DRAWS = 25


def _null_amplitude(control: np.ndarray, levels: np.ndarray, wanted: dict[object, int]) -> float:
    """Amplitude reached by control wells laid out the way ``wanted`` counts them, level by level.

    Two things have to match, or the floor is not the floor the concentration is read against.

    The layout, because per-plate normalization puts each plate's control median at zero: a group drawn from one
    plate starts out at the centre while a group spread over eight of them does not, and a group as large as a
    plate's control set reads a floor of zero.

    And the drawn wells have to be held out of the scale they are then measured in, as a treated well is. Scored
    against a baseline they helped define, control groups sit closer to it than any treated group can: with
    sixteen control wells that reads the floor 64% low, with thirty-two 30% low, and with the OASIS pilot's 256
    about 4% low.
    """
    pools = {level: np.flatnonzero(levels == level) for level in wanted}
    if any(pools[level].size < count for level, count in wanted.items()):
        return np.nan
    generator = np.random.default_rng(0)

    def draw() -> float:
        rows = np.concatenate([generator.choice(pools[level], count, replace=False) for level, count in wanted.items()])
        held_out = np.ones(control.shape[0], dtype=bool)
        held_out[rows] = False
        if held_out.sum() < 2:
            return np.nan
        baseline = _nanmedian(control[held_out])
        spread = MAD_TO_SIGMA * _nanmedian(np.abs(control[held_out] - baseline))
        with np.errstate(invalid="ignore"):
            return _amplitude((_nanmedian(control[rows]) - baseline) / np.where(spread > 0, spread, np.nan))

    return float(np.median([draw() for _ in range(_NULL_DRAWS)]))


def _viability(
    adata: AnnData, count_key: str, site_key: str | None, control: np.ndarray, levels: np.ndarray
) -> np.ndarray:
    """Each row's cell count against its own plate's controls, per field of view where the fields are known.

    Against the whole screen's controls it would not be a viability at all. Plates are seeded and imaged
    separately and their control counts differ by a factor of two on the OASIS pilot, so a plate that happens to
    be dense reads as a plate whose treated wells are dying.
    """
    obs = as_frame(adata.obs)
    if count_key not in obs:
        get_logger().info(
            "dose_direction: obs has no column %r, so no concentration is marked cytotoxic. "
            "mt.tl.aggregate and every well-level mt.ds dataset write Metadata_CellCount.",
            count_key,
        )
        return np.full(adata.n_obs, np.nan)

    counts = obs[count_key].to_numpy(dtype=float)
    if site_key is not None and site_key in obs:
        counts = counts / np.maximum(obs[site_key].to_numpy(dtype=float), 1)

    viability = np.full(adata.n_obs, np.nan)
    unreferenced = 0
    for level in pd.unique(levels):
        on = levels == level
        reference = float(np.nanmedian(counts[on & control])) if (on & control).any() else np.nan
        if np.isfinite(reference) and reference > 0:
            viability[on] = counts[on] / reference
        else:
            unreferenced += 1
    report_drop(
        "level(s) of the split with no control count to read viability against",
        unreferenced,
        len(pd.unique(levels)),
        remedy=f"check that every {count_key!r} is filled and that each level carries controls",
    )
    return viability


def _label_phases(ladder: list[dict], min_viability: float, reproducible: float | None) -> list[str]:
    """Label one compound's whole ladder at once.

    The window is a run, not a scatter. ``split_half_cosine`` over a handful of replicate wells is itself noisy,
    and thresholding each concentration on its own lets one lucky concentration open a window several steps below
    where the compound actually engages: on the OASIS pilot that put amperozide's onset at 3.7 uM instead of 33.
    A response that has started does not stop, so the window is the run of concentrations reaching the highest one
    that is not cytotoxic, and anything below the run is silent whatever its own cosine says.

    Cell loss is read first and separately. A concentration that has lost its cells reports the morphology of
    dying cells whatever else is true of it, which is why the US EPA's phenotypic pipeline drops those
    concentrations before fitting anything.
    """

    def active(row: dict) -> bool:
        # A comparison against NaN is False, so a concentration with no amplitude or no reproducible direction
        # fails these without a guard of its own.
        return bool(row["amplitude"] > row["amplitude_null"]) and (
            reproducible is None or bool(row["split_half_cosine"] >= reproducible)
        )

    phases = ["cytotoxic" if row["viability"] < min_viability else "silent" for row in ladder]
    highest = max((index for index, phase in enumerate(phases) if phase != "cytotoxic"), default=-1)
    for index in range(highest, -1, -1):
        if phases[index] == "cytotoxic" or not active(ladder[index]):
            break
        # Still moving if the step from the concentration below beats the noise two independent medians carry.
        moving = ladder[index]["step_amplitude"] > np.sqrt(2.0) * ladder[index]["amplitude_null"]
        phases[index] = "responding" if moving else "saturated"
    return phases


@inplace_or_copy()
def dose_direction(
    adata: AnnData,
    compound_key: str = "Metadata_Compound",
    dose_key: str = "Metadata_Concentration",
    reference: str | None = "negcon",
    split_by: str | None = "Metadata_Plate",
    min_doses: int = 4,
    count_key: str = "Metadata_CellCount",
    site_key: str | None = "Metadata_SiteCount",
    min_viability: float = 0.5,
    reproducible: float | None = 0.5,
    key_added: str = "dose_direction",
    copy: bool = False,
) -> AnnData | None:
    """Whether a compound's phenotype only grows with concentration, or turns into a different one.

    A dose series is usually summarised by one number per concentration, how far the wells sit from the controls.
    That number cannot tell a phenotype that is getting louder from a phenotype that is being replaced. This reads
    the direction as well as the distance: each concentration gets an ``amplitude``, its ``cosine_to_top`` against
    the highest concentration's profile, and a ``split_half_cosine`` that says whether its direction reproduces
    across replicates at all.

    The three read together. A concentration whose ``split_half_cosine`` is near zero has no direction to speak of,
    only noise, however large its amplitude. One that reproduces but sits at a low ``cosine_to_top`` is a real
    phenotype, and a different one from the top concentration's.

    ``phase`` turns that reading into a label, so the stretch of the ladder worth analysing can be subset out
    rather than described. A concentration is ``silent`` while nothing reproducible is happening, ``responding``
    while the profile is still moving from the concentration below it, ``saturated`` once it has stopped, and
    ``cytotoxic`` once the cells are gone and the profile is the morphology of dying cells. The label is
    broadcast to ``obs``, so ``adata[adata.obs["dose_direction_phase"] == "responding"]`` is the window, and
    :func:`dose_features`, :func:`~mantispy.tl.differential_features` and the rest work on it unchanged.

    Args:
        adata: Object carrying a compound and a dose per row, at well resolution.
        compound_key: ``obs`` column holding the compound identity.
        dose_key: ``obs`` column holding the concentration. Rows with a zero or missing dose are left out.
        reference: Rows that set each feature's baseline and spread. ``"negcon"`` reads ``Metadata_Control``.
        split_by: ``obs`` column whose levels split the replicates in two for ``split_half_cosine``, normally the plate. It also lays out the control groups ``amplitude_null`` is drawn from. ``None``, or a column with one level, splits the wells by position instead.
        min_doses: Distinct doses below which a compound is left out of the table. One concentration says nothing about how a response changes with concentration.
        count_key: ``obs`` column holding the cell count, which sets ``viability``. Without it no concentration is marked cytotoxic.
        site_key: ``obs`` column holding the number of fields that count covers, so a well missing a field does not read as cell loss. ``None`` compares the counts as they are.
        min_viability: Fraction of its own plate's control cell count below which a concentration is ``cytotoxic``. The US EPA's phenotypic pipeline drops a concentration that has lost more than half its cells before fitting anything.
        reproducible: ``split_half_cosine`` a concentration needs before it can be anything but ``silent``. ``None`` drops the requirement, which is what a screen with one well per concentration has to do, at the cost of calling noise a phenotype.
        key_added: Name for the output table.
        copy: Return a modified copy instead of mutating in place.

    Returns:
        ``None``, or the modified copy.
        Writes ``uns["mantispy"][key_added]``, one row per compound and concentration, with ``compound``, ``dose``,
        ``n_wells``, ``amplitude``, ``amplitude_null``, ``step_amplitude``, ``split_half_cosine``,
        ``cosine_to_top``, ``viability`` and ``phase``. The phase is broadcast to ``obs[key_added + "_phase"]``
        so the window can be subset like any other annotation.
        ``amplitude`` is the root-mean-square response over the features, in MADs of the controls, and
        ``amplitude_null`` is what control wells spread over the same plates in the same numbers reach, so the two
        are read against each other. ``step_amplitude`` is the same measure applied to the change from the
        concentration below, which is what tells a response that is still moving from one that has arrived.
        ``phase`` is one of ``DOSE_PHASES``.

    Raises:
        KeyError: ``obs`` has no ``compound_key``, no ``dose_key``, or no ``split_by`` column.
        ValueError: Fewer than two reference rows, so there is no scale to read a direction in.

    Notes:
        The cosines are taken over the features scaled by the controls' spread, so a feature the controls happen to
        measure loosely does not set the direction on its own. Features whose controls show no spread are left out.

        ``amplitude_null`` is the median over twenty-five draws, and is NaN when the controls cannot fill the
        layout, for instance when a plate carries treated wells but no vehicle.

        ``split_half_cosine`` needs at least two wells at a concentration, and reproduces the plate structure when
        ``split_by`` names it: halving by plate answers whether the direction survives a different plate, which is
        the harder and more useful question. With one well per concentration it is NaN, and the table then says
        nothing about whether any single concentration's direction is real.
    """
    obs = as_frame(adata.obs)
    _require_columns(obs, compound_key, dose_key, *([split_by] if split_by is not None else []))
    baseline, spread, keep, control = _control_scale(adata, reference)
    # Raw, not scaled: the floor holds each drawn group out of the scale it is measured in.
    control_values = get_matrix(adata, rows=np.flatnonzero(control))[:, keep].astype(np.float64)

    all_levels = obs[split_by].to_numpy() if split_by is not None else np.zeros(adata.n_obs)
    control_levels = all_levels[control]
    viable = _viability(adata, count_key, site_key, control, all_levels)
    phases = np.full(adata.n_obs, "", dtype=object)
    nulls: dict[tuple, float] = {}
    records: list[dict] = []
    for compound, (rows, doses) in _usable_doses(obs, compound_key, dose_key, control).items():
        n_doses = len(np.unique(doses))
        if n_doses < min_doses:
            get_logger().debug("dose_direction skipped %s: %d usable dose(s)", compound, n_doses)
            continue
        block = _z_rows(adata, rows, baseline, spread, keep)
        order, medians = _dose_medians(block, doses)
        levels, well_viability = all_levels[rows], viable[rows]

        ladder, covering = [], []
        for index, dose in enumerate(order):
            at = doses == dose
            # The floor is drawn with this concentration's own spread over plates, not from any group of that
            # size. setdefault would evaluate the draw whether or not the layout has been seen before.
            layout = dict(zip(*np.unique(levels[at], return_counts=True), strict=True))
            key = tuple(sorted(layout.items(), key=lambda item: str(item[0])))
            if key not in nulls:
                nulls[key] = _null_amplitude(control_values, control_levels, layout)
            # The step from the concentration below is what "still changing" means. The first one steps from the
            # controls, so its step is how far it has already come.
            amplitude = _amplitude(medians[index])
            here = well_viability[at]
            covering.append(rows[at])
            ladder.append(
                {
                    "compound": str(compound),
                    "dose": float(dose),
                    "n_wells": int(at.sum()),
                    "amplitude": amplitude,
                    "amplitude_null": nulls[key],
                    "step_amplitude": amplitude if index == 0 else _amplitude(medians[index] - medians[index - 1]),
                    "split_half_cosine": _split_half_cosine(block[at], levels[at]),
                    "cosine_to_top": _cosine(medians[index], medians[-1]),
                    "viability": float(np.nanmedian(here)) if np.isfinite(here).any() else np.nan,
                }
            )

        for row, phase, covered in zip(
            ladder, _label_phases(ladder, min_viability, reproducible), covering, strict=True
        ):
            row["phase"] = phase
            phases[covered] = phase
        records.extend(ladder)

    table = pd.DataFrame(records, columns=list(_DIRECTION_COLUMNS))
    adata.uns.setdefault("mantispy", {})[key_added] = table
    # A row that no concentration covers, a control or an unannotated well, is in no phase at all.
    phases[phases == ""] = None
    adata.obs[f"{key_added}_phase"] = pd.Categorical(phases, categories=list(DOSE_PHASES), ordered=False)
    get_logger().info(
        "dose_direction: %d compound(s) over %d concentration(s); %s",
        table["compound"].nunique(),
        len(table),
        table["phase"].value_counts().to_dict(),
    )
    return None


def dose_trajectory(
    adata: AnnData,
    compound_key: str = "Metadata_Compound",
    dose_key: str = "Metadata_Concentration",
    reference: str | None = "negcon",
    phase_key: str | None = "dose_direction_phase",
    phases: Sequence[str] = ("responding",),
    n_positions: int = 3,
) -> AnnData:
    """Each compound's whole path through its own responding window, on one comparable axis.

    Comparing two compounds at the same concentration compares one that has engaged against one that has not.
    Comparing them at their own strongest concentration throws away everything on the way there, which for a
    compound whose phenotype turns along its window is most of what it did. This reads each compound's window
    onto the same relative axis instead: position 0 is where it starts responding, position 1 is the top of its
    window, and the profile is interpolated in log concentration in between.

    The result is one row per compound and one column per feature and position, so the usual tools apply to it:
    :func:`~mantispy.tl.similarity` gives the compound-to-compound matrix, and a PCA or a clustering groups
    compounds by the shape of their response rather than by its size.

    Args:
        adata: Object carrying a compound, a dose and, normally, the phase column :func:`dose_direction` wrote.
        compound_key: ``obs`` column holding the compound identity.
        dose_key: ``obs`` column holding the concentration.
        reference: Rows that set each feature's baseline and spread.
        phase_key: ``obs`` column holding the phase, so the path is read over the window rather than the whole ladder. ``None``, or a column that is absent, uses every concentration the compound was dosed at.
        phases: Which phases count as the window.
        n_positions: Points the window is resampled to. Two is its ends; more describes the path between them, and cannot add detail a short window does not have.

    Returns:
        A new object of compounds by features-and-positions, at ``"perturbation"`` resolution.
        ``var`` carries ``feature`` and ``position``; ``obs`` carries ``n_doses`` and the window's ends.
        Compounds whose window holds fewer than two concentrations are left out, since a single point is not a path.

    Raises:
        KeyError: ``obs`` has no ``compound_key`` or no ``dose_key``.
        ValueError: ``n_positions`` is below two, or fewer than two reference rows.

    Notes:
        The axis is relative, so position 0 means a different concentration for every compound. That is the
        point, and it is also the caveat: two compounds matched here are matched on what they do in their own
        effective range, which is a claim about mechanism rather than about potency. Read it beside the benchmark
        doses from :func:`dose_features`, which is where the potency lives.

        How much this says over comparing the top of each window alone depends on how long the windows are. On
        the OASIS pilot, where most compounds respond over two or three of the ten concentrations, the two
        agree closely; the pairs they disagree on are the ones with long windows.
    """
    if n_positions < 2:
        raise ValueError(f"n_positions must be at least two, got {n_positions}")
    obs = as_frame(adata.obs)
    _require_columns(obs, compound_key, dose_key)
    baseline, spread, keep, control = _control_scale(adata, reference)

    in_window = np.ones(adata.n_obs, dtype=bool)
    if phase_key in obs:
        in_window = obs[phase_key].astype(str).isin(list(phases)).to_numpy()
    elif phase_key is not None:
        get_logger().info(
            "dose_trajectory: obs has no column %r, so the path is read over every concentration. "
            "Run mt.tl.dose_direction first to read it over the responding window.",
            phase_key,
        )

    grid = np.linspace(0.0, 1.0, n_positions)
    paths, records = [], []
    for compound, (rows, doses) in _usable_doses(obs, compound_key, dose_key, control).items():
        inside = in_window[rows]
        if not inside.any():
            continue
        rows, doses = rows[inside], doses[inside]
        order, z = _dose_medians(_z_rows(adata, rows, baseline, spread, keep), doses)
        if len(order) < 2:
            continue
        # Relative position through the window in log concentration, so the ends are 0 and 1 for every compound.
        position = np.log10(order)
        position = (position - position[0]) / (position[-1] - position[0])
        # One interval and weight per grid point serves every feature; np.interp per feature is the same
        # arithmetic done n_vars times over.
        below = np.clip(np.searchsorted(position, grid, side="right") - 1, 0, position.size - 2)
        weight = ((grid - position[below]) / (position[below + 1] - position[below]))[:, None]
        paths.append((z[below] * (1.0 - weight) + z[below + 1] * weight).ravel())
        records.append(
            {
                compound_key: str(compound),
                "n_doses": len(order),
                "window_low": float(order[0]),
                "window_high": float(order[-1]),
            }
        )

    names = np.asarray(adata.var_names)[keep]
    var = pd.DataFrame({"feature": np.tile(names, n_positions), "position": np.repeat(grid, names.size)})
    var.index = var["feature"] + "@" + var["position"].map("{:.2f}".format)
    result = ad.AnnData(
        X=np.array(paths, dtype=np.float32).reshape(len(paths), var.shape[0]),
        obs=pd.DataFrame(records, columns=[compound_key, "n_doses", "window_low", "window_high"]).set_axis(
            pd.RangeIndex(len(records)).astype(str)
        ),
        var=var,
    )
    stamp(result, resolution="perturbation")
    record_params(
        result,
        "dose_trajectory",
        {
            "compound_key": compound_key,
            "dose_key": dose_key,
            "reference": reference,
            "phase_key": phase_key,
            "phases": list(phases),
            "n_positions": n_positions,
        },
    )
    get_logger().info(
        "dose_trajectory: %d compound(s) over %d position(s); windows of %s concentration(s)",
        result.n_obs,
        n_positions,
        sorted(set(result.obs["n_doses"])),
    )
    return result
