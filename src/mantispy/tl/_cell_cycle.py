"""Cell-cycle position from DNA content: an ergodic-rate model fitted on reference cells."""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from anndata import AnnData

from mantispy._core._reduce import get_matrix, group_codes, group_offsets
from mantispy._core._stats import MAD_TO_SIGMA
from mantispy._core.frames import as_frame
from mantispy._core.logging import get_logger
from mantispy._core.masks import reference_mask
from mantispy._core.mutation import inplace_or_copy
from mantispy.tl._heterogeneity import _WELL, _on_dna_stain

CYCLE_PHASES = ("G1", "S", "G2M", "sub-2N", "above-4N")
QC_FLAGS = ("ok", "irregular", "small", "dim")
QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)
_GATES = np.log2([1.25, 1.75])  # the dna_state defaults, used only by the S share the gate asks for
_GRID = (np.arange(400) + 0.5) / 400
_LOG_PRIOR = np.log(2 * np.log(2)) - _GRID * np.log(2)
_DENSITY_GRID = np.linspace(-1, 2.5, 701)
_MIN_REFERENCE = 50
_QC_SPREAD = 4  # robust standard deviations below the reference median that flag a nucleus
_QC_TERM = {"irregular": 1, "small": 0, "dim": 2}  # the column of _shape_terms each check reads
_ORDER_COMPONENTS, _ORDER_NEIGHBOURS, _ORDER_SMOOTHING = 30, 30, 10.0


def _age(fraction: Any) -> Any:
    """Cycle position of the cell at a given cumulative fraction of an exponentially growing population.

    Ages in such a population have density 2 ln2 2^-t on [0, 1), so the cumulative fraction F maps to t = -log2(1 - F/2).
    """
    return -np.log2(1 - np.clip(fraction, 0, 1) / 2)


def _robust(values: np.ndarray) -> tuple[float, float]:
    """Median and the median absolute deviation scaled to a standard deviation.

    Falls back to the standard deviation when the deviation is zero, as it is for a value rounded so coarsely that
    most cells share it.
    """
    centre = float(np.median(values))
    return centre, MAD_TO_SIGMA * float(np.median(np.abs(values - centre))) or float(np.std(values))


def _gate(content: np.ndarray) -> dict:
    """Locate the 2N and 4N peaks and check for a continuous S population between them."""
    from scipy.stats import gaussian_kde

    values = np.clip(content, -1, 2.5)
    if np.ptp(values) == 0:  # a constant population has no density to estimate, and no cycle
        return {"peaks": [float(values[0]), float(values[0])], "is_peak": False, "passed": False}
    density = gaussian_kde(values)(_DENSITY_GRID)
    grid = _DENSITY_GRID
    low = np.flatnonzero(np.abs(grid) < 0.25)
    two = low[np.argmax(density[low])]
    window = np.flatnonzero((grid >= 0.6) & (grid <= 1.4))
    top = window[np.argmax(density[window])]
    valley = density[(grid >= 0.2) & (grid <= grid[top] - 0.1)].min() if grid[top] > 0.35 else density[top]
    floor = density[(grid >= 0.4) & (grid <= 0.7)].min()
    with np.errstate(divide="ignore", invalid="ignore"):  # a population with no 4N peak has zero density there
        report: dict[str, Any] = {
            "peaks": [float(grid[two]), float(grid[top])],
            "is_peak": bool(
                density[top] >= density[max(top - 20, 0)] and density[top] >= density[min(top + 20, grid.size - 1)]
            ),
            "peak_ratio": float(density[top] / density[two]),
            "valley_ratio": float(valley / density[top]),
            "s_floor": float(floor / density[top]),
            "s_share": float(np.mean((content >= _GATES[0]) & (content < _GATES[1]))),
        }
    report["passed"] = bool(
        report["is_peak"]
        and report["peak_ratio"] >= 0.05
        and report["valley_ratio"] <= 0.9
        and report["s_floor"] >= 0.1
        and report["s_share"] >= 0.03
    )
    return report


def _span(peaks: Sequence[float]) -> tuple[float, float]:
    """The DNA range the off-cycle density covers: two doublings beyond either peak."""
    return float(peaks[0]) - 2, float(peaks[1]) + 2


def _normal(x: np.ndarray, mean: np.ndarray | float, sd: float) -> np.ndarray:
    return np.exp(-0.5 * ((x - mean) / sd) ** 2) / (sd * np.sqrt(2 * np.pi))


def _fit_fractions(values: np.ndarray, peaks: Sequence[float], sd: float) -> tuple[list[float], float]:
    """The G1, S and G2M fractions and the off-cycle weight that best explain the reference's DNA content.

    The cycle density is a 2N peak holding the G1 fraction, a 4N peak holding the G2M fraction, and S cells spread
    along a ramp linear in the amount of DNA, weighted by the age distribution; a flat density over the span stands
    for cells off the cycle. The fractions are found on a grid, the off-cycle weight by expectation-maximization.
    """
    low, high = _span(peaks)
    inside = values[(values > low) & (values < high)]
    outside = values.size - inside.size
    counts, bins = np.histogram(inside, np.arange(low, high + 0.01, 0.01))
    centres = (bins[:-1] + bins[1:]) / 2
    step = (np.arange(50) + 0.5) / 50
    ramp = np.log2(2.0 ** peaks[0] + (2.0 ** peaks[1] - 2.0 ** peaks[0]) * step)
    two, four = _normal(centres, peaks[0], sd), _normal(centres, peaks[1], sd)
    along = _normal(centres[:, None], ramp[None], sd)
    flat = 1 / (high - low)

    best: tuple[float, list[float], float] = (-np.inf, [1.0, 0.0, 0.0], 0.0)
    for first in np.arange(0.05, 0.955, 0.01):
        second = np.arange(first + 0.01, 0.995, 0.01)
        start, end = _age(first), _age(second)
        weight = 2.0 ** -(start + (end - start)[:, None] * step[None])
        weight *= ((second - first) / weight.sum(axis=1))[:, None]
        cycle = first * two[None] + (1 - second)[:, None] * four[None] + weight @ along.T
        off = np.full(second.size, 0.02)
        for _ in range(30):
            mixed = (1 - off)[:, None] * cycle + off[:, None] * flat
            off = ((counts * off[:, None] * flat / mixed).sum(axis=1) + outside) / values.size
        mixed = (1 - off)[:, None] * cycle + off[:, None] * flat
        likelihood = (counts * np.log(mixed + 1e-300)).sum(axis=1) + outside * np.log(off * flat + 1e-300)
        index = int(np.argmax(likelihood))
        if likelihood[index] > best[0]:
            best = (
                float(likelihood[index]),
                [float(first), float(second[index] - first), float(1 - second[index])],
                float(off[index]),
            )
    return best[1], best[2]


def _template(model: Mapping) -> np.ndarray:
    """Expected DNA content along the cycle.

    G1 sits on the measured 2N peak and G2M on the measured 4N peak. Through S, the model takes DNA to be replicated
    at a constant rate, so the amount of DNA, not its logarithm, rises linearly between the two.
    """
    low, high = model["peaks"]
    edges = model["edges"]
    progress = np.clip((_GRID - edges[1]) / max(edges[2] - edges[1], 1e-6), 0, 1)
    return np.log2(2.0**low + (2.0**high - 2.0**low) * progress)


def _peak_shift(content: np.ndarray, peaks: Sequence[float], sd: float) -> tuple[float, str]:
    """How far a group's 2N and 4N peaks both sit from the reference's, or ``NaN`` and why not.

    ``NaN`` when the group lacks either peak, as an arrested well does, or when the gap between its peaks differs from
    the reference's, since only a shift both peaks share is a property of the measurement rather than of the cells.
    The 4N peak is looked for one reference gap above the group's own 2N peak, so a shifted well is still found.
    """
    from scipy.ndimage import gaussian_filter1d

    values = content[np.isfinite(content) & (content > -1.5) & (content < 2.5)]
    if values.size < _MIN_REFERENCE:
        return float("nan"), "too few cells"
    grid = _DENSITY_GRID
    step = grid[1] - grid[0]
    counts = np.histogram(values, bins=grid.size, range=(grid[0] - step / 2, grid[-1] + step / 2))[0]
    density = gaussian_filter1d(counts.astype(float), 0.5 * sd / step, mode="constant")
    reach = max(int(sd / step), 1)
    shifts: list[float] = []
    for index, (name, half) in enumerate((("2N", 0.75), ("4N", 0.45))):
        centre = peaks[index] + (shifts[0] if shifts else 0.0)
        window = np.flatnonzero(np.abs(grid - centre) <= half)
        top = window[np.argmax(density[window])]
        if not (
            window[0] < top < window[-1]
            and density[top] >= density[top - reach]
            and density[top] >= density[min(top + reach, grid.size - 1)]
            and density[top] >= 0.1 * density.max()
        ):
            return float("nan"), f"no {name} peak"
        shifts.append(float(grid[top] - peaks[index]))
    if abs(shifts[0] - shifts[1]) > sd:
        return float("nan"), "peaks moved apart"
    return float(np.mean(shifts)), ""


def _matrix(adata: AnnData, features: Sequence[str], layer: str | None) -> np.ndarray:
    """The named features as float64, after checking every one is there."""
    missing = [name for name in features if name not in adata.var_names]
    if missing:
        raise KeyError(f"the model reads features this object lacks: {missing[:5]}")
    return get_matrix(adata[:, list(features)], layer).astype(np.float64)


def _shape_terms(block: np.ndarray, content: np.ndarray) -> np.ndarray:
    """Per cell: log2 nuclear area, solidity, and log2 DNA per area, a stain-independent mean DNA intensity."""
    with np.errstate(divide="ignore", invalid="ignore"):
        log_area = np.log2(np.where(block[:, 0] > 0, block[:, 0], np.nan))
    return np.column_stack([log_area, block[:, 1], content - log_area])


def _posterior(content: np.ndarray, model: Mapping) -> tuple[np.ndarray, np.ndarray]:
    """Per finite DNA content, the probability of each of ``CYCLE_PHASES`` and the quantiles of the position."""
    expected, edges = _template(model), model["edges"]
    phase_of = np.digitize(_GRID, edges[1:3])
    low, high = _span(model["peaks"])
    off, flat, sd = float(model["off_cycle"]), 1 / (high - low), float(model["sd"])
    middle = float(np.mean(model["peaks"]))
    probabilities = np.empty((content.size, len(CYCLE_PHASES)))
    quantiles = np.empty((content.size, len(QUANTILES)))
    for start in range(0, content.size, 8192):
        chunk = slice(start, start + 8192)
        log_post = _LOG_PRIOR[None] - 0.5 * ((content[chunk, None] - expected[None]) / sd) ** 2
        peak = log_post.max(axis=1, keepdims=True)
        post = np.exp(log_post - peak)
        cycle = post.sum(axis=1) * np.exp(peak[:, 0]) / (_GRID.size * sd * np.sqrt(2 * np.pi))
        p_off = off * flat / ((1 - off) * cycle + off * flat)
        post /= post.sum(axis=1, keepdims=True)
        for index in range(3):
            probabilities[chunk, index] = (1 - p_off) * post[:, phase_of == index].sum(axis=1)
        # The flat background below the middle of the cycle holds sub-2N cells, above it cells beyond 4N.
        below = content[chunk] < middle
        probabilities[chunk, 3], probabilities[chunk, 4] = np.where(below, p_off, 0), np.where(below, 0, p_off)
        cdf = np.cumsum(post, axis=1)
        quantiles[chunk] = np.stack(
            [_GRID[np.minimum((cdf < q).sum(axis=1), _GRID.size - 1)] for q in QUANTILES], axis=1
        )
    return probabilities, quantiles


def _model(adata: AnnData, model: str | Mapping) -> Mapping:
    fitted = adata.uns.get("mantispy", {}).get(model) if isinstance(model, str) else model
    if fitted is None:
        raise KeyError(f"uns['mantispy'] has no {model!r}; run mt.tl.fit_cell_cycle first")
    if fitted["key"] not in adata.obs:
        raise KeyError(f"obs has no column {fitted['key']!r}; run mt.tl.dna_content first")
    return fitted


@inplace_or_copy(expects="object")
def fit_cell_cycle(
    adata: AnnData,
    key: str = "Metadata_DNAContent",
    reference: str | None = "negcon",
    qc_features: tuple[str, str] | None = None,
    layer: str | None = None,
    key_added: str = "cell_cycle",
    copy: bool = False,
) -> AnnData | None:
    """Fit a cell-cycle clock on the reference cells' DNA content.

    The reference cells' DNA content is modelled as a 2N peak (G1), a 4N peak (G2M) and S cells between them, each
    at the position the measurement puts it, plus a flat density for cells off the cycle. The fractions of the three
    phases are turned into a position in the cycle with the age distribution of an exponentially growing population
    (ergodic rate analysis, :cite:p:`Kafri_2013`). Position 0 is the start of G1 and 1 is division. Inside S, DNA content
    places a cell precisely. Inside the 2N and 4N plateaus it does not, and :func:`~mantispy.tl.project_cell_cycle`
    reports the whole plateau as the cell's uncertainty instead of inventing an order.

    The fit first checks that the reference cycles: its DNA content must show a 2N peak, a separate 4N peak and a
    continuous S population between them. A population that fails, such as an arrested one or a stain that does not
    resolve the 4N peak, gets a model marked as failed, and projecting it gives ``NaN``.

    Args:
        adata: Object-resolution object carrying the output of :func:`~mantispy.tl.dna_content`.
        key: ``obs`` column holding the DNA content, in doublings of the G1 peak.
        reference: The cells to fit on: ``"negcon"`` for the negative controls, the name of a boolean ``obs``
            column, or ``None`` for every cell.
        qc_features: The nuclear area and solidity features, as a pair. With them, the model learns from the
            reference how small, how irregular and how dim, per amount of DNA, a nucleus may be, and
            :func:`~mantispy.tl.project_cell_cycle` flags nuclei beyond that. ``None`` checks no shape.
        layer: Layer to read ``qc_features`` from, or ``None`` for ``X``. The model records it, and projection reads
            the same layer.
        key_added: Key the model is written under in ``uns["mantispy"]``.
        copy: Return a modified copy instead of mutating in place.

    Returns:
        ``None``, or the modified copy.
        Writes ``uns["mantispy"][key_added]``: the DNA ``key``, whether the reference ``passed`` the check and the
        ``check`` measures, the measured 2N and 4N ``peaks`` and the 2N peak width ``sd``. A model that passed also
        holds the reference ``fractions`` in G1, S and G2M, the cycle positions where G1, S and G2M begin and end
        as ``edges``, and the ``off_cycle`` weight. With ``qc_features`` it holds a ``qc`` entry with the features
        and the ``cutoffs`` on log2 area, solidity and log2 DNA per area, and the ``layer`` they were read from.

    Raises:
        KeyError: ``obs`` has no column ``key``; run :func:`~mantispy.tl.dna_content` first. Or a ``qc_features``
            feature is missing.
        ValueError: The reference holds too few cells with a DNA content to fit on.

    Notes:
        The check asks for a 4N peak between 0.6 and 1.4 doublings, at least a twentieth the height of the 2N peak.
        It asks for a valley between the peaks no higher than nine tenths of the 4N peak, a density across S of at
        least a tenth of the 4N peak, and at least 3% of the reference between 1.25 and 1.75 times the 2N amount of
        DNA.

        The model assumes that DNA content is flat through G1 and through G2M, that DNA is replicated at a steady rate
        through S, and that the reference grows in steady state, with every phase taking a fixed share of the cycle.
        It cannot tell G0 from G1, nor mitosis from G2, since both pairs share a DNA content.

    See Also:
        :func:`~mantispy.tl.phase_order` for an experimental order within a phase from the other features.
        :doc:`/tutorials/single_cells/cell_cycle` for a worked example, and
        :doc:`/tutorials/single_cells/cell_cycle_method` for how the model works, step by step.
    """
    if key not in adata.obs:
        raise KeyError(f"obs has no column {key!r}; run mt.tl.dna_content first")
    content = as_frame(adata.obs)[key].to_numpy(dtype=float)
    reference_rows = reference_mask(adata, reference) & np.isfinite(content)
    if reference_rows.sum() < _MIN_REFERENCE:
        raise ValueError(
            f"reference={reference!r} has {int(reference_rows.sum())} cells with a DNA content; "
            f"fit_cell_cycle needs at least {_MIN_REFERENCE}"
        )
    values = content[reference_rows]
    gate = _gate(values)
    peaks = gate.pop("peaks")
    passed = gate.pop("passed")
    near = values[np.abs(values - peaks[0]) < 0.3]
    sd = _robust(near)[1] if near.size else float(np.std(values))
    model: dict[str, Any] = {"key": key, "passed": passed, "check": gate, "peaks": peaks, "sd": sd}
    if passed:
        fractions, model["off_cycle"] = _fit_fractions(values, peaks, sd)
        model["fractions"], model["edges"] = fractions, _age(np.r_[0, np.cumsum(fractions)]).tolist()
    if qc_features is not None:
        terms = _shape_terms(_matrix(adata[reference_rows], qc_features, layer), values)
        cutoffs = []
        for column in terms.T:
            centre, spread = _robust(column[np.isfinite(column)])
            cutoffs.append(centre - _QC_SPREAD * spread)
        model["qc"] = {"features": [str(name) for name in qc_features], "cutoffs": cutoffs}
        if layer is not None:
            model["layer"] = layer
    if not passed:
        get_logger().warning(
            "fit_cell_cycle: the reference does not show separate 2N and 4N peaks with S cells between them %s; "
            "project_cell_cycle will give NaN",
            {name: round(value, 3) if isinstance(value, float) else value for name, value in gate.items()},
        )
    adata.uns.setdefault("mantispy", {})[key_added] = model
    return None


@inplace_or_copy(expects="object")
def project_cell_cycle(
    adata: AnnData,
    model: str | Mapping = "cell_cycle",
    by: Sequence[str] | str | None = _WELL,
    checks: Sequence[str] = QC_FLAGS[1:],
    key_added: str = "cell_cycle",
    copy: bool = False,
) -> AnnData | None:
    """Place every cell on the cycle of a model from :func:`~mantispy.tl.fit_cell_cycle`, with its uncertainty.

    Each cell gets a posterior over its position: the age distribution of the reference as the prior, times how well
    the cell's DNA content fits each position. From it come the probability of each phase, and of sitting below 2N
    or above 4N, off the cycle, and quantiles of the position. An S cell is placed tightly. A G1 or G2M cell spreads
    over its whole plateau, since DNA content does not order cells within it, so there its position is the prior.
    Perturbed cells are placed on the reference's cycle, so an arrest shows up as cells piling up at one position.

    Each ``by`` group, normally a well, is also checked against the reference. When its 2N and 4N peaks both sit off
    the reference's by more than the 2N peak width, the well reads its DNA stain differently, every position in it is
    biased by that shift, and a warning names it. The positions are not corrected, since a well with only one of the
    two peaks, such as an arrested one, cannot tell a shifted stain from a shifted population. Groups that cannot be
    checked are named in a warning with the reason.

    Args:
        adata: Object-resolution object carrying the DNA content column the model was fitted on.
        model: Key of the model in ``uns["mantispy"]``, or the model itself, such as one fitted on another object.
        by: The ``obs`` columns of a well, to check each well's peaks against the reference; ``None`` skips the check.
            The default uses whichever of ``Metadata_Plate`` and ``Metadata_Well`` the object has.
        checks: The shape checks to flag, out of ``"irregular"``, ``"small"`` and ``"dim"``, when the model holds
            ``qc_features``. Rounded mitotic cells can fail the solidity check as merged nuclei do, so whether to use
            ``"irregular"`` depends on the cell line and the question.
        key_added: Prefix of the outputs written.
        copy: Return a modified copy instead of mutating in place.

    Returns:
        ``None``, or the modified copy. Writes
        ``obsm[key_added + "_probabilities"]``, a frame with the probability of each of ``CYCLE_PHASES`` per cell;
        ``obs[key_added + "_phase"]``, the most probable of them, for convenience;
        ``obs[key_added + "_position"]``, the posterior median in [0, 1), 0 being the start of G1;
        ``obsm[key_added + "_quantiles"]``, a frame with the 5%, 25%, 50%, 75% and 95% posterior quantiles;
        ``obs[key_added + "_qc"]``, when the model holds ``qc_features``, the first of ``checks`` a nucleus fails:
        ``"irregular"`` for a solidity below the reference's, as merged nuclei and mitotic figures have, ``"small"``
        for an area below it, as fragments and debris have, ``"dim"`` for too little DNA per area, and ``"ok"``
        otherwise;
        ``uns["mantispy"][key_added]``, the model used, so plots read its phase edges from there; and
        ``uns["mantispy"][key_added + "_wells"]``, one row per ``by`` group with its ``n_cells`` and the ``shift``
        its 2N and 4N peaks share against the reference, in doublings, ``NaN`` when it could not be checked.
        A cell without a DNA content gets ``NaN`` positions and phases, as does every cell when the model failed its
        check. A cell off the cycle gets ``NaN`` positions. Columns of an earlier projection under ``key_added`` that
        this model does not write, and the orders of :func:`~mantispy.tl.phase_order`, are removed.

    Raises:
        KeyError: ``uns["mantispy"]`` has no entry ``model``, or ``obs`` lacks the DNA content column it was fitted on
            or a ``by`` column, or the object lacks a feature of the model's ``qc_features``.
        ValueError: A ``by`` column holds missing values, or ``checks`` names an unknown check.

    See Also:
        :doc:`/tutorials/single_cells/cell_cycle` for a worked example, and
        :doc:`/tutorials/single_cells/cell_cycle_method` for how the model works, step by step.
    """
    fitted = _model(adata, model)
    unknown = [name for name in checks if name not in QC_FLAGS[1:]]
    if unknown:
        raise ValueError(f"checks {unknown} are not among {list(QC_FLAGS[1:])}")
    content = as_frame(adata.obs)[fitted["key"]].to_numpy(dtype=float)
    qc_block = _matrix(adata, fitted["qc"]["features"], fitted.get("layer")) if "qc" in fitted else None
    if by == _WELL:
        by = [column for column in _WELL if column in adata.obs] or None
    codes, keys = group_codes(adata, by)
    passed = bool(fitted["passed"])
    known = np.isfinite(content)

    probabilities = np.full((adata.n_obs, len(CYCLE_PHASES)), np.nan)
    quantiles = np.full((adata.n_obs, len(QUANTILES)), np.nan)
    if passed and known.any():
        probabilities[known], quantiles[known] = _posterior(content[known], fitted)
    elif not passed:
        get_logger().warning("project_cell_cycle: the model failed its check; every position is NaN")

    placed = np.isfinite(probabilities).all(axis=1)
    phase = np.full(adata.n_obs, np.nan, dtype=object)
    phase[placed] = np.asarray(CYCLE_PHASES, dtype=object)[probabilities[placed].argmax(axis=1)]
    quantiles[~placed | np.isin(phase, CYCLE_PHASES[3:])] = np.nan

    table = pd.DataFrame(
        [key if isinstance(key, tuple) else (key,) for key in keys],
        columns=["group"] if by is None else ([by] if isinstance(by, str) else list(by)),
    )
    table["n_cells"] = np.bincount(codes, minlength=len(keys))
    table["shift"] = np.nan
    if passed and by is not None:
        peaks, sd = [float(peak) for peak in fitted["peaks"]], float(fitted["sd"])
        order, offsets = group_offsets(codes, len(keys))
        shifts = [_peak_shift(content[order[offsets[i] : offsets[i + 1]]], peaks, sd) for i in range(len(keys))]
        table["shift"] = [shift for shift, _ in shifts]
        shifted = table[table["shift"].abs() > sd]
        if len(shifted):
            get_logger().warning(
                "project_cell_cycle: %d %s group(s) have both their 2N and 4N peaks off the reference's, so their DNA "
                "stain reads differently and their positions carry the shift given, in doublings: %s",
                len(shifted),
                by,
                shifted.round(2).head(5).to_dict("records"),
            )
        unchecked: dict[str, list] = {}
        for name, (_, why) in zip(keys, shifts, strict=True):
            if why:
                unchecked.setdefault(why, []).append(name)
        if unchecked:
            get_logger().warning(
                "project_cell_cycle: %d %s group(s) could not be checked against the reference: %s",
                sum(map(len, unchecked.values())),
                by,
                {why: names[:5] for why, names in unchecked.items()},
            )

    stale = [f"{key_added}_{name}_order" for name in CYCLE_PHASES[:3]]
    if qc_block is None:
        stale.append(f"{key_added}_qc")
    else:
        terms = _shape_terms(qc_block, content)
        columns = [_QC_TERM[name] for name in QC_FLAGS[1:]]
        with np.errstate(invalid="ignore"):
            failed = (terms[:, columns] < np.asarray(fitted["qc"]["cutoffs"])[columns]) & np.isin(QC_FLAGS[1:], checks)
        flags = np.asarray(QC_FLAGS, dtype=object)[np.where(failed.any(axis=1), failed.argmax(axis=1) + 1, 0)]
        flags[~np.isfinite(terms).all(axis=1)] = np.nan
        adata.obs[f"{key_added}_qc"] = pd.Categorical(flags, categories=list(QC_FLAGS))
    adata.obs = adata.obs.drop(columns=stale, errors="ignore")

    names = adata.obs_names
    adata.obsm[f"{key_added}_probabilities"] = pd.DataFrame(probabilities, index=names, columns=list(CYCLE_PHASES))
    adata.obs[f"{key_added}_phase"] = pd.Categorical(phase, categories=list(CYCLE_PHASES))
    adata.obs[f"{key_added}_position"] = quantiles[:, 2]
    adata.obsm[f"{key_added}_quantiles"] = pd.DataFrame(
        quantiles, index=names, columns=[f"{round(100 * q)}%" for q in QUANTILES]
    )
    adata.uns.setdefault("mantispy", {})[key_added] = dict(fitted)
    adata.uns["mantispy"][f"{key_added}_wells"] = table
    return None


def _rank01(values: np.ndarray) -> np.ndarray:
    return (pd.Series(values).rank().to_numpy() - 0.5) / values.size


def _order_space(block: np.ndarray, entry: Mapping) -> tuple[np.ndarray, np.ndarray]:
    """The S-axis score of each cell and its place in the weighted space the neighbours are found in."""
    scaled = np.clip(np.nan_to_num((block - entry["median"]) / entry["scale"], nan=0.0, posinf=8, neginf=-8), -8, 8)
    components = np.asarray(entry["components"])
    centred = scaled - entry["mean"]
    return centred @ components.T @ np.asarray(entry["coef"]), (centred * entry["weights"]) @ components.T


def _fit_order(block: np.ndarray, content: np.ndarray, in_s: np.ndarray, in_phase: np.ndarray) -> dict:
    """The pieces of a phase order, fitted on reference cells on the cycle; ``in_s`` and ``in_phase`` mark subsets.

    Holds the scaling, principal axes, S-axis coefficients and feature weights, and the reference cells of the phase
    embedded and ordered along their nearest-neighbour graph.
    """
    from scipy.sparse import diags, eye
    from scipy.sparse.linalg import cg
    from sklearn.linear_model import Ridge
    from sklearn.neighbors import kneighbors_graph

    from mantispy.tl._dose import _spearman_against

    median = np.nanmedian(block, axis=0)
    spread = np.nanpercentile(block, 75, axis=0) - np.nanpercentile(block, 25, axis=0)
    scale = np.where(spread > 1e-8 * np.maximum(np.abs(median), 1), spread, np.nanstd(block, axis=0))
    keep = np.isfinite(scale) & (scale > 1e-12) & np.isfinite(median)
    entry: dict[str, Any] = {"median": median[keep], "scale": scale[keep], "kept": keep}
    scaled = np.clip(np.nan_to_num((block[:, keep] - entry["median"]) / entry["scale"], nan=0.0), -8, 8)
    entry["mean"] = scaled.mean(axis=0)
    centred = scaled - entry["mean"]
    # The principal axes from the small feature-by-feature matrix, not a decomposition of every cell.
    entry["components"] = np.linalg.eigh(centred.T @ centred)[1][:, ::-1][:, :_ORDER_COMPONENTS].T.copy()
    projected = centred @ entry["components"].T
    entry["coef"] = Ridge(alpha=float(in_s.sum())).fit(projected[in_s], content[in_s]).coef_
    entry["weights"] = np.abs(np.nan_to_num(_spearman_against(content[in_s], scaled[in_s])[0]))

    score, embedding = _order_space(block[in_phase][:, keep], entry)
    k = min(_ORDER_NEIGHBOURS, int(in_phase.sum()) - 1)
    graph = kneighbors_graph(embedding, n_neighbors=k)
    # Each cell's edges sum to log2(k), the weight a UMAP neighbourhood carries, so the smoothing keeps its strength.
    affinity = graph.maximum(graph.T) * (np.log2(k) / k)
    laplacian = diags(np.asarray(affinity.sum(axis=1)).ravel()) - affinity
    smoothed = cg(eye(embedding.shape[0]) + _ORDER_SMOOTHING * laplacian, _rank01(score), maxiter=300)[0]
    entry["embedding"], entry["order"] = embedding, _rank01(smoothed)
    return entry


@inplace_or_copy(expects="object")
def phase_order(
    adata: AnnData,
    phase: str = "G2M",
    model: str | Mapping = "cell_cycle",
    reference: str | None = "negcon",
    features: Sequence[str] | None = None,
    layer: str | None = None,
    anchor: str | Mapping | None = None,
    key_added: str | None = None,
    copy: bool = False,
) -> AnnData | None:
    """Experimental: order the cells within one phase from features other than DNA content.

    DNA content is flat through G1 and G2M, so :func:`~mantispy.tl.project_cell_cycle` cannot order cells within them.
    This orders them from the other features, without labels. On the reference cells on the cycle, the features are
    scaled robustly and reduced to their principal components, and a ridge regression of DNA content on those
    components, fitted on the S cells, gives an S axis. Each feature is weighted by its rank correlation with DNA
    content in S. The reference cells of ``phase`` start at their rank on the S axis, and that rank is smoothed along
    their nearest-neighbour graph in the weighted space. Each cell of ``phase`` then gets the mean order of its nearest
    reference cells, so a cell's order depends only on the reference and on the cell, and orders are comparable across
    objects placed against the same ``anchor``.

    The order assumes that whatever changes through S keeps changing in the same direction through ``phase``. It can
    follow cell size, DNA damage or any other feature that trends through S rather than progress, and nothing
    label-free tells which. Validate it against a marker of the phase or a paired modality before reading it.

    Args:
        adata: Object-resolution object carrying the DNA content column the model was fitted on.
        phase: The phase to order: ``"G1"``, ``"S"`` or ``"G2M"``. Cells are assigned to it by the model.
        model: Key of a :func:`~mantispy.tl.fit_cell_cycle` model in ``uns["mantispy"]``, or the model itself.
        reference: The cells the order is fitted on: ``"negcon"`` for the negative controls, the name of a boolean
            ``obs`` column, or ``None`` for every cell. Ignored with ``anchor``.
        features: The features to order on. ``None`` uses every feature except those measured on a DNA stain and the
            nuclear shape features of the DNA mask, so the order cannot read DNA content back.
        layer: Layer to read the features from, or ``None`` for ``X``. Ignored with ``anchor``, which records its own.
        anchor: An order written by an earlier call, its key in ``uns["mantispy"]`` or the entry itself. This
            object's cells are then placed against that call's reference cells instead of fitting new ones.
        key_added: Key of the order in ``obs`` and of its entry in ``uns["mantispy"]``. Defaults to
            ``model + "_" + phase + "_order"``, which :func:`~mantispy.pl.cell_cycle_clock` reads.
        copy: Return a modified copy instead of mutating in place.

    Returns:
        ``None``, or the modified copy.
        Writes ``obs[key_added]``, the order of every cell of ``phase`` from 0 to 1 and ``NaN`` for every other cell,
        and ``uns["mantispy"][key_added]``, holding the ``phase``, the ``features`` and ``layer``, their scaling, the
        principal axes, the S-axis coefficients, the feature weights, and the reference cells' ``embedding`` and
        ``order``.

    Raises:
        KeyError: The model, its DNA content column or a feature is missing.
        ValueError: ``phase`` is not G1, S or G2M, the model failed its check, or the reference holds fewer than
            50 S cells or 50 cells of ``phase``.

    Warns:
        UserWarning: Every call, since the order needs validating against a marker or a paired modality.

    See Also:
        :doc:`/tutorials/single_cells/cell_cycle_method` for each step, and where it worked and where it did not.
    """
    from sklearn.neighbors import NearestNeighbors

    if phase not in CYCLE_PHASES[:3]:
        raise ValueError(f"phase must be one of {list(CYCLE_PHASES[:3])}, got {phase!r}")
    fitted = _model(adata, model)
    if not fitted["passed"]:
        raise ValueError("the cell-cycle model failed its check, so it assigns no cell to a phase")
    if key_added is None:
        key_added = f"{model if isinstance(model, str) else 'cell_cycle'}_{phase}_order"
    content = as_frame(adata.obs)[fitted["key"]].to_numpy(dtype=float)
    called = np.full(adata.n_obs, -1)
    known = np.isfinite(content)
    called[known] = _posterior(content[known], fitted)[0].argmax(axis=1)
    index = CYCLE_PHASES.index(phase)

    if anchor is None:
        if features is None:
            stain = _on_dna_stain(adata)
            shape = adata.var_names.str.startswith("Nuclei_AreaShape")
            features = [str(name) for name in adata.var_names[~stain & ~shape]]
        rows = np.flatnonzero(reference_mask(adata, reference) & (called >= 0) & (called < 3))
        in_s, in_phase = called[rows] == 1, called[rows] == index
        if min(in_s.sum(), in_phase.sum()) < _MIN_REFERENCE:
            raise ValueError(
                f"the reference has {int(in_s.sum())} S cells and {int(in_phase.sum())} {phase} cells; "
                f"phase_order needs at least {_MIN_REFERENCE} of each"
            )
        block = _matrix(adata[rows], features, layer)
        entry = _fit_order(block, content[rows], in_s, in_phase)
        entry = {
            "phase": phase,
            "features": [name for name, kept in zip(features, entry.pop("kept"), strict=True) if kept],
            **entry,
        }
        if layer is not None:
            entry["layer"] = layer
    else:
        entry = adata.uns.get("mantispy", {}).get(anchor) if isinstance(anchor, str) else anchor
        if entry is None:
            raise KeyError(f"uns['mantispy'] has no {anchor!r}; run mt.tl.phase_order first")
        if entry["phase"] != phase:
            raise ValueError(f"the anchor orders {entry['phase']}, not {phase}")

    order = np.full(adata.n_obs, np.nan)
    rows = np.flatnonzero(called == index)
    if rows.size:
        embedding = _order_space(_matrix(adata[rows], entry["features"], entry.get("layer")), entry)[1]
        reference_embedding = np.asarray(entry["embedding"])
        k = min(_ORDER_NEIGHBOURS, reference_embedding.shape[0])
        neighbours = NearestNeighbors(n_neighbors=k).fit(reference_embedding).kneighbors(embedding)[1]
        order[rows] = np.asarray(entry["order"])[neighbours].mean(axis=1)
    warnings.warn(
        "phase_order is experimental: validate the order against a marker of the phase or a paired modality "
        "before reading it biologically",
        UserWarning,
        stacklevel=3,
    )
    adata.obs[key_added] = order
    adata.uns.setdefault("mantispy", {})[key_added] = dict(entry)
    return None
