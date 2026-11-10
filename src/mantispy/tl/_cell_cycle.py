"""Cell-cycle position from DNA content: an ergodic-rate model fitted on reference cells."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from anndata import AnnData

from mantispy._core._reduce import get_matrix, group_codes, group_offsets
from mantispy._core.frames import as_frame
from mantispy._core.logging import get_logger
from mantispy._core.masks import reference_mask
from mantispy._core.mutation import inplace_or_copy

CYCLE_PHASES = ("G1", "S", "G2M", "off-cycle")
QC_FLAGS = ("ok", "irregular", "small", "dim")
QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)
_GATES = np.log2([1.25, 1.75])  # the dna_state defaults, used only by the S share the gate asks for
_GRID = (np.arange(400) + 0.5) / 400
_LOG_PRIOR = np.log(2 * np.log(2)) - _GRID * np.log(2)
_MIN_REFERENCE = 50
_QC_SPREAD = 4  # robust standard deviations below the reference median that flag a nucleus
_G2M_COMPONENTS, _G2M_NEIGHBOURS, _G2M_SMOOTHING = 30, 30, 10.0


def _age(fraction: Any) -> Any:
    """Cycle position of the cell at a given cumulative fraction of an exponentially growing population.

    Ages in such a population have density 2 ln2 2^-t on [0, 1), so the cumulative fraction F maps to t = -log2(1 - F/2).
    """
    return -np.log2(1 - np.clip(fraction, 0, 1) / 2)


def _robust(values: np.ndarray) -> tuple[float, float]:
    """Median and the median absolute deviation scaled to a standard deviation."""
    centre = float(np.median(values))
    return centre, 1.4826 * float(np.median(np.abs(values - centre))) or 1.0


def _gate(content: np.ndarray) -> dict:
    """Locate the 2N and 4N peaks and check for a continuous S population between them."""
    from scipy.stats import gaussian_kde

    grid = np.linspace(-1, 2.5, 701)
    density = gaussian_kde(np.clip(content, -1, 2.5))(grid)
    low = np.flatnonzero(np.abs(grid) < 0.25)
    two = low[np.argmax(density[low])]
    window = np.flatnonzero((grid >= 0.6) & (grid <= 1.4))
    top = window[np.argmax(density[window])]
    is_peak = density[top] >= density[max(top - 20, 0)] and density[top] >= density[min(top + 20, grid.size - 1)]
    valley = density[(grid >= 0.2) & (grid <= grid[top] - 0.1)].min() if grid[top] > 0.35 else density[top]
    floor = density[(grid >= 0.4) & (grid <= 0.7)].min()
    report: dict[str, Any] = {
        "peaks": [float(grid[two]), float(grid[top])],
        "peak_ratio": float(density[top] / density[two]),
        "valley_ratio": float(valley / density[top]),
        "s_floor": float(floor / density[top]),
        "s_share": float(np.mean((content >= _GATES[0]) & (content < _GATES[1]))),
    }
    report["passed"] = bool(
        is_peak
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


def _template(model: Mapping) -> tuple[np.ndarray, np.ndarray]:
    """Expected DNA content along the cycle, and the cycle positions where G1, S and G2M begin and end.

    G1 sits on the measured 2N peak and G2M on the measured 4N peak. Through S, DNA is replicated at a constant rate,
    so the amount of DNA, not its logarithm, rises linearly between the two.
    """
    low, high = model["peaks"]
    edges = _age(np.r_[0, np.cumsum(model["fractions"])])
    progress = np.clip((_GRID - edges[1]) / max(edges[2] - edges[1], 1e-6), 0, 1)
    return np.log2(2.0**low + (2.0**high - 2.0**low) * progress), edges


def _peak_shift(content: np.ndarray, peaks: Sequence[float], sd: float) -> float:
    """How far a group's 2N and 4N peaks both sit from the reference's.

    ``NaN`` when the group lacks either peak, as an arrested well does, or when the two peaks move apart, since only a
    shift they share is a property of the measurement rather than of the cells.
    """
    from scipy.stats import gaussian_kde

    values = content[np.isfinite(content) & (content > -1.5) & (content < 2.5)]
    if values.size < _MIN_REFERENCE or values.std() == 0:
        return float("nan")
    grid = np.linspace(-1, 2.5, 701)
    density = gaussian_kde(values, bw_method=0.5 * sd / values.std())(grid)
    reach = max(int(sd / (grid[1] - grid[0])), 1)
    shifts = []
    for centre in peaks:
        window = np.flatnonzero(np.abs(grid - centre) <= 0.45)
        top = window[np.argmax(density[window])]
        is_peak = (
            window[0] < top < window[-1]
            and density[top] >= density[top - reach]
            and density[top] >= density[min(top + reach, grid.size - 1)]
            and density[top] >= 0.1 * density.max()
        )
        if not is_peak:
            return float("nan")
        shifts.append(grid[top] - centre)
    return float(np.mean(shifts)) if abs(shifts[0] - shifts[1]) <= sd else float("nan")


def _shape_terms(adata: AnnData, content: np.ndarray, features: Sequence[str], layer: str | None) -> np.ndarray:
    """Per cell: log2 nuclear area, solidity, and DNA content per area, a stain-independent mean DNA intensity."""
    matrix = get_matrix(adata[:, [str(name) for name in features]], layer).astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_area = np.log2(np.where(matrix[:, 0] > 0, matrix[:, 0], np.nan))
    return np.column_stack([log_area, matrix[:, 1], content - log_area])


def _qc(terms: np.ndarray, cutoffs: Sequence[float]) -> np.ndarray:
    """The first check each cell fails, as an index into ``QC_FLAGS``: irregular, then small, then dim."""
    with np.errstate(invalid="ignore"):
        failed = terms[:, [1, 0, 2]] < np.asarray(cutoffs)[[1, 0, 2]]
    return np.where(failed.any(axis=1), failed.argmax(axis=1) + 1, 0)


def _draw_positions(quantiles: np.ndarray, seed: int = 0) -> np.ndarray:
    """One cycle position per cell drawn from its posterior, given as the quantiles of ``QUANTILES``.

    The quantile function is interpolated linearly between the reported quantiles and extended along its outer
    segments to the ends of the cycle, so the outer tenth of draws spreads over the tails instead of collapsing onto
    the 5% and 95% quantiles.
    """
    quantiles = np.asarray(quantiles, dtype=float)
    levels = np.r_[0, QUANTILES, 1]
    below = quantiles[:, 0] - (quantiles[:, 1] - quantiles[:, 0]) * QUANTILES[0] / (QUANTILES[1] - QUANTILES[0])
    above = quantiles[:, -1] + (quantiles[:, -1] - quantiles[:, -2]) * (1 - QUANTILES[-1]) / (
        QUANTILES[-1] - QUANTILES[-2]
    )
    knots = np.column_stack([np.clip(below, 0, 1), quantiles, np.clip(above, 0, 1)])
    draw = np.random.default_rng(seed).random(quantiles.shape[0])
    upper = np.clip(np.searchsorted(levels, draw), 1, levels.size - 1)
    fraction = (draw - levels[upper - 1]) / (levels[upper] - levels[upper - 1])
    rows = np.arange(quantiles.shape[0])
    return knots[rows, upper - 1] + fraction * (knots[rows, upper] - knots[rows, upper - 1])


def _posterior(content: np.ndarray, model: Mapping) -> tuple[np.ndarray, np.ndarray]:
    """Per finite DNA content, the probability of each of ``CYCLE_PHASES`` and the quantiles of the position."""
    expected, edges = _template(model)
    phase_of = np.digitize(_GRID, edges[1:3])
    low, high = _span(model["peaks"])
    off, flat, sd = float(model["off_cycle"]), 1 / (high - low), float(model["sd"])
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
        probabilities[chunk, 3] = p_off
        cdf = np.cumsum(post, axis=1)
        quantiles[chunk] = np.stack(
            [_GRID[np.minimum((cdf < q).sum(axis=1), _GRID.size - 1)] for q in QUANTILES], axis=1
        )
    return probabilities, quantiles


def _rank01(values: np.ndarray) -> np.ndarray:
    return (pd.Series(values).rank().to_numpy() - 0.5) / values.size


def _dna_free_features(adata: AnnData) -> list[str]:
    """Every feature not measured on a DNA stain, so the G2M order cannot read DNA content back."""
    var = as_frame(adata.var)
    stains = ("dna", "dapi", "hoechst")
    return [
        str(name)
        for name in adata.var_names
        if not any(stain in str(name).lower() for stain in stains)
        and not any(stain in str(var.loc[name].get("channel")).lower() for stain in stains)
    ]


def _g2m_space(block: np.ndarray, g2m: Mapping) -> tuple[np.ndarray, np.ndarray]:
    """The robustly scaled features, clipped, and their projection on the reference's principal axes."""
    scaled = np.clip(np.nan_to_num((block - g2m["median"]) / g2m["scale"], nan=0.0, posinf=8, neginf=-8), -8, 8)
    return scaled, (scaled - g2m["mean"]) @ np.asarray(g2m["components"]).T


def _fit_g2m(block: np.ndarray, content: np.ndarray, in_s: np.ndarray, names: list[str]) -> dict:
    """The pieces of the G2M order, fitted on reference cells on the cycle; ``in_s`` marks those in S."""
    from scipy.stats import rankdata
    from sklearn.linear_model import Ridge

    median = np.nanmedian(block, axis=0)
    spread = np.nanpercentile(block, 75, axis=0) - np.nanpercentile(block, 25, axis=0)
    scale = np.where(spread > 1e-8 * np.maximum(np.abs(median), 1), spread, np.nanstd(block, axis=0))
    keep = np.isfinite(scale) & (scale > 1e-12) & np.isfinite(median)
    g2m: dict[str, Any] = {"features": [name for name, kept in zip(names, keep, strict=True) if kept]}
    g2m["median"], g2m["scale"] = median[keep], scale[keep]
    scaled = np.clip(np.nan_to_num((block[:, keep] - g2m["median"]) / g2m["scale"], nan=0.0), -8, 8)
    g2m["mean"] = scaled.mean(axis=0)
    components = np.linalg.svd(scaled - g2m["mean"], full_matrices=False)[2][:_G2M_COMPONENTS]
    g2m["components"] = components
    projected = (scaled - g2m["mean"]) @ components.T
    ridge = Ridge(alpha=float(in_s.sum())).fit(projected[in_s], content[in_s])
    g2m["coef"], g2m["intercept"] = ridge.coef_, float(ridge.intercept_)
    ranked = rankdata(scaled[in_s], axis=0)
    target = rankdata(content[in_s])
    centred = ranked - ranked.mean(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        correlation = (centred * (target - target.mean())[:, None]).sum(axis=0) / (
            np.sqrt((centred**2).sum(axis=0)) * np.sqrt(((target - target.mean()) ** 2).sum())
        )
    g2m["weights"] = np.abs(np.nan_to_num(correlation))
    return g2m


def _g2m_order(block: np.ndarray, g2m: Mapping) -> np.ndarray:
    """Order G2M cells along the S-phase axis, then smooth that order on their nearest-neighbour graph."""
    import anndata as ad
    import scanpy as sc
    from scipy.sparse import csr_matrix, diags, eye
    from scipy.sparse.linalg import cg

    scaled, projected = _g2m_space(block, g2m)
    start = _rank01(projected @ np.asarray(g2m["coef"]) + float(g2m["intercept"]))
    if block.shape[0] < 3:
        return start
    weighted = ((scaled - g2m["mean"]) * g2m["weights"]) @ np.asarray(g2m["components"]).T
    graph = ad.AnnData(np.ascontiguousarray(weighted, dtype=np.float32))
    # With fewer than 50 dimensions scanpy builds the graph on X itself, so no representation needs naming.
    sc.pp.neighbors(graph, n_neighbors=min(_G2M_NEIGHBOURS, block.shape[0] - 1))
    affinity = csr_matrix(graph.obsp["connectivities"])
    laplacian = diags(np.asarray(affinity.sum(axis=1)).ravel()) - affinity
    smoothed = cg(eye(block.shape[0]) + _G2M_SMOOTHING * laplacian, start, maxiter=300)[0]
    return _rank01(smoothed)


@inplace_or_copy(expects="object")
def fit_cell_cycle(
    adata: AnnData,
    key: str = "Metadata_DNAContent",
    reference: str | None = "negcon",
    qc_features: tuple[str, str] | None = None,
    refine_g2m: bool = False,
    layer: str | None = None,
    key_added: str = "cell_cycle",
    copy: bool = False,
) -> AnnData | None:
    """Fit a cell-cycle clock on the reference cells' DNA content.

    The reference cells' DNA content is modelled as a 2N peak (G1), a 4N peak (G2M) and S cells between them, each
    at the position the measurement puts it, plus a flat density for cells off the cycle. The fractions of the three
    phases are turned into a position in the cycle with the age distribution of an exponentially growing population
    (ergodic rate analysis, :cite:p:`Kafri_2013`). Position 0 is the start of G1 and 1 is division. Inside S, DNA content
    places a cell precisely; inside the 2N and 4N plateaus it does not, and :func:`~mantispy.tl.project_cell_cycle`
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
        refine_g2m: Experimental. Also fit a label-free order of the cells within G2M, which DNA content cannot
            order. Every feature not measured on a DNA stain is used: a ridge regression of DNA content on them,
            fitted on the reference's S cells, gives an axis that is extended into G2M, and
            :func:`~mantispy.tl.project_cell_cycle` smooths the order along it on the G2M cells' nearest-neighbour
            graph. In testing, the order followed a marker of G2 and mitosis on some datasets and not on others, and
            no label-free check told the two apart, so validate it against an orthogonal marker before reading it
            biologically.
        layer: Layer to read ``qc_features`` and the ``refine_g2m`` features from, or ``None`` for ``X``.
        key_added: Key the model is written under in ``uns["mantispy"]``.
        copy: Return a modified copy instead of mutating in place.

    Returns:
        ``None``, or the modified copy.
        Writes ``uns["mantispy"][key_added]``: the DNA ``key``, whether the reference ``passed`` the check, the
        measured 2N and 4N ``peaks``, the 2N peak width ``sd``, the reference ``fractions`` in G1, S and G2M, the
        ``off_cycle`` weight of cells off the cycle, with ``qc_features`` a ``qc`` entry holding the features
        and the ``cutoffs`` on log2 area, solidity and DNA per log2 area, and with ``refine_g2m`` a ``g2m`` entry
        holding the features, their scaling, the principal axes, the feature weights and the ridge coefficients.

    Raises:
        KeyError: ``obs`` has no column ``key``; run :func:`~mantispy.tl.dna_content` first.
        ValueError: The reference holds too few cells with a DNA content to fit on.

    Notes:
        The check asks for a 4N peak between 0.6 and 1.4 doublings, at least a twentieth the height of the 2N peak;
        a valley between the peaks no higher than nine tenths of the 4N peak; a density across S of at least a tenth
        of the 4N peak; and at least 3% of the reference between 1.25 and 1.75 times the 2N amount of DNA.

        The model assumes the reference grows in steady state, with every phase taking a fixed share of the cycle.
        It cannot tell G0 from G1, nor mitosis from G2, since both pairs share a DNA content.
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
    peaks = gate["peaks"]
    near = values[np.abs(values - peaks[0]) < 0.3]
    sd = _robust(near)[1] if near.size else float(np.std(values))
    fractions, off = _fit_fractions(values, peaks, sd)
    model = {"key": key, "passed": gate["passed"], "peaks": peaks, "sd": sd, "fractions": fractions, "off_cycle": off}
    if qc_features is not None:
        terms = _shape_terms(adata[reference_rows], values, qc_features, layer)
        cutoffs = []
        for column in terms.T:
            centre, spread = _robust(column[np.isfinite(column)])
            cutoffs.append(centre - _QC_SPREAD * spread)
        model["qc"] = {"features": [str(name) for name in qc_features], "cutoffs": cutoffs}
    if refine_g2m and gate["passed"]:
        called = _posterior(values, model)[0].argmax(axis=1)
        on, in_s = called != 3, called == 1
        if in_s.sum() < _MIN_REFERENCE:
            get_logger().warning(
                "fit_cell_cycle: the reference has %d S cells, too few to fit the G2M order; refine_g2m is skipped",
                int(in_s.sum()),
            )
        else:
            names = _dna_free_features(adata)
            block = get_matrix(adata[np.flatnonzero(reference_rows)[on]][:, names], layer).astype(np.float64)
            model["g2m"] = _fit_g2m(block, values[on], in_s[on], names)
    if not gate["passed"]:
        get_logger().warning(
            "fit_cell_cycle: the reference does not show separate 2N and 4N peaks with S cells between them %s; "
            "project_cell_cycle will give NaN",
            {name: round(value, 3) for name, value in gate.items() if name not in ("passed", "peaks")},
        )
    adata.uns.setdefault("mantispy", {})[key_added] = model
    return None


@inplace_or_copy(expects="object")
def project_cell_cycle(
    adata: AnnData,
    model: str | Mapping = "cell_cycle",
    by: Sequence[str] | str | None = ("Metadata_Plate", "Metadata_Well"),
    layer: str | None = None,
    key_added: str = "cell_cycle",
    copy: bool = False,
) -> AnnData | None:
    """Place every cell on the cycle of a model from :func:`~mantispy.tl.fit_cell_cycle`, with its uncertainty.

    Each cell gets a posterior over its position: the age distribution of the reference as the prior, times how well
    the cell's DNA content fits each position. From it come the probability of each phase, and of being off the
    cycle, and quantiles of the position. An S cell is placed tightly; a G1 or G2M cell spreads over its whole
    plateau, since DNA content does not order cells within it, so there its position is the prior. Perturbed cells are
    placed on the reference's cycle, so an arrest shows up as cells piling up at one position.

    Each ``by`` group, normally a well, is also checked against the reference: when its 2N and 4N peaks both sit off
    the reference's by more than the 2N peak width, the well reads its DNA stain differently, every position in it is
    biased by that shift, and a warning names it. The positions are not corrected, since a well with only one of the two
    peaks, such as an arrested one, cannot tell a shifted stain from a shifted population.

    Args:
        adata: Object-resolution object carrying the DNA content column the model was fitted on.
        model: Key of the model in ``uns["mantispy"]``, or the model itself, such as one fitted on another object.
        by: The ``obs`` columns of a well, to check each well's peaks against the reference; ``None`` skips the check.
        layer: Layer to read the model's ``qc_features`` and G2M features from, or ``None`` for ``X``.
        key_added: Prefix of the outputs written.
        copy: Return a modified copy instead of mutating in place.

    Returns:
        ``None``, or the modified copy. Writes
        ``obsm[key_added + "_probabilities"]``, a frame with the probability of G1, S, G2M and off-cycle per cell;
        ``obs[key_added + "_phase"]``, the most probable of the four, for convenience;
        ``obs[key_added + "_position"]``, the posterior median in [0, 1), 0 being the start of G1;
        ``obsm[key_added + "_quantiles"]``, a frame with the 5%, 25%, 50%, 75% and 95% posterior quantiles;
        ``obs[key_added + "_qc"]``, when the model holds ``qc_features``, the first shape check a nucleus fails out of
        ``QC_FLAGS``: ``"irregular"`` for a solidity below the reference's, as merged nuclei and mitotic figures
        have, ``"small"`` for an area below it, as fragments and debris have, ``"dim"`` for too little DNA per area,
        and ``"ok"`` otherwise;
        ``obs[key_added + "_g2m_order"]``, when the model was fitted with ``refine_g2m``, the experimental order of
        each cell called G2M within G2M, from 0 to 1, ``NaN`` for every other cell; and
        ``uns["mantispy"][key_added + "_wells"]``, one row per ``by`` group with its ``n_cells`` and the ``shift``
        its 2N and 4N peaks share against the reference, in doublings, ``NaN`` when it lacks either peak.
        A cell without a DNA content gets ``NaN`` throughout, as does every cell when the model failed its check.
        An off-cycle cell gets ``NaN`` positions.

    Raises:
        KeyError: ``uns["mantispy"]`` has no entry ``model``, or ``obs`` lacks the DNA content column it was fitted on
            or a ``by`` column.
        ValueError: A ``by`` column holds missing values.
    """
    fitted = adata.uns.get("mantispy", {}).get(model) if isinstance(model, str) else model
    if fitted is None:
        raise KeyError(f"uns['mantispy'] has no {model!r}; run mt.tl.fit_cell_cycle first")
    key = fitted["key"]
    if key not in adata.obs:
        raise KeyError(f"obs has no column {key!r}; run mt.tl.dna_content first")
    content = as_frame(adata.obs)[key].to_numpy(dtype=float)
    known = np.isfinite(content)
    codes, keys = group_codes(adata, by)
    passed = bool(fitted["passed"])
    peaks, sd = [float(peak) for peak in fitted["peaks"]], float(fitted["sd"])

    probabilities = np.full((adata.n_obs, len(CYCLE_PHASES)), np.nan)
    quantiles = np.full((adata.n_obs, len(QUANTILES)), np.nan)
    if passed and known.any():
        probabilities[known], quantiles[known] = _posterior(content[known], fitted)
    elif not passed:
        get_logger().warning("project_cell_cycle: the model failed its check; every position is NaN")

    placed = np.isfinite(probabilities).all(axis=1)
    phase = np.full(adata.n_obs, np.nan, dtype=object)
    phase[placed] = np.asarray(CYCLE_PHASES, dtype=object)[probabilities[placed].argmax(axis=1)]
    quantiles[~placed | (phase == "off-cycle")] = np.nan

    table = pd.DataFrame(
        [key if isinstance(key, tuple) else (key,) for key in keys],
        columns=["group"] if by is None else ([by] if isinstance(by, str) else list(by)),
    )
    table["n_cells"] = np.bincount(codes, minlength=len(keys))
    table["shift"] = np.nan
    if passed and by is not None:
        order, offsets = group_offsets(codes, len(keys))
        table["shift"] = [
            _peak_shift(content[order[offsets[index] : offsets[index + 1]]], peaks, sd) for index in range(len(keys))
        ]
        shifted = table[table["shift"].abs() > sd]
        if len(shifted):
            get_logger().warning(
                "project_cell_cycle: %d %s group(s) have both their 2N and 4N peaks off the reference's, so their DNA "
                "stain reads differently and their positions carry the shift given, in doublings: %s",
                len(shifted),
                by,
                shifted.round(2).head(5).to_dict("records"),
            )

    if "qc" in fitted:
        terms = _shape_terms(adata, content, fitted["qc"]["features"], layer)
        flags = np.asarray(QC_FLAGS, dtype=object)[_qc(terms, fitted["qc"]["cutoffs"])]
        flags[~np.isfinite(terms).all(axis=1)] = np.nan
        adata.obs[f"{key_added}_qc"] = pd.Categorical(flags, categories=list(QC_FLAGS))

    if "g2m" in fitted:
        order = np.full(adata.n_obs, np.nan)
        rows = np.flatnonzero(phase == "G2M")
        if rows.size:
            block = get_matrix(adata[rows][:, list(fitted["g2m"]["features"])], layer).astype(np.float64)
            order[rows] = _g2m_order(block, fitted["g2m"])
        get_logger().warning(
            "project_cell_cycle: the G2M order is experimental and cannot be validated without a marker of G2 and "
            "mitosis; check it against one before reading it biologically"
        )
        adata.obs[f"{key_added}_g2m_order"] = order

    names = adata.obs_names
    adata.obsm[f"{key_added}_probabilities"] = pd.DataFrame(probabilities, index=names, columns=list(CYCLE_PHASES))
    adata.obs[f"{key_added}_phase"] = pd.Categorical(phase, categories=list(CYCLE_PHASES))
    adata.obs[f"{key_added}_position"] = quantiles[:, 2]
    adata.obsm[f"{key_added}_quantiles"] = pd.DataFrame(
        quantiles, index=names, columns=[f"{round(100 * q)}%" for q in QUANTILES]
    )
    adata.uns.setdefault("mantispy", {})[f"{key_added}_wells"] = table
    return None
