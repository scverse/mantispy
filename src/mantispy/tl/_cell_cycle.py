"""Cell-cycle position from DNA content: an ergodic-rate model fitted on reference cells."""

from __future__ import annotations

import numpy as np
import pandas as pd
from anndata import AnnData

from mantispy._core._reduce import get_matrix
from mantispy._core.frames import as_frame
from mantispy._core.logging import get_logger
from mantispy._core.masks import reference_mask
from mantispy._core.mutation import inplace_or_copy

CYCLE_PHASES = ("G1", "S", "G2", "M", "off-cycle")
QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)
_GATES = np.log2([1.25, 1.75])  # the dna_state defaults: 2N below the first, 4N from the second
_GRID = (np.arange(400) + 0.5) / 400
_MIN_REFERENCE = 50


def _age(fraction: np.ndarray) -> np.ndarray:
    """Cycle position of the cell at a given cumulative fraction of an exponentially growing population.

    Ages in such a population have density 2 ln2 2^-t on [0, 1), so the cumulative fraction F maps to t = -log2(1 - F/2).
    """
    return -np.log2(1 - np.clip(fraction, 0, 1) / 2)


def _robust_scale(values: np.ndarray) -> tuple[float, float]:
    low, high = np.percentile(values, [25, 75])
    return float(np.median(values)), float(high - low) or 1.0


def _gate(content: np.ndarray) -> dict:
    """Separate 2N and 4N peaks with a continuous S population between them."""
    from scipy.stats import gaussian_kde

    grid = np.linspace(-1, 2.5, 701)
    density = gaussian_kde(np.clip(content, -1, 2.5))(grid)
    g1 = density[np.abs(grid) < 0.25].max()
    window = np.flatnonzero((grid >= 0.6) & (grid <= 1.4))
    top = window[np.argmax(density[window])]
    is_peak = density[top] >= density[max(top - 20, 0)] and density[top] >= density[min(top + 20, grid.size - 1)]
    valley = density[(grid >= 0.2) & (grid <= grid[top] - 0.1)].min() if grid[top] > 0.35 else density[top]
    floor = density[(grid >= 0.4) & (grid <= 0.7)].min()
    s_share = float(np.mean((content >= _GATES[0]) & (content < _GATES[1])))
    report = {
        "peak_4n": float(grid[top]),
        "peak_ratio": float(density[top] / g1),
        "valley_ratio": float(valley / density[top]),
        "s_floor": float(floor / density[top]),
        "s_share": s_share,
    }
    report["passed"] = bool(
        is_peak
        and report["peak_ratio"] >= 0.05
        and report["valley_ratio"] <= 0.9
        and report["s_floor"] >= 0.1
        and s_share >= 0.03
    )
    return report


def _condensation(adata: AnnData, features: tuple[str, str], layer: str | None) -> tuple[np.ndarray, np.ndarray]:
    """Log mean DNA intensity and log nuclear area; a condensed nucleus is bright and small."""
    matrix = get_matrix(adata[:, list(features)], layer)
    intensity, area = (matrix[:, index].astype(np.float64) for index in range(2))
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log(np.where(intensity > 0, intensity, np.nan)), np.log(np.where(area > 0, area, np.nan))


def _flag_model(log_intensity: np.ndarray, log_area: np.ndarray, rows: np.ndarray) -> dict | None:
    """Centre and scale of both terms among ``rows``, and a cutoff three robust deviations above the median score."""
    rows = rows & np.isfinite(log_intensity) & np.isfinite(log_area)
    if rows.sum() < 20:
        return None
    centre_i, scale_i = _robust_scale(log_intensity[rows])
    centre_a, scale_a = _robust_scale(log_area[rows])
    score = (log_intensity[rows] - centre_i) / scale_i - (log_area[rows] - centre_a) / scale_a
    median = float(np.nanmedian(score))
    cutoff = median + 3 * 1.4826 * float(np.nanmedian(np.abs(score - median)))
    return {"centre": [centre_i, centre_a], "scale": [scale_i, scale_a], "cutoff": cutoff}


def _flagged(log_intensity: np.ndarray, log_area: np.ndarray, flag: dict | None) -> np.ndarray:
    if flag is None:
        return np.zeros(log_intensity.shape, dtype=bool)
    (ci, ca), (si, sa) = flag["centre"], flag["scale"]
    with np.errstate(invalid="ignore"):
        return ((log_intensity - ci) / si - (log_area - ca) / sa) > flag["cutoff"]


@inplace_or_copy(expects="object")
def fit_cell_cycle(
    adata: AnnData,
    key: str = "Metadata_DNAContent",
    reference: str | None = "negcon",
    mitosis_features: tuple[str, str] | None = None,
    layer: str | None = None,
    key_added: str = "cell_cycle",
    copy: bool = False,
) -> AnnData | None:
    """Fit a cell-cycle clock on the reference cells' DNA content.

    The reference cells are ordered G1, S, G2 and M by DNA content, and their cumulative fraction is turned into a
    position in the cycle with the age distribution of an exponentially growing population (ergodic rate analysis,
    Kafri et al. 2013). Position 0 is the start of G1 and 1 is division. Inside S, DNA content places a cell
    precisely; inside the 2N and 4N plateaus it does not, and :func:`~mantispy.tl.project_cell_cycle` reports the
    whole plateau as the cell's uncertainty instead of inventing an order.

    The fit first checks that the reference cycles: its DNA content must show a 2N peak, a separate 4N peak and a
    continuous S population between them. A population that fails, such as an arrested one or a stain that does not
    resolve the 4N peak, gets a model marked as failed, and projecting it gives ``NaN``.

    Args:
        adata: Object-resolution object carrying the output of :func:`~mantispy.tl.dna_content`.
        key: ``obs`` column holding the DNA content, in doublings of the G1 peak.
        reference: The cells to fit on: ``"negcon"`` for the negative controls, the name of a boolean ``obs``
            column, or ``None`` for every cell.
        mitosis_features: The mean DNA intensity and nuclear area features, as a pair. With them, a 4N nucleus
            that is much brighter and smaller than the other 4N nuclei is called mitotic, and so is a 2N nucleus
            that stands out the same way among 2N nuclei, a daughter just out of division. ``None`` calls no
            mitosis.
        layer: Layer to read ``mitosis_features`` from, or ``None`` for ``X``. They need positive values.
        key_added: Key the model is written under in ``uns["mantispy"]``.
        copy: Return a modified copy instead of mutating in place.

    Returns:
        ``None``, or the modified copy.
        Writes ``uns["mantispy"][key_added]``: the DNA ``key``, the ``gate`` report (``passed``, the 4N peak position,
        the 4N to 2N peak height ratio, the valley and S-floor ratios to the 4N peak, and the share of cells in S),
        the 2N peak width ``sd``, the ``off_cycle`` DNA bounds, the reference ``fractions`` in G1, S, G2 and M,
        the 4N level ``g4``, and the ``mitosis`` flag model when ``mitosis_features`` is given.

    Raises:
        KeyError: ``obs`` has no column ``key``; run :func:`~mantispy.tl.dna_content` first.
        ValueError: The reference holds too few cells with a DNA content to fit on.

    Notes:
        The model assumes the reference grows in steady state, with every phase taking a fixed share of the cycle.
        It cannot tell G0 from G1. A cell beyond the off-cycle bounds, below 2N or above 4N, is left off the cycle.
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
    g1 = np.abs(values) < 0.3
    sd = 1.4826 * float(np.median(np.abs(values[g1] - np.median(values[g1])))) if g1.any() else float(np.std(values))
    off_cycle = [-4 * sd, gate["peak_4n"] + 4 * sd]
    on = (values >= off_cycle[0]) & (values <= off_cycle[1])
    phase = np.digitize(values, _GATES)

    model = {"key": key, "gate": gate, "sd": sd, "off_cycle": off_cycle}
    mitotic = np.zeros(values.size, dtype=bool)
    if mitosis_features is not None:
        log_intensity, log_area = (term[reference_rows] for term in _condensation(adata, mitosis_features, layer))
        flags = {
            name: _flag_model(log_intensity, log_area, on & (phase == index)) for name, index in (("4n", 2), ("2n", 0))
        }
        model["mitosis"] = {"features": list(mitosis_features)} | {
            name: f for name, f in flags.items() if f is not None
        }
        mitotic = (phase == 2) & _flagged(log_intensity, log_area, flags["4n"])
    phase[mitotic] = 3
    model["fractions"] = [float(np.mean(phase[on] == index)) for index in range(4)]
    model["g4"] = float(np.median(values[on & (phase >= 2)])) if (on & (phase >= 2)).any() else 1.0
    if not gate["passed"]:
        get_logger().warning(
            "fit_cell_cycle: the reference does not show separate 2N and 4N peaks with S cells between them %s; "
            "project_cell_cycle will give NaN",
            {name: round(value, 3) for name, value in gate.items() if name != "passed"},
        )
    adata.uns.setdefault("mantispy", {})[key_added] = model
    return None


def _template(model: dict) -> tuple[np.ndarray, np.ndarray]:
    """Expected DNA content along the cycle, and the cycle positions where G1, S, G2 and M begin and end.

    DNA is replicated at a constant rate, so it rises linearly between 2N and the 4N level; the S gates sit at the
    positions the reference fractions give them, and the ramp is extended through them to 2N and to 4N.
    """
    edges = _age(np.r_[0, np.cumsum(model["fractions"])])
    slope = (_GATES[1] - _GATES[0]) / max(edges[2] - edges[1], 1e-6)
    start = max(edges[1] - _GATES[0] / slope, 0.0)
    end = min(edges[2] + (model["g4"] - _GATES[1]) / slope, edges[3])
    expected = np.where(
        _GRID < start, 0.0, np.where(_GRID < end, model["g4"] * (_GRID - start) / max(end - start, 1e-6), model["g4"])
    )
    return expected, edges


@inplace_or_copy(expects="object")
def project_cell_cycle(
    adata: AnnData,
    model: str = "cell_cycle",
    layer: str | None = None,
    key_added: str = "cell_cycle",
    copy: bool = False,
) -> AnnData | None:
    """Place every cell on the cycle of a model from :func:`~mantispy.tl.fit_cell_cycle`, with its uncertainty.

    Each cell gets a posterior over its position: the age distribution of the reference as the prior, times how well
    the cell's DNA content fits each position. An S cell is placed tightly; a G1 or G2 cell spreads over its whole
    plateau, since DNA content does not order cells within it. Perturbed cells are placed on the reference's cycle,
    so an arrest shows up as cells piling up at one position.

    Args:
        adata: Object-resolution object carrying the model in ``uns["mantispy"][model]`` and the DNA content column
            it was fitted on. To project another object, copy the model entry into its ``uns["mantispy"]`` first.
        model: Key of the model in ``uns["mantispy"]``.
        layer: Layer to read the model's ``mitosis_features`` from, or ``None`` for ``X``.
        key_added: Prefix of the columns written.
        copy: Return a modified copy instead of mutating in place.

    Returns:
        ``None``, or the modified copy. Writes
        ``obs[key_added + "_position"]``, the posterior median in [0, 1), 0 being the start of G1;
        ``obsm[key_added + "_quantiles"]``, the 5%, 25%, 50%, 75% and 95% posterior quantiles as columns;
        ``obs[key_added + "_phase"]``, a categorical over ``CYCLE_PHASES``; and
        ``obs[key_added + "_mitotic"]``, true for a cell flagged as a mitotic 4N nucleus or a 2N daughter.
        A cell without a DNA content gets ``NaN`` throughout, as does every cell when the model failed its gate.
        An off-cycle cell gets the phase ``"off-cycle"`` and ``NaN`` positions.

    Raises:
        KeyError: ``uns["mantispy"]`` has no entry ``model``, or ``obs`` lacks the DNA content column it was fitted on.
    """
    fitted = adata.uns.get("mantispy", {}).get(model)
    if fitted is None:
        raise KeyError(f"uns['mantispy'] has no {model!r}; run mt.tl.fit_cell_cycle first")
    key = fitted["key"]
    if key not in adata.obs:
        raise KeyError(f"obs has no column {key!r}; run mt.tl.dna_content first")
    content = as_frame(adata.obs)[key].to_numpy(dtype=float)
    known = np.isfinite(content)
    low, high = fitted["off_cycle"]
    on = known & (content >= low) & (content <= high)

    phase = np.digitize(content, _GATES).astype(int)
    mitotic = np.zeros(adata.n_obs, dtype=bool)
    flags = fitted.get("mitosis")
    if flags is not None:
        first, second = flags["features"]
        log_intensity, log_area = _condensation(adata, (str(first), str(second)), layer)
        four = on & (phase == 2) & _flagged(log_intensity, log_area, flags.get("4n"))
        two = on & (phase == 0) & _flagged(log_intensity, log_area, flags.get("2n"))
        phase[four] = 3
        mitotic = four | two

    quantiles = np.full((adata.n_obs, len(QUANTILES)), np.nan)
    passed = bool(fitted["gate"]["passed"])
    if passed and on.any():
        expected, edges = _template(fitted)
        log_prior = np.log(2 * np.log(2)) - _GRID * np.log(2)
        in_m = _GRID >= edges[3]
        rows = np.flatnonzero(on)
        for start in range(0, rows.size, 8192):
            chunk = rows[start : start + 8192]
            log_post = log_prior[None] - 0.5 * ((content[chunk, None] - expected[None]) / fitted["sd"]) ** 2
            if flags is not None:
                # A flagged nucleus sits in M; an unflagged one may still be mitotic, since the flag misses many.
                log_post += np.where(
                    phase[chunk, None] == 3, np.where(in_m, 0.0, np.log(0.02)), np.where(in_m, np.log(0.3), 0.0)
                )
            post = np.exp(log_post - log_post.max(axis=1, keepdims=True))
            cdf = np.cumsum(post, axis=1)
            cdf /= cdf[:, -1:]
            quantiles[chunk] = np.stack(
                [_GRID[np.minimum((cdf < q).sum(axis=1), _GRID.size - 1)] for q in QUANTILES], axis=1
            )
    elif not passed:
        get_logger().warning("project_cell_cycle: model %r failed its gate; every position is NaN", model)

    labels = np.asarray(CYCLE_PHASES, dtype=object)[np.where(on, phase, 4)]
    labels[~known] = np.nan
    if not passed:
        labels[:] = np.nan
    adata.obs[f"{key_added}_position"] = quantiles[:, 2]
    adata.obsm[f"{key_added}_quantiles"] = quantiles
    adata.obs[f"{key_added}_phase"] = pd.Categorical(labels, categories=list(CYCLE_PHASES))
    adata.obs[f"{key_added}_mitotic"] = mitotic
    return None
