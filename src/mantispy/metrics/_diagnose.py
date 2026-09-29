from __future__ import annotations

import warnings

import anndata as ad
import numpy as np
import pandas as pd
from anndata import AnnData

from mantispy._core._reduce import get_matrix, group_codes, group_offsets
from mantispy._core._stats import _default_well_block, benjamini_hochberg
from mantispy._core.frames import as_frame
from mantispy._core.masks import reference_mask
from mantispy._core.schema import get_resolution, stamp

#: How far the observed null rate may exceed the nominal one before it is a failure.
TOLERANCE = 2.0


def _verdict(ok: bool, warn: bool = False) -> str:
    return "pass" if ok else ("warn" if warn else "FAIL")


def _empirical_null(controls: AnnData, size: int, n_draws: int, seed: int, block: str | None) -> np.ndarray:
    """P-values from relabeling control wells as a treatment of ``size`` wells."""
    from mantispy.tl._differential import differential_features

    values = get_matrix(controls)
    pvalues = []
    for draw in range(n_draws):
        rng = np.random.default_rng(seed + draw)
        picked = rng.choice(controls.n_obs, size=size, replace=False)
        labels = np.full(controls.n_obs, "__reference__", dtype=object)
        labels[picked] = "__pseudo__"
        obs = as_frame(controls.obs).copy()
        obs["Metadata_Perturbation"] = labels
        obs["Metadata_Control"] = labels == "__reference__"
        scratch = ad.AnnData(X=values.copy(), obs=obs, var=as_frame(controls.var).copy())
        stamp(scratch, resolution="well")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            differential_features(scratch, block=block, key_added="__null__")
        table = scratch.uns["mantispy"]["__null__"]
        pvalues.append(table.loc[table["group"] == "__pseudo__", "pvalue"].to_numpy())
    return np.concatenate(pvalues) if pvalues else np.empty(0)


def _empirical_hit_rate(
    controls: AnnData, size: int, n_draws: int, seed: int, n_permutations: int, alpha: float
) -> dict[str, int]:
    """Count the control-only pseudo-treatments each hit caller calls."""
    from mantispy.tl._distance import edistance
    from mantispy.tl._hits import hit_calling

    values = get_matrix(controls)
    called = {"hit_calling": 0, "edistance": 0}
    for draw in range(n_draws):
        rng = np.random.default_rng(seed + draw)
        picked = rng.choice(controls.n_obs, size=size, replace=False)
        labels = np.where(np.isin(np.arange(controls.n_obs), picked), "__pseudo__", "__reference__")
        obs = as_frame(controls.obs).copy()
        obs["Metadata_Perturbation"] = labels
        obs["Metadata_Control"] = labels == "__reference__"
        scratch = ad.AnnData(X=values.copy(), obs=obs, var=as_frame(controls.var).copy())
        stamp(scratch, resolution="well")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for name, function in (("hit_calling", hit_calling), ("edistance", edistance)):
                function(scratch, n_permutations=n_permutations, seed=draw, key_added=f"__{name}__")
                table = scratch.uns["mantispy"][f"__{name}__"]
                row = table.loc[table["group"] == "__pseudo__", "pvalue"]
                called[name] += int((row.to_numpy() < alpha).sum())
    return {name: int(value) for name, value in called.items()}


def _empirical_cell_hit_rate(
    controls: AnnData,
    well_codes: np.ndarray,
    pseudo_wells: int,
    n_draws: int,
    seed: int,
    n_permutations: int,
    alpha: float,
) -> dict[str, dict[str, int]]:
    """Count control-only pseudo-treatments each hit caller calls, under two nulls side by side.

    A pseudo-treatment is every cell of a random subset of control wells, kept whole, since the well
    structure is what makes the two nulls differ. Each caller runs twice per draw:

    - ``well-block``: the caller with its default block, which draws whole wells for the null, the
      design's exchangeable unit. This is the calibrated rate.
    - ``cell-shuffle``: the same caller given a block that puts every cell in its own group, so its own
      well-block permutation engine resamples single cells. Cells within a well are not independent
      replicates, so this rate is anti-conservative and is the number to distrust.

    Both go through the caller's one code path, so the difference is the null unit alone.
    ``hit_calling`` runs ``method="ks"``, the method its docstring recommends at cell resolution.
    """
    from mantispy.tl._distance import edistance
    from mantispy.tl._hits import hit_calling

    values = get_matrix(controls)
    wells = np.unique(well_codes)
    # One block per cell: whole-block resampling then draws single cells, a free cell shuffle.
    cell_block = np.arange(controls.n_obs)
    called = {
        "hit_calling": {"well-block": 0, "cell-shuffle": 0},
        "edistance": {"well-block": 0, "cell-shuffle": 0},
    }
    for draw in range(n_draws):
        rng = np.random.default_rng(seed + draw)
        picked = rng.choice(wells, size=pseudo_wells, replace=False)
        labels = np.where(np.isin(well_codes, picked), "__pseudo__", "__reference__")
        obs = as_frame(controls.obs).copy()
        obs["Metadata_Perturbation"] = labels
        obs["Metadata_Control"] = labels == "__reference__"
        obs["__cell__"] = cell_block
        scratch = ad.AnnData(X=values.copy(), obs=obs, var=as_frame(controls.var).copy())
        stamp(scratch, resolution="cell")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for mode, block in (("well-block", None), ("cell-shuffle", "__cell__")):
                hit_calling(
                    scratch, method="ks", block=block, n_permutations=n_permutations, seed=draw, key_added="__null__"
                )
                edistance(scratch, block=block, n_permutations=n_permutations, seed=draw, key_added="__edist__")
                for name, key in (("hit_calling", "__null__"), ("edistance", "__edist__")):
                    table = scratch.uns["mantispy"][key]
                    row = table.loc[table["group"] == "__pseudo__", "pvalue"].to_numpy()
                    called[name][mode] += int((row < alpha).sum())
    return called


def _diagnose_cell(
    adata: AnnData,
    groupby: str,
    reference: str | None,
    n_draws: int,
    alpha: float,
    seed: int,
    n_permutations: int,
) -> pd.DataFrame:
    """Cell-resolution calibration: expose cell-within-well pseudoreplication.

    Real control cells are relabeled as pseudo-treatments of whole control wells, and the false positive
    rate of each hit caller is reported under the naive cell-shuffle null and the well-block null side by
    side. See :func:`_empirical_cell_hit_rate` and the ``Notes`` of :func:`diagnose_testing`.
    """
    from scipy import stats

    is_control = reference_mask(adata, reference)
    codes, keys = group_codes(adata, groupby)
    well_codes = _default_well_block(adata, block=None)
    if well_codes is None:
        raise ValueError(
            "diagnose_testing at cell resolution needs a complete (Metadata_Plate, Metadata_Well) with "
            "replicated wells to draw the well-block null; add it, or aggregate to wells with mt.tl.aggregate"
        )

    # Wells per treatment, the analogue of the well-level sizes, so the pseudo-treatment matches yours.
    treatment_wells = [
        int(np.unique(well_codes[codes == index]).size) for index in range(len(keys)) if not is_control[codes == index].all()
    ]
    scored = [count for count in treatment_wells if count >= 2]
    controls = adata[is_control].copy()
    control_wells = _default_well_block(controls, block=None)
    n_control_wells = int(np.unique(control_wells).size) if control_wells is not None else 0
    if not scored or n_control_wells < 4:
        raise ValueError("need at least one treatment on two wells and four reference wells to measure a null against")

    typical = int(np.median(scored))
    # Cap the pseudo-treatment at half the control wells so the rest can serve as the reference.
    pseudo_wells = max(min(typical, n_control_wells // 2), 2)
    counts = _empirical_cell_hit_rate(controls, control_wells, pseudo_wells, n_draws, seed, n_permutations, alpha)

    # A calibrated test calls a pseudo-treatment at rate alpha, so the count over n_draws is
    # Binomial(n_draws, alpha) and the cutoff is its 95th percentile, as the well-level hit-caller rows use.
    critical = int(stats.binom.ppf(0.95, n_draws, alpha))
    rows = []
    for name in ("hit_calling", "edistance"):
        well_block = counts[name]["well-block"]
        cell_shuffle = counts[name]["cell-shuffle"]
        rows.append(
            {
                "check": f"{name} well-block null rate",
                "value": f"{well_block} of {n_draws}",
                "expected": f"<= {critical}",
                "verdict": _verdict(well_block <= critical, warn=well_block <= critical + 1),
                "note": (
                    f"{name} called a control-only pseudo-treatment of {pseudo_wells} wells at p<{alpha} in "
                    f"{well_block} of {n_draws} draws, drawing whole wells for the null as the cell design "
                    f"requires; a calibrated test exceeds {critical} about 5% of the time by chance. This is the "
                    "rate to trust."
                ),
            }
        )
        rows.append(
            {
                "check": f"{name} cell-shuffle null rate",
                "value": f"{cell_shuffle} of {n_draws}",
                "expected": f"> {critical} (inflated)",
                "verdict": _verdict(False, warn=True),
                "note": (
                    f"the naive null that permutes single cells instead of whole wells, called in {cell_shuffle} of "
                    f"{n_draws} draws. Cells within a well share the well and are not independent replicates, so "
                    "this rate is expected to be inflated above the well-block rate; it is shown to make the "
                    "pseudoreplication visible, not to pass or fail."
                ),
            }
        )
    return pd.DataFrame(rows, columns=["check", "value", "expected", "verdict", "note"])


def diagnose_testing(
    adata: AnnData,
    groupby: str = "Metadata_Perturbation",
    reference: str | None = "negcon",
    block: str | None = "Metadata_Plate",
    n_draws: int = 8,
    alpha: float = 0.05,
    seed: int = 0,
    n_permutations: int = 200,
) -> pd.DataFrame:
    """Check whether differential testing is calibrated on this screen.

    Args:
        adata: Well-level profiles after the normalization and transform you plan to test with, since the results depend on both.
        groupby: As in :func:`~mantispy.tl.differential_features`.
        reference: As in :func:`~mantispy.tl.differential_features`.
        block: As in :func:`~mantispy.tl.differential_features`.
        n_draws: Pseudo-treatments drawn from the controls for the empirical null.
            More draws resolve the false positive rate better and take longer.
        alpha: Nominal rate the null is compared against.
        seed: Seed for choosing which control wells stand in for a treatment.
            The two hit callers' permutation nulls are seeded by the draw index instead, so they are identical across calls that differ only in ``seed``.
        n_permutations: Null size for the two hit callers; smaller is faster and coarser.

    Returns:
        A frame with columns ``check``, ``value``, ``expected``, ``verdict`` and ``note``, one row per check that ran, where a ``FAIL`` verdict means the check does not hold on this data.
        The empirical-null rows are absent when every null p-value came back non-finite, and the two hit-caller rows need at least eight reference wells.

    Raises:
        ValueError: No treatment has two wells, or there are fewer than four reference wells, leaving nothing to measure a null against.

    Notes:
        At cell resolution the checks change, because cells within a well are not independent replicates and the well-level checks describe well-level testing. Control cells are relabeled as pseudo-treatments of whole control wells and put through the two cell-resolution hit callers under two nulls:

        ``hit_calling well-block null rate`` / ``edistance well-block null rate``
            The caller drawing whole wells for its null, the design's exchangeable unit.
            The count is compared against the upper tail of ``Binomial(n_draws, alpha)`` as the well-level hit-caller rows are.
            This is the rate to trust.
        ``hit_calling cell-shuffle null rate`` / ``edistance cell-shuffle null rate``
            The same caller with the null permuting single cells instead of whole wells.
            Cells within a well share the well, so this shrinks the null spread by the cell count rather than the well count and is expected to be inflated above the well-block rate.
            It is diagnostic, shown to make the pseudoreplication visible rather than to pass or fail.

        At well resolution the checks and what each one detects:

        ``null p < 0.05`` / ``null p < 0.01``
            Control wells relabeled as treatments of the size yours have.
            The rate should match the nominal one.
            Heavy tails distort small p-values first, so a test can be calibrated at 0.05 and not at 0.01, which is closer to the range a false discovery rate works in.
        ``null discoveries``
            How many of those null p-values survive Benjamini-Hochberg.
            A count above zero means the q-values on the real data are optimistic by roughly that much.
        ``hit_calling null rate`` / ``edistance null rate``
            The same relabeling applied to the two hit callers, counted over ``n_draws`` draws.
            Both are permutation tests that are not fully calibrated at small control counts, so the count is compared against the upper tail of ``Binomial(n_draws, alpha)`` instead of a fixed rate.
            At eight draws the smallest non-zero rate is 0.125, and a threshold below that would fail a calibrated screen a third of the time.
            Raising ``n_draws`` sharpens the answer and moves the cutoff with it.
        ``rank test resolution``
            The smallest p-value a Mann-Whitney test can return at your replication, compared with what multiple-testing correction requires.
            With three wells against 14 reference wells the floor is 2.9e-03 whatever the effect size, and :func:`~mantispy.tl.effect_size` then silently calls nothing.
        ``excess kurtosis``
            How far the features are from the normality a t-test assumes.
            It predicts the null checks above but is not a verdict on its own, since heavy tails matter less with enough wells per group.
        ``wells per treatment`` and ``treatments sharing a {block} with the reference``
            The replicate structure the other checks depend on.
            A treatment whose wells share no block with the reference cannot be tested.
    """
    from scipy import stats

    if get_resolution(adata) == "cell":
        return _diagnose_cell(adata, groupby, reference, n_draws, alpha, seed, n_permutations)

    obs = as_frame(adata.obs)
    is_control = reference_mask(adata, reference)
    codes, keys = group_codes(adata, groupby)
    sizes = np.bincount(codes[~is_control], minlength=len(keys))
    scored = sizes[sizes >= 2]
    if not scored.size or is_control.sum() < 4:
        raise ValueError("need at least one treatment with two wells and four reference wells")

    typical = int(np.median(scored))
    n_tests = int(scored.size) * adata.n_vars
    threshold = alpha / max(n_tests, 1)
    rows = []

    rows.append(
        {
            "check": "wells per treatment",
            "value": f"{typical} (min {int(scored.min())})",
            "expected": ">= 3",
            "verdict": _verdict(bool(scored.min() >= 3), warn=bool(scored.min() >= 2)),
            "note": "the unit that was randomized, and the sample size of every test",
        }
    )

    if block is not None and block in obs.columns:
        blocks = obs[block].to_numpy()
        control_blocks = set(blocks[is_control])
        order, offsets = group_offsets(codes, len(keys))
        stranded: list[str] = []
        spans: list[int] = []
        for index in range(len(keys)):
            if sizes[index] < 2:
                continue
            group = order[offsets[index] : offsets[index + 1]]
            seen = set(blocks[group[~is_control[group]]])
            spans.append(len(seen))
            if not seen & control_blocks:
                stranded.append(str(keys[index]))
        rows.append(
            {
                "check": f"treatments sharing a {block} with the reference",
                "value": f"{len(spans) - len(stranded)} of {len(spans)}",
                "expected": "all",
                "verdict": _verdict(not stranded),
                "note": "a treatment on plates with no controls cannot be told from its plate"
                if stranded
                else f"median {int(np.median(spans))} {block} per treatment",
            }
        )

    values = get_matrix(adata).astype(np.float64)
    finite = np.isfinite(values).all(axis=0)
    centred = values[:, finite] - values[:, finite].mean(axis=0)
    variance = np.mean(centred**2, axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        kurtosis = float(np.nanmean(np.mean(centred**4, axis=0) / np.where(variance > 0, variance**2, np.nan) - 3.0))
    rows.append(
        {
            "check": "excess kurtosis",
            "value": f"{kurtosis:.1f}",
            "expected": "0 (Gaussian)",
            "verdict": _verdict(kurtosis <= 20, warn=True),
            "note": "heavy tails break the small p-values first; mt.pp.rank_int removes them",
        }
    )

    # Distinct values, because scipy falls back to the normal approximation on tied samples.
    n_control = int(is_control.sum())
    floor = float(
        stats.mannwhitneyu(np.arange(typical, dtype=float), np.arange(typical, typical + n_control, dtype=float)).pvalue
    )
    needed = int(np.ceil(floor * n_tests / alpha))
    rows.append(
        {
            "check": "rank test resolution",
            "value": f"{floor:.1e}",
            "expected": f"< {threshold:.1e}",
            "verdict": _verdict(floor < threshold),
            "note": f"smallest p a Mann-Whitney can return with {typical} wells against {n_control} reference wells; "
            + (
                f"the top of {n_tests:,} tests needs {threshold:.1e}"
                if floor < threshold
                else f"under BH nothing is called until {needed:,} of {n_tests:,} tests reach it at once"
            ),
        }
    )

    controls = adata[is_control].copy()
    # Cap a pseudo-treatment at half the control wells so the rest can serve as the reference.
    pseudo_size = max(min(typical, controls.n_obs // 2), 2)
    null = _empirical_null(controls, pseudo_size, n_draws, seed, block)
    null = null[np.isfinite(null)]
    if null.size:
        for level in (0.05, 0.01):
            observed = float(np.mean(null < level))
            rows.append(
                {
                    "check": f"null p < {level}",
                    "value": f"{observed * 100:.1f}%",
                    "expected": f"{level * 100:.0f}%",
                    "verdict": _verdict(observed <= level * TOLERANCE),
                    "note": f"control wells relabeled as {pseudo_size}-well treatments, {n_draws} draws"
                    + ("" if pseudo_size == typical else f" (capped from {typical}: only {controls.n_obs} controls)"),
                }
            )
        discoveries = int((benjamini_hochberg(null) < alpha).sum())
        rows.append(
            {
                "check": "null discoveries",
                "value": f"{discoveries} of {null.size:,}",
                "expected": "0",
                "verdict": _verdict(discoveries == 0),
                "note": "findings on data where there is nothing to find",
            }
        )

    if controls.n_obs >= 8:
        counts = _empirical_hit_rate(controls, pseudo_size, n_draws, seed, n_permutations, alpha)
        critical = int(stats.binom.ppf(0.95, n_draws, alpha))
        for name, count in counts.items():
            rows.append(
                {
                    "check": f"{name} null rate",
                    "value": f"{count} of {n_draws}",
                    "expected": f"<= {critical}",
                    "verdict": _verdict(count <= critical, warn=count <= critical + 1),
                    "note": (
                        f"{name} called a control-only pseudo-treatment of {pseudo_size} wells at "
                        f"p<{alpha} in {count} of {n_draws} draws; a calibrated test exceeds "
                        f"{critical} about 5% of the time by chance. Raise n_draws for a sharper "
                        "answer; the cutoff moves with it."
                    ),
                }
            )

    return pd.DataFrame(rows, columns=["check", "value", "expected", "verdict", "note"])
