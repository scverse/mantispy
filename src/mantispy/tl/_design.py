"""Experimental design diagnostics: replicate saturation and cytotoxicity."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from anndata import AnnData

from mantispy._core._numba import MEDIAN, grouped_stat
from mantispy._core._reduce import group_codes, group_offsets, group_rows, representation
from mantispy._core.frames import as_frame
from mantispy._core.logging import get_logger
from mantispy._core.masks import held_out_reference, reference_mask
from mantispy._core.mutation import inplace_or_copy


def viability(
    obs: pd.DataFrame,
    count_key: str,
    site_key: str | None,
    is_control: np.ndarray,
    levels: np.ndarray | None = None,
) -> np.ndarray:
    """Each row's cell count against the controls', per field of view where the fields are known.

    ``levels`` names a grouping each row is scored inside, normally the plate. Plates are seeded and imaged
    separately, and on the OASIS pilot their control counts differ by half, so a sparse plate otherwise reads as
    one whose treated wells are dying. ``None`` pools every control row.

    A level whose controls carry no usable count gets NaN rather than an error, so the caller decides whether
    that is fatal.
    """
    counts = obs[count_key].to_numpy(dtype=float)
    if site_key is not None and site_key in obs:
        counts = counts / np.maximum(obs[site_key].to_numpy(dtype=float), 1)

    scored = np.full(counts.shape, np.nan)
    for level in pd.unique(np.zeros(len(obs)) if levels is None else levels):
        inside = np.ones(len(obs), dtype=bool) if levels is None else levels == level
        referenced = inside & is_control
        centre = float(np.nanmedian(counts[referenced])) if referenced.any() else np.nan
        if np.isfinite(centre) and centre > 0:
            scored[inside] = counts[inside] / centre
    return scored


def signature_stability(
    profiles: np.ndarray, members: list[np.ndarray], depth: int, generator: np.random.Generator
) -> float:
    """Median correlation between two independent ``depth``-replicate signatures per group.

    Draws ``2 * depth`` replicates per group, splits them into halves and correlates the two medians.
    This shows at what depth a signature stops changing when a different set of wells is drawn.
    It needs no labels and is defined from a depth of one.

    Args:
        profiles: The feature matrix, one row per replicate.
        members: Row indices of each group, one array per group.
        depth: Replicates per half, so a group contributes only with ``2 * depth`` rows.
        generator: Source of the random draw.

    Returns:
        The median over the contributing groups, or ``NaN`` when no group could contribute.
        A group is left out when it has fewer than ``2 * depth`` replicates, or when either half has fewer than two finite features or no spread, which leaves the correlation undefined.
    """
    scores = []
    for rows in members:
        if rows.size < 2 * depth:
            continue
        drawn = generator.choice(rows, size=2 * depth, replace=False)
        # Both halves in one grouped-median call.
        # np.nanmedian dispatches per feature slice, and this runs n_draws times for every depth of every group.
        halves = grouped_stat(profiles[drawn], np.repeat([0, 1], depth).astype(np.int32), 2, MEDIAN)
        left, right = halves[0], halves[1]
        usable = np.isfinite(left) & np.isfinite(right)
        if usable.sum() < 2 or np.ptp(left[usable]) == 0 or np.ptp(right[usable]) == 0:
            continue
        scores.append(float(np.corrcoef(left[usable], right[usable])[0, 1]))
    return float(np.median(scores)) if scores else float("nan")


def signature_convergence(
    profiles: np.ndarray, members: list[np.ndarray], depth: int, generator: np.random.Generator
) -> float:
    """Median correlation between a ``depth``-replicate signature and the full one.

    Defined up to one less than the group size, whereas ``signature_stability`` needs twice the depth and gives a single point at three replicates per treatment, as in BBBC021.
    The subset is part of the full set it is compared against, so the correlation is optimistic.
    Read the shape of the curve rather than its height.

    Args:
        profiles: The feature matrix, one row per replicate.
        members: Row indices of each group, one array per group.
        depth: Replicates in the subset, so a group contributes only with more than ``depth`` rows.
        generator: Source of the random draw.

    Returns:
        The median over the contributing groups, or ``NaN`` when no group could contribute.
        A group is left out when it has ``depth`` or fewer replicates, or when the two signatures share fewer than two finite features or have no spread.
    """
    scores = []
    for rows in members:
        if rows.size <= depth:
            continue
        whole = grouped_stat(profiles[rows], np.zeros(rows.size, dtype=np.int32), 1, MEDIAN)[0]
        drawn = generator.choice(rows, size=depth, replace=False)
        part = grouped_stat(profiles[drawn], np.zeros(depth, dtype=np.int32), 1, MEDIAN)[0]
        usable = np.isfinite(whole) & np.isfinite(part)
        if usable.sum() < 2 or np.ptp(whole[usable]) == 0 or np.ptp(part[usable]) == 0:
            continue
        scores.append(float(np.corrcoef(whole[usable], part[usable])[0, 1]))
    return float(np.median(scores)) if scores else float("nan")


METRICS = {"signature_stability": signature_stability, "convergence": signature_convergence}


@inplace_or_copy(expects=("well", "perturbation"))
def replicate_saturation(
    adata: AnnData,
    groupby: str = "Metadata_Perturbation",
    metric: str | Callable[[np.ndarray, np.ndarray, int, np.random.Generator], float] = "signature_stability",
    max_replicates: int | None = None,
    min_groups: int = 3,
    n_draws: int = 5,
    use_rep: str | None = None,
    seed: int = 0,
    key_added: str = "replicate_saturation",
    copy: bool = False,
) -> AnnData | None:
    """Score how much a group's signature improves with each additional replicate.

    Args:
        adata: Well-level profiles with several replicates per group.
        groupby: The column whose groups are the replicate sets.
        metric: ``"signature_stability"`` correlates two disjoint subsets of this depth. It is unbiased but needs ``2 * depth`` replicates, so it stops early on a screen with three. ``"convergence"`` correlates a subset of this depth with the group's full signature. It is defined up to one less than the group size and optimistic by construction. A callable ``(profiles, codes, depth, generator) -> float`` can score anything else, such as MOA retrieval or mAP.
        max_replicates: Deepest subset to try. ``None`` derives it from ``min_groups``.
        min_groups: Number of groups that must be able to supply a depth for it to be scored. The statistic is a median over the contributing groups, and the largest group is usually the negative controls. On 132 JUMP plates, taking the range from the largest group gives 4252 depths, and past about 66 only DMSO contributes. ``min_groups=1`` takes the range from the largest group.
        n_draws: Random subsets per depth. The spread across draws is reported as ``std``.
        use_rep: Score ``obsm[use_rep]`` instead of ``X``.
        seed: Seed for reproducibility.
        key_added: Name for the output table.
        copy: Return a modified copy instead of mutating in place.

    Returns:
        ``None``, or the modified copy.
        Writes ``uns["mantispy"][key_added]`` with ``n_replicates``, ``mean``, ``std`` and ``n_draws``.
        The ``n_draws`` column counts the draws that produced a finite score, which can be fewer than the ``n_draws`` asked for, and the first depth at which every draw is ``NaN`` ends the table.

    Raises:
        ValueError: ``metric`` is neither a callable nor one of ``METRICS``.

    Notes:
        A curve that still climbs steeply at the deepest depth means the screen is under-replicated, which informs the design of the next experiment.
        Three replicates, as in BBBC021, give one point with the default metric and two with ``"convergence"``.
    """
    profiles = representation(adata, use_rep)
    codes, keys = group_codes(adata, groupby)
    sizes = np.bincount(codes, minlength=len(keys))
    if isinstance(metric, str) and metric not in METRICS:
        raise ValueError(f"metric must be one of {tuple(METRICS)} or a callable, got {metric!r}")

    if max_replicates is not None:
        deepest = max_replicates
    else:
        # The deepest depth that `min_groups` groups can still supply.
        # The largest group is usually the negative controls, 8505 wells on JUMP against a median group size of 132.
        ranked = np.sort(sizes)[::-1]
        pivot = int(ranked[min(min_groups, ranked.size) - 1]) if ranked.size else 0
        deepest = max(pivot - 1 if metric == "convergence" else pivot // 2, 1)

    # Group the rows once; `codes == group` inside the loop is an O(n_obs) scan per group, repeated n_draws * deepest times.
    members = group_rows(codes, len(keys))

    records = []
    for depth in range(1, deepest + 1):
        generator = np.random.default_rng(seed + depth)
        if isinstance(metric, str):
            values = np.array([METRICS[metric](profiles, members, depth, generator) for _ in range(n_draws)])
        else:
            values = np.array([metric(profiles, codes, depth, generator) for _ in range(n_draws)])
        if np.isnan(values).all():
            get_logger().info("replicate_saturation: no group has enough replicates at depth %d; stopping", depth)
            break
        records.append(
            {
                "n_replicates": depth,
                "mean": float(np.nanmean(values)),
                "std": float(np.nanstd(values)),
                "n_draws": int(np.isfinite(values).sum()),
            }
        )

    adata.uns.setdefault("mantispy", {})[key_added] = pd.DataFrame(
        records, columns=["n_replicates", "mean", "std", "n_draws"]
    )
    return None


@inplace_or_copy(expects=("well", "perturbation"))
def cytotoxicity(
    adata: AnnData,
    groupby: str = "Metadata_Perturbation",
    reference: str | None = "negcon",
    count_key: str = "Metadata_CellCount",
    site_key: str | None = "Metadata_SiteCount",
    distance_key: str = "hits_row_distance",
    min_viability: float = 0.7,
    key_added: str = "cytotoxicity",
    copy: bool = False,
) -> AnnData | None:
    """Flag perturbations that both lost cells and moved away from the controls.

    Args:
        adata: Profiles carrying a per-well cell count and a per-row distance from the controls.
        groupby: The column defining a perturbation.
        reference: Rows whose median cell count defines a viability of 1.0.
        count_key: ``obs`` column holding the cell count.
        site_key: ``obs`` column holding the number of fields of view that count covers.
            Where present, viability compares cells per field, so a well missing a field does not read as cell loss. ``None`` compares the counts as they are.
        distance_key: ``obs`` column holding the per-row distance from the controls, as written by :func:`~mantispy.tl.hit_calling`. Its group-level sibling ``hits_distance`` is one number repeated over each group's rows, so the median below would return the value it was handed.
        min_viability: Fraction of the control cell count below which a group counts as having lost cells.
        key_added: Name for the outputs.
        copy: Return a modified copy instead of mutating in place.

    Returns:
        ``None``, or the modified copy.
        Writes ``uns["mantispy"][key_added]`` with ``group``, ``viability``, ``distance``, ``n_obs`` and ``suspect``, and broadcasts ``obs[key_added + "_suspect"]``.

    Raises:
        KeyError: ``obs`` has no ``count_key`` or no ``distance_key``.
        ValueError: The reference rows have no usable cell count to normalize viability against.

    Notes:
        A group is suspect when its viability is below ``min_viability`` and its median distance is above that of the controls.
        Cell loss alone is a phenotype, and a large distance alone is a hit.
        Together they are suspect because a well with a fifth of its cells has a noisier median and drifts from the controls regardless of the biology.
        On a synthetic plate with one purely cytotoxic perturbation and its morphology effect removed, that perturbation's distance was 21.1 against 7.0 for the controls.

        Run it whichever feature block a hit was read off. Cell loss moves a profile away from the controls however it is measured, so a screen's most distant perturbations are partly a cytotoxicity ranking on CellProfiler features and on learned embeddings alike.

        Where the two differ is the geometry rather than the ranking. An embedding of the whole field encodes how full the well is, and on every trained model of :func:`~mantispy.ds.jump_lite` the cell count lands on the first component, while averaging per-cell measurements over a well leaves it as one signal among many. That costs distances, neighbourhoods and batch correction rather than this flag, and :doc:`/tutorials/multisite/learned_embeddings` measures both.

        The flag is a diagnostic and does not correct the distances.
        How much cytotoxicity confounds a screen varies.
        Over the pki dose series, the rank correlation between phenotype distance and cell loss is +0.79 (p < 1e-8) and the four strongest hits have viabilities of 0.27 to 0.68.
        Over rohban2017's ORF overexpression the same correlation is +0.00 (p = 0.95).
        Measure it on your own screen.

        The cell count is a baseline in its own right. Across three bioactivity benchmarks, a model given only the cell count often matched one given the whole Cell Painting profile, because many assays' actives simply lower it :cite:p:`Seal_2025`.
        Predicting two cytotoxicity readouts in hepatocytes, the profiles did no better than cell count, plate and well position on LDH release :cite:p:`Ewald_2026`.

    References:
        :cite:t:`Seal_2025`.
        :cite:t:`Ewald_2026`.
    """
    obs = as_frame(adata.obs)
    if count_key not in obs:
        raise KeyError(
            f"obs has no column {count_key!r}; mt.tl.aggregate and every well-level mt.ds dataset write "
            "Metadata_CellCount, or name another column with count_key="
        )
    if distance_key not in obs:
        raise KeyError(
            f"obs has no column {distance_key!r}; run mt.tl.hit_calling first, which writes "
            "obs['hits_row_distance'], or name another column"
        )

    is_control = reference_mask(adata, reference)
    # Pooled: cytotoxicity reads one screen-wide control level, where dose_direction reads one per plate.
    scored = viability(obs, count_key, site_key, is_control)
    distances = obs[distance_key].to_numpy(dtype=float)
    control_distance = float(np.nanmedian(distances[held_out_reference(adata, is_control, distance_key)]))
    if not np.isfinite(scored).any():
        raise ValueError(f"the reference rows have no usable {count_key!r} to normalize viability against")

    codes, keys = group_codes(adata, groupby)
    order, offsets = group_offsets(codes, len(keys))
    records = []
    for index, key in enumerate(keys):
        rows = order[offsets[index] : offsets[index + 1]]
        fraction = float(np.nanmedian(scored[rows]))
        distance = float(np.nanmedian(distances[rows]))
        records.append(
            {
                "group": str(key),
                "n_obs": int(rows.size),
                "viability": fraction,
                "distance": distance,
                "suspect": bool(fraction < min_viability and distance > control_distance),
            }
        )

    table = pd.DataFrame(records)
    adata.uns.setdefault("mantispy", {})[key_added] = table
    adata.obs[f"{key_added}_suspect"] = table.set_index("group")["suspect"].reindex(obs[groupby].astype(str)).to_numpy()
    get_logger().info("cytotoxicity flagged %d of %d groups as suspect", int(table["suspect"].sum()), len(table))
    return None
