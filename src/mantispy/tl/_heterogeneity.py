"""Single-cell heterogeneity beyond the well median."""

from __future__ import annotations

import warnings
from collections.abc import Sequence

import anndata as ad
import numpy as np
import pandas as pd
from anndata import AnnData

from mantispy._core._distance import pairwise_sqeuclidean
from mantispy._core._reduce import get_matrix, group_codes, group_offsets, representation
from mantispy._core._stats import (
    _default_replicate,
    _is_object_resolution,
    _split_wells,
    benjamini_hochberg,
    permutation_pvalue,
    split_reference,
    unit_permutation_pvalue,
)
from mantispy._core.features import annotation
from mantispy._core.frames import as_frame
from mantispy._core.logging import get_logger, report_drop
from mantispy._core.masks import control_annotated, reference_mask
from mantispy._core.mutation import inplace_or_copy
from mantispy._core.schema import stamp
from mantispy.tl._aggregate import _group_obs
from mantispy.tl._hits import _ks_replicate_null, ks_statistic

DNA_STATES = ("2N", "S", "4N")

_DISPERSION_MIN_CONTROLS = 8
_WELL = ("Metadata_Plate", "Metadata_Well")


def _cluster_labels(obs: pd.DataFrame, cluster_key: str) -> tuple[pd.Series, np.ndarray]:
    """The cluster column as strings, and which cells the clustering actually assigned.

    Read through here rather than ``astype(str)`` directly: a missing value becomes a cluster literally called ``"nan"`` on pandas 2, and makes ``sorted`` raise on pandas 3.
    """
    assigned = obs[cluster_key].notna().to_numpy()
    return obs[cluster_key].astype(str), assigned


def cluster_composition(
    adata: AnnData,
    cluster_key: str = "leiden",
    by: Sequence[str] | str = ("Metadata_Plate", "Metadata_Well"),
    reference: str | None = "negcon",
) -> AnnData:
    """Fraction of each well's cells in each cluster, as a well-level object.

    Args:
        adata: Clustered single-cell object.
        cluster_key: ``obs`` column holding the cluster label.
        by: Columns defining a row of the result, normally the well.
        reference: Controls to test each well's composition against, or ``None`` to skip the test.

    Returns:
        A new object with wells as rows and clusters as columns, holding the fraction of each well's cells in each cluster.
        A cell the clustering left unassigned is left out of the fractions, and a group in which no cell was assigned is left out altogether.
        It is a well-level mantispy object, so :func:`~mantispy.tl.map`, :func:`~mantispy.pp.normalize` and the plots accept it.
        ``uns["mantispy"]["composition_test"]`` holds a chi-square test of each well against the pooled control composition, with ``group``, ``statistic``, ``pvalue`` and ``qvalue``.
        ``uns["mantispy"]["composition_dispersion"]`` holds the factor the controls' own spread contributed, described below.

    Raises:
        KeyError: ``obs`` has no column ``cluster_key``.
        ValueError: No cell carries a cluster, so there is no composition to take.

    Notes:
        A well with few cells has a noisy composition.
        The chi-square test is computed on counts and accounts for this, but the fractions in ``X`` do not.
        Filter with :func:`~mantispy.pp.well_qc` first.

        Chi-square alone asks whether a well's cells are a multinomial draw from the control composition, and wells vary beyond that: seeding, position and edge effects all move a composition without any perturbation.
        The statistic is therefore divided by the dispersion the control wells show, ``mean(control statistic) / df``, floored at one.
        Without that correction, wells drawn from a single composition with mild jitter were called at q = 9e-10, 10 of 20 of them.

        The dispersion is estimated leave-one-out (each control well is scored against the pooled composition of the *other* controls, never one that includes itself), and the scaled statistic is referred to an F distribution rather than chi-square, so the uncertainty in that estimate widens the tail.
        Without either, the pure-null false positive rate ran above nominal and worse with fewer controls (0.138, 0.092, 0.070, 0.059 at 8, 16, 32, 64 control wells against a nominal 0.05); with both it sits at or below nominal.
        Below 8 wells the estimate is too noisy to trust and the function warns.
        Power falls accordingly: a composition shift of a few percentage points is not separable from well-to-well variation, and reporting it as significant was the bug.

        Clusters no control cell reached are left out of the test, since the controls give them no expected frequency.
        Their fractions stay in ``X``, and :func:`subpopulation_hits` compares within a cluster.

        ``Metadata_CellCount`` is every cell of the well, since :func:`~mantispy.tl.cytotoxicity` reads it to tell a hit from cell loss.
        The fractions are over the cells the clustering assigned, counted in ``Metadata_ClusteredCellCount``; the two differ when the clustering left cells out, so recovering the counts from ``X`` needs the latter.

        The test holds one row per well, in the order of the rows of the returned object.
        A well with no cells in the clusters the controls occupy gets ``NaN``.
        So does every well when the controls occupy fewer than two clusters, since a chi-square over a single category has no degrees of freedom.
        That case warns, because the table keeps its row per well and would otherwise read as no well's composition differing.
    """
    if cluster_key not in adata.obs:
        raise KeyError(f"obs has no column {cluster_key!r}; cluster first, e.g. sc.tl.leiden(adata)")

    columns = [by] if isinstance(by, str) else list(by)
    codes, keys = group_codes(adata, columns)
    named, assigned = _cluster_labels(as_frame(adata.obs), cluster_key)
    labels = sorted(named[assigned].unique())
    if not labels:
        raise ValueError(
            f"no cell carries a {cluster_key!r}, so there is no cluster to take a composition over. "
            "Run the clustering first, or name the column that holds it with cluster_key="
        )
    report_drop(
        "cell(s)",
        int((~assigned).sum()),
        int(assigned.size),
        remedy=f"they have no {cluster_key!r}, so they are left out of the fractions",
    )

    counts = np.zeros((len(keys), len(labels)))
    np.add.at(counts, (codes[assigned], pd.Categorical(named, categories=labels).codes[assigned]), 1)

    totals = counts.sum(axis=1)
    measured = totals > 0
    report_drop(
        "group(s)",
        int((~measured).sum()),
        int(measured.size),
        remedy=f"no cell in them carries a {cluster_key!r}, so they have no composition",
    )
    counts, totals = counts[measured], totals[measured]
    fractions = counts / totals[:, None]

    # Every cell, not only the clustered ones, or a group that merely lost cluster labels reads as cell loss in tl.cytotoxicity.
    cell_count = np.bincount(codes, minlength=len(keys)).astype(int)
    obs = _group_obs(adata, columns, keys, codes, {"Metadata_CellCount": cell_count})[measured]
    obs["Metadata_ClusteredCellCount"] = totals.astype(int)
    obs.index = pd.Index([str(row) for row in range(len(obs))])
    var = annotation(
        pd.Index(labels), object="Cluster", feature_group="Composition", feature=labels, feature_kind="derived"
    )

    result = ad.AnnData(X=fractions, obs=obs, var=var)
    stamp(result, resolution="well", grouped_by=columns)
    test, dispersion = _composition_test(result, counts, reference)
    result.uns["mantispy"]["composition_test"] = test
    result.uns["mantispy"]["composition_dispersion"] = dispersion
    return result


def _composition_test(composition: AnnData, counts: np.ndarray, reference: str | None) -> tuple[pd.DataFrame, float]:
    """Chi-square each well's cluster counts against the pooled control composition, at the scale the controls vary on."""
    from scipy.stats import chisquare, f

    empty = pd.DataFrame(columns=["group", "statistic", "pvalue", "qvalue"])
    if reference is None:
        return empty, np.nan
    if reference == "negcon" and not control_annotated(composition):
        get_logger().info("cluster_composition: no Metadata_Control_Type, so no composition test")
        return empty, np.nan

    is_control = reference_mask(composition, reference)
    if not is_control.any():
        return empty, np.nan

    # Flooring unreached clusters at epsilon instead gives p exactly zero and expected counts scipy refuses.
    pooled = counts[is_control].sum(axis=0)
    reached = pooled > 0
    comparable = int(reached.sum()) >= 2
    share = pooled[reached] / pooled[reached].sum()
    if not comparable:
        get_logger().warning(
            "cluster_composition: the controls occupy %d of %d clusters, so every p-value is NaN",
            int(reached.sum()),
            len(reached),
            deep="cluster on fewer components or at a higher resolution, or compare whole populations with "
            "mt.tl.hit_calling or mt.tl.edistance",
        )
    elif not reached.all():
        get_logger().debug(
            "cluster_composition: %d cluster(s) hold no control cells, so the composition test leaves them out",
            int((~reached).sum()),
        )

    labels = as_frame(composition.obs)
    naming = [column for column in ("Metadata_Plate", "Metadata_Well") if column in labels]
    names = (
        labels[naming].astype(str).agg("/".join, axis=1).to_numpy()
        if naming
        else composition.obs_names.astype(str).to_numpy()
    )

    dof = max(int(reached.sum()) - 1, 1)

    # A control well is in the pool it is scored against here, so its reported statistic is informational only.
    statistics = np.full(composition.n_obs, np.nan)
    for row in range(composition.n_obs):
        observed = counts[row][reached]
        if comparable and observed.sum() >= 1:
            statistics[row] = chisquare(observed, share * observed.sum()).statistic

    # Leave-one-out, because a control scored against a pool that includes itself biases the dispersion low.
    control_rows = np.flatnonzero(is_control)
    calibration = np.full(control_rows.size, np.nan)
    if comparable:
        for position, control_row in enumerate(control_rows):
            observed = counts[control_row][reached]
            others = pooled[reached] - counts[control_row][reached]
            if observed.sum() >= 1 and others.min() > 0:
                calibration[position] = chisquare(observed, others / others.sum() * observed.sum()).statistic
    calibration = calibration[np.isfinite(calibration)]

    # Below one the controls are tighter than multinomial; scaling down would only invent hits.
    dispersion = max(1.0, float(np.mean(calibration) / dof)) if calibration.size else np.nan
    if comparable and calibration.size < _DISPERSION_MIN_CONTROLS:
        get_logger().warning(
            "cluster_composition: the dispersion comes from only %d control well(s), so p-values are too small",
            calibration.size,
            deep=f"use at least {_DISPERSION_MIN_CONTROLS} control wells, or read the statistic as a ranking",
        )

    # Quasi-likelihood F-test, since the dispersion is estimated; each finite control statistic adds ~dof denominator df.
    residual_df = int(calibration.size) * dof
    pvalues = (
        f.sf(statistics / (dof * dispersion), dof, residual_df)
        if np.isfinite(dispersion)
        else np.full(len(statistics), np.nan)
    )
    table = pd.DataFrame(
        {"group": names.astype(str), "statistic": statistics, "pvalue": np.where(np.isnan(statistics), np.nan, pvalues)}
    )
    table["qvalue"] = benjamini_hochberg(table["pvalue"].to_numpy()) if len(table) else []
    return table, dispersion


def _on_dna_stain(adata: AnnData) -> np.ndarray:
    """Per feature, whether its name or ``var["channel"]`` names a DNA stain: DNA, DAPI or Hoechst."""
    pattern = "dna|dapi|hoechst"
    stain = adata.var_names.str.lower().str.contains(pattern)
    if "channel" in adata.var:
        stain |= as_frame(adata.var)["channel"].astype(str).str.lower().str.contains(pattern).to_numpy()
    return np.asarray(stain, dtype=bool)


def _dna_intensity(adata: AnnData, dna_feature: str | None, layer: str | None, caller: str) -> np.ndarray:
    """The integrated DNA intensity of every cell, refusing values that are mostly not positive."""
    if dna_feature is None:
        names = adata.var_names
        candidates = list(
            names[_on_dna_stain(adata) & names.str.contains("IntegratedIntensity") & ~names.str.contains("Edge")]
        )
        if not candidates:
            raise KeyError(
                "could not find an integrated DNA intensity feature; pass dna_feature= explicitly "
                "(the usual name is Nuclei_Intensity_IntegratedIntensity_DNA)"
            )
        # The nucleus is where DNA is measured best, so its intensity wins over a whole-cell one.
        dna_feature = min(candidates, key=lambda name: not name.startswith("Nuclei_"))
        get_logger().info("%s using %s", caller, dna_feature)

    values = get_matrix(adata, layer)[:, adata.var_names.get_loc(dna_feature)].astype(np.float64)
    positive = np.isfinite(values) & (values > 0)
    if positive.sum() < 0.95 * adata.n_obs:
        raise ValueError(
            f"{int((~positive).sum())} of {adata.n_obs} values of {dna_feature!r} are not positive, so their "
            f"log cannot be fitted. {caller} needs the raw intensities; run it before mt.pp.normalize, "
            "or keep them with normalize(keep_raw=True) and pass layer='raw'."
        )
    return values


def _g1_peak(log_values: np.ndarray) -> float:
    """Where one group's G1 cells sit on a log2 DNA axis: its highest density peak, or the peak one doubling below that."""
    from scipy.stats import gaussian_kde

    low, high = np.percentile(log_values, [0.5, 99.5])
    if high <= low:
        return float(np.median(log_values))
    grid = np.linspace(low, high, 512)
    density = gaussian_kde(log_values)(grid)
    top = int(np.argmax(density))
    # A well arrested in G2 has its highest peak at 4N; a substantial peak one doubling below it is then the G1 peak.
    peaks = np.flatnonzero((density[1:-1] > density[:-2]) & (density[1:-1] >= density[2:])) + 1
    below = peaks[(np.abs(grid[peaks] - grid[top] + 1) <= 0.25) & (density[peaks] >= 0.1 * density[top])]
    if below.size:
        top = int(below[np.argmax(density[below])])
    return float(grid[top])


@inplace_or_copy(expects="object")
def dna_content(
    adata: AnnData,
    dna_feature: str | None = None,
    by: Sequence[str] | str | None = ("Metadata_Plate", "Metadata_Well"),
    layer: str | None = None,
    min_cells: int = 50,
    reference: str | None = None,
    key_added: str = "Metadata_DNAContent",
    copy: bool = False,
) -> AnnData | None:
    """DNA content of each cell, in doublings of the G1 peak of its group.

    The integrated DNA intensity is log2-transformed, and the G1 (2N) peak of each ``by`` group is subtracted.
    A cell with a 2N genome then sits near 0 and a cell with a 4N genome near 1, whatever the staining intensity of its well.

    Args:
        adata: Single-cell object holding raw, unnormalized intensities.
        dna_feature: The integrated DNA intensity feature.
            Found when omitted as an integrated intensity, not its edge, whose name or ``var["channel"]`` names DNA, DAPI or Hoechst, the ``Nuclei_`` one first.
        by: Find the G1 peak separately within each group, normally the well, since staining intensity does not carry across wells.
        layer: Read this layer instead of ``X``, for example ``"raw"`` after ``normalize(keep_raw=True)``.
        min_cells: A group with fewer usable cells than this, or fewer ``reference`` cells when given, gets no G1
            peak, and its cells get ``NaN``.
        reference: Find each group's G1 peak on these cells only, and apply it to every cell of the group:
            ``"negcon"`` for the negative controls, or the name of a boolean ``obs`` column. ``None`` uses every cell.
            Pass it when a treatment can arrest whole wells, so their peak is not taken for G1; ``by`` must then
            group treated and control cells together, normally by plate, or by well in a pooled screen.
        key_added: ``obs`` column written.
        copy: Return a modified copy instead of mutating in place.

    Returns:
        ``None``, or the modified copy.
        Writes ``obs[key_added]``, and ``uns["mantispy"]["dna_content"]`` with one row per group: the ``by`` columns, ``n_cells`` (the usable cells the peak was found on) and ``g1_peak`` (log2 of the integrated intensity at the G1 peak, ``NaN`` for a group below ``min_cells``).

    Raises:
        KeyError: ``dna_feature`` was not given and no integrated DNA intensity feature could be found.
        ValueError: More than 5% of the feature's values are not positive, so their log cannot be taken.

    Notes:
        The G1 peak is the highest peak of a Gaussian kernel density estimate of the group's log2 intensities.
        In a population mostly in G2, such as a well arrested in mitosis, the highest peak is the 4N one.
        So when a second peak lies one doubling below the highest, within a quarter of a doubling, and is at least a tenth as high, that lower peak is taken as G1.
        A sub-G1 peak of fragmented nuclei at half the G1 intensity would be mistaken for G1 by the same rule; check ``g1_peak`` across wells.

        The intensity is used as measured.
        CellProfiler's integrated intensity carries no background subtraction, so a strong background lowers the ratio between the 4N and 2N peaks below two.
        Subtract it before calling this if your pipeline measured it.

        A cell whose intensity is missing, zero or negative gets ``NaN``, as does every cell of a group below ``min_cells``.

        A stain that does not resolve a 4N peak from the 2N one gives a single broad peak, and its mode is not a G1 peak.
        The content then still ranks cells by DNA, but :func:`~mantispy.tl.dna_state` gates do not mean 2N, S and 4N; look at the histogram first.
    """
    values = _dna_intensity(adata, dna_feature, layer, "dna_content")
    positive = np.isfinite(values) & (values > 0)
    report_drop(
        "cell(s)",
        int((~positive).sum()),
        adata.n_obs,
        remedy="their DNA intensity is missing or not positive, so their DNA content is NaN",
    )
    log_values = np.full(adata.n_obs, np.nan)
    log_values[positive] = np.log2(values[positive])

    codes, keys = group_codes(adata, by)
    order, offsets = group_offsets(codes, len(keys))
    fit_on = reference_mask(adata, reference)
    content = np.full(adata.n_obs, np.nan)
    peaks = np.full(len(keys), np.nan)
    counts = np.zeros(len(keys), dtype=int)
    for index in range(len(keys)):
        rows = order[offsets[index] : offsets[index + 1]]
        usable = rows[positive[rows]]
        anchors = usable[fit_on[usable]]
        counts[index] = anchors.size
        if anchors.size < min_cells:
            continue
        peaks[index] = _g1_peak(log_values[anchors])
        content[usable] = log_values[usable] - peaks[index]

    small = counts < min_cells
    if small.any():
        counted = "usable cells" if reference is None else f"usable {reference!r} reference cells"
        hint = (
            ""
            if reference is None
            else ", and by= must group the reference with the other cells, e.g. by='Metadata_Plate'"
        )
        get_logger().warning(
            "dna_content: %d of %d group(s) have fewer than %d %s, too few to place a G1 peak (%s)",
            int(small.sum()),
            len(keys),
            min_cells,
            counted,
            ", ".join(map(str, list(keys[small])[:5])),
            deep=f"their cells are left as NaN{hint}",
        )

    columns = ["group"] if by is None else ([by] if isinstance(by, str) else list(by))
    table = pd.DataFrame([key if isinstance(key, tuple) else (key,) for key in keys], columns=columns)
    table["n_cells"] = counts
    table["g1_peak"] = peaks
    adata.obs[key_added] = content
    adata.uns.setdefault("mantispy", {})["dna_content"] = table
    return None


@inplace_or_copy(expects="object")
def dna_state(
    adata: AnnData,
    key: str = "Metadata_DNAContent",
    cutoffs: tuple[float, float] = (1.25, 1.75),
    key_added: str = "Metadata_DNAState",
    copy: bool = False,
) -> AnnData | None:
    """Call each cell 2N, S or 4N from fixed gates on its DNA content.

    Args:
        adata: Single-cell object carrying the output of :func:`~mantispy.tl.dna_content`.
        key: ``obs`` column holding the DNA content, in doublings of the G1 peak.
        cutoffs: The lower and upper gate, in multiples of the G1 peak intensity.
            A cell below the lower gate is 2N, at or above the upper gate 4N, and in between S.
            The defaults sit a quarter of the way from each peak towards the other, the usual gates on a DNA content histogram.
        key_added: ``obs`` column written.
        copy: Return a modified copy instead of mutating in place.

    Returns:
        ``None``, or the modified copy.
        Writes ``obs[key_added]`` as a categorical over ``DNA_STATES``, ``NaN`` where the DNA content is ``NaN``.
        Writes ``uns["mantispy"]["dna_state"]`` with the ``key`` read and the ``cutoffs`` applied, so the call can be reproduced.

    Raises:
        KeyError: ``obs`` has no column ``key``.
        ValueError: ``cutoffs`` is not two finite, positive, increasing values.

    Notes:
        Fixed gates cannot tell G0 from G1, or a late-S cell from a G2 cell whose stain is a little dim.
        A cell above 4N, from polyploidy or a doublet, is called 4N.
    """
    if key not in adata.obs:
        raise KeyError(f"obs has no column {key!r}; run mt.tl.dna_content first")
    gates = np.asarray(cutoffs, dtype=float)
    if gates.shape != (2,) or not np.isfinite(gates).all() or gates[0] <= 0 or gates[0] >= gates[1]:
        raise ValueError(f"cutoffs must be two finite, positive, increasing values, got {cutoffs!r}")

    content = as_frame(adata.obs)[key].to_numpy(dtype=float)
    codes = np.digitize(content, np.log2(gates))
    codes[np.isnan(content)] = -1
    adata.obs[key_added] = pd.Categorical.from_codes(codes, categories=pd.Index(DNA_STATES))
    adata.uns.setdefault("mantispy", {})["dna_state"] = {"key": key, "cutoffs": gates}
    return None


@inplace_or_copy(expects="object")
def subpopulation_hits(
    adata: AnnData,
    cluster_key: str = "leiden",
    groupby: str = "Metadata_Perturbation",
    reference: str | None = "negcon",
    use_rep: str | None = None,
    min_cells: int = 3,
    seed: int = 0,
    key_added: str = "subpopulation_hits",
    copy: bool = False,
    *,
    replicate: str | Sequence[str] | None = None,
    n_permutations: int = 1000,
) -> AnnData | None:
    """Test each perturbation against the controls within each cluster.

    An effect confined to one cell state is diluted by the other states in a well median.
    Comparing cells within a cluster avoids that.

    Args:
        adata: Clustered single-cell object.
        cluster_key: ``obs`` column holding the cluster label.
        groupby: ``obs`` column holding the perturbation.
        reference: Which rows are the controls, ``"negcon"`` or the name of a boolean ``obs`` column.
        use_rep: Measure in ``obsm[use_rep]`` instead of ``X``.
        min_cells: Skip a (cluster, group) pair with fewer cells than this on either side.
            A cluster needs twice as many controls, and at least four, since half of them place the centroid and half supply the distances tested against.
            The reference group's own row needs four times as many, since it comes from a second split of the held-out half.
        seed: Seed for the split of a cluster's controls and the permutation null.
        key_added: Name for the output table.
        copy: Return a modified copy instead of mutating in place.
        replicate: ``obs`` column, or sequence of columns, whose groups are the design's exchangeable unit: the well in an arrayed screen, the guide (``"Metadata_sgRNA"``) in a pooled one.
            The permutation null then draws whole replicates rather than cells, since cells within a well are not independent replicates (see Notes).
            Left ``None``, it defaults to the physical well on an object stamped cell resolution with replicated wells; it warns and falls back to the analytic KS approximation when no complete well column is present, and raises on a pooled design, where some well holds both reference and tested cells.
        n_permutations: Size of the replicate permutation null.
            Ignored on the analytic fallback, which has no permutation draw.

    Returns:
        ``None``, or the modified copy.
        Writes ``uns["mantispy"][key_added]`` with ``cluster``, ``group``, ``n_cells``, ``statistic`` (KS), ``pvalue`` and ``qvalue``.

    Raises:
        KeyError: ``obs`` has no column ``cluster_key``.
        ValueError: ``replicate`` is not given on a pooled design at cell resolution.

    Notes:
        Each cell is reduced to its Euclidean distance from the control centroid of its own cluster, and a KS test compares the treated cells' distances with the controls'.
        This is close to ``hit_calling(method="ks")``, restricted to comparable cells.
        A distance is used rather than a single feature so that the test means the same on every dataset.

        Cells within a well are not independent replicates, so the p-value comes from a permutation over whole replicates: whole wells, or whatever ``replicate`` names, are drawn for the pseudo-treated side and the KS statistic recomputed, the same unit :func:`~mantispy.tl.hit_calling` resamples.
        Without a usable well column at cell resolution it warns and falls back to the analytic KS approximation on cell counts, which is anti-conservative for the same reason.
        In a pooled screen every well holds every perturbation, so drawing whole wells cannot separate a group from the reference; the exchangeable unit is the guide, and the call raises until ``replicate`` names it.

        A cluster's controls are split in half, as in :func:`~mantispy.tl.hit_calling`.
        One half places the centroid and the other supplies the distances tested against, so the null is out of sample.
        Controls measured against a centroid they defined themselves sit closer to it than any other group can.
        That bias grows with features per control, which is the shape of real Cell Painting data.
        On pure noise with few controls and many features, a null drawn from the rows that placed the centroid calls far more pseudo-treatments than the nominal rate at raw ``p < 0.05``, and the split stays near it.

        The controls carry a perturbation label of their own, so one row of the table is the reference group against itself.
        That row is computed from the held-out half alone, split once more, at random, so that the cells tested and the cells they are tested against are different cells.
        It is therefore a draw from the null, rather than a sample compared with part of itself measured against a centre half of it placed.
        A cluster with too few controls to split twice has no such row.
    """
    if cluster_key not in adata.obs:
        raise KeyError(f"obs has no column {cluster_key!r}; cluster first, e.g. sc.tl.leiden(adata)")

    values = representation(adata, use_rep)
    obs = as_frame(adata.obs)
    is_control = reference_mask(adata, reference)
    named, assigned = _cluster_labels(obs, cluster_key)
    clusters = named.where(assigned).to_numpy()
    groups = obs[groupby].astype(str).to_numpy()

    replicate_codes = _default_replicate(adata, replicate=replicate, reference=is_control)
    if replicate_codes is None and replicate is None and _is_object_resolution(adata):
        warnings.warn(
            "replicate=None at cell resolution treats cells as independent replicates, which makes p-values too "
            "small. Pass replicate= a well column, or aggregate with mt.tl.aggregate.",
            UserWarning,
            stacklevel=3,
        )

    generator = np.random.default_rng(seed)
    # A child generator, so halving the reference group leaves the stream the cluster splits draw from untouched.
    half_generator = generator.spawn(1)[0]
    needed = max(2 * min_cells, 4)

    records: list[dict[str, object]] = []
    for cluster in pd.unique(clusters):
        in_cluster = np.flatnonzero(clusters == cluster)
        controls = in_cluster[is_control[in_cluster]]
        if controls.size < needed:
            continue

        fit_rows, null_rows = split_reference(controls, generator)
        centre = np.nanmean(values[fit_rows], axis=0, keepdims=True)
        distance = np.sqrt(pairwise_sqeuclidean(np.nan_to_num(values[in_cluster]), np.nan_to_num(centre))).ravel()
        fitted = np.isin(in_cluster, fit_rows)
        held_out = is_control[in_cluster] & ~fitted
        cluster_replicates = replicate_codes[in_cluster] if replicate_codes is not None else None

        for group in pd.unique(groups[in_cluster]):
            in_group = (groups[in_cluster] == group) & ~fitted
            shared = np.flatnonzero(in_group & held_out)
            if cluster_replicates is not None:
                # Halve the reference group by whole wells so its null draws a strict subset like any other group.
                in_group[shared[_split_wells(cluster_replicates[shared], half_generator)]] = False
            else:
                # The reference group's held-out cells are halved at random because cells are ordered by plate and well.
                in_group[half_generator.permutation(shared)[: shared.size // 2]] = False
            control_mask = held_out & ~in_group
            treated, control_distance = distance[in_group], distance[control_mask]
            if treated.size < min_cells or control_distance.size < min_cells:
                continue
            statistic = float(ks_statistic(treated[None, :], control_distance)[0])
            if cluster_replicates is not None:
                pool = np.flatnonzero(in_group | control_mask)
                null = _ks_replicate_null(
                    distance,
                    cluster_replicates,
                    pool,
                    np.flatnonzero(in_group),
                    control_distance,
                    n_permutations,
                    seed,
                    len(records),
                )
                pvalue = float(permutation_pvalue(np.array([statistic]), null[None, :])[0])
            else:
                # Two-sample KS p-value, the asymptotic form scipy uses for large samples.
                n, m = treated.size, control_distance.size
                effective = np.sqrt(n * m / (n + m))
                pvalue = float(min(1.0, 2.0 * np.exp(-2.0 * (effective * statistic) ** 2)))
            records.append(
                {
                    "cluster": str(cluster),
                    "group": str(group),
                    "n_cells": int(treated.size),
                    "statistic": statistic,
                    "pvalue": pvalue,
                }
            )

    table = pd.DataFrame(records, columns=["cluster", "group", "n_cells", "statistic", "pvalue"])
    if table.empty or int(table.groupby("cluster")["group"].nunique().max()) <= 1:
        get_logger().warning(
            "subpopulation_hits: no cluster held %d controls and %d cells of another group, so nothing was compared",
            needed,
            min_cells,
            deep="cluster on fewer components, or test whole populations with mt.tl.edistance",
        )
    table["qvalue"] = benjamini_hochberg(table["pvalue"].to_numpy()) if len(table) else []
    adata.uns.setdefault("mantispy", {})[key_added] = table
    return None


@inplace_or_copy(expects="object")
def composition_shift(
    adata: AnnData,
    key: str,
    groupby: str = "Metadata_Perturbation",
    reference: str = "negcon",
    block: Sequence[str] | str | None = _WELL,
    replicate: str | None = "Metadata_sgRNA",
    n_permutations: int = 999,
    seed: int = 0,
    key_added: str = "composition_shift",
    copy: bool = False,
) -> AnnData | None:
    """Change in each group's share of cells per category against the reference, compared within blocks.

    The category is a label per cell, such as a cell-cycle state or a cluster. Within each ``block`` (a well on a plate),
    the group's share of cells in every category is compared with the reference's, and the differences are
    averaged over blocks. The distance is the total variation distance of that average, half the sum of its
    absolute values: 0 when the group's composition matches the reference, 1 when they share no category.

    The p-value comes from relabelling ``replicate`` (the guide) as group or reference, one relabelling per
    permutation applied in every block, so the guide is the replicate even though every well of a pooled
    screen holds every guide. For one row per guide, pass the guide column as both ``groupby`` and ``replicate``:
    each guide is then compared with the reference guides.

    Args:
        adata: Object-resolution object carrying the category in ``obs[key]``.
        key: ``obs`` column holding a label per cell. A cell without a value is left out.
        groupby: ``obs`` column naming the groups to score, one row of the result each.
        reference: The rows to compare against: ``"negcon"`` for the negative controls, or the name of a
            boolean ``obs`` column. The reference has no row of its own.
        block: ``obs`` column or columns the comparison is paired within, normally the plate and well, so the same
            well position on two plates stays two blocks; ``None`` pools all rows. The default uses whichever of the
            two columns the object has.
        replicate: ``obs`` column naming the independent replicate the permutation relabels, normally the guide.
            ``None`` relabels single cells, which treats every cell as independent and warns.
            Each replicate must belong to one group.
        n_permutations: Permutations for the null.
        seed: Seed for the permutations.
        key_added: Key the table is written under in ``uns["mantispy"]``.
        copy: Return a modified copy instead of mutating in place.

    Returns:
        ``None``, or the modified copy.
        Writes ``uns["mantispy"][key_added]`` with one row per group: ``group``, the ``distance``, ``n_blocks``
        that held both the group and the reference, ``pvalue``, the Benjamini-Hochberg ``qvalue`` over the
        groups, ``min_pvalue``, the smallest p-value the design could reach, and one column per category with
        the group's share minus the reference's.

    Raises:
        KeyError: ``obs`` lacks a ``block`` column.
        ValueError: ``n_permutations`` is not positive, ``reference`` selects no cells, a ``block`` column holds
            missing values, or a ``replicate`` spans more than one group.
    """
    if n_permutations < 1:
        raise ValueError(f"n_permutations must be positive, got {n_permutations}")
    obs = as_frame(adata.obs)
    category = obs[key].astype(object).where(obs[key].notna())
    labels = sorted(category.dropna().unique(), key=str)
    known = category.notna().to_numpy()
    onehot = np.full((adata.n_obs, len(labels)), np.nan)
    onehot[known] = (category[known].to_numpy()[:, None] == np.asarray(labels, dtype=object)[None, :]).astype(float)

    reference_rows = reference_mask(adata, reference)
    if not reference_rows.any():
        raise ValueError(f"reference={reference!r} selects no cells to compare against")
    if block == _WELL:  # an object without plates pairs within wells alone
        block = [column for column in _WELL if column in obs] or list(_WELL)
    blocks = group_codes(adata, block)[0]
    if replicate is None:
        warnings.warn(
            "replicate=None treats cells as independent replicates, which makes p-values far too small. Pass "
            "the guide column as replicate=.",
            UserWarning,
            stacklevel=3,
        )
        units = np.arange(adata.n_obs).astype(str)
    else:
        spans = obs.groupby(replicate, observed=True)[groupby].nunique()
        if (spans > 1).any():
            raise ValueError(
                f"{int((spans > 1).sum())} values of {replicate!r} span more than one group of {groupby!r}"
            )
        units = obs[replicate].astype(str).to_numpy()
    rng = np.random.default_rng(seed)

    def mean_difference(sums: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
        return np.mean([group / group.sum() - reference / reference.sum() for group, reference in sums], axis=0)

    def distance(sums: list[tuple[np.ndarray, np.ndarray]]) -> float:
        return 0.5 * float(np.abs(mean_difference(sums)).sum())

    values = obs[groupby].to_numpy()
    rows = []
    for group in pd.unique(values[~reference_rows]):
        group_rows = (values == group) & ~reference_rows
        observed = [
            (onehot[group_rows & in_block & known].sum(axis=0), onehot[reference_rows & in_block & known].sum(axis=0))
            for in_block in (blocks == value for value in pd.unique(blocks))
        ]
        observed = [(g, r) for g, r in observed if g.sum() > 0 and r.sum() > 0]
        if not observed:
            rows.append({"group": group, "distance": np.nan, "n_blocks": 0, "pvalue": np.nan, "min_pvalue": np.nan})
            continue
        pvalue, floor = unit_permutation_pvalue(
            onehot, group_rows, reference_rows, blocks, units, distance, n_permutations, rng
        )
        rows.append(
            {
                "group": group,
                "distance": distance(observed),
                "n_blocks": len(observed),
                "pvalue": pvalue,
                "min_pvalue": floor,
                **dict(zip(labels, mean_difference(observed), strict=True)),
            }
        )

    table = pd.DataFrame(rows, columns=["group", "distance", "n_blocks", "pvalue", "min_pvalue", *labels])
    table.insert(4, "qvalue", benjamini_hochberg(table["pvalue"].to_numpy()))
    adata.uns.setdefault("mantispy", {})[key_added] = table
    return None


@inplace_or_copy(expects="object")
def neighbors_local_density(
    adata: AnnData,
    k: int = 15,
    by: str = "Metadata_ImageID",
    key_added: str = "Metadata_LocalDensity",
    copy: bool = False,
) -> AnnData | None:
    """Local cell density: how many cells per unit area surround each cell in its field of view.

    The density is the ``k``-nearest-neighbour estimate, ``k`` divided by the area of the disc reaching the ``k``-th nearest cell, in cells per square unit of ``Metadata_Center_X`` and ``Metadata_Center_Y`` (usually pixels).
    Higher means more crowded.
    Local crowding changes morphology independently of treatment.
    Consider regressing it out with :func:`~mantispy.pp.regress_out` before attributing a change to a perturbation.

    Args:
        adata: Single-cell object carrying ``Metadata_Center_X`` and ``Metadata_Center_Y``.
        k: Number of neighbors whose disc sets the density; a field with fewer cells uses all of them.
        by: ``obs`` column identifying the field of view.
            Neighbors are searched within each field only, since coordinates from different images are not comparable.
        key_added: ``obs`` column written.
        copy: Return a modified copy instead of mutating in place.

    Returns:
        ``None``, or the modified copy.
        Writes ``obs[key_added]``, and a cell alone in its field, or one sharing its position with all of its neighbours, gets ``NaN``.

    Raises:
        KeyError: ``obs`` is missing ``Metadata_Center_X`` or ``Metadata_Center_Y``.
    """
    from sklearn.neighbors import NearestNeighbors

    needed = {"Metadata_Center_X", "Metadata_Center_Y"}
    if not needed <= set(adata.obs.columns):
        raise KeyError(f"obs needs {sorted(needed)}; read with mt.io.read_profiles, which keeps them")

    coordinates = as_frame(adata.obs)[["Metadata_Center_X", "Metadata_Center_Y"]].to_numpy(dtype=float)
    codes, keys = group_codes(adata, by)
    density = np.full(adata.n_obs, np.nan)

    order, offsets = group_offsets(codes, len(keys))
    for index in range(len(keys)):
        rows = order[offsets[index] : offsets[index + 1]]
        if rows.size < 2:
            continue
        neighbours = min(k + 1, rows.size)
        distances, _ = NearestNeighbors(n_neighbors=neighbours).fit(coordinates[rows]).kneighbors(coordinates[rows])
        # k-nearest-neighbour density estimate: the cells counted over the area of the disc that reaches them.
        reach = distances[:, -1]
        with np.errstate(divide="ignore"):
            density[rows] = np.where(reach > 0, (neighbours - 1) / (np.pi * reach**2), np.nan)

    adata.obs[key_added] = density
    if np.isnan(density).any():
        get_logger().debug(
            "neighbors_local_density: %d cell(s) have no measurable neighbourhood in their field, left as NaN",
            int(np.isnan(density).sum()),
        )
    return None
