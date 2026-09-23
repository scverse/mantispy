"""Single-cell heterogeneity beyond the well median.

A perturbation that strongly shifts ten percent of cells looks like a small change in the average.
These tools work on the distribution of cells instead, so they need single-cell profiles.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence

import anndata as ad
import numpy as np
import pandas as pd
from anndata import AnnData

from mantispy._core._distance import pairwise_sqeuclidean
from mantispy._core._reduce import get_matrix, group_codes, group_offsets, representation
from mantispy._core._stats import benjamini_hochberg, split_reference
from mantispy._core.features import empty_annotation
from mantispy._core.frames import as_frame
from mantispy._core.logging import get_logger, report_drop
from mantispy._core.masks import reference_mask
from mantispy._core.mutation import inplace_or_copy
from mantispy._core.schema import stamp
from mantispy.tl._aggregate import _group_obs
from mantispy.tl._hits import ks_statistic

PHASES = ("G1", "S", "G2M")

#: Control wells needed before the spread between them estimates the dispersion the composition test divides by.
#: Below this the estimate is itself noise, and the test stays anti-conservative.
_DISPERSION_MIN_CONTROLS = 8


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
        A cell the clustering left unassigned is left out of both the fraction and the count, since it belongs to no cluster.
        It is a well-level mantispy object, so :func:`~mantispy.tl.map`, :func:`~mantispy.pp.normalize` and the plots accept it.
        ``uns["mantispy"]["composition_test"]`` holds a chi-square test of each well against the pooled control composition, with ``group``, ``statistic``, ``pvalue`` and ``qvalue``.
        ``uns["mantispy"]["composition_dispersion"]`` holds the factor the controls' own spread contributed, described below.

    Raises:
        KeyError: ``obs`` has no column ``cluster_key``.

    Notes:
        A well with few cells has a noisy composition.
        The chi-square test is computed on counts and accounts for this, but the fractions in ``X`` do not.
        Filter with :func:`~mantispy.pp.well_qc` first.

        Chi-square alone asks whether a well's cells are a multinomial draw from the control composition, and wells
        vary beyond that: seeding, position and edge effects all move a composition without any perturbation.
        The statistic is therefore divided by the dispersion the control wells show, ``mean(control statistic) / df``,
        floored at one. Without that correction, wells drawn from a single composition with mild jitter were called
        at q = 9e-10, 10 of 20 of them.

        The correction is only as good as the dispersion estimate. With 16 control wells the false positive rate ran
        near 0.10 against a nominal 0.05 in simulation, and with 32 it ran near 0.06; below
        ``_DISPERSION_MIN_CONTROLS`` wells the function warns. Power falls accordingly: a composition shift of a few
        percentage points is not separable from well-to-well variation, and reporting it as significant was the bug.

        Clusters no control cell reached are left out of the test, since the controls give them no expected frequency.
        Their fractions stay in ``X``, and :func:`subpopulation_hits` compares within a cluster.

        The test holds one row per well, in the order of the rows of the returned object.
        A well with no cells in the clusters the controls occupy gets ``NaN``.
        So does every well when the controls occupy fewer than two clusters, since a chi-square over a single category has no degrees of freedom.
        That case warns, because the table keeps its row per well and would otherwise read as no well's composition differing.
    """
    if cluster_key not in adata.obs:
        raise KeyError(f"obs has no column {cluster_key!r}; cluster first, e.g. sc.tl.leiden(adata)")

    columns = [by] if isinstance(by, str) else list(by)
    codes, keys = group_codes(adata, columns)
    clusters = as_frame(adata.obs)[cluster_key]
    # A cell the clustering left unassigned belongs to no cluster. Dropping the missing values before the
    # labels are read keeps it from becoming a cluster of its own: sorted() cannot order NaN against the
    # names and raises on pandas 3, while on pandas 2 astype(str) first turned it into a cluster literally
    # called "nan". The same drop is owed to subpopulation_hits, which still reads its labels this way.
    assigned = clusters.notna().to_numpy()
    named = clusters.astype(str)
    labels = sorted(named[assigned].unique())
    report_drop(
        "cell(s)",
        int((~assigned).sum()),
        int(assigned.size),
        remedy=f"they have no {cluster_key!r}, so they are left out of the fractions",
    )
    if not labels:
        raise ValueError(
            f"no cell carries a {cluster_key!r}, so there is no cluster to take a composition over. "
            "Run the clustering first, or name the column that holds it with cluster_key="
        )

    counts = np.zeros((len(keys), len(labels)))
    # An unassigned cell stringifies to a name no label matches, so its code is already -1 and the
    # mask below drops it; there is nothing left for a `.where` to do.
    membership = pd.Categorical(named, categories=labels).codes
    np.add.at(counts, (codes[assigned], membership[assigned]), 1)
    totals = counts.sum(axis=1, keepdims=True)
    # A well none of whose cells were assigned was not measured, so its composition is unknown.
    # Zero in every cluster would say it was measured and found empty everywhere.
    fractions = np.divide(counts, totals, out=np.full_like(counts, np.nan), where=totals > 0)

    # Every cell in the well, not only the clustered ones: Metadata_CellCount is tl.cytotoxicity's
    # default count_key, and a well that merely lost cluster labels must not read as cell loss.
    cell_count = np.bincount(codes, minlength=len(keys)).astype(int)
    obs = _group_obs(adata, columns, keys, codes, {"Metadata_CellCount": cell_count})
    obs.index = pd.Index([str(row) for row in range(len(obs))])
    # A cluster fraction is not a CellProfiler measurement, so the schema's annotation columns come
    # from the same helper every such object uses, and the three that mean something here are set.
    var = empty_annotation(pd.Index(labels))
    # pd.Categorical, not the bare value: df[column] = value replaces the column, which would drop
    # the category dtype empty_annotation supplies these columns in.
    var["object"] = pd.Categorical(["Cluster"] * len(labels))
    var["feature_group"] = pd.Categorical(["Composition"] * len(labels))
    var["feature"] = pd.Categorical(labels)

    result = ad.AnnData(X=fractions.astype(np.float32), obs=obs, var=var)
    stamp(result, resolution="well")
    test, dispersion = _composition_test(result, counts, reference)
    result.uns["mantispy"]["composition_test"] = test
    result.uns["mantispy"]["composition_dispersion"] = dispersion
    return result


def _composition_test(composition: AnnData, counts: np.ndarray, reference: str | None) -> tuple[pd.DataFrame, float]:
    """Chi-square each well's cluster counts against the pooled control composition, at the scale the controls vary on."""
    from scipy.stats import chi2, chisquare

    empty = pd.DataFrame(columns=["group", "statistic", "pvalue", "qvalue"])
    if reference is None:
        return empty, np.nan
    if reference == "negcon" and "Metadata_Control" not in composition.obs:
        get_logger().info("cluster_composition: no Metadata_Control, so no composition test")
        return empty, np.nan

    is_control = reference_mask(composition, reference)
    if not is_control.any():
        return empty, np.nan

    # A cluster no control cell reached has no expected frequency.
    # Flooring it at epsilon made a single treated cell there a chi-square of 1e10 and p exactly zero, and with enough such clusters the expected counts stopped summing to the observed ones, which scipy refuses.
    pooled = counts[is_control].sum(axis=0)
    reached = pooled > 0
    comparable = int(reached.sum()) >= 2
    share = pooled[reached] / pooled[reached].sum()
    if not comparable:
        warnings.warn(
            f"the controls occupy {int(reached.sum())} of {len(reached)} clusters, so the composition test has no "
            "second category to set a well against and every statistic, pvalue and qvalue is NaN. The clustering "
            "is following the perturbation rather than a cell state the conditions share: cluster on fewer "
            "components so that they mix, or at a higher resolution so that the controls' own cluster splits into "
            "sub-states. To compare whole populations instead, use mt.tl.hit_calling or mt.tl.edistance.",
            UserWarning,
            stacklevel=3,
        )
    elif not reached.all():
        get_logger().info(
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

    statistics = np.full(composition.n_obs, np.nan)
    for row in range(composition.n_obs):
        observed = counts[row][reached]
        # A well with no cells in the clusters the controls occupy has no composition to set against theirs.
        # Its fractions are still in X.
        if comparable and observed.sum() >= 1:
            statistics[row] = chisquare(observed, share * observed.sum()).statistic

    # Chi-square asks whether a well's cells are a multinomial draw from the control composition.
    # Wells also differ from one another, so the counts are overdispersed and the test is anti-conservative:
    # on wells drawn from one composition with mild jitter it called 10 of 20 at q < 0.05, down to q = 9e-10.
    # The controls measure that extra spread, and dividing by it is the usual quasi-likelihood correction.
    dof = max(int(reached.sum()) - 1, 1)
    control_statistics = statistics[is_control]
    control_statistics = control_statistics[np.isfinite(control_statistics)]
    # Below one the controls are tighter than multinomial; scaling down would only invent hits.
    dispersion = max(1.0, float(np.mean(control_statistics) / dof)) if control_statistics.size else np.nan
    if comparable and control_statistics.size < _DISPERSION_MIN_CONTROLS:
        warnings.warn(
            f"the dispersion the composition test calibrates against comes from {control_statistics.size} control "
            f"well(s), too few to estimate it; p-values are anti-conservative by however much the wells vary. "
            f"Use at least {_DISPERSION_MIN_CONTROLS}, or read the statistic as a ranking rather than a test.",
            UserWarning,
            stacklevel=3,
        )

    pvalues = chi2.sf(statistics / dispersion, dof) if np.isfinite(dispersion) else np.full(len(statistics), np.nan)
    table = pd.DataFrame(
        {"group": names.astype(str), "statistic": statistics, "pvalue": np.where(np.isnan(statistics), np.nan, pvalues)}
    )
    table["qvalue"] = benjamini_hochberg(table["pvalue"].to_numpy()) if len(table) else []
    return table, dispersion


@inplace_or_copy(expects="cell")
def cell_cycle_phase(
    adata: AnnData,
    dna_feature: str | None = None,
    by: str | None = "Metadata_Plate",
    layer: str | None = None,
    key_added: str = "Metadata_CellCyclePhase",
    copy: bool = False,
) -> AnnData | None:
    """Assign G1, S or G2M from integrated DNA intensity.

    A two-component Gaussian mixture is fitted to log DNA intensity within each ``by`` group.
    The lower component is G1 and the upper is G2M.
    Cells that neither component claims with probability above 0.9 are called S.

    Args:
        adata: Single-cell object holding raw, unnormalized intensities.
        dna_feature: The integrated DNA intensity feature. Found from the feature names and ``var["channel"]`` when omitted.
        by: Fit separately within each group, normally the plate, since staining intensity does not carry across plates.
        layer: Read this layer instead of ``X``, for example ``"raw"`` after ``normalize(keep_raw=True)``.
        key_added: ``obs`` column written.
        copy: Return a modified copy instead of mutating in place.

    Returns:
        ``None``, or the modified copy.
        Writes ``obs[key_added]`` as a categorical over ``PHASES``.

    Raises:
        KeyError: ``dna_feature`` was not given and no integrated DNA intensity feature could be found.
        ValueError: More than 5% of the feature's values are not positive, so their log cannot be fitted.

    Notes:
        This is a heuristic.
        It needs a well-resolved DNA stain and enough cells per group (a group with fewer than 20 usable cells is left as S), and it cannot tell G0 from G1.
        It also needs raw intensities, because after :func:`~mantispy.pp.normalize` the values are z-scores and about half are negative.
        Check the result before relying on it.
    """
    from sklearn.mixture import GaussianMixture

    var = as_frame(adata.var)
    if dna_feature is None:
        candidates = [
            name
            for name in adata.var_names
            if "IntegratedIntensity" in name and "DNA" in str(var.loc[name].get("channel"))
        ]
        if not candidates:
            raise KeyError(
                "could not find an integrated DNA intensity feature; pass dna_feature= explicitly "
                "(the usual name is Nuclei_Intensity_IntegratedIntensity_DNA)"
            )
        dna_feature = candidates[0]
        get_logger().info("cell_cycle_phase using %s", dna_feature)

    values = get_matrix(adata, layer)[:, adata.var_names.get_loc(dna_feature)].astype(np.float64)
    positive = np.isfinite(values) & (values > 0)
    # Raw integrated intensities are positive, so many non-positive values mean normalized data.
    if positive.sum() < 0.95 * adata.n_obs:
        raise ValueError(
            f"{int((~positive).sum())} of {adata.n_obs} values of {dna_feature!r} are not positive, so their "
            "log cannot be fitted. cell_cycle_phase needs the raw intensities; run it before mt.pp.normalize, "
            "or keep them with normalize(keep_raw=True) and pass layer='raw'."
        )

    codes, keys = group_codes(adata, by)
    phases = np.full(adata.n_obs, "S", dtype=object)

    for index in range(len(keys)):
        rows = np.flatnonzero(codes == index)
        usable = rows[positive[rows]]
        if usable.size < 20:
            get_logger().warning("cell_cycle_phase: %s has %d usable cells; left as S", keys[index], usable.size)
            continue

        log_intensity = np.log(values[usable]).reshape(-1, 1)
        mixture = GaussianMixture(n_components=2, random_state=0).fit(log_intensity)
        probability = mixture.predict_proba(log_intensity)
        lower = int(np.argmin(mixture.means_.ravel()))
        assignment = np.where(probability.argmax(axis=1) == lower, "G1", "G2M")
        phases[usable] = np.where(probability.max(axis=1) > 0.9, assignment, "S")

    adata.obs[key_added] = pd.Categorical(phases, categories=list(PHASES))
    counted = pd.Series(phases).value_counts()
    get_logger().info("cell_cycle_phase: %s", counted.to_dict())
    return None


@inplace_or_copy(expects="cell")
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
        min_cells: Skip a (cluster, group) pair with fewer cells than this on either side. A cluster needs twice as many controls, and at least four, since half of them place the centroid and half supply the distances tested against. The reference group's own row needs four times as many, since it comes from a second split of the held-out half.
        seed: Seed for the split of a cluster's controls.
        key_added: Name for the output table.
        copy: Return a modified copy instead of mutating in place.

    Returns:
        ``None``, or the modified copy.
        Writes ``uns["mantispy"][key_added]`` with ``cluster``, ``group``, ``n_cells``, ``statistic`` (KS), ``pvalue`` and ``qvalue``.

    Raises:
        KeyError: ``obs`` has no column ``cluster_key``.

    Notes:
        Each cell is reduced to its Euclidean distance from the control centroid of its own cluster, and a KS test compares the treated cells' distances with the controls'.
        This is close to ``hit_calling(method="ks")``, restricted to comparable cells.
        A distance is used rather than a single feature so that the test means the same on every dataset.

        A cluster's controls are split in half, as in :func:`~mantispy.tl.hit_calling`.
        One half places the centroid and the other supplies the distances tested against, so the null is out of sample.
        Controls measured against a centroid they defined themselves sit closer to it than any other group can.
        That bias grows with features per control, which is the shape of real Cell Painting data.
        On pure noise with 36 controls and 120 features, a null drawn from the rows that placed the centroid called 0.40 of pseudo-treatments at raw ``p < 0.05``, and the split called 0.03.

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
    clusters = obs[cluster_key].astype(str).to_numpy()
    groups = obs[groupby].astype(str).to_numpy()

    generator = np.random.default_rng(seed)
    # Halving the reference group's held-out cells draws from a child of the seeded generator, so that it leaves the stream the cluster splits come from where it was.
    half_generator = generator.spawn(1)[0]
    # Half the controls place the centroid and half form the distances tested against, so the null is out of sample like every group (see Notes).
    needed = max(2 * min_cells, 4)

    records = []
    for cluster in pd.unique(clusters):
        in_cluster = np.flatnonzero(clusters == cluster)
        controls = in_cluster[is_control[in_cluster]]
        if controls.size < needed:
            continue

        fit_rows, null_rows = split_reference(controls, generator)
        centre = np.nanmean(values[fit_rows], axis=0, keepdims=True)
        distance = np.sqrt(pairwise_sqeuclidean(np.nan_to_num(values[in_cluster]), np.nan_to_num(centre))).ravel()
        fitted = np.isin(in_cluster, fit_rows)
        # The split partitions the cluster's control cells, so the held-out half is the controls that did not place the centroid.
        held_out = is_control[in_cluster] & ~fitted

        for group in pd.unique(groups[in_cluster]):
            # A cell that placed the centroid sits closer to it than one that did not, so it stays off the tested side.
            in_group = (groups[in_cluster] == group) & ~fitted
            # The reference group is the controls under their own perturbation label.
            # Half its held-out cells are the sample and half are what it is tested against, drawn at random because the cells are ordered by plate and well.
            shared = np.flatnonzero(in_group & held_out)
            in_group[half_generator.permutation(shared)[: shared.size // 2]] = False
            treated, control_distance = distance[in_group], distance[held_out & ~in_group]
            if treated.size < min_cells or control_distance.size < min_cells:
                continue
            statistic = float(ks_statistic(treated[None, :], control_distance)[0])
            # Two-sample KS p-value, the asymptotic form scipy uses for large samples.
            n, m = treated.size, control_distance.size
            effective = np.sqrt(n * m / (n + m))
            records.append(
                {
                    "cluster": str(cluster),
                    "group": str(group),
                    "n_cells": int(n),
                    "statistic": statistic,
                    "pvalue": float(min(1.0, 2.0 * np.exp(-2.0 * (effective * statistic) ** 2))),
                }
            )

    table = pd.DataFrame(records, columns=["cluster", "group", "n_cells", "statistic", "pvalue"])
    if table.empty or int(table.groupby("cluster")["group"].nunique().max()) <= 1:
        warnings.warn(
            f"no cluster held at least {needed} controls, half to place its centroid and half to test against, "
            f"and {min_cells} cells of another group, so nothing was compared. The clustering separates the "
            "perturbations, leaving no shared cell state to compare within. Cluster on fewer components, or "
            "test whole populations with mt.tl.edistance.",
            UserWarning,
            stacklevel=3,
        )
    table["qvalue"] = benjamini_hochberg(table["pvalue"].to_numpy()) if len(table) else []
    adata.uns.setdefault("mantispy", {})[key_added] = table
    return None


@inplace_or_copy(expects="cell")
def neighbors_local_density(
    adata: AnnData,
    k: int = 15,
    by: str = "Metadata_ImageNumber",
    key_added: str = "Metadata_LocalDensity",
    copy: bool = False,
) -> AnnData | None:
    """Mean distance to the ``k`` nearest cells in the same field of view.

    Local crowding changes morphology independently of treatment.
    Consider regressing it out with :func:`~mantispy.pp.regress_out` before attributing a change to a perturbation.

    Args:
        adata: Single-cell object carrying ``Metadata_Center_X`` and ``Metadata_Center_Y``.
        k: Number of neighbors averaged over.
        by: ``obs`` column identifying the field of view. Neighbors are searched within each field only, since coordinates from different images are not comparable.
        key_added: ``obs`` column written.
        copy: Return a modified copy instead of mutating in place.

    Returns:
        ``None``, or the modified copy.
        Writes ``obs[key_added]``, and a cell alone in its field gets ``NaN``.

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
        density[rows] = distances[:, 1:].mean(axis=1)

    adata.obs[key_added] = density
    if np.isnan(density).any():
        get_logger().info(
            "neighbors_local_density: %d cell(s) alone in their field, left as NaN", int(np.isnan(density).sum())
        )
    return None
