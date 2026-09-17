"""Single-cell heterogeneity beyond the well median.

A perturbation that strongly shifts ten percent of cells looks like a small change in the
average. These tools work on the distribution of cells instead, so they need single-cell
profiles.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence

import anndata as ad
import numpy as np
import pandas as pd
from anndata import AnnData
from scipy.stats import chisquare

from mantispy._core._distance import pairwise_sqeuclidean
from mantispy._core._reduce import get_matrix, group_codes, representation
from mantispy._core._stats import benjamini_hochberg
from mantispy._core.frames import as_frame
from mantispy._core.logging import get_logger
from mantispy._core.masks import reference_mask
from mantispy._core.mutation import inplace_or_copy
from mantispy._core.schema import stamp
from mantispy.tl._aggregate import _group_obs
from mantispy.tl._hits import ks_statistic

PHASES = ("G1", "S", "G2M")


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
        A new object with wells as rows and clusters as columns, holding the fraction of each
        well's cells in each cluster. It is a well-level mantispy object, so
        :func:`~mantispy.tl.map`, :func:`~mantispy.pp.normalize` and the plots accept it.
        ``uns["mantispy"]["composition_test"]`` holds a chi-square test of each well against
        the pooled control composition.

    Notes:
        A well with few cells has a noisy composition. The chi-square test is computed on
        counts and accounts for this, but the fractions in ``X`` do not. Filter with
        :func:`~mantispy.pp.well_qc` first.
    """
    if cluster_key not in adata.obs:
        raise KeyError(f"obs has no column {cluster_key!r}; cluster first, e.g. sc.tl.leiden(adata)")

    columns = [by] if isinstance(by, str) else list(by)
    codes, keys = group_codes(adata, columns)
    clusters = as_frame(adata.obs)[cluster_key].astype(str)
    labels = sorted(clusters.unique())

    counts = np.zeros((len(keys), len(labels)))
    membership = pd.Categorical(clusters, categories=labels).codes
    np.add.at(counts, (codes, membership), 1)
    totals = counts.sum(axis=1, keepdims=True)
    fractions = np.divide(counts, totals, out=np.zeros_like(counts), where=totals > 0)

    obs = _group_obs(adata, columns, keys, codes, counts.sum(axis=1).astype(int))
    obs.index = pd.Index([str(row) for row in range(len(obs))])
    var = pd.DataFrame(index=pd.Index(labels))
    var["object"], var["feature_group"], var["feature"] = "Cluster", "Composition", labels
    for column in ("channel", "scale", "angle", "gray_levels", "radial_bin", "params"):
        var[column] = np.nan
    var["is_feature"] = True

    result = ad.AnnData(X=fractions.astype(np.float32), obs=obs, var=var)
    stamp(result, resolution="well")
    result.uns["mantispy"]["composition_test"] = _composition_test(result, counts, reference)
    return result


def _composition_test(composition: AnnData, counts: np.ndarray, reference: str | None) -> pd.DataFrame:
    """Chi-square each well's cluster counts against the pooled control composition."""
    empty = pd.DataFrame(columns=["group", "statistic", "pvalue", "qvalue"])
    if reference is None:
        return empty
    if reference == "negcon" and "Metadata_Control" not in composition.obs:
        get_logger().info("cluster_composition: no Metadata_Control, so no composition test")
        return empty

    is_control = reference_mask(composition, reference)
    if not is_control.any():
        return empty

    share = counts[is_control].sum(axis=0)
    share = np.where(share == 0, 1e-9, share) / max(share.sum(), 1e-9)

    labels = as_frame(composition.obs)
    naming = [column for column in ("Metadata_Plate", "Metadata_Well") if column in labels]
    names = (
        labels[naming].astype(str).agg("/".join, axis=1).to_numpy()
        if naming
        else composition.obs_names.astype(str).to_numpy()
    )

    records = []
    for row in range(composition.n_obs):
        observed = counts[row]
        if observed.sum() < 1:
            continue
        statistic, pvalue = chisquare(observed, share * observed.sum())
        records.append({"group": str(names[row]), "statistic": float(statistic), "pvalue": float(pvalue)})

    table = pd.DataFrame(records)
    table["qvalue"] = benjamini_hochberg(table["pvalue"].to_numpy()) if len(table) else []
    return table


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
    The lower component is G1 and the upper is G2M. Cells that neither component claims with
    probability above 0.9 are called S.

    Args:
        adata: Single-cell object holding raw, unnormalized intensities.
        dna_feature: The integrated DNA intensity feature. Found from the feature names and
            ``var["channel"]`` when omitted.
        by: Fit separately within each group, normally the plate, since staining intensity
            does not carry across plates.
        layer: Read this layer instead of ``X``, for example ``"raw"`` after
            ``normalize(keep_raw=True)``.
        key_added: ``obs`` column written.
        copy: Return a modified copy instead of mutating in place.

    Returns:
        ``None``, or the modified copy.

    Notes:
        This is a heuristic. It needs a well-resolved DNA stain and enough cells per group
        (a group with fewer than 20 usable cells is left as S), and it cannot tell G0 from G1.
        It also needs raw intensities, because after :func:`~mantispy.pp.normalize` the values
        are z-scores and about half are negative. Check the result before relying on it.
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
        min_cells: Skip a (cluster, group) pair with fewer cells than this on either side.
        key_added: Name for the output table.
        copy: Return a modified copy instead of mutating in place.

    Returns:
        ``None``, or the modified copy. Writes ``uns["mantispy"][key_added]`` with
        ``cluster``, ``group``, ``n_cells``, ``statistic`` (KS), ``pvalue`` and ``qvalue``.

    Notes:
        Each cell is reduced to its Euclidean distance from the control centroid of its own
        cluster, and a KS test compares the treated cells' distances with the controls'. This
        is close to ``hit_calling(method="ks")``, restricted to comparable cells. A distance is
        used rather than a single feature so that the test means the same on every dataset.
    """
    if cluster_key not in adata.obs:
        raise KeyError(f"obs has no column {cluster_key!r}; cluster first, e.g. sc.tl.leiden(adata)")

    values = representation(adata, use_rep)
    obs = as_frame(adata.obs)
    is_control = reference_mask(adata, reference)
    clusters = obs[cluster_key].astype(str).to_numpy()
    groups = obs[groupby].astype(str).to_numpy()

    records = []
    for cluster in pd.unique(clusters):
        in_cluster = np.flatnonzero(clusters == cluster)
        controls = in_cluster[is_control[in_cluster]]
        if controls.size < min_cells:
            continue

        centre = np.nanmean(values[controls], axis=0, keepdims=True)
        distance = np.sqrt(pairwise_sqeuclidean(np.nan_to_num(values[in_cluster]), np.nan_to_num(centre))).ravel()
        control_distance = distance[is_control[in_cluster]]

        for group in pd.unique(groups[in_cluster]):
            treated = distance[groups[in_cluster] == group]
            if treated.size < min_cells:
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
            f"no cluster held at least {min_cells} controls and {min_cells} cells of another group, so nothing "
            "was compared. The clustering separates the perturbations, leaving no shared cell state to compare "
            "within. Cluster on fewer components, or test whole populations with mt.tl.edistance.",
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

    Local crowding changes morphology independently of treatment. Consider regressing it out
    with :func:`~mantispy.pp.regress_out` before attributing a change to a perturbation.

    Args:
        adata: Single-cell object carrying ``Metadata_Center_X`` and ``Metadata_Center_Y``.
        k: Number of neighbors averaged over.
        by: ``obs`` column identifying the field of view. Neighbors are searched within each
            field only, since coordinates from different images are not comparable.
        key_added: ``obs`` column written.
        copy: Return a modified copy instead of mutating in place.

    Returns:
        ``None``, or the modified copy. A cell alone in its field gets ``NaN``.
    """
    from sklearn.neighbors import NearestNeighbors

    needed = {"Metadata_Center_X", "Metadata_Center_Y"}
    if not needed <= set(adata.obs.columns):
        raise KeyError(f"obs needs {sorted(needed)}; read with mt.io.read_profiles, which keeps them")

    coordinates = as_frame(adata.obs)[["Metadata_Center_X", "Metadata_Center_Y"]].to_numpy(dtype=float)
    codes, keys = group_codes(adata, by)
    density = np.full(adata.n_obs, np.nan)

    for index in range(len(keys)):
        rows = np.flatnonzero(codes == index)
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
