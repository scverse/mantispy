"""Cluster composition, cell cycle, subpopulation hits and local density."""

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scanpy as sc

import mantispy as mt
from mantispy._core.schema import stamp


@pytest.fixture
def clustered():
    cells = mt.ds.synthetic_plate(n_wells=48, n_cells=40, n_features=20, n_perturbations=3, effect_size=4.0, seed=0)
    mt.pp.normalize(cells, by="Metadata_Plate", reference="negcon")
    sc.pp.pca(cells, n_comps=10)
    sc.pp.neighbors(cells)
    sc.tl.leiden(cells, key_added="leiden", flavor="igraph", n_iterations=2)
    return cells


def test_composition_rows_are_wells_and_sum_to_one(clustered):
    composition = mt.tl.cluster_composition(clustered)
    assert composition.n_obs == clustered.obs.groupby(["Metadata_Plate", "Metadata_Well"], observed=True).ngroups
    assert composition.n_vars == clustered.obs["leiden"].nunique()
    np.testing.assert_allclose(np.asarray(composition.X).sum(axis=1), 1.0, atol=1e-5)
    assert mt.io.validate(composition).ok, mt.io.validate(composition).errors


@pytest.mark.filterwarnings("ignore:the controls occupy")
def test_a_cell_with_no_cluster_is_left_out_rather_than_made_into_one(clustered, caplog):
    """The label an unassigned cell contributed either crashed sorted(), on pandas 3, where NaN cannot be
    ordered against the cluster names, or became a cluster literally called "nan" on pandas 2."""
    import logging

    clusters = clustered.obs["leiden"].astype(str)
    unassigned = np.zeros(clustered.n_obs, dtype=bool)
    unassigned[:5] = True
    clustered.obs["leiden"] = pd.Categorical(np.where(unassigned, None, clusters))

    # info, not warning: report_drop escalates only at half the input or more, and this is 5 of 1920.
    with caplog.at_level(logging.INFO, logger="mantispy"):
        composition = mt.tl.cluster_composition(clustered)
    assert "dropped 5 of 1920 cell(s); they have no 'leiden'" in " ".join(
        record.getMessage() for record in caplog.records
    )

    assert list(composition.var_names) == sorted(set(clusters[~unassigned]))
    assert "nan" not in set(composition.var_names)
    assert "None" not in set(composition.var_names)
    # The fractions are over the cells that were assigned, so they still sum to one per well.
    totals = np.asarray(composition.X).sum(axis=1)
    np.testing.assert_allclose(totals[totals > 0], 1.0, atol=1e-5)
    assert mt.io.validate(composition).ok, mt.io.validate(composition).errors


def test_composition_carries_metadata_and_tests_against_the_controls(clustered):
    values = np.asarray(clustered.X, dtype=float)
    is_control = clustered.obs["Metadata_Control"].to_numpy(dtype=bool)
    # At effect_size=4 leiden gives one cluster per perturbation, so the controls occupy one of
    # them and no composition can be set against theirs. Split on distance from the control
    # centre at the control median instead, a state both conditions hold cells in.
    distance = np.linalg.norm(values - np.nanmean(values[is_control], axis=0), axis=1)
    clustered.obs["state"] = np.where(distance > np.median(distance[is_control]), "far", "near")

    composition = mt.tl.cluster_composition(clustered, cluster_key="state")
    assert "Metadata_Perturbation" in composition.obs
    test = composition.uns["mantispy"]["composition_test"]
    assert {"group", "statistic", "pvalue", "qvalue"} <= set(test.columns)
    assert len(test) == composition.n_obs, "one row per well, in the order of the rows"

    control = composition.obs["Metadata_Control"].to_numpy(dtype=bool)
    statistic = test["statistic"].to_numpy()
    # Every treated well's composition departs further than any control well's.
    assert statistic[~control].min() > statistic[control].max()
    assert (test["qvalue"].to_numpy()[~control] < 0.05).all()


def test_controls_confined_to_one_cluster_warn_rather_than_return_a_table_of_nan(clustered):
    """A chi-square over a single category has no degrees of freedom, so the test is NaN.

    A full-looking table of NaN handed back without a word reads as no well's composition
    differing from the controls'.
    """
    leiden = clustered.obs["leiden"].astype(str)
    assert leiden[clustered.obs["Metadata_Control"].to_numpy(dtype=bool)].nunique() == 1
    with pytest.warns(UserWarning, match="controls occupy 1 of"):
        composition = mt.tl.cluster_composition(clustered)
    test = composition.uns["mantispy"]["composition_test"]
    assert len(test) == composition.n_obs
    assert test["pvalue"].isna().all()


def test_cell_cycle_splits_a_bimodal_dna_content(clustered):
    rng = np.random.default_rng(0)
    values = clustered.X.copy()
    doubled = rng.random(clustered.n_obs) < 0.4
    values[:, 0] = np.where(doubled, 2.0, 1.0) + rng.normal(0, 0.05, clustered.n_obs)
    clustered.X = values

    mt.tl.cell_cycle_phase(clustered, dna_feature=clustered.var_names[0])
    phases = clustered.obs["Metadata_CellCyclePhase"].astype(str)
    assert set(phases.unique()) <= set(mt.tl._heterogeneity.PHASES)
    # The G2M share should recover the 40% that were doubled, give or take the S band.
    assert 0.3 < (phases == "G2M").mean() < 0.5
    assert (phases[doubled] == "G2M").mean() > 0.8


def test_cell_cycle_refuses_normalized_values(clustered):
    """After normalize the values are z-scores, and half of them have no logarithm."""
    with pytest.raises(ValueError, match="raw intensities"):
        mt.tl.cell_cycle_phase(clustered, dna_feature=clustered.var_names[0])


def test_subpopulation_hits_finds_an_effect_inside_a_shared_state(clustered):
    """Clusters have to mix the perturbations for there to be anything to compare."""
    clustered.obs["state"] = np.where(np.random.default_rng(0).random(clustered.n_obs) < 0.5, "a", "b")
    mt.tl.subpopulation_hits(clustered, cluster_key="state")

    table = clustered.uns["mantispy"]["subpopulation_hits"]
    assert {"cluster", "group", "n_cells", "statistic", "pvalue", "qvalue"} <= set(table.columns)
    assert (table[table["group"] != "DMSO"]["qvalue"] < 0.05).any()
    assert table[table["group"] == "DMSO"]["statistic"].mean() < table[table["group"] != "DMSO"]["statistic"].mean()


def test_a_clustering_that_separates_the_perturbations_has_nothing_to_test(clustered):
    """At effect_size=4 leiden gives one cluster per perturbation, so no cluster holds
    both controls and treated cells. An empty table without a warning would read as
    'no subpopulation effects'."""
    with pytest.warns(UserWarning, match="no shared cell state"):
        mt.tl.subpopulation_hits(clustered)
    table = clustered.uns["mantispy"]["subpopulation_hits"]
    assert set(table["group"]) <= {"DMSO"}  # only the controls compared against themselves


def test_local_density_is_computed_within_a_field(clustered):
    rng = np.random.default_rng(0)
    clustered.obs["Metadata_Center_X"] = rng.uniform(0, 1000, clustered.n_obs)
    clustered.obs["Metadata_Center_Y"] = rng.uniform(0, 1000, clustered.n_obs)
    mt.tl.neighbors_local_density(clustered, k=5)
    density = clustered.obs["Metadata_LocalDensity"].to_numpy()
    assert np.isfinite(density).all() and (density > 0).all()

    # A cell alone in its field has no neighbours and gets NaN.
    clustered.obs["Metadata_ImageNumber"] = np.arange(clustered.n_obs)
    mt.tl.neighbors_local_density(clustered, k=5)
    assert clustered.obs["Metadata_LocalDensity"].isna().all()


def test_round_trip(clustered, tmp_path):
    composition = mt.tl.cluster_composition(clustered)
    mt.io.write(composition, tmp_path / "composition.h5ad")
    loaded = mt.io.read(tmp_path / "composition.h5ad")
    assert loaded.n_vars == composition.n_vars
    assert len(loaded.uns["mantispy"]["composition_test"]) == composition.n_obs
    # The annotation columns come from _core.features.empty_annotation (#103), which supplies the
    # empty ones as categoricals precisely so that the h5ad writer keeps them.
    assert list(loaded.var.columns) == list(composition.var.columns)
    assert set(loaded.var["feature_group"]) == {"Composition"}
    for column in ("channel", "radial_bin", "params"):
        assert loaded.var[column].isna().all(), column
        # The dtype is the point of the comment above, so assert it: the float-NaN construction this
        # replaced round-trips identically and would otherwise pass.
        assert isinstance(loaded.var[column].dtype, pd.CategoricalDtype), column


def _noise_cells(
    n_control: int,
    n_features: int,
    per_group: int,
    n_groups: int = 12,
    seed: int = 0,
    n_plates: int = 1,
    plate_sd: float = 0.0,
):
    """Single cells with nothing in them: one shared cluster, controls, and pseudo-treatments.

    The rows are plate-major with each plate's controls first, the way a screen is read, so
    `plate_sd` gives the controls a between-plate offset that follows their row order.
    """
    generator = np.random.default_rng(seed)
    labels, plate_of = [], []
    for plate in range(n_plates):
        block = ["DMSO"] * (n_control // n_plates) + [
            f"p{index:02d}" for index in range(n_groups) for _ in range(per_group // n_plates)
        ]
        labels += block
        plate_of += [plate] * len(block)
    obs = pd.DataFrame(
        {
            "Metadata_Perturbation": labels,
            "Metadata_Control": [label == "DMSO" for label in labels],
            "Metadata_Plate": [f"P{plate + 1}" for plate in plate_of],
            "Metadata_Well": [f"A{index:04d}" for index in range(len(labels))],
            "state": "one",
        },
        index=[str(index) for index in range(len(labels))],
    )
    values = generator.standard_normal((len(labels), n_features))
    values += generator.normal(scale=plate_sd, size=(n_plates, n_features))[np.array(plate_of)]
    adata = ad.AnnData(
        X=values.astype(np.float32),
        obs=obs,
        var=pd.DataFrame(index=[f"Cells_AreaShape_f{index}" for index in range(n_features)]),
    )
    stamp(adata, resolution="cell")
    return adata


def test_subpopulation_hits_does_not_call_pure_noise():
    """The controls that define a cluster's centroid must not also supply the distances tested against.

    In sample they sit closer to the centroid than any other group can, and the bias grows
    with features per control: at 36 controls and 120 features the raw false positive rate
    was 0.39 against the in-sample null and 0.00 against a split reference.
    """
    called = total = 0
    for seed in range(10):
        adata = _noise_cells(n_control=36, n_features=120, per_group=200, seed=seed)
        mt.tl.subpopulation_hits(adata, cluster_key="state")
        table = adata.uns["mantispy"]["subpopulation_hits"]
        pseudo = table[table["group"] != "DMSO"]
        called += int((pseudo["pvalue"] < 0.05).sum())
        total += len(pseudo)
    assert total == 120
    assert called / total < 0.1, f"{called}/{total} pure-noise pseudo-treatments called at raw p < 0.05"


def test_the_reference_group_is_tested_only_on_cells_that_did_not_place_the_centroid():
    """The controls carry a perturbation label too, so one row of the table is the reference group against itself.

    Tested on every control cell, that row compares a sample with the half of itself it is
    tested against, measured from a centre the other half placed.
    """
    adata = _noise_cells(n_control=36, n_features=120, per_group=200, n_groups=1, seed=0)
    mt.tl.subpopulation_hits(adata, cluster_key="state")
    table = adata.uns["mantispy"]["subpopulation_hits"].set_index("group")

    # Eighteen of the 36 controls place the centroid, and the held-out half is split again.
    assert int(table.loc["DMSO", "n_cells"]) <= 9
    assert np.isfinite(float(table.loc["DMSO", "statistic"]))
    # A group that is not the reference keeps every cell it has.
    assert int(table.loc["p00", "n_cells"]) == 200


def test_the_reference_group_is_not_halved_along_the_plate_order():
    """The reference group's held-out cells are halved at random, not by row order.

    Cells arrive ordered by plate and well, so the first half in row order is one set of
    plates and the second is another. With the controls carrying a between-plate offset,
    that makes the reference row a KS test between plates rather than a draw from the null,
    and it is called on most seeds instead of one in twenty.
    """
    called = 0
    for seed in range(15):
        adata = _noise_cells(n_control=360, n_features=8, per_group=60, n_groups=2, n_plates=6, plate_sd=3.0, seed=seed)
        mt.tl.subpopulation_hits(adata, cluster_key="state", seed=seed)
        reference = adata.uns["mantispy"]["subpopulation_hits"].query("group == 'DMSO'")
        assert len(reference) == 1, "one cluster, so one reference row per seed"
        called += int((reference["pvalue"] < 0.05).sum())
    assert called <= 3, f"{called}/15 reference rows called at raw p < 0.05, where 0.05 is calibrated"


def test_wells_that_vary_around_one_composition_are_not_called_hits():
    """Wells differ from each other, so cluster counts are overdispersed relative to multinomial.

    A chi-square on raw counts assumes each well is a multinomial draw from the control
    composition and called 10 of 20 null wells, down to q = 9e-10. The dispersion the controls
    show has to set the scale the test measures a well against.
    """
    rng = np.random.default_rng(0)
    shares = rng.dirichlet(np.full(4, 40.0), size=32)
    layout = {
        f"{'A' if index < 16 else 'B'}{index % 16 + 1:02d}": dict(
            enumerate(np.bincount(rng.choice(4, size=300, p=share), minlength=4))
        )
        for index, share in enumerate(shares)
    }

    composition = mt.tl.cluster_composition(_clustered_wells(layout, n_clusters=4))
    test = composition.uns["mantispy"]["composition_test"]
    treated = ~composition.obs["Metadata_Control"].to_numpy(dtype=bool)

    called = int((test["qvalue"].to_numpy()[treated] < 0.05).sum())
    assert called <= 1, f"{called}/16 wells of the control composition called at q < 0.05"
    assert composition.uns["mantispy"]["composition_dispersion"] > 1.0, "the controls are overdispersed"


def test_a_composition_test_without_the_controls_to_calibrate_it_warns():
    layout = {"A01": {0: 50, 1: 50}, "A02": {0: 52, 1: 48}, "B01": {0: 80, 1: 20}}
    with pytest.warns(UserWarning, match="dispersion"):
        mt.tl.cluster_composition(_clustered_wells(layout, n_clusters=2))


def _clustered_wells(layout: dict[str, dict[int, int]], n_clusters: int):
    """Cells labeled by well and cluster; wells whose name starts with A are the controls."""
    rows = [
        {
            "Metadata_Plate": "P1",
            "Metadata_Well": well,
            "Metadata_Perturbation": "DMSO" if well.startswith("A") else "pert",
            "Metadata_Control": well.startswith("A"),
            "leiden": str(cluster),
        }
        for well, clusters in layout.items()
        for cluster, count in clusters.items()
        for _ in range(count)
    ]
    obs = pd.DataFrame(rows, index=[str(index) for index in range(len(rows))])
    adata = ad.AnnData(
        X=np.random.default_rng(0).standard_normal((len(rows), 4)).astype(np.float32),
        obs=obs,
        var=pd.DataFrame(index=[f"Cells_AreaShape_f{index}" for index in range(4)]),
    )
    assert obs["leiden"].nunique() == n_clusters
    stamp(adata, resolution="cell")
    return adata


def test_a_shifted_composition_departs_further_than_a_control_well_does():
    """A test restricted to the clusters the controls occupy still has to separate a shifted composition from an unshifted one."""
    layout = {
        "A01": {0: 50, 1: 50},
        "A02": {0: 52, 1: 48},
        "A03": {0: 48, 1: 52},
        "B01": {0: 80, 1: 20},
        "B02": {0: 78, 1: 22},
    }
    composition = mt.tl.cluster_composition(_clustered_wells(layout, n_clusters=2))
    test = composition.uns["mantispy"]["composition_test"]
    control = composition.obs["Metadata_Control"].to_numpy(dtype=bool)

    statistic = test["statistic"].to_numpy()
    assert statistic[~control].min() > statistic[control].max()
    assert (test["qvalue"].to_numpy()[~control] < 0.05).all()


def test_a_cluster_the_controls_never_reached_does_not_fabricate_a_hit():
    """An expected count floored at 1e-9 turned five cells in a treatment-only cluster into chi-square 7.5e10 and p exactly 0."""
    layout = {
        "A01": {0: 50, 1: 50},
        "A02": {0: 60, 1: 40},
        "A03": {0: 55, 1: 45},
        "B01": {0: 50, 1: 45, 2: 5},  # five cells where no control cell was seen
        "B02": {0: 52, 1: 48},
    }
    composition = mt.tl.cluster_composition(_clustered_wells(layout, n_clusters=3))
    test = composition.uns["mantispy"]["composition_test"].set_index("group")

    assert float(test.loc["P1/B01", "statistic"]) < 1e3
    assert float(test.loc["P1/B01", "pvalue"]) > 0.0
    assert float(test.loc["P1/B01", "qvalue"]) > 0.0
    # The fractions still show the cluster, which is where a treatment-only state belongs.
    row = composition.obs_names[composition.obs["Metadata_Well"].astype(str).to_numpy() == "B01"][0]
    assert float(composition[row, "2"].X[0, 0]) == pytest.approx(0.05)


def test_many_unreached_clusters_do_not_break_the_chi_square():
    """Thirty-eight expected counts floored at 1e-9 outweighed two pooled control cells, and scipy refused the test outright."""
    layout = {"A01": {0: 1, 1: 1}, "B01": {0: 10, 5: 3}, "B02": dict.fromkeys(range(40), 1)}
    composition = mt.tl.cluster_composition(_clustered_wells(layout, n_clusters=40))
    test = composition.uns["mantispy"]["composition_test"].set_index("group")

    assert np.isfinite(test["statistic"].to_numpy()).all()
    # Ten cells in cluster 0 against an even control split over clusters 0 and 1.
    assert float(test.loc["P1/B01", "statistic"]) == pytest.approx(10.0)


@pytest.mark.filterwarnings("ignore:the controls occupy")
def test_the_cell_count_covers_every_cell_not_only_the_clustered_ones(clustered):
    """Metadata_CellCount is tl.cytotoxicity's default count_key, and validate warns that without it
    cytotoxicity cannot separate a hit from cell loss. Summing the fractions' counts made it the number
    of *clustered* cells, so a well the clustering merely left cells out of read as cell loss."""
    clusters = clustered.obs["leiden"].astype(str)
    unassigned = np.zeros(clustered.n_obs, dtype=bool)
    unassigned[::10] = True
    clustered.obs["leiden"] = pd.Categorical(np.where(unassigned, None, clusters))

    composition = mt.tl.cluster_composition(clustered)

    actual = clustered.obs.groupby(["Metadata_Plate", "Metadata_Well"], observed=True).size()
    assert composition.obs["Metadata_CellCount"].tolist() == actual.tolist()


@pytest.mark.filterwarnings("ignore:the controls occupy")
def test_a_well_with_no_assigned_cell_is_left_out(clustered):
    """Zero in every cluster said the well was measured and found empty everywhere; NaN said it was
    unknown, and tl.map refuses an object with missing values although the Returns clause promises
    tl.map accepts it. A well with nothing to measure is dropped, like any other empty group."""
    clusters = clustered.obs["leiden"].astype(str)
    wells = clustered.obs["Metadata_Well"].to_numpy()
    blanked = wells == wells[0]
    clustered.obs["leiden"] = pd.Categorical(np.where(blanked, None, clusters))

    composition = mt.tl.cluster_composition(clustered)

    assert wells[0] not in set(composition.obs["Metadata_Well"])
    assert not np.isnan(np.asarray(composition.X)).any(), "the result stays mappable"
    mt.tl.map(composition, mode="activity", null_size=50)


def test_the_drop_is_reported_only_when_something_is_dropped(clustered, caplog):
    """report_drop ran before the no-cluster refusal, so an unclustered object was told its cells
    were 'left out of the fractions' immediately before being told there are no fractions."""
    import logging

    clustered.obs["leiden"] = pd.Categorical([None] * clustered.n_obs)
    with caplog.at_level(logging.INFO, logger="mantispy"), pytest.raises(ValueError, match="no cell"):
        mt.tl.cluster_composition(clustered)
    assert not [record for record in caplog.records if "left out of the fractions" in record.getMessage()]


def test_a_clustering_that_assigned_nothing_is_refused(clustered):
    """labels == [] gave an (n_wells, 0) object, which io.validate rejects -- a tool returning
    something validate will not accept is the defect this change set out to remove."""
    clustered.obs["leiden"] = pd.Categorical([None] * clustered.n_obs)
    with pytest.raises(ValueError, match="no cell"):
        mt.tl.cluster_composition(clustered)


@pytest.mark.filterwarnings("ignore:the controls occupy")
def test_the_annotation_columns_that_carry_values_stay_categorical(clustered):
    """empty_annotation makes the text columns categorical; df[column] = value replaces the column
    rather than setting into it, so the three that carry values silently lost the dtype."""
    composition = mt.tl.cluster_composition(clustered)
    for column in ("object", "feature_group", "feature"):
        assert isinstance(composition.var[column].dtype, pd.CategoricalDtype), column
