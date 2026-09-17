"""Cluster composition, cell cycle, subpopulation hits and local density."""

import numpy as np
import pytest
import scanpy as sc

import mantispy as mt


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


def test_composition_carries_metadata_and_tests_against_the_controls(clustered):
    composition = mt.tl.cluster_composition(clustered)
    assert "Metadata_Perturbation" in composition.obs
    test = composition.uns["mantispy"]["composition_test"]
    assert {"group", "statistic", "pvalue", "qvalue"} <= set(test.columns)
    assert len(test) == composition.n_obs
    # A treated well's composition should depart from the controls more often than a
    # control well's does.
    control = composition.obs["Metadata_Control"].to_numpy(dtype=bool)
    assert test["statistic"].to_numpy()[~control].mean() > test["statistic"].to_numpy()[control].mean()


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
