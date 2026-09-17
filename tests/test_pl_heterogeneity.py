"""Plots for cell-state composition, cell cycle and subpopulation effects."""

import matplotlib
import numpy as np
import pytest
import scanpy as sc

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import mantispy as mt


@pytest.fixture(scope="module")
def clustered():
    cells = mt.ds.synthetic_plate(n_wells=48, n_cells=30, n_features=20, n_perturbations=3, effect_size=4.0, seed=0)
    mt.pp.normalize(cells, by="Metadata_Plate", reference="negcon", keep_raw=True)
    sc.pp.pca(cells, n_comps=10)
    sc.pp.neighbors(cells)
    sc.tl.leiden(cells, key_added="leiden", flavor="igraph", n_iterations=2)
    cells.obs["state"] = np.where(np.random.default_rng(0).random(cells.n_obs) < 0.5, "a", "b")
    mt.tl.subpopulation_hits(cells, cluster_key="state")
    return cells


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def test_composition_and_subpopulation_plots_draw(clustered):
    composition = mt.tl.cluster_composition(clustered)
    assert isinstance(mt.pl.cluster_composition(composition), matplotlib.axes.Axes)
    assert isinstance(mt.pl.subpopulation_hits(clustered), matplotlib.axes.Axes)


def test_cell_cycle_plot_draws_one_panel_per_plate(clustered):
    # Synthetic features are centred, so give the raw layer a positive bimodal DNA column.
    rng = np.random.default_rng(0)
    raw = clustered.layers["raw"].copy()
    raw[:, 0] = np.where(rng.random(clustered.n_obs) < 0.4, 2.0, 1.0) + rng.normal(0, 0.05, clustered.n_obs)
    clustered.layers["raw"] = raw

    mt.tl.cell_cycle_phase(clustered, dna_feature=clustered.var_names[0], layer="raw")
    axes = np.asarray(mt.pl.cell_cycle(clustered, dna_feature=clustered.var_names[0], layer="raw"))
    assert axes.size == clustered.obs["Metadata_Plate"].nunique()


def test_feature_distributions_gains_a_ridge_kind(clustered):
    axes = mt.pl.feature_distributions(clustered, features=list(clustered.var_names[:2]), kind="ridge")
    assert axes[0, 0].collections  # filled density curves; a histogram would draw patches instead
    assert np.asarray(axes).size >= 2
    with pytest.raises(ValueError, match="'ecdf', 'hist' or 'ridge'"):
        mt.pl.feature_distributions(clustered, features=[clustered.var_names[0]], kind="violin")


def test_an_empty_subpopulation_table_is_refused_rather_than_drawn(clustered):
    empty = clustered.copy()
    with pytest.warns(UserWarning):
        mt.tl.subpopulation_hits(empty, cluster_key="leiden")
    empty.uns["mantispy"]["subpopulation_hits"] = empty.uns["mantispy"]["subpopulation_hits"].iloc[:0]
    with pytest.raises(ValueError, match="is empty"):
        mt.pl.subpopulation_hits(empty)


def test_density_plot_draws_and_says_what_to_run_first(clustered):
    rng = np.random.default_rng(0)
    clustered.obs["Metadata_Center_X"] = rng.uniform(0, 1000, clustered.n_obs)
    clustered.obs["Metadata_Center_Y"] = rng.uniform(0, 1000, clustered.n_obs)

    with pytest.raises(KeyError, match="mt.tl.neighbors_local_density"):
        mt.pl.density(clustered, feature=clustered.var_names[0])

    mt.tl.neighbors_local_density(clustered, k=5)
    assert isinstance(mt.pl.density(clustered, feature=clustered.var_names[0]), matplotlib.axes.Axes)
