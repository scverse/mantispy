"""Plots for hits, effects, dose and mechanism.

One smoke test over the namespace, plus assertions where a plot computes something or has to explain itself.
"""

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import mantispy as mt
from mantispy.io._profiles import from_dataframe


@pytest.fixture(scope="module")
def scored():
    cells = mt.ds.synthetic_plate(
        n_plates=2, n_wells=96, n_cells=10, n_features=25, n_perturbations=4, effect_size=4.0, seed=0
    )
    mt.pp.normalize(cells, by="Metadata_Plate", reference="negcon")
    wells = mt.tl.aggregate(cells, min_cells=0)
    wells.obs["Metadata_Compound"] = wells.obs["Metadata_Perturbation"].astype(str).to_numpy()
    wells.obs["Metadata_MOA"] = wells.obs["Metadata_Perturbation"].astype(str).to_numpy()
    wells.obs["Metadata_Concentration"] = np.tile([0.1, 1.0, 10.0, 100.0], wells.n_obs)[: wells.n_obs]

    mt.tl.hit_calling(wells, n_permutations=15)  # plot fixture: the tests draw and check non-mutation, not a p-value
    mt.tl.effect_size(wells)
    mt.tl.enrich(wells, by="feature_group", tmin=2)
    mt.tl.nn_moa_classify(wells, scheme="nn")
    mt.tl.moa_enrichment(wells, k=5)
    mt.tl.dose_response(wells, min_doses=4)
    mt.tl.edistance(wells, reference=None)
    return wells


@pytest.mark.parametrize(
    "draw",
    [
        lambda a, ax: mt.pl.hits(a, ax=ax),
        lambda a, ax: mt.pl.effect_sizes(a, group="pert00", ax=ax),
        lambda a, ax: mt.pl.feature_volcano(a, group="pert00", ax=ax),
        lambda a, ax: mt.pl.dose_response(a, compound="pert00", ax=ax),
        lambda a, ax: mt.pl.moa_confusion(a, ax=ax),
        lambda a, ax: mt.pl.moa_enrichment(a, group="pert00", ax=ax),
        lambda a, ax: mt.pl.distance_heatmap(a, ax=ax),
        lambda a, ax: mt.pl.sets_heatmap(a, groupby="Metadata_Perturbation", ax=ax),
    ],
    ids=["hits", "effects", "volcano", "dose", "confusion", "enrichment", "distances", "sets"],
)
def test_every_plot_draws_and_changes_nothing(scored, draw):
    columns, values = list(scored.obs.columns), scored.X.copy()
    # Passing ax makes the plot hand it back, so this still asserts on the drawn Axes.
    _, ax = plt.subplots()
    assert draw(scored, ax) is ax
    assert list(scored.obs.columns) == columns
    np.testing.assert_array_equal(scored.X, values)
    plt.close(ax.figure)


def test_the_hits_plot_draws_the_threshold_that_colored_the_points():
    """Reading the threshold under the table key rather than the function name drew a run called at q < 0.25 against a line labelled q = 0.05, with a point colored as a hit below it."""
    # 96 wells over 3 perturbations leaves 24 controls; fewer let the over-calling of #60 pass as the effect.
    cells = mt.ds.synthetic_plate(
        n_plates=1, n_wells=96, n_cells=6, n_features=8, n_perturbations=3, effect_size=3.0, seed=0
    )
    wells = mt.tl.aggregate(cells, min_cells=0)
    mt.tl.hit_calling(wells, n_permutations=15, threshold=0.25)  # the plot marks the threshold; not a p-value assertion

    _, ax = plt.subplots()
    mt.pl.hits(wells, ax=ax)
    line = next(drawn for drawn in ax.get_lines() if str(drawn.get_label()).startswith("q ="))
    assert line.get_label() == "q = 0.25"

    height = float(line.get_ydata()[0])
    assert height == pytest.approx(-np.log10(0.25))
    called = next(group for group in ax.collections if group.get_label() == "hit").get_offsets()
    assert called.shape[0] and (called[:, 1] >= height).all()


def test_plots_say_what_to_run_first():
    fresh = mt.tl.aggregate(mt.ds.synthetic_plate(n_wells=8, n_cells=4, n_features=8, seed=0), min_cells=0)
    with pytest.raises(KeyError, match="mt.tl.hit_calling"):
        mt.pl.hits(fresh)
    with pytest.raises(KeyError, match="mt.tl.effect_size"):
        mt.pl.effect_sizes(fresh, group="DMSO")
    with pytest.raises(KeyError, match="mt.tl.enrich"):
        mt.pl.sets_heatmap(fresh, groupby="Metadata_Perturbation")


def test_design_plots_draw_and_say_what_to_run_first(scored):
    mt.tl.replicate_saturation(scored, n_draws=2, max_replicates=3)
    scored.obs["Metadata_CellCount"] = np.where(
        scored.obs["Metadata_Perturbation"].astype(str) == "pert00", 10.0, 100.0
    )
    mt.tl.cytotoxicity(scored)
    _, saturation_ax = plt.subplots()
    assert isinstance(mt.pl.replicate_saturation(scored, ax=saturation_ax), matplotlib.axes.Axes)
    _, cytotoxicity_ax = plt.subplots()
    assert isinstance(mt.pl.cytotoxicity(scored, ax=cytotoxicity_ax), matplotlib.axes.Axes)

    fresh = mt.tl.aggregate(mt.ds.synthetic_plate(n_wells=8, n_cells=4, n_features=8, seed=0), min_cells=0)
    with pytest.raises(KeyError, match="mt.tl.replicate_saturation"):
        mt.pl.replicate_saturation(fresh)
    with pytest.raises(KeyError, match="mt.tl.cytotoxicity"):
        mt.pl.cytotoxicity(fresh)


@pytest.fixture
def inhibitor_adata():
    """One compound whose response falls from 10 to 2 with an EC50 of 1."""
    from mantispy.tl._dose import four_parameter_logistic

    doses = np.repeat(np.geomspace(0.01, 100.0, 8), 3)
    frame = pd.DataFrame(
        {
            "Metadata_Compound": "cpd",
            "Metadata_Concentration": doses,
            "Metadata_Plate": "P1",
            "Metadata_Well": [f"A{i + 1:02d}" for i in range(doses.size)],
            "Cells_AreaShape_Area": 1.0,
            "Cells_AreaShape_Perimeter": 2.0,
        }
    )
    adata = from_dataframe(frame)
    adata.obs["hits_row_distance"] = four_parameter_logistic(np.log10(doses), 10.0, 2.0, 0.0, 1.0)
    mt.tl.dose_response(adata)
    return adata


def test_the_direction_plot_bands_the_ladder_by_phase(phenotypes):
    mt.tl.dose_direction(phenotypes)
    _, ax = plt.subplots()
    mt.pl.dose_direction(phenotypes, compound="grows", ax=ax)
    table = phenotypes.uns["mantispy"]["dose_direction"]
    drawn_compound = table[table["compound"] == "grows"]

    assert len(ax.patches) == len(drawn_compound)
    colours = {patch.get_facecolor() for patch in ax.patches}
    assert len(colours) == drawn_compound["phase"].nunique(), "each phase present gets its own colour"
    assert len(ax.lines) == 4, "two curves and the two floors"
    plt.close(ax.figure)


def test_the_direction_plot_says_which_compounds_it_has(phenotypes):
    mt.tl.dose_direction(phenotypes)
    with pytest.raises(KeyError, match="grows"):
        mt.pl.dose_direction(phenotypes, compound="not_dosed")
    plt.close("all")


def test_one_stray_well_does_not_flatten_the_rest_onto_the_baseline(inhibitor_adata):
    """A distance from the controls has a long right tail, so a linear axis hides the response."""
    _, ax = plt.subplots()
    mt.pl.dose_response(inhibitor_adata, compound="cpd", ax=ax)
    assert ax.get_yscale() == "log"

    # A response that reaches zero has no log scale to be drawn on.
    inhibitor_adata.obs.loc[inhibitor_adata.obs.index[0], "hits_row_distance"] = 0.0
    _, linear_ax = plt.subplots()
    mt.pl.dose_response(inhibitor_adata, compound="cpd", ax=linear_ax)
    assert linear_ax.get_yscale() == "linear"
