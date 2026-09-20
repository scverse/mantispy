"""Plots for hits, effects, dose and mechanism.

One smoke test over the namespace, plus assertions where a plot computes something or
has to explain itself.
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

    mt.tl.hit_calling(wells, n_permutations=100)
    mt.tl.effect_size(wells)
    mt.tl.enrich(wells, by="feature_group", tmin=2)
    mt.tl.nn_moa_classify(wells, scheme="nn")
    mt.tl.moa_enrichment(wells, k=5)
    mt.tl.dose_response(wells, min_doses=4)
    mt.tl.edistance(wells, reference=None)
    return wells


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


@pytest.mark.parametrize(
    "draw",
    [
        lambda a: mt.pl.hits(a),
        lambda a: mt.pl.effect_sizes(a, group="pert00"),
        lambda a: mt.pl.feature_volcano(a, group="pert00"),
        lambda a: mt.pl.dose_response(a, compound="pert00"),
        lambda a: mt.pl.moa_confusion(a),
        lambda a: mt.pl.moa_enrichment(a, group="pert00"),
        lambda a: mt.pl.distance_heatmap(a),
        lambda a: mt.pl.sets_heatmap(a, groupby="Metadata_Perturbation"),
    ],
    ids=["hits", "effects", "volcano", "dose", "confusion", "enrichment", "distances", "sets"],
)
def test_every_plot_draws_and_changes_nothing(scored, draw):
    columns, values = list(scored.obs.columns), scored.X.copy()
    axes = np.asarray(draw(scored))
    assert all(isinstance(axis, matplotlib.axes.Axes) for axis in axes.ravel())
    assert list(scored.obs.columns) == columns
    np.testing.assert_array_equal(scored.X, values)


def test_the_hits_plot_marks_the_threshold_the_run_used(scored):
    ax = mt.pl.hits(scored)
    assert any(abs(float(line.get_ydata()[0]) + np.log10(0.05)) < 1e-9 for line in ax.get_lines())


def test_the_hits_plot_draws_the_threshold_that_colored_the_points():
    """Reading the threshold under the table key rather than the function name drew a run called at q < 0.25 against a line labelled q = 0.05, with a point colored as a hit below it."""
    # 96 wells over 3 perturbations leaves 24 controls, so the reference is large enough for the null to
    # mean something. At 48 wells over 11 it left four, two of which formed the whole null, and the hit this
    # asserts on was the over-calling of #60 rather than the effect.
    cells = mt.ds.synthetic_plate(
        n_plates=1, n_wells=96, n_cells=6, n_features=8, n_perturbations=3, effect_size=3.0, seed=0
    )
    wells = mt.tl.aggregate(cells, min_cells=0)
    mt.tl.hit_calling(wells, n_permutations=200, threshold=0.25)

    ax = mt.pl.hits(wells)
    line = next(drawn for drawn in ax.get_lines() if str(drawn.get_label()).startswith("q ="))
    assert line.get_label() == "q = 0.25"

    height = float(line.get_ydata()[0])
    assert height == pytest.approx(-np.log10(0.25))
    # Every point colored as a hit has to sit on or above the line that called it.
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


def test_asking_for_a_group_that_is_not_there_lists_what_is(scored):
    with pytest.raises(KeyError, match="it holds"):
        mt.pl.effect_sizes(scored, group="not_a_perturbation")


def test_design_plots_draw_and_say_what_to_run_first(scored):
    mt.tl.replicate_saturation(scored, n_draws=2, max_replicates=3)
    scored.obs["Metadata_CellCount"] = np.where(
        scored.obs["Metadata_Perturbation"].astype(str) == "pert00", 10.0, 100.0
    )
    mt.tl.cytotoxicity(scored)
    assert isinstance(mt.pl.replicate_saturation(scored), matplotlib.axes.Axes)
    assert isinstance(mt.pl.cytotoxicity(scored), matplotlib.axes.Axes)

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


def test_an_inhibitory_curve_is_drawn_the_way_the_data_runs(inhibitor_adata):
    ax = mt.pl.dose_response(inhibitor_adata, compound="cpd")
    line = next(line for line in ax.get_lines() if line.get_label().startswith("EC50"))
    drawn = line.get_ydata()
    assert drawn[0] > drawn[-1], "the curve runs uphill while the data runs downhill"


def test_the_direction_plot_bands_the_ladder_by_phase(phenotypes):
    mt.tl.dose_direction(phenotypes)
    ax = mt.pl.dose_direction(phenotypes, compound="grows")
    table = phenotypes.uns["mantispy"]["dose_direction"]
    drawn_compound = table[table["compound"] == "grows"]

    # One band per concentration of the compound drawn, plus the two lines it reads against.
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
    assert mt.pl.dose_response(inhibitor_adata, compound="cpd").get_yscale() == "log"

    # A response that reaches zero has no log scale to be drawn on.
    inhibitor_adata.obs.loc[inhibitor_adata.obs.index[0], "hits_row_distance"] = 0.0
    assert mt.pl.dose_response(inhibitor_adata, compound="cpd").get_yscale() == "linear"
