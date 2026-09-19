"""Plot tests.

One smoke test covering the whole namespace, plus assertions where a plot computes
something (the plate grid, the QC dashboard).
"""

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import mantispy as mt
from mantispy.ds import synthetic_plate


@pytest.fixture
def plotted():
    adata = synthetic_plate(n_plates=2, n_wells=96, n_cells=5, n_features=10, seed=0)
    mt.pp.calculate_qc_metrics(adata)
    mt.pp.normalize(adata, keep_raw=True)
    return adata


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


@pytest.mark.parametrize(
    "draw",
    [
        lambda a: mt.pl.plate(a, color=a.var_names[0]),
        lambda a: mt.pl.cell_counts(a),
        lambda a: mt.pl.feature_distributions(a, features=list(a.var_names[:2])),
        lambda a: mt.pl.nan_matrix(a),
        lambda a: mt.pl.qc(a),
    ],
    ids=["plate", "cell_counts", "feature_distributions", "nan_matrix", "qc"],
)
def test_every_plot_draws_and_does_not_mutate(plotted, draw):
    before = (list(plotted.obs.columns), list(plotted.var.columns), plotted.X.copy())
    result = np.asarray(draw(plotted))
    assert result.size >= 1
    assert all(isinstance(axis, matplotlib.axes.Axes) for axis in result.ravel())
    assert (list(plotted.obs.columns), list(plotted.var.columns)) == before[:2]
    np.testing.assert_array_equal(plotted.X, before[2])


def test_cell_counts_draws_the_count_profiles_carry(plotted):
    """A well is one row of a profile object; counting rows would draw one cell per well."""
    wells = mt.tl.aggregate(plotted, min_cells=0)
    ax = mt.pl.cell_counts(wells)
    assert {float(y) for line in ax.get_lines() for y in line.get_ydata()} == {5.0}
    # An unknown count is left out of its box rather than blanking it.
    wells.obs["n"] = np.where(wells.obs_names == wells.obs_names[0], np.nan, 2.0)
    ax = mt.pl.cell_counts(wells, count_key="n")
    assert {float(y) for line in ax.get_lines() for y in line.get_ydata()} == {2.0}
    # A missing label is a box of its own.
    wells.obs["group"] = pd.array(["a"] * (wells.n_obs - 1) + [None], dtype="string")
    ax = mt.pl.cell_counts(wells, groupby="group")
    assert [label.get_text() for label in ax.get_xticklabels()] == ["a", "<NA>"]

    del wells.obs["Metadata_CellCount"]
    with pytest.raises(KeyError, match="count_key="):
        mt.pl.cell_counts(wells)
    # qc leaves out the panel it has nothing for.
    assert np.asarray(mt.pl.qc(wells)).size == 4


def test_plate_grid_matches_the_detected_format(plotted):
    ax = mt.pl.plate(plotted, color=plotted.var_names[0], plate="Plate01")
    assert isinstance(ax, matplotlib.axes.Axes)
    assert ax.images[0].get_array().shape == (8, 12)


def test_each_plate_is_drawn_on_its_own_grid(plotted):
    """The grid was sized from every plate's wells at once, so the smaller plate was drawn three quarters empty.

    jump_target2 is the real case: source_9 runs the map on 1536-well plates and the other ten on 384-well ones.
    """
    wells = plotted.obs["Metadata_Well"].astype(str).to_numpy()
    larger = (plotted.obs["Metadata_Plate"] == "Plate02").to_numpy()
    # One well in the far corner is enough; the format is the smallest one that holds every well.
    wells[np.flatnonzero(larger)[0]] = "P24"
    plotted.obs["Metadata_Well"] = wells

    axes = mt.pl.plate(plotted, color=plotted.var_names[0])
    assert [axis.images[0].get_array().shape for axis in axes] == [(8, 12), (16, 24)]


def test_plate_one_panel_per_plate_and_rejects_unknown_color(plotted):
    assert np.asarray(mt.pl.plate(plotted, color=plotted.var_names[0])).size == 2
    with pytest.raises(KeyError, match="nonexistent"):
        mt.pl.plate(plotted, color="nonexistent")
    with pytest.raises(KeyError, match="Plate09"):
        mt.pl.plate(plotted, color=plotted.var_names[0], plate="Plate09")

    plotted.obs["Metadata_Plate"] = plotted.obs["Metadata_Plate"].str[-2:].astype(int)
    assert mt.pl.plate(plotted, color=plotted.var_names[0], plate=1).get_title() == "1"


def test_plates_fill_a_grid_on_one_shared_scale():
    many = synthetic_plate(n_plates=5, n_wells=96, n_cells=2, n_features=3, seed=0)
    axes = mt.pl.plate(many, color=many.var_names[0], ncols=2)
    assert len(axes) == 5
    assert axes[0].get_subplotspec().get_gridspec().get_geometry() == (3, 2)
    assert len({axis.images[0].get_clim() for axis in axes}) == 1
    assert len(axes[0].figure.axes) == 6, "five panels and one colorbar"

    apart = mt.pl.plate(many, color=many.var_names[0], share_colorbar=False)
    assert len({axis.images[0].get_clim() for axis in apart}) == 5

    # A plate with nothing measured leaves the others' scale alone, and a norm replaces it.
    values = many.X.copy()
    values[(many.obs["Metadata_Plate"] == "Plate01").to_numpy(), 0] = np.nan
    many.X = values
    assert np.isfinite(mt.pl.plate(many, color=many.var_names[0])[1].images[0].get_clim()).all()
    normed = mt.pl.plate(many, color=many.var_names[0], norm=matplotlib.colors.Normalize(0, 1))
    assert normed[1].images[0].get_clim() == (0, 1)


def test_groupby_orders_and_titles_the_plates(plotted):
    plotted.obs["Metadata_Batch"] = np.where(plotted.obs["Metadata_Plate"] == "Plate01", "batch10", "batch2")
    axes = mt.pl.plate(plotted, color=plotted.var_names[0], groupby="Metadata_Batch")
    assert [axis.get_title() for axis in axes] == ["batch2 · Plate02", "batch10 · Plate01"]  # digits sort as numbers

    with pytest.raises(KeyError, match="Metadata_Nope"):
        mt.pl.plate(plotted, color=plotted.var_names[0], groupby="Metadata_Nope")
    plotted.obs["Metadata_Batch"] = np.where(
        plotted.obs["Metadata_Plate"] == "Plate01", "b", np.arange(plotted.n_obs) % 2
    )
    with pytest.raises(ValueError, match="varies within a plate"):
        mt.pl.plate(plotted, color=plotted.var_names[0], groupby="Metadata_Batch")
    # Only the plates drawn have to be constant in it.
    assert (
        mt.pl.plate(plotted, color=plotted.var_names[0], plate="Plate01", groupby="Metadata_Batch").get_title()
        == "b · Plate01"
    )
    plotted.obs["Metadata_Batch"] = None
    with pytest.raises(ValueError, match="missing values"):
        mt.pl.plate(plotted, color=plotted.var_names[0], groupby="Metadata_Batch")


def test_feature_distributions_shows_raw_and_current(plotted):
    axes = mt.pl.feature_distributions(plotted, features=list(plotted.var_names[:2]))
    assert axes.shape == (2, 2)  # raw and current, two features


def test_qc_survives_an_all_nan_feature(plotted):
    """log10 of a NaN variance must not break the whole dashboard."""
    values = plotted.X.copy()
    values[:, 0] = np.nan
    plotted.X = values
    mt.pp.calculate_qc_metrics(plotted)
    assert np.asarray(mt.pl.qc(plotted)).size == 4
