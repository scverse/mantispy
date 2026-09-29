"""Plot tests.

One smoke test covering the whole namespace, plus assertions where a plot computes something (the plate grid, the QC dashboard).
"""

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

import matplotlib.pyplot as plt

import mantispy as mt
from mantispy.ds import synthetic_plate


def _panels(figure):
    """The drawn panels of an owned figure, in panel order, without the shared colorbar axes."""
    return [axis for axis in figure.axes if axis.images]


@pytest.fixture
def plotted():
    adata = synthetic_plate(n_plates=2, n_wells=96, n_cells=5, n_features=10, seed=0)
    mt.pp.calculate_qc_metrics(adata)
    mt.pp.normalize(adata, keep_raw=True)
    return adata


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
    opened = set(plt.get_fignums())
    # Each plot owns its figure, so it draws and returns None rather than the Axes.
    assert draw(plotted) is None
    assert set(plt.get_fignums()) - opened, "the plot drew its own figure"
    assert (list(plotted.obs.columns), list(plotted.var.columns)) == before[:2]
    np.testing.assert_array_equal(plotted.X, before[2])
    plt.close("all")


def test_cell_counts_draws_the_count_profiles_carry(plotted):
    """A well is one row of a profile object; counting rows would draw one cell per well."""
    wells = mt.tl.aggregate(plotted, min_cells=0)
    _, ax = plt.subplots()
    mt.pl.cell_counts(wells, ax=ax)
    assert {float(y) for line in ax.get_lines() for y in line.get_ydata()} == {5.0}
    # An unknown count is left out of its box rather than blanking it.
    wells.obs["n"] = np.where(wells.obs_names == wells.obs_names[0], np.nan, 2.0)
    _, ax = plt.subplots()
    mt.pl.cell_counts(wells, count_key="n", ax=ax)
    assert {float(y) for line in ax.get_lines() for y in line.get_ydata()} == {2.0}
    wells.obs["group"] = pd.array(["a"] * (wells.n_obs - 1) + [None], dtype="string")
    _, ax = plt.subplots()
    mt.pl.cell_counts(wells, groupby="group", ax=ax)
    assert [label.get_text() for label in ax.get_xticklabels()] == ["a", "<NA>"]

    del wells.obs["Metadata_CellCount"]
    with pytest.raises(KeyError, match="count_key="):
        mt.pl.cell_counts(wells)
    # qc leaves out the panel it has nothing for; it owns its figure, so inspect the grid it drew.
    assert mt.pl.qc(wells) is None
    assert len(plt.gcf().axes) == 4
    plt.close("all")


def test_plate_one_panel_per_plate_and_rejects_unknown_color(plotted):
    # plate owns the figure over both plates, so it draws and returns None; count the panels it drew.
    assert mt.pl.plate(plotted, color=plotted.var_names[0]) is None
    assert len(_panels(plt.gcf())) == 2
    plt.close("all")
    with pytest.raises(KeyError, match="nonexistent"):
        mt.pl.plate(plotted, color="nonexistent")
    with pytest.raises(KeyError, match="Plate09"):
        mt.pl.plate(plotted, color=plotted.var_names[0], plate="Plate09")

    plotted.obs["Metadata_Plate"] = plotted.obs["Metadata_Plate"].str[-2:].astype(int)
    # A single plate can take a caller ax, which is handed back for its title.
    _, ax = plt.subplots()
    mt.pl.plate(plotted, color=plotted.var_names[0], plate=1, ax=ax)
    assert ax.get_title() == "1"


def test_plates_fill_a_grid_on_one_shared_scale():
    many = synthetic_plate(n_plates=5, n_wells=96, n_cells=2, n_features=3, seed=0)
    assert mt.pl.plate(many, color=many.var_names[0], ncols=2) is None
    figure = plt.gcf()
    axes = _panels(figure)
    assert len(axes) == 5
    assert axes[0].get_subplotspec().get_gridspec().get_geometry() == (3, 2)
    assert len({axis.images[0].get_clim() for axis in axes}) == 1
    assert len(figure.axes) == 6, "five panels and one colorbar"
    plt.close("all")

    assert mt.pl.plate(many, color=many.var_names[0], share_colorbar=False) is None
    apart = _panels(plt.gcf())
    assert len({axis.images[0].get_clim() for axis in apart}) == 5
    plt.close("all")

    values = many.X.copy()
    values[(many.obs["Metadata_Plate"] == "Plate01").to_numpy(), 0] = np.nan
    many.X = values
    assert mt.pl.plate(many, color=many.var_names[0]) is None
    assert np.isfinite(_panels(plt.gcf())[1].images[0].get_clim()).all()
    plt.close("all")
    assert mt.pl.plate(many, color=many.var_names[0], norm=matplotlib.colors.Normalize(0, 1)) is None
    assert _panels(plt.gcf())[1].images[0].get_clim() == (0, 1)
    plt.close("all")


def test_groupby_orders_and_titles_the_plates(plotted):
    plotted.obs["Metadata_Batch"] = np.where(plotted.obs["Metadata_Plate"] == "Plate01", "batch10", "batch2")
    assert mt.pl.plate(plotted, color=plotted.var_names[0], groupby="Metadata_Batch") is None
    axes = _panels(plt.gcf())
    assert [axis.get_title() for axis in axes] == ["batch2 · Plate02", "batch10 · Plate01"]  # digits sort as numbers
    plt.close("all")

    with pytest.raises(KeyError, match="Metadata_Nope"):
        mt.pl.plate(plotted, color=plotted.var_names[0], groupby="Metadata_Nope")
    plotted.obs["Metadata_Batch"] = np.where(
        plotted.obs["Metadata_Plate"] == "Plate01", "b", np.arange(plotted.n_obs) % 2
    )
    with pytest.raises(ValueError, match="varies within a plate"):
        mt.pl.plate(plotted, color=plotted.var_names[0], groupby="Metadata_Batch")
    # Only the plates drawn have to be constant in it; a single plate takes a caller ax.
    _, ax = plt.subplots()
    mt.pl.plate(plotted, color=plotted.var_names[0], plate="Plate01", groupby="Metadata_Batch", ax=ax)
    assert ax.get_title() == "b · Plate01"
    plt.close("all")
    plotted.obs["Metadata_Batch"] = None
    with pytest.raises(ValueError, match="missing values"):
        mt.pl.plate(plotted, color=plotted.var_names[0], groupby="Metadata_Batch")
