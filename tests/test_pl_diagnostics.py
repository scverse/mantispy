"""The 0.2 diagnostic and feature-space plots."""

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

import matplotlib.pyplot as plt

import mantispy as mt
from mantispy.ds import synthetic_plate


@pytest.fixture
def diagnosed():
    adata = synthetic_plate(
        n_plates=2,
        n_wells=48,
        n_cells=10,
        n_features=20,
        n_images_per_well=2,
        n_bad_images=4,
        n_correlated_pairs=3,
        row_gradient=2.0,
        seed=0,
    )
    mt.pp.calculate_qc_metrics(adata)
    mt.pp.image_qc(adata)
    mt.pp.outliers(adata, method="mad")
    mt.pp.feature_select(adata)
    return adata


@pytest.mark.parametrize(
    "draw",
    [
        lambda a: mt.pl.plate_effects(a, axes=plt.subplots(a.obs["Metadata_Plate"].nunique(), 2, squeeze=False)[1]),
        lambda a: mt.pl.image_qc(a, ax=plt.subplots()[1]),
        lambda a: mt.pl.control_drift(a, ax=plt.subplots()[1]),
        lambda a: mt.pl.outliers(a, axes=plt.subplots(1, 2)[1]),
        lambda a: mt.pl.feature_correlation(a, ax=plt.subplots()[1]),
        lambda a: mt.pl.feature_groups(a, ax=plt.subplots()[1]),
    ],
    ids=["plate_effects", "image_qc", "control_drift", "outliers", "feature_correlation", "feature_groups"],
)
def test_every_plot_draws_and_does_not_mutate(diagnosed, draw):
    before = (list(diagnosed.obs.columns), list(diagnosed.var.columns), diagnosed.X.copy())
    # Passing axes makes each plot hand them back, so this still asserts on the drawn Axes.
    result = np.asarray(draw(diagnosed))
    assert all(isinstance(axis, matplotlib.axes.Axes) for axis in result.ravel())
    assert (list(diagnosed.obs.columns), list(diagnosed.var.columns)) == before[:2]
    np.testing.assert_array_equal(diagnosed.X, before[2])
    plt.close("all")


def test_plots_say_what_to_run_first(diagnosed):
    fresh = synthetic_plate(n_wells=8, n_cells=4, n_features=8, seed=0)
    with pytest.raises(KeyError, match="mt.pp.image_qc"):
        mt.pl.image_qc(fresh)
    with pytest.raises(KeyError, match="mt.pp.outliers"):
        mt.pl.outliers(fresh)


def test_well_level_pass_maps_reuse_pl_plate(diagnosed):
    """No dedicated pl.well_qc: the plate heatmap already draws any obs column."""
    mt.pp.well_qc(diagnosed, min_cells=5)
    # plate owns the figure over several plates, so it draws and returns None; check the panels it drew.
    assert mt.pl.plate(diagnosed, color="qc_well_pass") is None
    panels = [axis for axis in plt.gcf().axes if axis.images]
    assert len(panels) == diagnosed.obs["Metadata_Plate"].nunique()
    plt.close("all")


def test_control_drift_reads_an_infinity_as_missing(diagnosed):
    """Regression test for #65: filled as the largest float, one infinite control was drawn at 3e38."""
    control = np.flatnonzero(diagnosed.obs["Metadata_Control"].to_numpy(dtype=bool))[0]
    drawn = []
    for value in (np.inf, np.nan):
        adata = diagnosed.copy()
        adata.X[control, 0] = value
        _, ax = plt.subplots()
        mt.pl.control_drift(adata, ax=ax)
        drawn.append(np.vstack([points.get_offsets() for points in ax.collections]))
        plt.close(ax.figure)
    np.testing.assert_array_equal(*drawn)


def test_control_drift_names_the_minimum_it_draws(diagnosed):
    """Regression test for #54: n_components=1 fitted, then failed on embedding[:, 1] with a bare IndexError."""
    with pytest.raises(ValueError, match="n_components must be at least 2"):
        mt.pl.control_drift(diagnosed, n_components=1)
