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


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


@pytest.mark.parametrize(
    "draw",
    [
        lambda a: mt.pl.plate_effects(a),
        lambda a: mt.pl.image_qc(a),
        lambda a: mt.pl.control_drift(a),
        lambda a: mt.pl.outliers(a),
        lambda a: mt.pl.feature_correlation(a),
        lambda a: mt.pl.feature_groups(a),
    ],
    ids=["plate_effects", "image_qc", "control_drift", "outliers", "feature_correlation", "feature_groups"],
)
def test_every_plot_draws_and_does_not_mutate(diagnosed, draw):
    before = (list(diagnosed.obs.columns), list(diagnosed.var.columns), diagnosed.X.copy())
    result = np.asarray(draw(diagnosed))
    assert all(isinstance(axis, matplotlib.axes.Axes) for axis in result.ravel())
    assert (list(diagnosed.obs.columns), list(diagnosed.var.columns)) == before[:2]
    np.testing.assert_array_equal(diagnosed.X, before[2])


def test_plate_effects_has_a_row_and_column_panel_per_plate(diagnosed):
    assert mt.pl.plate_effects(diagnosed).shape == (2, 2)


def test_feature_correlation_is_ordered_by_annotation(diagnosed):
    """Ordering by feature group is what makes the block structure readable."""
    ax = mt.pl.feature_correlation(diagnosed, key=None)
    assert ax.images[0].get_array().shape == (diagnosed.n_vars, diagnosed.n_vars)
    assert len(ax.get_xticklabels()) == diagnosed.var["feature_group"].nunique()


def test_plots_say_what_to_run_first(diagnosed):
    fresh = synthetic_plate(n_wells=8, n_cells=4, n_features=8, seed=0)
    with pytest.raises(KeyError, match="mt.pp.image_qc"):
        mt.pl.image_qc(fresh)
    with pytest.raises(KeyError, match="mt.pp.outliers"):
        mt.pl.outliers(fresh)


def test_well_level_pass_maps_reuse_pl_plate(diagnosed):
    """No dedicated pl.well_qc: the plate heatmap already draws any obs column."""
    mt.pp.well_qc(diagnosed, min_cells=5)
    assert np.asarray(mt.pl.plate(diagnosed, color="qc_well_pass")).size == 2


def test_control_drift_reads_an_infinity_as_missing(diagnosed):
    """Regression test for #65: filled as the largest float, one infinite control was drawn at 3e38."""
    control = np.flatnonzero(diagnosed.obs["Metadata_Control"].to_numpy(dtype=bool))[0]
    drawn = []
    for value in (np.inf, np.nan):
        adata = diagnosed.copy()
        adata.X[control, 0] = value
        axes = mt.pl.control_drift(adata)
        drawn.append(np.vstack([points.get_offsets() for points in axes.collections]))
    np.testing.assert_array_equal(*drawn)
