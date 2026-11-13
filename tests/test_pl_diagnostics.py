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
    assert mt.pl.plate(diagnosed, color="qc_well_fail") is None
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


def _tall_controls(n_controls=600, n_features=20, seed=0):
    """Far more control rows than features, so control_drift's auto solver takes the covariance path."""
    import anndata as ad
    import pandas as pd

    rng = np.random.default_rng(seed)
    values = rng.standard_normal((n_controls, n_features)) @ rng.standard_normal((n_features, n_features))
    obs = pd.DataFrame(
        {
            "Metadata_Plate": np.where(np.arange(n_controls) % 2 == 0, "p0", "p1"),
            "Metadata_Control_Type": "negcon",
        },
        index=[str(index) for index in range(n_controls)],
    )
    return ad.AnnData(X=values.astype(np.float32), obs=obs)


def test_control_drift_auto_matches_the_full_solver_embedding():
    """scikit-learn's covariance and full solvers share a sign convention, so the scatter is identical."""
    from mantispy._core._pca import pca_solver

    assert pca_solver(600, 20, fallback="auto") == "covariance_eigh"
    adata = _tall_controls()
    auto = mt.pl.control_drift(adata, ax=plt.subplots()[1])
    full = mt.pl.control_drift(adata, svd_solver="full", ax=plt.subplots()[1])
    # Same groups in the same order, so each scatter's points line up for a direct comparison.
    for auto_group, full_group in zip(auto.collections, full.collections, strict=True):
        np.testing.assert_allclose(auto_group.get_offsets().data, full_group.get_offsets().data, rtol=1e-4, atol=1e-4)
    plt.close("all")
