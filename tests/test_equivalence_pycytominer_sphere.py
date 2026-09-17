"""Spherize equivalence against pycytominer."""

import numpy as np
import pytest

import mantispy as mt
from mantispy.ds import synthetic_plate

pycytominer = pytest.importorskip("pycytominer")


@pytest.fixture
def wells():
    # Enough control wells that the control matrix is comfortably full rank:
    # 96 wells over 9 perturbations gives ~11 DMSO wells per plate against 8 features.
    cells = synthetic_plate(n_plates=2, n_wells=96, n_cells=10, n_features=8, seed=3)
    return mt.tl.aggregate(cells, min_cells=0)


@pytest.mark.parametrize("method", ["ZCA", "ZCA-cor", "PCA", "PCA-cor"])
def test_sphere_matches_pycytominer(wells, method):
    frame = mt.get.to_dataframe(wells)
    result = pycytominer.normalize(
        profiles=frame,
        features=list(wells.var_names),
        meta_features=[c for c in frame.columns if c.startswith("Metadata_")],
        method="spherize",
        spherize_method=method,
        spherize_center=True,
        samples="Metadata_Control == True",
    )
    # PCA leaves the feature basis, so pycytominer renames the columns to PC1..PCn;
    # compare positionally rather than by name.
    expected = result[[c for c in result.columns if not c.startswith("Metadata_")]].to_numpy(np.float64)

    mt.pp.sphere(wells, method=method, reference="negcon", key_added="sphered")
    np.testing.assert_allclose(np.asarray(wells.layers["sphered"], dtype=np.float64), expected, rtol=1e-4, atol=1e-4)


def test_pca_warns_that_x_is_no_longer_features(wells):
    """var still describes features while X holds components, so PCA warns."""
    with pytest.warns(UserWarning, match="rotates out of the feature basis"):
        mt.pp.sphere(wells, method="PCA", reference="negcon")
    mt.pp.sphere(wells, method="ZCA-cor", reference="negcon")  # no warning


def test_controls_become_white(wells):
    mt.pp.sphere(wells, method="ZCA-cor", reference="negcon")
    controls = np.asarray(wells[wells.obs["Metadata_Control"]].X, dtype=np.float64)
    covariance = np.cov(controls, rowvar=False)
    np.testing.assert_allclose(np.diag(covariance), 1.0, atol=0.15)
    assert np.abs(covariance - np.diag(np.diag(covariance))).max() < 0.2


def test_rank_deficient_reference_is_refused(wells):
    """Regularising through a rank-deficient control set would give a transform
    dominated by noise directions, so pycytominer raises and so do we."""
    values = wells.X.copy()
    values[:, 1] = values[:, 0]  # a perfectly dependent feature
    wells.X = values
    with pytest.raises(ValueError, match="not full rank"):
        mt.pp.sphere(wells, method="ZCA", reference="negcon")


def test_underdetermined_reference_warns(wells):
    """Fewer control rows than features means a padded, noise-amplifying transform."""
    narrow = wells[:, : wells.n_vars].copy()
    few = narrow[narrow.obs["Metadata_Control"].to_numpy()].copy()
    few = few[: narrow.n_vars - 2].copy()
    few.obs["Metadata_Control"] = True
    with pytest.warns(UserWarning, match="fewer rows than features"):
        mt.pp.sphere(few, method="ZCA", reference="negcon")


def test_constant_feature_is_reported_for_the_cor_variants(wells):
    values = wells.X.copy()
    values[:, 0] = 1.0
    wells.X = values
    with pytest.raises(ValueError, match="zero variance"):
        mt.pp.sphere(wells, method="ZCA-cor", reference="negcon")


def test_a_missing_reference_value_is_named_rather_than_an_svd_failure(wells):
    """A NaN in a control row is named in the error instead of surfacing as "SVD did not converge"."""
    values = np.asarray(wells.X, dtype=np.float32).copy()
    control = np.flatnonzero(wells.obs["Metadata_Control"].to_numpy())
    values[control[0], 0] = np.nan
    wells.X = values

    with pytest.raises(ValueError, match="sphering needs a complete reference"):
        mt.pp.sphere(wells, method="ZCA-cor", reference="negcon")
