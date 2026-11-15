"""PHATE parity with the phate package, a test-only dependency."""

import anndata as ad
import numpy as np
import pytest
from scipy.spatial import procrustes
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr

import mantispy as mt

phate = pytest.importorskip("phate")


def _manifold(n, n_features, seed):
    """Six clusters on a curved surface, roughly the shape principal components of profiles take."""
    rng = np.random.default_rng(seed)
    latent = rng.normal(size=(6, 5))[rng.integers(0, 6, n)] * 3 + rng.normal(size=(n, 5))
    return np.tanh(latent @ rng.normal(size=(5, n_features))) + 0.2 * rng.normal(size=(n, n_features))


def _reference(x, **kwargs):
    operator = phate.PHATE(n_pca=None, random_state=0, verbose=0, **kwargs)
    embedding = operator.fit_transform(x)
    return operator, embedding


@pytest.mark.parametrize(("knn", "decay", "seed"), [(5, 40, 0), (10, 15, 1)])
def test_the_exact_path_matches_phate_for_the_same_t(knn, decay, seed):
    x = _manifold(500, 12, seed)
    operator, embedding = _reference(x, knn=knn, decay=decay, n_landmark=None)
    adata = ad.AnnData(np.zeros((x.shape[0], 1)), obsm={"X_pca": x})
    mt.tl.phate(adata, knn=knn, decay=decay, n_landmark=None, t=int(operator.optimal_t))

    # phate rounds its distances to single precision, so the kernel agrees to that.
    np.testing.assert_allclose(adata.obsp["phate_connectivities"].toarray(), operator.graph.K.toarray(), atol=1e-5)
    np.testing.assert_allclose(adata.obsm["X_phate_potential"], operator.diff_potential, atol=1e-3)
    # Both minimise the same stress, SMACOF here and SGD in phate, which stops short of the minimum.
    assert procrustes(adata.obsm["X_phate"], embedding)[2] < 5e-3


def test_the_landmark_path_tracks_phate():
    """Landmarks are k-means clusters on both sides, so they differ; the potential distances and layout still agree."""
    x = _manifold(3000, 10, 2)
    operator, embedding = _reference(x, n_landmark=300)
    adata = ad.AnnData(np.zeros((x.shape[0], 1)), obsm={"X_pca": x})
    mt.tl.phate(adata, n_landmark=300, t=int(operator.optimal_t))

    rows = np.arange(0, x.shape[0], 5)
    ours, theirs = adata.obsm["X_phate_potential"][rows], np.asarray(operator.diff_potential)[rows]
    assert spearmanr(pdist(ours), pdist(theirs))[0] > 0.95
    assert procrustes(adata.obsm["X_phate"], embedding)[2] < 0.05
