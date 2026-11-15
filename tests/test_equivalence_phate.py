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


def _reference(x, random_state=0, **kwargs):
    operator = phate.PHATE(n_pca=None, random_state=random_state, verbose=0, **kwargs)
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


def test_the_landmark_path_tracks_phate_as_closely_as_phate_tracks_itself():
    """Landmarks are k-means clusters, so they move with the seed in phate too; that spread sets the tolerance.

    The two implementations choose landmarks from different diffusion coordinates, which costs a little more agreement
    than a change of seed, hence the margin.
    """
    x = _manifold(3000, 10, 2)
    t = 25
    first, embedding = _reference(x, n_landmark=300, t=t)
    second, _ = _reference(x, n_landmark=300, t=t, random_state=1)
    adata = ad.AnnData(np.zeros((x.shape[0], 1)), obsm={"X_pca": x})
    mt.tl.phate(adata, n_landmark=300, t=t)

    rows = np.arange(0, x.shape[0], 5)
    theirs = pdist(np.asarray(first.diff_potential)[rows])
    across_seeds = spearmanr(theirs, pdist(np.asarray(second.diff_potential)[rows]))[0]
    ours = spearmanr(theirs, pdist(adata.obsm["X_phate_potential"][rows]))[0]
    assert ours > across_seeds - 0.05
    assert procrustes(adata.obsm["X_phate"], embedding)[2] < 0.05
