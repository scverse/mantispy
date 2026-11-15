"""PHATE embedding, its affinity graph and its diffusion potential."""

import logging

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from scipy.spatial import procrustes
from scipy.spatial.distance import cdist

import mantispy as mt
from mantispy._core import _phate
from mantispy.tl import _phate as tl_phate


def _blobs(n_per=40, n_groups=3, n_features=6, spread=0.5, seed=0):
    rng = np.random.default_rng(seed)
    centres = rng.normal(size=(n_groups, n_features)) * 3
    values = np.repeat(centres, n_per, axis=0) + rng.normal(scale=spread, size=(n_groups * n_per, n_features))
    n = values.shape[0]
    adata = ad.AnnData(
        np.zeros((n, 1), dtype=np.float32),
        obs=pd.DataFrame(
            {"group": np.repeat([f"g{i}" for i in range(n_groups)], n_per)}, index=[str(i) for i in range(n)]
        ),
    )
    adata.obsm["X_pca"] = values
    return adata


def _dense_kernel(x, knn, decay, metric="euclidean"):
    """The alpha-decay affinity straight from all pairwise distances, as the paper defines it."""
    d = cdist(x, x, metric=metric)
    bandwidth = np.sort(d, axis=1)[:, knn]
    directed = np.exp(-((d / bandwidth[:, None]) ** decay))
    directed[directed < _phate.THRESHOLD] = 0.0
    return (directed + directed.T) / 2


def test_writes_the_embedding_and_the_scanpy_neighbors_layout():
    adata = _blobs()
    mt.tl.phate(adata, knn=5)
    assert adata.obsm["X_phate"].shape == (adata.n_obs, 2)
    assert adata.obsm["X_phate_potential"].shape == (adata.n_obs, adata.n_obs)
    graph = adata.obsp["phate_connectivities"]
    assert sp.issparse(graph) and abs(graph - graph.T).max() == 0
    np.testing.assert_allclose(graph.diagonal(), 1.0)
    assert adata.obsp["phate_distances"].diagonal().max() == 0
    entry = adata.uns["phate"]
    assert entry["connectivities_key"] == "phate_connectivities"
    assert entry["distances_key"] == "phate_distances"
    assert entry["params"]["t"] >= 1 and entry["params"]["knn"] == 5


def test_the_kernel_keeps_every_affinity_above_the_threshold():
    """A tight cloud puts many rows inside the kernel's reach, past the first neighbour search; none may be cut off."""
    rng = np.random.default_rng(1)
    x = np.vstack([rng.normal(scale=0.01, size=(60, 3)), rng.normal(size=(60, 3)) + 5])
    for decay, metric in [(40, "euclidean"), (5, "euclidean"), (40, "cosine")]:
        adata = ad.AnnData(np.zeros((x.shape[0], 1)), obsm={"X_pca": x})
        mt.tl.phate(adata, knn=5, decay=decay, metric=metric, t=3)
        np.testing.assert_allclose(
            adata.obsp["phate_connectivities"].toarray(), _dense_kernel(x, 5, decay, metric), atol=1e-12
        )


def test_a_plain_knn_kernel_without_decay():
    adata = _blobs()
    mt.tl.phate(adata, knn=4, decay=None, t=5)
    directed = (adata.obsp["phate_connectivities"].toarray() > 0).sum(axis=1)
    # Each row reaches itself and four neighbours, and more once symmetrised.
    assert directed.min() >= 5
    assert "decay" not in adata.uns["phate"]["params"]


def test_the_exact_path_is_deterministic_and_order_free():
    adata = _blobs()
    mt.tl.phate(adata, t=10)
    first = adata.obsm["X_phate"].copy()
    mt.tl.phate(adata, t=10)
    np.testing.assert_array_equal(first, adata.obsm["X_phate"])
    order = np.random.default_rng(0).permutation(adata.n_obs)
    shuffled = adata[order].copy()
    mt.tl.phate(shuffled, t=10)
    np.testing.assert_allclose(np.abs(shuffled.obsm["X_phate"]), np.abs(first[order]), atol=1e-6)


def test_landmarks_place_every_row_and_follow_the_seed():
    adata = _blobs(n_per=100, spread=1.5)
    mt.tl.phate(adata, n_landmark=60, t=8, random_state=3)
    assert adata.uns["phate"]["params"]["n_landmark"] <= 60
    assert adata.obsm["X_phate_potential"].shape == (adata.n_obs, adata.uns["phate"]["params"]["n_landmark"])
    first = adata.obsm["X_phate"].copy()
    mt.tl.phate(adata, n_landmark=60, t=8, random_state=3)
    np.testing.assert_array_equal(first, adata.obsm["X_phate"])
    # Landmarks approximate the exact walk, so the layout matches the exact one up to rotation and scale.
    mt.tl.phate(adata, n_landmark=None, t=8)
    assert procrustes(first, adata.obsm["X_phate"])[2] < 0.1


def test_the_potential_is_stored_up_to_a_size_unless_asked(monkeypatch):
    adata = _blobs()
    monkeypatch.setattr(tl_phate, "POTENTIAL_MAX_OBS", adata.n_obs - 1)
    mt.tl.phate(adata, t=5)
    assert "X_phate_potential" not in adata.obsm
    mt.tl.phate(adata, t=5, store_potential=True, n_landmark=20)
    assert adata.obsm["X_phate_potential"].shape[0] == adata.n_obs


def test_an_automatic_t_is_recorded_and_a_given_t_is_kept():
    adata = _blobs()
    mt.tl.phate(adata, t=7)
    assert adata.uns["phate"]["params"]["t"] == 7
    mt.tl.phate(adata)
    assert 1 <= adata.uns["phate"]["params"]["t"] <= 100


def test_duplicate_rows_still_embed():
    adata = _blobs()
    adata.obsm["X_pca"][:10] = adata.obsm["X_pca"][0]
    mt.tl.phate(adata, knn=5, t=5)
    assert np.isfinite(adata.obsm["X_phate"]).all()


def test_a_disconnected_graph_warns_and_stays_finite(caplog):
    adata = _blobs(n_groups=2, spread=0.05)
    adata.obsm["X_pca"][40:] += 1e3
    with caplog.at_level(logging.WARNING, logger="scverse.mantispy"):
        mt.tl.phate(adata, t=5)
    assert any("disconnected" in record.getMessage() for record in caplog.records)
    assert np.isfinite(adata.obsm["X_phate_potential"]).all()
    mt.tl.cluster(adata, method="leiden", neighbors_key="phate")
    assert adata.obs["cluster"].nunique() >= 2


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [({"knn": 200}, "more rows"), ({"metric": "manhattan"}, "metric"), ({"gamma": 2}, "gamma"), ({"t": 0}, "t must")],
)
def test_bad_arguments_raise(kwargs, match):
    with pytest.raises(ValueError, match=match):
        mt.tl.phate(_blobs(), **kwargs)


def test_missing_values_raise():
    adata = _blobs()
    adata.obsm["X_pca"][0, 0] = np.nan
    with pytest.raises(ValueError, match="missing"):
        mt.tl.phate(adata)


def test_the_knee_of_two_lines_is_their_break():
    x = np.arange(100.0)
    for brk in (10.5, 40.5, 70.5):
        y = np.where(x < brk, -3.0 * x, -3.0 * brk - 0.2 * (x - brk))
        assert _phate.knee(y) == int(brk) + 1


def test_smacof_lowers_the_stress_of_classical_mds():
    rng = np.random.default_rng(0)
    d = cdist(*(2 * [rng.normal(size=(50, 6))]))

    def stress(y):
        return ((cdist(y, y) - d) ** 2).sum()

    start = _phate.classic_mds(d, 2)
    assert stress(_phate.smacof(d, start)) < stress(start)
