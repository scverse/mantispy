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
    # scanpy's graphs have no self-loops.
    assert graph.diagonal().max() == 0
    distances = adata.obsp["phate_distances"]
    assert distances.diagonal().max() == 0
    # Each row lists its five nearest other rows, nearest first, as sc.pp.neighbors does.
    np.testing.assert_array_equal(np.diff(distances.indptr), 5)
    expected = np.sort(cdist(adata.obsm["X_pca"], adata.obsm["X_pca"]), axis=1)[:, 1:6]
    np.testing.assert_allclose(distances.data.reshape(-1, 5), expected, rtol=1e-12)
    entry = adata.uns["phate"]
    assert entry["connectivities_key"] == "phate_connectivities"
    assert entry["distances_key"] == "phate_distances"
    assert entry["params"]["t"] >= 1 and entry["params"]["knn"] == 5
    assert entry["params"]["n_neighbors"] == 6 and entry["params"]["use_rep"] == "X_pca"


@pytest.mark.filterwarnings("ignore:The argument neighbors_key is deprecated:FutureWarning")
def test_scanpy_graph_tools_run_on_the_stored_graph():
    sc = pytest.importorskip("scanpy")
    # Overlapping groups, so the graph is connected and PAGA finds edges between them.
    blobs = _blobs(spread=1.5)
    adata = ad.AnnData(blobs.obsm["X_pca"], obs=blobs.obs)
    mt.tl.phate(adata, knn=10, t=5, use_rep=None)
    adata.obs["group"] = adata.obs["group"].astype("category")
    sc.tl.umap(adata, neighbors_key="phate")
    assert adata.obsm["X_umap"].shape == (adata.n_obs, 2)
    # use_rep=None is recorded as X, so umap neither reads nor computes a PCA.
    assert "X_pca" not in adata.obsm
    sc.tl.paga(adata, groups="group", neighbors_key="phate")
    assert adata.uns["paga"]["connectivities"].nnz > 0
    sc.tl.diffmap(adata, neighbors_key="phate")
    assert np.isfinite(adata.obsm["X_diffmap"]).all()
    adata.uns["iroot"] = 0
    sc.tl.dpt(adata, neighbors_key="phate")
    assert np.isfinite(adata.obs["dpt_pseudotime"]).all()


def test_the_kernel_keeps_every_affinity_above_the_threshold():
    """A tight cloud puts many rows inside the kernel's reach, past the first neighbour search; none may be cut off."""
    rng = np.random.default_rng(1)
    x = np.vstack([rng.normal(scale=0.01, size=(60, 3)), rng.normal(size=(60, 3)) + 5])
    for decay, metric in [(40, "euclidean"), (5, "euclidean"), (40, "cosine")]:
        adata = ad.AnnData(np.zeros((x.shape[0], 1)), obsm={"X_pca": x})
        mt.tl.phate(adata, knn=5, decay=decay, metric=metric, t=3)
        np.testing.assert_allclose(
            adata.obsp["phate_connectivities"].toarray() + np.eye(x.shape[0]),
            _dense_kernel(x, 5, decay, metric),
            atol=1e-12,
        )


def test_a_far_row_widens_only_its_own_neighbourhood():
    """A row far from the rest reaches every row; the others keep the width of the first search."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=(400, 10))
    x[0] += 30
    _, _, indptr, _, _ = _phate.knn_with_tail(x, 5, 40)
    widths = np.diff(indptr)
    assert widths[0] == x.shape[0]
    assert np.median(widths[1:]) == 4 * 6
    assert indptr[-1] < 2 * 4 * 6 * x.shape[0]
    adata = ad.AnnData(np.zeros((x.shape[0], 1)), obsm={"X_pca": x})
    mt.tl.phate(adata, knn=5, t=3)
    np.testing.assert_allclose(
        adata.obsp["phate_connectivities"].toarray() + np.eye(x.shape[0]), _dense_kernel(x, 5, 40), atol=1e-12
    )


def test_neighbours_are_exact_for_near_duplicates_far_from_the_mean():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(1000, 10))
    x[:100] = 100 * np.eye(10)[0] + rng.normal(scale=1e-6, size=(100, 10))
    _, dist = _phate.nearest_neighbours(x, 6)
    np.testing.assert_allclose(dist, np.sort(cdist(x, x), axis=1)[:, :6], rtol=1e-6, atol=1e-12)


def test_one_far_row_keeps_the_others_in_single_precision(monkeypatch):
    screened = []
    original = _phate._screen

    def spy(x, rows, k):
        screened.append((x.dtype, rows.size))
        return original(x, rows, k)

    monkeypatch.setattr(_phate, "_screen", spy)
    rng = np.random.default_rng(0)
    x = rng.normal(size=(2000, 30))
    x[0] *= 1000
    _phate.nearest_neighbours(x, 24)
    assert sum(size for dtype, size in screened if dtype == np.float64) <= 10


def test_a_plain_knn_kernel_without_decay():
    adata = _blobs()
    mt.tl.phate(adata, knn=4, decay=None, t=5)
    x = adata.obsm["X_pca"]
    directed = np.zeros((x.shape[0],) * 2)
    np.put_along_axis(directed, np.argsort(cdist(x, x), axis=1)[:, :5], 1.0, axis=1)
    expected = (directed + directed.T) / 2
    np.fill_diagonal(expected, 0.0)
    np.testing.assert_array_equal(adata.obsp["phate_connectivities"].toarray(), expected)
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
    mt.tl.phate(adata, n_landmark=100, t=8, seed=3)
    assert adata.uns["phate"]["params"]["n_landmark"] <= 100
    assert adata.uns["phate"]["params"]["seed"] == 3
    assert adata.obsm["X_phate_potential"].shape == (adata.n_obs, adata.uns["phate"]["params"]["n_landmark"])
    first = adata.obsm["X_phate"].copy()
    mt.tl.phate(adata, n_landmark=100, t=8, seed=3)
    np.testing.assert_array_equal(first, adata.obsm["X_phate"])
    # Landmarks approximate the exact walk, so the layout matches the exact one up to rotation and scale.
    mt.tl.phate(adata, n_landmark=None, t=8)
    assert procrustes(first, adata.obsm["X_phate"])[2] < 0.1


def test_the_potential_is_stored_up_to_a_size_unless_asked(monkeypatch):
    adata = _blobs()
    # The exact potential has a column per row; twenty landmarks make it six times narrower.
    monkeypatch.setattr(tl_phate, "POTENTIAL_MAX_BYTES", adata.n_obs * 20 * 4)
    mt.tl.phate(adata, t=5)
    assert "X_phate_potential" not in adata.obsm
    mt.tl.phate(adata, t=5, n_landmark=20)
    assert adata.obsm["X_phate_potential"].shape[0] == adata.n_obs
    mt.tl.phate(adata, t=5, store_potential=True)
    assert adata.obsm["X_phate_potential"].shape == (adata.n_obs, adata.n_obs)


def test_a_rerun_without_the_potential_removes_the_old_one():
    adata = _blobs()
    mt.tl.phate(adata, t=5)
    mt.tl.phate(adata, knn=10, t=5, store_potential=False)
    assert "X_phate_potential" not in adata.obsm


def test_an_automatic_t_is_recorded_and_a_given_t_is_kept():
    adata = _blobs()
    mt.tl.phate(adata, t=7)
    assert adata.uns["phate"]["params"]["t"] == 7
    mt.tl.phate(adata)
    # The knee of the von Neumann entropy of the walk's spectrum, recomputed from the stored graph.
    kernel = adata.obsp["phate_connectivities"].toarray() + np.eye(adata.n_obs)
    scale = 1.0 / np.sqrt(kernel.sum(axis=1))
    spectrum = np.abs(np.linalg.eigvalsh(kernel * scale[:, None] * scale[None, :]))
    spectrum = spectrum[spectrum > 0]
    entropy = [-(p * np.log(p)).sum() for p in (spectrum**t / (spectrum**t).sum() for t in range(1, 101))]
    assert adata.uns["phate"]["params"]["t"] == _phate.knee(np.array(entropy))


def test_duplicate_rows_still_embed():
    adata = _blobs()
    adata.obsm["X_pca"][:10] = adata.obsm["X_pca"][0]
    mt.tl.phate(adata, knn=5, t=5)
    assert np.isfinite(adata.obsm["X_phate"]).all()


def test_identical_rows_embed_at_one_point():
    adata = ad.AnnData(np.zeros((30, 1)), obsm={"X_pca": np.ones((30, 4))})
    mt.tl.phate(adata, knn=5, t=5)
    np.testing.assert_array_equal(adata.obsm["X_phate"], 0.0)


def test_potential_rows_apart_by_rounding_alone_are_one_point():
    """BLAS sums the walk's powers in a platform-dependent order, so identical rows can differ in the last bit."""
    row = np.random.default_rng(0).uniform(1, 16, 300)
    potential = np.repeat(row[None], 20, axis=0)
    potential[::2] = np.nextafter(potential[::2], np.inf)
    np.testing.assert_array_equal(_phate.potential_distances(potential), 0.0)
    apart = np.random.default_rng(1).normal(size=(50, 300))
    np.testing.assert_allclose(_phate.potential_distances(apart), cdist(apart, apart), rtol=1e-10, atol=1e-12)


def test_classical_mds_pads_axes_past_the_points():
    d = cdist(*(2 * [np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 2.0]])]))
    coordinates = _phate.classic_mds(d, 5)
    assert coordinates.shape == (3, 5)
    np.testing.assert_allclose(cdist(coordinates, coordinates), d, atol=1e-12)


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
    [
        ({"knn": 200}, "more rows"),
        ({"knn": 5.0}, "knn"),
        ({"metric": "manhattan"}, "metric"),
        ({"gamma": 2}, "gamma"),
        ({"t": 0}, "t must"),
        ({"t": 2.5}, "t must"),
        ({"decay": np.nan}, "decay"),
        ({"decay": np.inf}, "decay"),
        ({"seed": None}, "seed"),
        ({"seed": np.random.default_rng(0)}, "seed"),
        ({"seed": 2**32}, "seed"),
        ({"n_components": 120}, "n_components"),
        ({"n_components": 20, "n_landmark": 20}, "n_components"),
        ({"n_landmark": 50.0}, "n_landmark"),
    ],
)
def test_bad_arguments_raise_before_writing(kwargs, match):
    adata = _blobs()
    with pytest.raises(ValueError, match=match):
        mt.tl.phate(adata, **kwargs)
    assert "X_phate" not in adata.obsm and "phate_connectivities" not in adata.obsp


def test_missing_values_raise():
    adata = _blobs()
    adata.obsm["X_pca"][0, 0] = np.nan
    with pytest.raises(ValueError, match="missing"):
        mt.tl.phate(adata)


def test_values_too_large_to_square_raise():
    adata = _blobs()
    adata.obsm["X_pca"][0] = 1e200
    with pytest.raises(ValueError, match="too large"):
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
