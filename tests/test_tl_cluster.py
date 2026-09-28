"""Hierarchical clustering of per-perturbation profiles."""

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import adjusted_rand_score

import mantispy as mt


@pytest.fixture
def planted():
    """Twenty-four profiles in three well-separated planted groups of eight."""
    rng = np.random.default_rng(0)
    centers = rng.normal(size=(3, 30)) * 5.0
    values = np.repeat(centers, 8, axis=0) + rng.normal(scale=0.3, size=(24, 30))
    adata = ad.AnnData(
        X=values.astype(np.float32),
        obs=pd.DataFrame({"Metadata_Perturbation": [f"p{i}" for i in range(24)]}, index=[str(i) for i in range(24)]),
        var=pd.DataFrame(index=[f"f{i}" for i in range(30)]),
    )
    adata.uns["mantispy"] = {"schema_version": "0.1", "resolution": "perturbation"}
    return adata


def test_auto_cut_recovers_the_planted_groups(planted):
    truth = np.repeat([0, 1, 2], 8)
    mt.tl.cluster(planted, use_rep=None)

    summary = planted.uns["mantispy"]["cluster"]
    assert summary["n_clusters"] == 3
    assert adjusted_rand_score(truth, planted.obs["cluster"].to_numpy()) == pytest.approx(1.0)
    # One merge per pair reduced, four columns, so the tree is complete over every profile.
    assert planted.uns["mantispy"]["cluster_linkage"].shape == (23, 4)
    assert summary["labels"] == list(planted.obs_names)
    assert planted.obs["cluster"].dtype.name == "category"


def test_n_clusters_overrides_the_auto_cut(planted):
    mt.tl.cluster(planted, use_rep=None, n_clusters=3, key_added="c3")
    assert planted.obs["c3"].nunique() == 3
    assert adjusted_rand_score(np.repeat([0, 1, 2], 8), planted.obs["c3"].to_numpy()) == pytest.approx(1.0)


def test_distance_cut_and_n_clusters_are_mutually_exclusive(planted):
    with pytest.raises(ValueError, match="at most one"):
        mt.tl.cluster(planted, use_rep=None, distance_cut=1.0, n_clusters=3)


def test_a_missing_representation_is_reported(planted):
    with pytest.raises(KeyError, match="X_pca"):
        mt.tl.cluster(planted)  # default use_rep="X_pca", which this object does not carry


def test_leiden_path_writes_labels_but_no_tree(planted):
    mt.tl.cluster(planted, use_rep=None, method="leiden", key_added="leiden")
    assert planted.obs["leiden"].nunique() >= 1
    assert planted.obs["leiden"].dtype.name == "category"


def test_distance_cut_sets_the_granularity(planted):
    mt.tl.cluster(planted, use_rep=None, distance_cut=0.5, key_added="dc")
    summary = planted.uns["mantispy"]["dc"]
    assert summary["distance_cut"] == 0.5
    assert summary["n_clusters"] == planted.obs["dc"].nunique()


def test_two_runs_keep_separate_linkage_trees(planted):
    mt.tl.cluster(planted, use_rep=None, key_added="a")
    mt.tl.cluster(planted, use_rep=None, metric="euclidean", linkage="ward", key_added="b")
    store = planted.uns["mantispy"]
    assert "a_linkage" in store and "b_linkage" in store
    assert not np.allclose(store["a_linkage"], store["b_linkage"])


def test_two_rows_collapse_to_one_cluster(planted):
    two = planted[:2].copy()
    mt.tl.cluster(two, use_rep=None)  # no 2-way silhouette exists at n=2
    assert two.obs["cluster"].nunique() == 1


def test_a_degenerate_profile_is_reported(planted):
    planted.X[0] = 0.0  # a constant profile has an undefined correlation distance
    with pytest.raises(ValueError, match="not finite"):
        mt.tl.cluster(planted, use_rep=None)


def test_stability_criterion_recovers_the_planted_groups(planted):
    truth = np.repeat([0, 1, 2], 8)
    mt.tl.cluster(planted, use_rep=None, criterion="stability")

    summary = planted.uns["mantispy"]["cluster"]
    assert summary["n_clusters"] == 3
    assert adjusted_rand_score(truth, planted.obs["cluster"].to_numpy()) == pytest.approx(1.0)
    assert np.isfinite(summary["distance_cut"])
    assert np.isfinite(summary["stability"])

    with pytest.raises(ValueError, match="criterion must be"):
        mt.tl.cluster(planted, use_rep=None, criterion="bogus", key_added="bad")


def test_stability_criterion_does_not_over_segment_moderate_groups():
    """Three planted groups at a modest separation: the old count-weighted score over-segmented here."""
    rng = np.random.default_rng(7)
    centers = rng.normal(size=(3, 12))
    values = np.repeat(centers, 8, axis=0) + rng.normal(scale=0.35, size=(24, 12))
    adata = ad.AnnData(
        X=values.astype(np.float32),
        obs=pd.DataFrame({"Metadata_Perturbation": [f"p{i}" for i in range(24)]}, index=[str(i) for i in range(24)]),
        var=pd.DataFrame(index=[f"f{i}" for i in range(12)]),
    )
    adata.uns["mantispy"] = {"schema_version": "0.1", "resolution": "perturbation"}

    truth = np.repeat([0, 1, 2], 8)
    mt.tl.cluster(adata, use_rep=None, criterion="stability")

    summary = adata.uns["mantispy"]["cluster"]
    assert summary["n_clusters"] == 3
    assert adjusted_rand_score(truth, adata.obs["cluster"].to_numpy()) > 0.95
    assert 0.0 <= summary["stability"] <= 1.0


def test_stability_window_restricts_the_sweep_to_a_height_band():
    """Two super-groups split into finer sub-groups: unbounded stability sits on the coarse plateau."""
    rng = np.random.default_rng(11)
    direction = rng.normal(size=40)
    direction /= np.linalg.norm(direction)
    supers = np.stack([direction, -direction])  # anti-correlated, so the super merge sits near height 2
    subs = np.repeat(supers, 3, axis=0) + rng.normal(size=(6, 40)) * 0.06
    values = np.repeat(subs, 8, axis=0) + rng.normal(scale=0.015, size=(48, 40))
    adata = ad.AnnData(
        X=values.astype(np.float32),
        obs=pd.DataFrame({"Metadata_Perturbation": [f"p{i}" for i in range(48)]}, index=[str(i) for i in range(48)]),
        var=pd.DataFrame(index=[f"f{i}" for i in range(40)]),
    )
    adata.uns["mantispy"] = {"schema_version": "0.1", "resolution": "perturbation"}

    truth = np.repeat(np.arange(6), 8)

    unbounded = adata.copy()
    mt.tl.cluster(unbounded, use_rep=None, criterion="stability")
    n_unbounded = unbounded.uns["mantispy"]["cluster"]["n_clusters"]

    windowed = adata.copy()
    lo, hi = 0.05, 0.25
    mt.tl.cluster(windowed, use_rep=None, criterion="stability", stability_window=(lo, hi))
    summary = windowed.uns["mantispy"]["cluster"]
    assert summary["n_clusters"] == 6
    assert summary["n_clusters"] > n_unbounded
    assert adjusted_rand_score(truth, windowed.obs["cluster"].to_numpy()) == pytest.approx(1.0)
    assert lo <= summary["distance_cut"] <= hi

    with pytest.raises(ValueError, match="stability_window"):
        mt.tl.cluster(adata, use_rep=None, criterion="stability", stability_window=(0.6, 0.3), key_added="bad")
    with pytest.raises(ValueError, match="does not overlap"):
        mt.tl.cluster(adata, use_rep=None, criterion="stability", stability_window=(100.0, 200.0), key_added="bad")
    with pytest.raises(ValueError, match="stability_window"):
        mt.tl.cluster(adata, use_rep=None, criterion="stability", stability_window=("a", "b"), key_added="bad")
    with pytest.raises(ValueError, match="stability_window"):
        mt.tl.cluster(adata, use_rep=None, criterion="stability", stability_window=0.5, key_added="bad")


def test_stability_window_lifts_the_cluster_ceiling():
    """With more than 25 well-separated groups a window keeps the many-cluster cut the count ceiling would drop."""
    from scipy.cluster.hierarchy import linkage as scipy_linkage
    from scipy.spatial.distance import pdist

    rng = np.random.default_rng(3)
    centers = rng.normal(size=(30, 40)) * 5.0  # 30 distinct, near-uncorrelated directions
    values = np.repeat(centers, 2, axis=0) + rng.normal(scale=0.01, size=(60, 40))  # 30 tight pairs, n_obs=60
    adata = ad.AnnData(
        X=values.astype(np.float32),
        obs=pd.DataFrame({"Metadata_Perturbation": [f"p{i}" for i in range(60)]}, index=[str(i) for i in range(60)]),
        var=pd.DataFrame(index=[f"f{i}" for i in range(40)]),
    )
    adata.uns["mantispy"] = {"schema_version": "0.1", "resolution": "perturbation"}

    # Reproduce the default correlation/average tree to find the 30-cluster plateau: the 30 within-pair merges are the
    # lowest heights, so between the 30th and 31st sorted merge exactly 30 clusters stand.
    heights = np.sort(scipy_linkage(pdist(values, metric="correlation"), method="average")[:, 2])
    span = float(heights[30] - heights[29])
    lo, hi = float(heights[29]) + span * 0.2, float(heights[29]) + span * 0.8

    truth = np.repeat(np.arange(30), 2)  # 30 planted pairs of two
    windowed = adata.copy()
    mt.tl.cluster(windowed, use_rep=None, criterion="stability", stability_window=(lo, hi))
    summary = windowed.uns["mantispy"]["cluster"]
    # The window brackets the 30-cluster plateau, so dropping the ceiling recovers the exact 30-cluster cut instead of
    # the 1-cluster fallback the ceiling forced when every in-window cut exceeded 25 clusters.
    assert summary["n_clusters"] == 30
    assert adjusted_rand_score(truth, windowed.obs["cluster"].to_numpy()) == pytest.approx(1.0)
    assert 0.0 <= summary["stability"] <= 1.0
    assert lo <= summary["distance_cut"] <= hi


def test_copy_leaves_the_input_alone(planted):
    result = mt.tl.cluster(planted, use_rep=None, copy=True)
    assert "cluster" in result.obs
    assert "cluster" not in planted.obs
