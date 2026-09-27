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


def test_copy_leaves_the_input_alone(planted):
    result = mt.tl.cluster(planted, use_rep=None, copy=True)
    assert "cluster" in result.obs
    assert "cluster" not in planted.obs
