"""The dendrogram plot for the stored clustering tree."""

import anndata as ad
import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")
from matplotlib.axes import Axes

import mantispy as mt


@pytest.fixture
def clustered():
    """A small clustered object carrying the linkage tree."""
    rng = np.random.default_rng(0)
    centers = rng.normal(size=(3, 30)) * 5.0
    values = np.repeat(centers, 6, axis=0) + rng.normal(scale=0.3, size=(18, 30))
    adata = ad.AnnData(
        X=values.astype(np.float32),
        obs=pd.DataFrame({"Metadata_Perturbation": [f"p{i}" for i in range(18)]}, index=[f"p{i}" for i in range(18)]),
        var=pd.DataFrame(index=[f"f{i}" for i in range(30)]),
    )
    adata.uns["mantispy"] = {"schema_version": "0.1", "resolution": "perturbation"}
    mt.tl.cluster(adata, use_rep=None)
    return adata


def test_dendrogram_returns_axes_with_every_leaf(clustered):
    ax = mt.pl.dendrogram(clustered)
    assert isinstance(ax, Axes)
    ticks = [text.get_text() for text in ax.get_xticklabels()]
    assert len(ticks) == clustered.n_obs
    assert set(ticks) == set(clustered.obs_names)


def test_dendrogram_needs_a_linkage_tree():
    adata = ad.AnnData(X=np.zeros((3, 2), dtype=np.float32))
    adata.uns["mantispy"] = {"schema_version": "0.1", "resolution": "perturbation"}
    with pytest.raises(KeyError, match="cluster_linkage"):
        mt.pl.dendrogram(adata)
