"""Within-group centring and standardising, such as per well."""

import anndata as ad
import numpy as np
import pandas as pd

import mantispy as mt


def _grouped(X, **columns):
    n = X.shape[0]
    obs = pd.DataFrame({**columns}, index=[str(i) for i in range(n)])
    return ad.AnnData(X=np.asarray(X, dtype=np.float32), obs=obs)


def test_center_within_zeros_each_group_mean():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 3)) + np.r_[np.zeros((10, 3)), np.ones((10, 3))]
    adata = _grouped(X, Metadata_Well=["A"] * 10 + ["B"] * 10)
    mt.pp.center_within(adata, by="Metadata_Well")
    for well in ("A", "B"):
        block = np.asarray(adata[adata.obs["Metadata_Well"] == well].X)
        assert np.allclose(block.mean(axis=0), 0.0, atol=1e-5)


def test_center_within_reference_centres_on_the_reference_subset():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 2))
    X[10:] += 5.0
    adata = _grouped(X, Metadata_Well=["A"] * 20, is_reference=[True] * 10 + [False] * 10)
    mt.pp.center_within(adata, by="Metadata_Well", reference="is_reference")
    reference = np.asarray(adata[adata.obs["is_reference"]].X)
    assert np.allclose(reference.mean(axis=0), 0.0, atol=1e-5)


def test_center_within_zscore_gives_unit_variance_and_survives_a_constant_feature():
    rng = np.random.default_rng(0)
    X = np.column_stack([rng.normal(2.0, 3.0, size=30), np.full(30, 7.0)])  # second feature is constant
    adata = _grouped(X, Metadata_Well=["A"] * 30)
    mt.pp.center_within(adata, by="Metadata_Well", method="zscore")
    out = np.asarray(adata.X)
    assert abs(out[:, 0].std() - 1.0) < 1e-5
    assert np.isfinite(out[:, 1]).all() and np.allclose(out[:, 1], 0.0)  # constant -> centred, not divided by zero
