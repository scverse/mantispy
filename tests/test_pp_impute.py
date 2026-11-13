import logging

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from sklearn.impute import KNNImputer

import mantispy as mt


def _with_gaps(n_obs=300, n_vars=12, fraction=0.05, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_obs, n_vars))
    X[rng.random(X.shape) < fraction] = np.nan
    obs = pd.DataFrame(
        {"Metadata_Plate": np.repeat(["p1", "p2", "p3"], n_obs // 3), "is_ref": np.arange(n_obs) % 4 == 0},
        index=[f"c{i}" for i in range(n_obs)],
    )
    return AnnData(X=X, obs=obs, var=pd.DataFrame(index=[f"f{j}" for j in range(n_vars)]))


def _brieflow_knn(features: pd.DataFrame, batch_size=1000, sample_size=10000) -> pd.DataFrame:
    """The imputation step of Brieflow's ``missing_values_filter`` (workflow/lib/aggregate/filter.py, v1.5.1)."""
    features = features.copy()
    remaining_cols_with_na = features.columns[features.isna().any()].tolist()
    has_na_mask = features[remaining_cols_with_na].isna().any(axis=1).to_numpy()
    na_positions = np.flatnonzero(has_na_mask)
    non_na_positions = np.flatnonzero(~has_na_mask)
    col_positions = np.array([features.columns.get_loc(c) for c in remaining_cols_with_na])
    np.random.seed(42)
    for i in range(0, len(na_positions), batch_size):
        batch_pos = na_positions[i : i + batch_size]
        sampled_non_na_pos = np.random.choice(
            non_na_positions, size=min(sample_size, len(non_na_positions)), replace=False
        )
        combined_pos = np.concatenate([batch_pos, sampled_non_na_pos])
        imputed_values = KNNImputer(n_neighbors=5).fit_transform(features.iloc[combined_pos, col_positions].to_numpy())
        features.iloc[batch_pos, col_positions] = imputed_values[: len(batch_pos)]
    return features


def test_knn_matches_sklearn_and_brieflow():
    """With every row a donor, the result is sklearn's KNNImputer on the whole matrix, and Brieflow's when one batch holds every gap."""
    adata = _with_gaps()
    expected = KNNImputer(n_neighbors=5).fit_transform(adata.X)
    brieflow = _brieflow_knn(pd.DataFrame(adata.X)).to_numpy()
    mt.pp.impute(adata, n_donors=None)
    np.testing.assert_allclose(adata.X, expected, rtol=1e-12)
    np.testing.assert_allclose(adata.X, brieflow, rtol=1e-12)


def test_knn_keeps_inf_and_fills_nan():
    """The contract leaves inf in place; knn used to raise on it (KNNImputer rejects inf)."""
    adata = _with_gaps()
    adata.X[0, 0] = np.inf
    adata.X[0, 1] = np.nan
    mt.pp.impute(adata, n_donors=None)
    assert np.isinf(adata.X[0, 0])  # the inf is left exactly as it was
    assert not np.isnan(adata.X[0, 1])  # the NaN beside it is filled
    assert not np.isnan(adata.X).any()  # every other gap is filled too


def test_knn_takes_values_only_from_reference_rows_of_the_same_group():
    adata = _with_gaps()
    reference = adata.obs["is_ref"].to_numpy()
    plate = adata.obs["Metadata_Plate"].to_numpy()
    adata.X[:, 0] = np.where(reference, 100.0 * (1 + (plate == "p2")), -1.0)
    adata.X[~reference & (np.arange(adata.n_obs) % 7 == 1), 0] = np.nan
    gaps = np.isnan(adata.X[:, 0])
    mt.pp.impute(adata, reference="is_ref", by="Metadata_Plate", n_donors=10)
    np.testing.assert_array_equal(adata.X[gaps, 0], np.where(plate[gaps] == "p2", 200.0, 100.0))
    assert not np.isnan(adata.X).any()


def test_reference_median_is_the_group_median_of_the_reference_rows():
    adata = _with_gaps()
    original = adata.X.copy()
    mt.pp.impute(adata, method="reference_median", reference="is_ref", by="Metadata_Plate")
    for plate in ("p1", "p3"):
        rows = (adata.obs["Metadata_Plate"] == plate).to_numpy()
        median = np.nanmedian(original[rows & adata.obs["is_ref"].to_numpy()], axis=0)
        gaps = np.isnan(original[rows])
        np.testing.assert_allclose(adata.X[rows][gaps], np.broadcast_to(median, gaps.shape)[gaps])


def test_zero_and_the_imputed_count_and_key_added():
    adata = _with_gaps()
    original = adata.X.copy()
    mt.pp.impute(adata, method="zero", key_added="filled")
    np.testing.assert_array_equal(adata.X, original)
    assert "n_imputed" not in adata.var
    np.testing.assert_array_equal(adata.var["n_imputed_filled"], np.isnan(original).sum(axis=0))
    np.testing.assert_array_equal(adata.layers["filled"], np.nan_to_num(original, nan=0.0))


@pytest.mark.parametrize("method", ["knn", "reference_median"])
def test_an_unmeasured_feature_stays_missing_with_a_warning(method, caplog):
    adata = _with_gaps()
    adata.X[:, 3] = np.nan
    with caplog.at_level(logging.WARNING, logger="scverse.mantispy"):
        mt.pp.impute(adata, method=method)
    assert np.isnan(adata.X[:, 3]).all()
    assert adata.var["n_imputed"].iloc[3] == 0
    assert not np.isnan(np.delete(adata.X, 3, axis=1)).any()
    assert "stay missing" in caplog.text


@pytest.mark.parametrize("method", ["knn", "reference_median"])
def test_a_single_row_cannot_donate_to_itself(method, caplog):
    adata = _with_gaps()[:1].copy()
    adata.X[0, :2] = np.nan
    with caplog.at_level(logging.WARNING, logger="scverse.mantispy"):
        mt.pp.impute(adata, method=method)
    assert np.isnan(adata.X[0, :2]).all() and not np.isnan(adata.X[0, 2:]).any()
    assert "stay missing" in caplog.text


def test_no_missing_values_leaves_x_untouched_without_copying():
    """With nothing to fill, impute must not copy X or rewrite it; n_imputed is still written as all zeros."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 12)).astype(np.float32)
    adata = AnnData(
        X=X,
        obs=pd.DataFrame(
            {"Metadata_Plate": np.repeat(["p1", "p2", "p3"], 100), "is_ref": np.arange(300) % 4 == 0},
            index=[f"c{i}" for i in range(300)],
        ),
        var=pd.DataFrame(index=[f"f{j}" for j in range(12)]),
    )
    before = adata.X
    assert not np.isnan(before).any()
    mt.pp.impute(adata, n_donors=None)
    assert np.shares_memory(adata.X, before)  # no copy of the matrix was made
    np.testing.assert_array_equal(adata.X, X)  # the values are exactly as they were
    np.testing.assert_array_equal(adata.var["n_imputed"], np.zeros(adata.n_vars, dtype=np.int64))


def test_no_missing_values_still_writes_the_requested_layer():
    """key_added must create the documented layer even when nothing is filled."""
    rng = np.random.default_rng(1)
    X = rng.normal(size=(120, 6)).astype(np.float32)
    adata = AnnData(X=X, var=pd.DataFrame(index=[f"f{j}" for j in range(6)]))
    mt.pp.impute(adata, key_added="filled", n_donors=None)
    np.testing.assert_array_equal(adata.layers["filled"], X)
    np.testing.assert_array_equal(adata.var["n_imputed_filled"], np.zeros(adata.n_vars, dtype=np.int64))


def test_rejects_no_reference_rows_and_a_bad_method():
    adata = _with_gaps()
    adata.obs["none"] = False
    with pytest.raises(ValueError, match="no reference rows selected"):
        mt.pp.impute(adata, reference="none")
    adata.obs["p1_only"] = (adata.obs["Metadata_Plate"] == "p1").to_numpy()
    with pytest.raises(ValueError, match=r"no reference rows in group\(s\): \['p2', 'p3'\]"):
        mt.pp.impute(adata, method="reference_median", reference="p1_only", by="Metadata_Plate")
    with pytest.raises(ValueError, match="method must be"):
        mt.pp.impute(adata, method="mean")
