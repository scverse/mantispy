import logging

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

import mantispy as mt
from mantispy.ds import synthetic_plate


@pytest.fixture
def adata():
    return synthetic_plate(n_wells=16, n_cells=40, n_features=15, seed=0)


@pytest.mark.parametrize("method", ["ecod", "isolation_forest", "lof", "mad"])
def test_every_method_writes_a_flag_and_a_score(adata, method):
    mt.pp.outliers(adata, method=method, contamination=0.05)
    assert adata.obs["qc_outlier"].dtype == bool
    assert 0 < adata.obs["qc_outlier"].sum() < adata.n_obs
    assert np.isfinite(adata.obs["qc_outlier_score"]).all()


def test_score_cutoff_thresholds_absolutely(adata):
    """With method='mad' the score is a robust z, so 5 is the familiar rule."""
    mt.pp.outliers(adata, method="mad", score_cutoff=5.0)
    assert (adata.obs["qc_outlier_score"][adata.obs["qc_outlier"]] > 5.0).all()
    mt.pp.outliers(adata, method="mad", score_cutoff=1e9, key_added="none_flagged")
    assert not adata.obs["none_flagged"].any()


def test_rejects_a_bad_method_or_contamination(adata):
    with pytest.raises(ValueError, match="method must be"):
        mt.pp.outliers(adata, method="nonsense")
    with pytest.raises(ValueError, match="contamination"):
        mt.pp.outliers(adata, contamination=1.5)
    with pytest.raises(ValueError, match="ecod_aggregation must be"):
        mt.pp.outliers(adata, ecod_aggregation="nonsense")


def test_ecod_does_not_depend_on_which_way_round_a_feature_is_measured(adata):
    """#72: CellProfiler's feature directions are arbitrary, so the default aggregation must not care about them.

    ecod_aggregation="paper" does, and fails this.
    """
    flipped = adata.copy()
    flipped.X[:, ::2] *= -1
    mt.pp.outliers(adata)
    mt.pp.outliers(flipped)
    np.testing.assert_allclose(flipped.obs["qc_outlier_score"], adata.obs["qc_outlier_score"], rtol=1e-12)


def test_key_picks_the_features_scored_and_key_added_names_the_output(adata):
    """`key` is the input var flag; an extreme value in a feature it leaves out cannot be flagged."""
    adata.X[0, 0] = 1e6
    adata.var["usable"] = [name != adata.var_names[0] for name in adata.var_names]
    mt.pp.outliers(adata, method="mad", score_cutoff=50.0, key="usable", key_added="flagged_usable")
    mt.pp.outliers(adata, method="mad", score_cutoff=50.0, key=None, key_added="flagged_all")
    assert not adata.obs["flagged_usable"].iloc[0]
    assert adata.obs["flagged_all"].iloc[0]


def test_outliers_join_qc_fail_so_filter_cells_drops_them(adata):
    # filter_cells reads qc_fail, so an outlier left only in qc_outlier survived filtering (#272).
    mt.pp.calculate_qc_metrics(adata)
    failed_before = adata.obs["qc_fail"].to_numpy(dtype=bool).copy()
    mt.pp.outliers(adata, method="mad", contamination=0.05)
    flagged = adata.obs["qc_outlier"].to_numpy(dtype=bool)
    assert (adata.obs["qc_fail"].to_numpy(dtype=bool) == (failed_before | flagged)).all()
    kept = mt.pp.filter_cells(adata, min_cells_per_well=0, copy=True)
    assert not kept.obs["qc_outlier"].any()


def test_lof_flags_what_sklearn_and_brieflow_flag():
    """Brieflow's intensity_filter (workflow/lib/aggregate/filter.py, v1.5.1) keeps fit_predict(...) == 1."""
    from sklearn.neighbors import LocalOutlierFactor

    adata = synthetic_plate(n_wells=25, n_cells=40, n_features=6, seed=1)
    X = np.asarray(adata.X, dtype=np.float64)
    brieflow = LocalOutlierFactor(contamination=0.01, n_jobs=-1).fit_predict(X) == -1
    mt.pp.outliers(adata, method="lof", contamination=0.01)
    assert brieflow.sum() == 10
    np.testing.assert_array_equal(adata.obs["qc_outlier"].to_numpy(), brieflow)


def _brute_knn(X: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Each row's k nearest other rows by squared distance summed in feature order, ties broken by row."""
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]
    rdist = np.zeros((n, n))
    for f in range(X.shape[1]):
        diff = X[:, None, f] - X[None, :, f]
        rdist += diff * diff
    rdist[np.arange(n), np.arange(n)] = np.inf
    order = np.lexsort((np.broadcast_to(np.arange(n), (n, n)), rdist), axis=1)
    return order[:, :k], np.take_along_axis(rdist, order[:, : k + 1], axis=1)


def _gridded(rng, shape):
    return rng.integers(0, 5, size=shape).astype(np.float64)


def _duplicated(rng, shape):
    base = rng.normal(size=shape)
    return np.vstack([base, base[: shape[0] // 4]])


def _normal(rng, shape):
    return rng.normal(size=shape)


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
@pytest.mark.parametrize("make", [_normal, _duplicated, _gridded])
@pytest.mark.parametrize("shape", [(3000, 8), (700, 3), (90, 2), (6, 4)])
def test_lof_kdtree_neighbours_match_a_brute_force_search(dtype, make, shape):
    """The numba KD-tree finds exactly the brute-force neighbours, in distance-then-row order, and flags k-th ties."""
    from mantispy._core._lof import _knn, n_chunks, tree_levels

    X = np.ascontiguousarray(make(np.random.default_rng(0), shape), dtype=dtype)
    coords = np.ascontiguousarray(X, dtype=np.float64)
    n = coords.shape[0]
    k = min(20, n - 1)
    m = min(k + 1, n - 1)
    levels = tree_levels(n)
    perm, _, neighbors, k_rdist, tied = _knn(coords, levels, k, m, n_chunks(levels))
    found = np.empty((n, k), dtype=np.int64)
    found[perm] = perm[neighbors]
    kth = np.empty(n)
    kth[perm] = k_rdist
    truth, rdist = _brute_knn(coords, k)
    np.testing.assert_array_equal(kth, rdist[:, k - 1])
    if m > k:
        assert tied == bool(np.any(np.sqrt(rdist[:, k - 1]) == np.sqrt(rdist[:, k])))
    np.testing.assert_array_equal(found, truth)


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
@pytest.mark.parametrize("shape", [(1000, 4), (500, 8), (40, 6), (22, 2), (21, 2), (3, 5)])
def test_lof_fast_path_matches_sklearn_exactly(dtype, shape):
    """The KD-tree fast path reproduces sklearn's negative_outlier_factor_ on ordinary data."""
    from sklearn.neighbors import LocalOutlierFactor

    from mantispy._core._lof import lof_negative_outlier_factor
    from mantispy.pp._outliers import _LOF_NEIGHBORS

    n = shape[0]
    X = np.ascontiguousarray(np.random.default_rng(0).normal(size=shape), dtype=dtype)
    fast = lof_negative_outlier_factor(X, _LOF_NEIGHBORS)
    assert fast is not None  # continuous data has no ties, so the fast path is taken
    assert fast.dtype == dtype
    truth = LocalOutlierFactor(n_neighbors=min(_LOF_NEIGHBORS, n - 1)).fit(X).negative_outlier_factor_
    # Distances agree with sklearn's to the last bit only where both compile the sum of squares the same way,
    # which depends on the platform; elsewhere they differ by a rounding step, far below anything that moves a flag.
    rtol, atol = (1e-12, 1e-12) if dtype is np.float64 else (1e-5, 1e-6)
    np.testing.assert_allclose(fast, truth, rtol=rtol, atol=atol)


def test_lof_matches_sklearn_on_duplicates_and_ties():
    """Duplicate rows and gridded ties make the neighbour set ambiguous; the result must still equal
    sklearn's, via the fall-back, end to end through pp.outliers."""
    from sklearn.neighbors import LocalOutlierFactor

    rng = np.random.default_rng(1)
    base = rng.normal(size=(300, 5))
    duplicated = np.vstack([base, base[:40]])  # 40 exact duplicate rows
    gridded = rng.integers(0, 5, size=(400, 4)).astype(np.float64)  # many equidistant ties
    for X in (duplicated, gridded):
        n = X.shape[0]
        truth = -LocalOutlierFactor(n_neighbors=min(20, n - 1)).fit(X).negative_outlier_factor_
        adata = AnnData(
            X=X.copy(),
            obs=pd.DataFrame(index=[f"c{i}" for i in range(n)]),
            var=pd.DataFrame(index=[f"f{j}" for j in range(X.shape[1])]),
        )
        mt.pp.outliers(adata, method="lof", contamination=0.05)
        np.testing.assert_allclose(adata.obs["qc_outlier_score"].to_numpy(), truth, rtol=1e-12, atol=1e-12)


def test_lof_outliers_scores_match_sklearn_on_a_realistic_dataset():
    """End to end: pp.outliers(lof) equals a direct sklearn LOF on a synthetic plate."""
    from sklearn.neighbors import LocalOutlierFactor

    adata = synthetic_plate(n_wells=25, n_cells=80, n_features=8, seed=3)
    X = np.ascontiguousarray(adata.X)
    truth = -LocalOutlierFactor(n_neighbors=20).fit(X).negative_outlier_factor_
    mt.pp.outliers(adata, method="lof", contamination=0.01)
    rtol, atol = (1e-12, 1e-12) if X.dtype == np.float64 else (1e-5, 1e-6)
    np.testing.assert_allclose(adata.obs["qc_outlier_score"].to_numpy(), truth, rtol=rtol, atol=atol)


def test_lof_does_not_depend_on_the_thread_count():
    """Threads come from numba's pool and the work is split into a fixed number of chunks, so scores are identical."""
    import numba

    adata = synthetic_plate(n_wells=25, n_cells=40, n_features=6, seed=1)
    one = adata.copy()
    many = adata.copy()
    threads = numba.get_num_threads()
    numba.set_num_threads(1)
    try:
        mt.pp.outliers(one, method="lof", contamination=0.01)
    finally:
        numba.set_num_threads(threads)
    mt.pp.outliers(many, method="lof", contamination=0.01)
    np.testing.assert_array_equal(one.obs["qc_outlier"].to_numpy(), many.obs["qc_outlier"].to_numpy())
    np.testing.assert_array_equal(one.obs["qc_outlier_score"].to_numpy(), many.obs["qc_outlier_score"].to_numpy())


def test_outliers_has_no_n_jobs(adata):
    with pytest.raises(TypeError, match="n_jobs"):
        mt.pp.outliers(adata, method="lof", n_jobs=-1)


def test_lof_sampled_fit_is_reproducible_and_agrees_with_the_full_fit():
    """Fitting LOF on a seeded sample bounds its cost; it reflags the same clear outliers as a full fit,
    and the same seed reflags the same cells."""
    rng = np.random.default_rng(0)
    core = rng.normal(size=(1000, 6))
    # Isolated outliers, each far in its own direction, so neither mode masks them as a dense clump.
    directions = rng.normal(size=(12, 6))
    far = 12.0 * directions / np.linalg.norm(directions, axis=1, keepdims=True)
    X = np.vstack([core, far])
    outlier_rows = np.arange(1000, 1012)
    make = lambda: AnnData(
        X=X.copy(),
        obs=pd.DataFrame(index=[f"c{i}" for i in range(X.shape[0])]),
        var=pd.DataFrame(index=[f"f{j}" for j in range(6)]),
    )

    full = make()
    mt.pp.outliers(full, method="lof", contamination=0.03, n_fit=None)
    sampled = make()
    mt.pp.outliers(sampled, method="lof", contamination=0.03, n_fit=200, seed=0)
    again = make()
    mt.pp.outliers(again, method="lof", contamination=0.03, n_fit=200, seed=0)

    s_flag = sampled.obs["qc_outlier"].to_numpy()
    f_flag = full.obs["qc_outlier"].to_numpy()
    np.testing.assert_array_equal(s_flag, again.obs["qc_outlier"].to_numpy())  # reproducible
    assert s_flag[outlier_rows].all() and f_flag[outlier_rows].all()  # both catch the obvious outliers
    jaccard = (s_flag & f_flag).sum() / (s_flag | f_flag).sum()
    assert jaccard > 0.7


def test_lof_rejects_a_bad_n_fit():
    adata = synthetic_plate(n_wells=4, n_cells=10, n_features=5, seed=0)
    with pytest.raises(ValueError, match="n_fit"):
        mt.pp.outliers(adata, method="lof", n_fit=1)


def test_lof_scores_a_group_of_one_or_two_rows():
    # One well of one cell and one of two, so both the single-row path and the real two-row (n_neighbors=1) path run.
    X = np.random.default_rng(0).normal(size=(3, 4))
    obs = pd.DataFrame({"Metadata_Well": ["A", "B", "B"]}, index=["c0", "c1", "c2"])
    adata = AnnData(X=X, obs=obs, var=pd.DataFrame(index=[f"f{j}" for j in range(4)]))
    assert sorted(adata.obs["Metadata_Well"].value_counts().tolist()) == [1, 2]
    mt.pp.outliers(adata, method="lof", by="Metadata_Well")
    assert np.isfinite(adata.obs["qc_outlier_score"]).all()
    assert adata.obs["qc_outlier"].sum() == 2  # one flagged per group


def test_lof_excludes_all_missing_rows_and_still_flags_real_outliers(caplog):
    """#2: filling all-missing rows with zero collapses them onto the origin, inflating nearby cells' LOF so the
    real outliers go unflagged. They must be excluded from the fit, left unscored, and the true outliers flagged."""
    rng = np.random.default_rng(0)
    n_core, n_missing, n_outliers, n_vars = 200, 30, 6, 6
    core = rng.normal(size=(n_core, n_vars))
    far = rng.normal(size=(n_outliers, n_vars)) + 15.0
    gone = np.full((n_missing, n_vars), np.nan)
    X = np.vstack([core, far, gone])
    obs = pd.DataFrame(index=[f"c{i}" for i in range(X.shape[0])])
    adata = AnnData(X=X, obs=obs, var=pd.DataFrame(index=[f"f{j}" for j in range(n_vars)]))
    outlier_rows = np.arange(n_core, n_core + n_outliers)
    missing_rows = np.arange(n_core + n_outliers, X.shape[0])

    with caplog.at_level(logging.WARNING, logger="scverse.mantispy"):
        mt.pp.outliers(adata, method="lof", contamination=0.03)

    flagged = adata.obs["qc_outlier"].to_numpy()
    scores = adata.obs["qc_outlier_score"].to_numpy()
    assert flagged[outlier_rows].all()  # the genuine outliers are flagged
    assert not flagged[missing_rows].any()  # the all-missing rows are not
    assert np.isnan(scores[missing_rows]).all()  # and are left unscored
    assert np.isfinite(scores[: n_core + n_outliers]).all()
    assert f"{n_missing} cell" in caplog.text
