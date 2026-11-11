"""feature_select equivalence against pycytominer.

Semantics follow the pinned pycytominer source rather than its docs.
Two points are easy to get wrong, and both are covered below.
`variance_threshold` is an sklearn variance cut; the frequency/uniqueness rule is `frequency_threshold`.
`correlation_threshold` judges each pair independently against a ranking computed once, using the signed correlation.
"""

import anndata as ad
import numpy as np
import pandas as pd
import pytest

import mantispy as mt
from mantispy.ds import synthetic_plate

pycytominer = pytest.importorskip("pycytominer")


@pytest.fixture
def wells():
    cells = synthetic_plate(
        n_plates=2,
        n_wells=48,
        n_cells=10,
        n_features=30,
        n_correlated_pairs=4,
        n_constant_features=3,
        seed=7,
    )
    return mt.tl.aggregate(cells, min_cells=0)


def _kept_by_pycytominer(wells, operation, **kwargs):
    frame = mt.get.to_dataframe(wells)
    result = pycytominer.feature_select(profiles=frame, features=list(wells.var_names), operation=[operation], **kwargs)
    return {c for c in result.columns if not c.startswith("Metadata_")}


def _kept_by_mantispy(wells, operation, **kwargs):
    mt.pp.feature_select(wells, operations=(operation,), **kwargs)
    return set(wells.var_names[wells.var["selected"]])


@pytest.mark.parametrize(
    ("operation", "ours", "theirs"),
    [
        ("variance_threshold", {}, {}),
        ("frequency_threshold", {}, {}),
        ("correlation_threshold", {"corr_threshold": 0.9}, {"corr_threshold": 0.9}),
        ("correlation_threshold", {"corr_threshold": 0.7}, {"corr_threshold": 0.7}),
        ("correlation_threshold", {"corr_threshold": 0.5}, {"corr_threshold": 0.5}),
        ("drop_na_columns", {}, {}),
        ("blocklist", {}, {}),
        ("drop_outliers", {"outlier_cutoff": 500.0}, {"outlier_cutoff": 500.0}),
        (
            "noise_removal",
            {"noise_removal_perturb_groups": "Metadata_Perturbation"},
            {"noise_removal_perturb_groups": "Metadata_Perturbation", "noise_removal_stdev_cutoff": 0.8},
        ),
    ],
    ids=[
        "variance",
        "frequency",
        "corr0.9",
        "corr0.7",
        "corr0.5",
        "drop_na",
        "blocklist",
        "outliers",
        "noise",
    ],
)
def test_each_operation_removes_the_same_features(wells, operation, ours, theirs):
    expected = _kept_by_pycytominer(wells, operation, **theirs)
    actual = _kept_by_mantispy(wells, operation, **ours)
    assert actual == expected, {
        "only_ours": sorted(actual - expected),
        "only_theirs": sorted(expected - actual),
    }


def test_negatively_correlated_pair_is_kept(wells):
    """pycytominer thresholds the signed correlation, so corr = -1.0 keeps both."""
    values = wells.X.copy()
    values[:, 1] = -values[:, 0]
    wells.X = values
    kept = _kept_by_mantispy(wells, "correlation_threshold", corr_threshold=0.9)
    assert {wells.var_names[0], wells.var_names[1]} <= kept
    assert kept == _kept_by_pycytominer(wells, "correlation_threshold", corr_threshold=0.9)


PYCYTOMINER_ORDER = ["variance_threshold", "correlation_threshold", "drop_na_columns", "blocklist"]


def test_pycytominer_order_gives_pycytominer_selection(wells):
    """Passing pycytominer's operations in pycytominer's order reproduces its selection."""
    frame = mt.get.to_dataframe(wells)
    expected = pycytominer.feature_select(profiles=frame, features=list(wells.var_names), operation=PYCYTOMINER_ORDER)
    expected_kept = {c for c in expected.columns if not c.startswith("Metadata_")}

    mt.pp.feature_select(wells, operations=PYCYTOMINER_ORDER)
    actual = set(wells.var_names[wells.var["selected"]])
    assert actual == expected_kept, {
        "only_ours": sorted(actual - expected_kept),
        "only_theirs": sorted(expected_kept - actual),
    }


def _order_sensitive_wells():
    """A correlated pair whose survivor depends on whether a mostly-missing feature is still in the ranking."""
    rng = np.random.default_rng(3)
    n = 200
    base = rng.normal(size=n)
    a = base + rng.normal(scale=0.1, size=n)
    b = base + rng.normal(scale=0.1, size=n)
    # Mostly missing, and where present it tracks a closely, raising a's total correlation.
    gappy = a + rng.normal(scale=0.05, size=n)
    gappy[rng.random(n) < 0.5] = np.nan
    noise = rng.normal(size=(n, 3))
    X = np.column_stack([a, b, gappy, noise])
    names = [f"Cells_Intensity_MeanIntensity_Ch{i}" for i in range(X.shape[1])]
    obs = pd.DataFrame(
        {"Metadata_Plate": "P1", "Metadata_Well": [f"W{i:03d}" for i in range(n)]},
        index=[f"w{i}" for i in range(n)],
    )
    return ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=names))


@pytest.mark.parametrize(
    "operations",
    [PYCYTOMINER_ORDER, ["drop_na_columns", "variance_threshold", "correlation_threshold", "blocklist"]],
    ids=["pycytominer-order", "drop-na-first"],
)
def test_any_order_matches_pycytominer_run_in_that_order(operations):
    wells = _order_sensitive_wells()
    frame = wells.to_df().join(wells.obs)
    expected = pycytominer.feature_select(profiles=frame, features=list(wells.var_names), operation=operations)
    expected_kept = {c for c in expected.columns if not c.startswith("Metadata_")}

    mt.pp.feature_select(wells, operations=operations)
    assert set(wells.var_names[wells.var["selected"]]) == expected_kept


def test_the_order_decides_which_member_of_a_correlated_pair_is_kept():
    wells = _order_sensitive_wells()
    corr_first = mt.pp.feature_select(wells, operations=PYCYTOMINER_ORDER, copy=True)
    na_first = mt.pp.feature_select(
        wells, operations=["drop_na_columns", "variance_threshold", "correlation_threshold"], copy=True
    )
    assert set(wells.var_names[corr_first.var["selected"]]) != set(wells.var_names[na_first.var["selected"]])


def test_subset_features_preserves_values(wells):
    mt.pp.feature_select(wells)
    subset = mt.pp.subset_features(wells)
    assert subset.n_vars == int(wells.var["selected"].sum())
    for name in subset.var_names:
        np.testing.assert_array_equal(subset[:, name].X, wells[:, name].X)
