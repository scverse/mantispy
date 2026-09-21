"""feature_select equivalence against pycytominer.

Semantics follow the pinned pycytominer source rather than its docs. Two points are easy
to get wrong, and both are covered below. `variance_threshold` is an sklearn variance cut;
the frequency/uniqueness rule is `frequency_threshold`. `correlation_threshold` judges
each pair independently against a ranking computed once, using the signed correlation.
"""

import numpy as np
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


def test_default_pipeline_equivalence(wells):
    """The default is pycytominer's own plus drop_degenerate, which removes nothing normalize did not flag."""
    assert "degenerate_scale" not in wells.var or not wells.var["degenerate_scale"].any()
    frame = mt.get.to_dataframe(wells)
    operations = [operation for operation in mt.pp._select.DEFAULT_OPERATIONS if operation != "drop_degenerate"]
    expected = pycytominer.feature_select(profiles=frame, features=list(wells.var_names), operation=operations)
    expected_kept = {c for c in expected.columns if not c.startswith("Metadata_")}

    mt.pp.feature_select(wells)
    actual = set(wells.var_names[wells.var["selected"]])
    assert actual == expected_kept, {
        "only_ours": sorted(actual - expected_kept),
        "only_theirs": sorted(expected_kept - actual),
    }


def test_subset_features_preserves_values(wells):
    mt.pp.feature_select(wells)
    subset = mt.pp.subset_features(wells)
    assert subset.n_vars == int(wells.var["selected"].sum())
    for name in subset.var_names:
        np.testing.assert_array_equal(subset[:, name].X, wells[:, name].X)
