"""Behaviour of feature_select beyond equivalence, which test_equivalence_* covers."""

import pytest

import mantispy as mt


def test_writes_a_bool_column_and_a_per_operation_count(wells):
    mt.pp.feature_select(wells)
    assert wells.var["selected"].dtype == bool
    counts = wells.uns["mantispy"]["feature_select"]
    assert set(counts) == set(mt.pp._select.DEFAULT_OPERATIONS)
    assert all(isinstance(value, int) for value in counts.values())


def test_nothing_is_dropped_or_reordered(wells):
    before = list(wells.var_names)
    mt.pp.feature_select(wells)
    assert list(wells.var_names) == before


def test_key_added_and_unknown_operation(wells):
    mt.pp.feature_select(wells, key_added="my_key")
    assert "my_key" in wells.var
    with pytest.raises(ValueError, match="unknown operation"):
        mt.pp.feature_select(wells, operations=("nonsense",))


def test_noise_removal_needs_its_grouping_column(wells):
    wells.obs = wells.obs.drop(columns="Metadata_Perturbation")
    with pytest.raises(KeyError, match="group replicates by"):
        mt.pp.feature_select(wells, operations=("noise_removal",))


def test_subset_features_requires_the_key(wells):
    with pytest.raises(KeyError, match="feature_select"):
        mt.pp.subset_features(wells, key="missing")


def test_correlated_copies_are_broken_up(cells):
    """The synthetic plate injects near-duplicate features; selection must split them."""
    from mantispy.ds import synthetic_plate

    plate = synthetic_plate(n_wells=48, n_cells=10, n_features=20, n_correlated_pairs=3, seed=0)
    profiles = mt.tl.aggregate(plate, min_cells=0)
    mt.pp.feature_select(profiles, operations=("correlation_threshold",), corr_threshold=0.9)
    selected = set(profiles.var_names[profiles.var["selected"]])
    for original, copy in profiles.uns["mantispy"]["truth"]["correlated_pairs"]:
        assert not ({original, copy} <= selected), f"{original} and {copy} both survived"


def test_an_unknown_blocklist_name_is_an_error(wells):
    """Falling back to the bundled list would return the wrong feature set."""
    with pytest.raises(ValueError, match="unknown blocklist"):
        mt.pp.feature_select(wells, operations=("blocklist",), blocklist="not_a_real_list")
    with pytest.raises(ValueError, match="unknown blocklist"):
        mt.pp.filter_features(wells, blocklist="not_a_real_list")


def test_the_same_input_always_gives_the_same_mask(wells):
    """No operation draws a random number, so a staged selection stays reproducible."""
    import numpy as np

    masks = []
    for _ in range(2):
        trial = wells.copy()
        mt.pp.feature_select(trial)
        masks.append(trial.var["selected"].to_numpy())

    assert np.array_equal(*masks)
