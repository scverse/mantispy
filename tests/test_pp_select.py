"""Behaviour of feature_select beyond equivalence, which test_equivalence_* covers."""

import anndata as ad
import numpy as np
import pandas as pd
import pytest

import mantispy as mt


def test_writes_a_bool_column_and_a_per_operation_count(wells):
    mt.pp.feature_select(wells)
    assert wells.var["selected"].dtype == bool
    counts = wells.uns["mantispy"]["feature_select"]
    assert set(counts) == set(mt.pp._select.DEFAULT_OPERATIONS)
    assert all(isinstance(value, int) for value in counts.values())


def test_counts_are_what_each_operation_removes_on_its_own(wells):
    """A feature two operations both remove is counted by both, in whichever order they run."""
    values = wells.X.copy()
    values[:, 0] = 1.0  # Constant, so variance_threshold removes it.
    values[: int(0.6 * wells.n_obs), 0] = np.nan  # And mostly missing, so drop_na_columns removes it too.
    wells.X = values

    alone = {}
    for operation in ("drop_na_columns", "variance_threshold"):
        trial = wells.copy()
        mt.pp.feature_select(trial, operations=(operation,))
        alone[operation] = int((~trial.var["selected"].to_numpy()).sum())
    assert min(alone.values()) > 0, "neither operation removes anything; the test proves nothing"

    for operations in (("drop_na_columns", "variance_threshold"), ("variance_threshold", "drop_na_columns")):
        run = wells.copy()
        mt.pp.feature_select(run, operations=operations)
        assert dict(run.uns["mantispy"]["feature_select"]) == alone, f"counts changed with order {operations}"


def _one_feature_per_operation():
    """Four replicate groups in which drop_na_columns, drop_outliers and noise_removal each have exactly one feature to remove."""
    rng = np.random.default_rng(0)
    n_obs, n_groups = 20, 4
    per_group = n_obs // n_groups

    plain = rng.normal(scale=0.2, size=n_obs)  # no NaN, small, quiet within a group: every operation keeps it
    noisy = rng.normal(scale=5.0, size=n_obs)  # within-group standard deviation above the 0.8 cutoff
    ratio = np.repeat(np.arange(1, n_groups + 1) * 1000.0, per_group)  # above the 500 cutoff, flat within a group
    sparse = np.full(n_obs, np.nan)  # missing in 60% of rows, two values per group so no group is all-NaN
    for group in range(n_groups):
        start = group * per_group
        sparse[start] = group * 0.5
        sparse[start + 1] = group * 0.5 + 0.1

    obs = pd.DataFrame(
        {"Metadata_Perturbation": np.repeat([f"g{group}" for group in range(n_groups)], per_group)},
        index=[str(index) for index in range(n_obs)],
    )
    adata = ad.AnnData(X=np.column_stack([plain, sparse, ratio, noisy]).astype(np.float32), obs=obs)
    adata.var_names = [
        "Cells_AreaShape_Plain",
        "Cells_AreaShape_Sparse",
        "Cells_Intensity_Ratio",
        "Cells_AreaShape_Jitter",
    ]
    return adata


def test_each_operation_counts_the_features_it_actually_removed():
    """The counts are the only thing that notices one of these three operations turning into a no-op that selects every feature."""
    adata = _one_feature_per_operation()
    operations = ("drop_na_columns", "drop_outliers", "noise_removal")

    mt.pp.feature_select(adata, operations=operations)

    assert adata.uns["mantispy"]["feature_select"] == dict.fromkeys(operations, 1)
    assert sorted(adata.var_names[~adata.var["selected"]]) == [
        "Cells_AreaShape_Jitter",
        "Cells_AreaShape_Sparse",
        "Cells_Intensity_Ratio",
    ]


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
