"""Stratified downsampling and Chatterjee feature selection."""

import numpy as np
import pytest

import mantispy as mt


@pytest.fixture
def plate():
    return mt.ds.synthetic_plate(n_wells=24, n_cells=40, n_features=20, n_perturbations=3, effect_size=3.0, seed=0)


def test_downsample_caps_each_group(plate):
    small = mt.pp.downsample(plate, n_per_group=10)
    counts = small.obs.groupby(["Metadata_Plate", "Metadata_Well"], observed=True).size()
    assert counts.max() == 10
    assert small.n_obs == 24 * 10


def test_downsample_keeps_small_groups_whole(plate):
    assert mt.pp.downsample(plate, n_per_group=10_000).n_obs == plate.n_obs


def test_downsample_is_reproducible_and_leaves_the_source_alone(plate):
    before = plate.n_obs
    first = mt.pp.downsample(plate, n_per_group=5, seed=1)
    second = mt.pp.downsample(plate, n_per_group=5, seed=1)
    assert plate.n_obs == before
    np.testing.assert_array_equal(first.obs_names.to_numpy(), second.obs_names.to_numpy())
    assert first.uns["mantispy"]["params"]["downsample"]["n_per_group"] == 5


def test_stratifying_keeps_the_rare_group_represented(plate):
    small = mt.pp.downsample(plate, n_per_group=12, groupby=("Metadata_Plate",), stratify="Metadata_Perturbation")
    shares = small.obs["Metadata_Perturbation"].value_counts(normalize=True)
    assert shares.max() - shares.min() < 0.25
    assert small.n_obs <= 12 * plate.obs["Metadata_Plate"].nunique()


def test_chatterjee_keeps_the_features_that_depend_on_the_group(plate):
    mt.pp.feature_select_chatterjee(plate, threshold=0.05)
    affected = {name for names in plate.uns["mantispy"]["truth"]["affected_features"].values() for name in names}
    kept = set(plate.var_names[plate.var["selected_chatterjee"].to_numpy()])
    assert affected <= kept, affected - kept


def test_chatterjee_writes_a_flag_and_the_statistic(plate):
    mt.pp.feature_select_chatterjee(plate)
    assert plate.var["selected_chatterjee"].dtype == bool
    assert plate.var["chatterjee_xi"].between(-1, 1).all()


def test_chatterjee_sees_a_non_monotonic_dependence_that_correlation_misses(plate):
    """A dependence that is high at both extremes and low in the middle, which correlation misses."""
    codes = plate.obs["Metadata_Perturbation"].cat.codes.to_numpy()
    values = plate.X.copy()
    values[:, 0] = np.abs(codes - codes.max() / 2) + np.random.default_rng(0).normal(0, 0.05, plate.n_obs)
    plate.X = values

    mt.pp.feature_select_chatterjee(plate)
    # With four groups the fold puts two of them at the same level, which caps xi well
    # below 1, but it stays an order of magnitude above the correlation.
    assert plate.var["chatterjee_xi"].to_numpy()[0] > 0.4
    assert abs(np.corrcoef(codes, values[:, 0])[0, 1]) < 0.2
