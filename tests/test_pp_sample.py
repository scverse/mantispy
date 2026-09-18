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


def test_chatterjee_rejects_a_constant_feature_it_cannot_measure(plate):
    """Ranking y's ties in x-order reads the group order back out of a tied column, which
    scored two all-zero features at 0.998, above a genuine dose-response feature at 0.752,
    so selection kept exactly the unmeasurable features it exists to reject. Zero-inflated
    Zernike, Granularity and RadialDistribution columns hit this routinely."""
    codes = plate.obs["Metadata_Perturbation"].cat.codes.to_numpy()
    values = plate.X.copy()
    values[:, 1] = 0.0
    values[:, 2] = 0.0
    values[:, 3] = 2.0 * codes + np.random.default_rng(1).normal(0.0, 0.2, plate.n_obs)
    plate.X = values

    mt.pp.feature_select_chatterjee(plate)
    scores = plate.var["chatterjee_xi"].to_numpy()
    assert scores[1] < 0.1 and scores[2] < 0.1, "a constant feature carries no information"
    assert scores[3] > 0.5, "and the dose-response feature still scores high"
    assert not plate.var["selected_chatterjee"].to_numpy()[[1, 2]].any()


def test_chatterjee_does_not_read_missingness_as_a_dependence(plate):
    """Ranking sorts NaN last, so a feature that is merely unmeasured in one group is read as a
    step function of the group: pure noise missing in one of four groups scored 0.34, above the
    0.1 threshold and in reach of the largest score a real screen produces, so selection kept a
    feature for being absent. Scoring only the rows where it was measured is what the coefficient
    is defined on."""
    codes = plate.obs["Metadata_Perturbation"].cat.codes.to_numpy()
    generator = np.random.default_rng(2)
    values = plate.X.copy()
    values[:, 4] = generator.normal(0.0, 1.0, plate.n_obs)
    values[codes == 1, 4] = np.nan
    values[:, 5] = 2.0 * codes + generator.normal(0.0, 0.2, plate.n_obs)
    plate.X = values

    mt.pp.feature_select_chatterjee(plate)
    scores = plate.var["chatterjee_xi"].to_numpy()
    selected = plate.var["selected_chatterjee"].to_numpy()
    assert scores[4] < 0.1, "noise missing in one group does not depend on the group"
    assert scores[5] > 0.5, "and a real group effect still scores high"
    assert not selected[4]
    assert selected[5]


def test_chatterjee_refuses_a_feature_it_has_too_few_measurements_of(plate):
    """Twelve finite values leave too few pairs for xi to be told from its own noise floor, so the
    feature scores NaN rather than a number that a threshold would compare."""
    values = plate.X.copy()
    values[:, 6] = np.nan
    values[:12, 6] = np.arange(12.0)
    plate.X = values

    mt.pp.feature_select_chatterjee(plate)
    assert np.isnan(plate.var["chatterjee_xi"].to_numpy()[6])
    assert not plate.var["selected_chatterjee"].to_numpy()[6]


def test_chatterjee_scores_a_gapped_column_the_same_as_scoring_it_alone():
    """One missing value used to send every column through the per-column loop; the complete columns are ranked together and only the gapped ones over their own rows.

    The two paths have to agree exactly, because a feature's score decides whether selection keeps it.
    Both break their ties from the same seed, so scoring a column alongside others has to give what scoring it alone gives.
    """
    from mantispy.pp._chatterjee import chatterjee_xi

    generator = np.random.default_rng(0)
    x = generator.normal(size=300)
    values = np.column_stack([index * x + generator.normal(0.0, 1.0, 300) for index in range(6)])
    values[:5, 1] = np.nan  # gaps in some columns and none in others
    values[100:140, 4] = np.nan
    values[:-10, 5] = np.nan  # and one feature measured too rarely to score

    together = chatterjee_xi(x, values)
    apart = [chatterjee_xi(x, values[:, [column]])[0] for column in range(values.shape[1])]

    np.testing.assert_array_equal(together, apart)
    assert np.isfinite(together[:5]).all()
    assert np.isnan(together[5]), "ten finite values leave too few pairs to score"
