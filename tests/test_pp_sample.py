"""Stratified downsampling and Chatterjee feature selection."""

import numpy as np
import pytest

import mantispy as mt


@pytest.fixture
def plate():
    return mt.ds.synthetic_plate(n_wells=24, n_cells=40, n_features=20, n_perturbations=3, effect_size=3.0, seed=0)


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


def test_chatterjee_does_not_read_missingness_as_a_dependence(plate):
    """Ranking sorts NaN last, so a feature that is merely unmeasured in one group is read as a step function of the group: pure noise missing in one of four groups scored 0.34, above the 0.1 threshold and in reach of the largest score a real screen produces, so selection kept a feature for being absent.
    Scoring only the rows where it was measured is what the coefficient is defined on."""
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


def test_chatterjee_handles_ties_in_y_as_chatterjee_does():
    """Regression test for #71: breaking y's ties at random made the score depend on the seed, and scored a step function of x at 0.51."""
    from scipy.stats import rankdata

    from mantispy.pp._chatterjee import chatterjee_xi

    generator = np.random.default_rng(0)
    n = 300
    x = generator.normal(size=n)
    values = np.column_stack([np.round(x + generator.normal(0.0, 1.0, n)), (x > 0).astype(float)])

    scores = chatterjee_xi(x, values)
    for seed in range(1, 4):
        np.testing.assert_array_equal(chatterjee_xi(x, values, seed=seed), scores)
    assert scores[1] > 0.98, "a step function of x is a function of x"

    # Chatterjee's equation 1.1, which the m=1 form matches up to a term of order 1/n.
    ranks = rankdata(values[np.argsort(x), 0], method="max")
    at_or_above = n + 1 - rankdata(values[:, 0], method="min")
    expected = 1 - n * np.abs(np.diff(ranks)).sum() / (2 * (at_or_above * (n - at_or_above)).sum())
    assert scores[0] == pytest.approx(expected, abs=3 / n)
