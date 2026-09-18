"""Consensus signatures, one profile per perturbation."""

import numpy as np
import pytest

import mantispy as mt


@pytest.fixture
def profiles():
    cells = mt.ds.synthetic_plate(
        n_plates=2, n_wells=48, n_cells=20, n_features=15, n_perturbations=4, effect_size=3.0, seed=0
    )
    return mt.tl.aggregate(cells, min_cells=0)


def test_consensus_has_one_row_per_perturbation(profiles):
    result = mt.tl.consensus(profiles)
    assert result.n_obs == profiles.obs["Metadata_Perturbation"].nunique()
    assert result.uns["mantispy"]["resolution"] == "perturbation"
    assert mt.io.validate(result).ok, mt.io.validate(result).errors
    assert "Metadata_ReplicateCount" in result.obs
    assert bool(result.obs.loc[result.obs["Metadata_Perturbation"] == "DMSO", "Metadata_Control"].iloc[0])


def test_modz_is_dragged_far_less_than_a_mean_by_one_bad_replicate(profiles):
    """modz is a weighted mean, so it is compared with the unweighted mean. A median is
    more robust still (see the tl.consensus docstring) and would not show what the
    weighting adds."""
    rows = np.flatnonzero((profiles.obs["Metadata_Perturbation"] == "pert00").to_numpy())
    corrupted = profiles.copy()
    values = corrupted.X.copy()
    values[rows[0]] = 50.0
    corrupted.X = values

    reference = mt.tl.consensus(profiles, method="modz")
    position = list(reference.obs["Metadata_Perturbation"]).index("pert00")
    modz = np.abs(mt.tl.consensus(corrupted, method="modz").X[position] - reference.X[position]).mean()
    mean = np.abs(np.asarray(corrupted.X, dtype=float)[rows].mean(axis=0) - reference.X[position]).mean()
    assert modz < mean / 10, (modz, mean)

    weights = mt.tl.consensus(corrupted).uns["mantispy"]["consensus_weights"]
    assert weights["weight"].to_numpy()[rows[0]] < 0.01


def test_weights_are_recorded_and_normalised(profiles):
    result = mt.tl.consensus(profiles)
    weights = result.uns["mantispy"]["consensus_weights"]
    assert len(weights) == profiles.n_obs
    np.testing.assert_allclose(weights.groupby("group")["weight"].sum(), 1.0, atol=1e-3)


def test_min_replicates_drops_groups(profiles):
    assert mt.tl.consensus(profiles, min_replicates=1000).n_obs == 0


def test_one_missing_feature_does_not_evict_a_replicate():
    """A replicate with one missing feature is weighted slightly down, not excluded.

    scipy's rankdata propagates NaN across a whole row by default, which would push the
    replicate to min_weight. Partial missingness is normal in CellProfiler output, since
    Zernike and radial features are undefined below an object size.
    """
    from mantispy.tl._consensus import modz_weights

    rng = np.random.default_rng(0)
    block = rng.normal(size=(4, 20))
    block[1:] = block[0] + rng.normal(0, 0.1, (3, 20))  # four replicates that agree
    clean = modz_weights(block)

    holed = block.copy()
    holed[2, 5] = np.nan  # one missing value, of eighty
    weights = modz_weights(holed)

    assert weights[2] > 0.15, "one gap in twenty features must not evict the replicate"
    # Not "the weight must fall": a gap carries no information either way, and requiring it
    # to fall is satisfied just as well by a fill that fabricates agreement instead.
    assert weights[2] == pytest.approx(clean[2], abs=0.01), "one gap of eighty barely moves the weight"
    assert weights.sum() == pytest.approx(1.0, abs=1e-3)


def test_a_gap_two_replicates_share_does_not_make_them_agree():
    """A gap the same in two replicates must not correlate them: zero-filled ranks gave the pair the two largest weights.

    ``similarity_matrix`` fills missing values with zero, which is below every rank, so two
    replicates with the same gap looked alike there. Structured missingness is the common
    case in CellProfiler output, where the same features are undefined in the same wells.
    """
    from mantispy.tl._consensus import modz_weights

    rng = np.random.default_rng(0)
    block = rng.normal(size=(4, 60))  # four replicates that agree on nothing
    holed = block.copy()
    holed[np.ix_([0, 1], np.arange(20))] = np.nan

    clean, weights = modz_weights(block), modz_weights(holed)
    assert np.abs(weights - clean).max() < 0.05, f"the gap moved the weights from {clean} to {weights}"
    assert set(np.argsort(weights)[-2:]) == set(np.argsort(clean)[-2:]), "and must not re-rank the replicates"
