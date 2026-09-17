"""Which features can be trusted, and which only follow the batch."""

import numpy as np
import pytest

import mantispy as mt


@pytest.fixture
def replicated():
    """Several replicates per perturbation, so an ICC is estimable."""
    cells = mt.ds.synthetic_plate(
        n_plates=2, n_wells=96, n_cells=20, n_features=30, n_perturbations=7, effect_size=3.0, seed=0
    )
    return mt.tl.aggregate(cells, min_cells=0)


def test_reproducible_features_score_higher_than_noise(replicated):
    """A feature the perturbations move consistently must beat one that is pure noise."""
    values = replicated.X.copy()
    values[:, 0] = np.random.default_rng(0).standard_normal(replicated.n_obs)
    replicated.X = values

    mt.pp.feature_reproducibility(replicated)
    icc = replicated.var["icc"]
    assert icc.iloc[0] < 0.1
    assert icc.max() > 0.4
    assert replicated.var["icc_selected"].dtype == bool
    assert icc.between(-1, 1).all()


def test_icc_is_the_between_group_share_of_variance(replicated):
    """Checked against the textbook formula on a hand-built balanced design."""
    from mantispy.pp._feature_qc import intraclass_correlation

    rng = np.random.default_rng(0)
    groups, replicates = 8, 5
    codes = np.repeat(np.arange(groups), replicates)
    level = rng.normal(0, 3.0, groups)[codes]
    values = (level + rng.normal(0, 1.0, groups * replicates))[:, None]

    between = np.var([values[codes == g].mean() for g in range(groups)], ddof=1) * replicates
    within = np.mean([values[codes == g].var(ddof=1) for g in range(groups)])
    expected = (between - within) / (between + (replicates - 1) * within)
    np.testing.assert_allclose(intraclass_correlation(values, codes, groups)[0], expected, rtol=1e-10)


def test_batch_sensitivity_finds_an_injected_batch_effect(replicated):
    values = replicated.X.copy()
    batch = (replicated.obs["Metadata_Plate"] == "Plate01").to_numpy()
    values[batch, 0] += 20.0  # feature 0 depends on the plate and nothing else
    replicated.X = values

    mt.pp.feature_batch_sensitivity(replicated, batch_key="Metadata_Plate")
    assert bool(replicated.var["batch_sensitive"].iloc[0])
    assert replicated.var["batch_qvalue"].iloc[0] < replicated.var["batch_qvalue"].median()


def test_batch_sensitivity_needs_more_than_one_batch(replicated):
    """Returning all-ones would look like a clean result rather than an undefined test."""
    replicated.obs["Metadata_OneBatch"] = "only"
    with pytest.raises(ValueError, match="at least two batches"):
        mt.pp.feature_batch_sensitivity(replicated, batch_key="Metadata_OneBatch")
