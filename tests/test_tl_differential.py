"""Differential features at well level: calibration, blocking, and the layouts that cannot work."""

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import stats

import mantispy as mt
from mantispy._core.schema import stamp
from mantispy.tl._differential import squeeze_variances


def test_the_null_is_calibrated(well_profiles):
    """Controls split arbitrarily must not produce a pile of significant features."""
    adata = well_profiles(seed=1)
    mt.tl.differential_features(adata)
    table = adata.uns["mantispy"]["differential"]
    assert (table["qvalue"] < 0.05).mean() < 0.02, "a pure null should give almost nothing"
    assert (table["pvalue"] < 0.05).mean() < 0.12, "and the raw p-values should be roughly uniform"


def test_blocking_on_plate_finds_what_ignoring_it_misses(well_profiles):
    """Without blocking, plate variance stays in the residual and costs power."""
    adata = well_profiles(effect=1.2, affected=40, seed=2)
    mt.tl.differential_features(adata, key_added="blocked")
    mt.tl.differential_features(adata, block=None, key_added="unblocked")

    truth = {f"Cells_AreaShape_F{i}" for i in range(40)}
    found = {
        key: set(adata.uns["mantispy"][key].query("qvalue < 0.05")["feature"]) & truth
        for key in ("blocked", "unblocked")
    }
    assert len(found["blocked"]) > len(found["unblocked"])


def test_cells_are_refused_with_the_fix_named(well_profiles):
    """Testing per cell is the mistake this function exists to prevent."""
    cells = mt.ds.synthetic_plate(n_wells=24, n_cells=20, n_features=20, seed=0)
    with pytest.raises(ValueError, match="mt.tl.aggregate"):
        mt.tl.differential_features(cells)


def test_a_group_confounded_with_its_plate_is_skipped_not_scored(well_profiles):
    """Treatment and plate become the same variable, and no test can separate them."""
    rng = np.random.default_rng(3)
    n_features = 200
    # Two plates, with treatment fully determined by plate. The plate shift is the only
    # difference between the groups, so nothing here is a treatment effect.
    plate_shift = rng.normal(scale=1.5, size=(2, n_features))
    values = np.concatenate(
        [rng.normal(size=(8, n_features)) + plate_shift[0], rng.normal(size=(8, n_features)) + plate_shift[1]]
    )
    treated = np.arange(16) < 8
    obs = pd.DataFrame(
        {
            "Metadata_Plate": np.where(treated, "P0", "P1"),
            "Metadata_Well": [f"A{i:02d}" for i in range(16)],
            "Metadata_Perturbation": np.where(treated, "compound", "DMSO"),
            "Metadata_Control": ~treated,
        },
        index=[str(i) for i in range(16)],
    )
    adata = ad.AnnData(X=values.astype(np.float32), obs=obs)
    adata.var_names = [f"Cells_AreaShape_F{i}" for i in range(n_features)]
    stamp(adata, resolution="well")

    with pytest.raises(ValueError, match="no group had enough replicates"):
        mt.tl.differential_features(adata)

    # Without blocking the confound goes unnoticed and the plate difference is scored as
    # a treatment effect.
    mt.tl.differential_features(adata, block=None, key_added="unblocked")
    table = adata.uns["mantispy"]["unblocked"]
    assert (table["qvalue"] < 0.05).mean() > 0.1, "the unblocked test reports the plate as biology"


def test_shrinkage_is_stronger_with_fewer_replicates(well_profiles):
    """Empirical Bayes should lean on the prior when there is little to go on."""
    rng = np.random.default_rng(0)
    variances = np.exp(rng.normal(0, 1.0, size=500))
    tight, _ = squeeze_variances(variances, df=2)
    loose, _ = squeeze_variances(variances, df=40)
    assert np.var(np.log(tight)) < np.var(np.log(loose))


def test_the_contrast_is_the_plain_difference_in_means(well_profiles):
    """Moderation changes the variance, never the estimate, so the coefficient is exact."""
    adata = well_profiles(n_plates=1, per_plate=40, n_features=100, plate_sd=0.0, effect=0.7, affected=20, seed=5)
    mt.tl.differential_features(adata, block=None)
    table = adata.uns["mantispy"]["differential"]

    values = np.asarray(adata.X, dtype=np.float64)
    control = adata.obs["Metadata_Control"].to_numpy()
    expected = values[~control].mean(axis=0) - values[control].mean(axis=0)
    np.testing.assert_allclose(table["difference"].to_numpy(), expected, rtol=1e-6, atol=1e-6)

    # And it ranks features the same way an ordinary t-test does, having only rescaled.
    _, plain = stats.ttest_ind(values[~control], values[control], axis=0)
    assert stats.spearmanr(table["pvalue"], plain).statistic > 0.99


def test_the_diagnostic_catches_what_the_docstrings_claim(well_profiles):
    """The battery has to reach the right verdict on layouts built to fail each check."""
    healthy = well_profiles(n_plates=4, per_plate=16, n_features=100, seed=7)
    report = mt.metrics.diagnose_testing(healthy, n_draws=3).set_index("check")["verdict"]
    assert report["null p < 0.05"] == "pass"
    assert report["null discoveries"] == "pass"

    # Two wells per treatment across many features: a rank test cannot reach the
    # threshold that many tests demand, whatever the effect size.
    rng = np.random.default_rng(8)
    n_features, n_groups = 200, 12
    labels = ["DMSO"] * 24 + [f"pert{i:02d}" for i in range(n_groups) for _ in range(2)]
    obs = pd.DataFrame(
        {
            "Metadata_Plate": [f"P{i % 4}" for i in range(len(labels))],
            "Metadata_Well": [f"A{i % 24 + 1:02d}" for i in range(len(labels))],
            "Metadata_Perturbation": labels,
            "Metadata_Control": [label == "DMSO" for label in labels],
        },
        index=[str(i) for i in range(len(labels))],
    )
    thin = ad.AnnData(X=rng.normal(size=(len(labels), n_features)).astype(np.float32), obs=obs)
    thin.var_names = [f"Cells_AreaShape_F{i}" for i in range(n_features)]
    stamp(thin, resolution="well")

    thin_report = mt.metrics.diagnose_testing(thin, n_draws=3).set_index("check")["verdict"]
    assert thin_report["rank test resolution"] == "FAIL", "two wells cannot reach any FDR threshold"


def test_the_diagnostic_names_a_confounded_layout(well_profiles):
    """Treatments that share no plate with the reference fail, since no test can separate treatment from plate."""
    adata = well_profiles(n_plates=4, per_plate=8, n_features=60, seed=9)
    adata.obs["Metadata_Plate"] = np.where(adata.obs["Metadata_Control"].to_numpy(), "P2", "P0")
    report = mt.metrics.diagnose_testing(adata, n_draws=2).set_index("check")["verdict"]
    assert report["treatments sharing a Metadata_Plate with the reference"] == "FAIL"


def test_the_reported_floor_is_the_floor_effect_size_actually_reaches(well_profiles):
    """The diagnostic uses the same arithmetic as the function it diagnoses.

    `rank test resolution` predicts whether `tl.effect_size`'s p-values can clear
    multiple-testing correction, so its floor has to be computed the way effect_size
    computes p-values. A floor computed differently can be orders of magnitude off and fail
    a screen on which effect_size calls many features.
    """
    adata = well_profiles(n_plates=4, per_plate=16, n_features=40, seed=11)
    treated = ~adata.obs["Metadata_Control"].to_numpy()
    groups = adata.obs.loc[treated, "Metadata_Perturbation"].astype(str)
    first = groups.iloc[0]

    # Push one group clear of every control in one feature: the largest effect the rank
    # test can be shown, so effect_size's smallest p is the floor by construction.
    values = np.asarray(adata.X, dtype=np.float64)
    rows = (adata.obs["Metadata_Perturbation"].astype(str) == first).to_numpy()
    values[rows, 0] = values[:, 0].max() + 1.0 + np.arange(int(rows.sum()))
    adata.X = values.astype(np.float32)

    mt.tl.effect_size(adata, key_added="e")
    reached = float(adata.uns["mantispy"]["e"]["pvalue"].min())

    report = mt.metrics.diagnose_testing(adata, n_draws=2).set_index("check")
    assert float(report.loc["rank test resolution", "value"]) == pytest.approx(reached, rel=0.05)
