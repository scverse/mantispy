"""Dose-response trends and curve fits."""

import numpy as np
import pandas as pd
import pytest

import mantispy as mt
from mantispy.tl._dose import four_parameter_logistic


@pytest.fixture
def dosed():
    """A plate where one compound has a real dose response and another has none."""
    adata = mt.ds.synthetic_plate(n_wells=96, n_cells=20, n_features=10, n_perturbations=1, seed=0)
    rng = np.random.default_rng(0)
    n = adata.n_obs
    compound = np.where(np.arange(n) % 2 == 0, "active", "flat")
    dose = np.tile([0.01, 0.1, 1.0, 10.0, 100.0], n)[:n]

    adata.obs["Metadata_Compound"] = compound
    adata.obs["Metadata_Concentration"] = dose
    response = rng.normal(0, 0.05, n)
    active = compound == "active"
    # A sigmoid response. A response linear in log10(dose) still gives a clean Spearman
    # trend, but without a plateau a four-parameter logistic has no EC50 to find and
    # fit_ok would refuse it.
    response[active] += four_parameter_logistic(np.log10(dose[active]), 0.0, 10.0, 0.0, 1.5)
    adata.obs["hits_distance"] = response
    return adata


def test_dose_response_separates_a_real_trend_from_none(dosed):
    mt.tl.dose_response(dosed, min_doses=4)
    table = dosed.uns["mantispy"]["dose_response"].set_index("compound")
    assert abs(table.loc["active", "spearman"]) > 0.8
    assert abs(table.loc["flat", "spearman"]) < 0.4
    assert table.loc["active", "qvalue"] < table.loc["flat", "qvalue"]
    assert (table["n_doses"] == 5).all()
    assert bool(table.loc["active", "fit_ok"])


def test_too_few_doses_skips_the_curve_but_keeps_the_trend(dosed):
    """A four-parameter curve fitted to three points is not a usable result."""
    mt.tl.dose_response(dosed, min_doses=99)
    table = dosed.uns["mantispy"]["dose_response"]
    assert not table["fit_ok"].any()
    assert table["spearman"].notna().all()
    assert table["ec50"].isna().all()


def test_a_missing_response_column_says_what_to_run(dosed):
    with pytest.raises(KeyError, match="mt.tl.hit_calling"):
        mt.tl.dose_response(dosed, response="not_computed")


def test_the_curve_recovers_a_known_ec50():
    from mantispy.tl._dose import _fit_curve

    log_dose = np.log10(np.geomspace(0.001, 100, 24))
    truth = four_parameter_logistic(log_dose, bottom=0.0, top=10.0, log_ec50=np.log10(0.5), hill=1.5)
    noisy = truth + np.random.default_rng(0).normal(0, 0.05, log_dose.size)
    ec50, hill, bottom, top, r_squared, ok = _fit_curve(log_dose, noisy, 0.8)
    assert ok
    assert ec50 == pytest.approx(0.5, rel=0.15)
    assert hill == pytest.approx(1.5, rel=0.2)


def test_round_trip(dosed, tmp_path):
    mt.tl.dose_response(dosed, min_doses=4)
    mt.io.write(dosed, tmp_path / "dose.h5ad")
    assert len(mt.io.read(tmp_path / "dose.h5ad").uns["mantispy"]["dose_response"]) == 2


def _inhibitor():
    """A clean IC50 series: the response falls from 10 to 2 with an EC50 of 1."""
    from mantispy.tl._dose import four_parameter_logistic

    doses = np.repeat(np.geomspace(0.01, 100.0, 8), 3)
    response = four_parameter_logistic(np.log10(doses), 10.0, 2.0, 0.0, 1.0)
    frame = pd.DataFrame(
        {
            "Metadata_Compound": "cpd",
            "Metadata_Concentration": doses,
            "Metadata_Plate": "P1",
            "Metadata_Well": [f"A{i:03d}" for i in range(doses.size)],
            "hits_distance": response,
        }
    )
    return frame


def test_the_fitted_asymptotes_are_kept():
    """Re-deriving the asymptotes from the data's min/max forces bottom < top and draws
    every inhibitory curve mirrored."""
    from mantispy.tl._dose import _fit_curve

    frame = _inhibitor()
    ec50, hill, bottom, top, r_squared, ok = _fit_curve(
        np.log10(frame["Metadata_Concentration"].to_numpy()), frame["hits_distance"].to_numpy(), 0.8
    )
    assert ok
    assert bottom > top, "an inhibitor runs downhill; the fit must be allowed to say so"
    assert ec50 == pytest.approx(1.0, rel=0.2)


def test_fit_ok_refuses_a_curve_that_only_the_optimiser_believes():
    """Four parameters converge on almost any six points, so convergence is not a verdict."""
    from mantispy.tl._dose import _fit_curve

    rng = np.random.default_rng(0)
    log_dose = np.log10(np.array([0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]))

    converged = sum(np.isfinite(_fit_curve(log_dose, rng.normal(0, 1, 6), 0.8)[0]) for _ in range(60))
    accepted = sum(_fit_curve(log_dose, rng.normal(0, 1, 6), 0.8)[5] for _ in range(60))
    assert converged >= 55, "the optimiser really does succeed on noise; that is the point"
    assert accepted <= 10, "fit_ok must not"

    # And a real curve still passes, with the EC50 it was built from.
    truth = four_parameter_logistic(log_dose, 0.0, 10.0, 0.0, 1.5)
    ec50, _, _, _, r_squared, ok = _fit_curve(log_dose, truth + rng.normal(0, 0.2, 6), 0.8)
    assert ok
    assert r_squared > 0.99
    assert ec50 == pytest.approx(1.0, rel=0.3)
