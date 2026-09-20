"""Dose-response trends and curve fits."""

import warnings

import numpy as np
import pandas as pd
import pytest

import mantispy as mt
from mantispy.io._profiles import from_dataframe
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
    adata.obs["hits_row_distance"] = response
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
            "hits_row_distance": response,
        }
    )
    return frame


def test_the_fitted_asymptotes_are_kept():
    """Re-deriving the asymptotes from the data's min/max forces bottom < top and draws
    every inhibitory curve mirrored."""
    from mantispy.tl._dose import _fit_curve

    frame = _inhibitor()
    ec50, hill, bottom, top, r_squared, ok = _fit_curve(
        np.log10(frame["Metadata_Concentration"].to_numpy()), frame["hits_row_distance"].to_numpy(), 0.8
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


def test_the_hit_call_separates_a_real_curve_from_a_noisy_one(dosed):
    """fit_ok asks whether the optimiser converged on something curve-shaped.

    The hit call asks whether the response is large next to the controls' own spread, which is
    the question a screener is actually asking.
    """
    mt.tl.dose_response(dosed, min_doses=4)
    table = dosed.uns["mantispy"]["dose_response"].set_index("compound")
    assert table.loc["active", "hitcall"] >= 0.9, "a ten-fold response over a 0.05 baseline is a hit"
    assert table.loc["flat", "hitcall"] < 0.9


def test_pure_noise_does_not_reach_the_hit_call_threshold():
    """On noise alone the optimiser still converges; 12 of 200 such fits passed fit_ok."""
    rng = np.random.default_rng(0)
    doses = np.repeat([0.01, 0.1, 1.0, 10.0, 100.0, 1000.0], 3)
    called = 0
    for _ in range(40):
        adata = _dosed_wells(doses, rng.normal(0, 1.0, len(doses)), controls=rng.normal(0, 1.0, 12))
        mt.tl.dose_response(adata, min_doses=4)
        table = adata.uns["mantispy"]["dose_response"].set_index("compound")
        if "c" in table.index and float(table.loc["c", "hitcall"]) >= 0.9:
            called += 1
    assert called <= 2, f"{called}/40 noise-only compounds called active at hitcall >= 0.9"


def test_without_controls_the_hit_call_is_left_out_rather_than_guessed(dosed):
    del dosed.obs["Metadata_Control"]
    mt.tl.dose_response(dosed, min_doses=4)
    table = dosed.uns["mantispy"]["dose_response"]
    assert table["hitcall"].isna().all()

    mt.tl.dose_response(dosed, min_doses=4, cutoff=1.0)
    table = dosed.uns["mantispy"]["dose_response"].set_index("compound")
    assert table.loc["active", "hitcall"] >= 0.9, "an explicit cutoff is enough to call one"


def _dosed_wells(conc, resp, controls=()):
    """One compound over `conc`, plus optional control wells, as a well-level mantispy object."""
    n, m = len(conc), len(controls)
    frame = pd.DataFrame(
        {
            "Metadata_Plate": "P1",
            "Metadata_Well": [f"{chr(65 + index // 24)}{index % 24 + 1:02d}" for index in range(n + m)],
            "Metadata_Compound": ["c"] * n + ["DMSO"] * m,
            "Metadata_Concentration": np.concatenate([np.asarray(conc, dtype=float), np.zeros(m)]),
            "Metadata_Control": [False] * n + [True] * m,
            "Cells_AreaShape_a": np.zeros(n + m),
            "Cells_AreaShape_b": np.zeros(n + m),
        }
    )
    adata = from_dataframe(frame, resolution="well")
    # The response is an obs column, not a measurement; from_dataframe would read the name as a feature.
    adata.obs["hits_row_distance"] = np.concatenate([np.asarray(resp, dtype=float), np.asarray(controls, dtype=float)])
    return adata


def test_the_cutoff_comes_from_the_controls_that_did_not_fit_the_transform():
    """Regression for #83.

    The controls that fitted the centroid and covariance sit closer to the centroid they placed,
    so their spread is narrower than the held-out half's. Pooling them shrinks the MAD, which is
    the whole cutoff, and every curve then clears a bar that is too low.
    """
    conc = np.array([0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0])
    resp = np.array([0.0, 0.2, 0.1, 0.4, 0.7, 0.9, 0.6, 1.2])
    # Two halves of one control population, one measured against a centroid it helped place.
    fitted, held_out = np.linspace(-0.1, 0.1, 12), np.linspace(-1.0, 1.0, 12)

    adata = _dosed_wells(conc, resp, controls=np.concatenate([fitted, held_out]))
    adata.obs["hits_reference_held_out"] = np.arange(adata.n_obs) >= adata.n_obs - len(held_out)

    mt.tl.dose_response(adata, min_doses=4)
    honest = float(adata.uns["mantispy"]["dose_response"].set_index("compound").loc["c", "hitcall"])

    del adata.obs["hits_reference_held_out"]
    mt.tl.dose_response(adata, min_doses=4)
    pooled = float(adata.uns["mantispy"]["dose_response"].set_index("compound").loc["c", "hitcall"])

    assert honest < 0.5 < pooled, "the narrow half of the controls must not set the bar the curve clears"


def _one_compound(conc, resp, cutoff):
    adata = _dosed_wells(conc, resp)
    mt.tl.dose_response(adata, min_doses=4, reference=None, cutoff=cutoff)
    return adata.uns["mantispy"]["dose_response"].iloc[0]


def test_the_hit_call_grades_the_same_curve_against_the_cutoff_it_is_given():
    """The response tcplfit2 documents as its own example, read against three cutoffs.

    A hit call is a confidence, not a verdict: the same curve is a clear hit against a low
    cutoff, borderline when its top only just clears, and no hit when it does not.
    """
    conc = np.array([0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0])
    resp = np.array([0.0, 0.2, 0.1, 0.4, 0.7, 0.9, 0.6, 1.2])

    clear, borderline, out_of_reach = (float(_one_compound(conc, resp, c)["hitcall"]) for c in (0.2, 1.0, 2.0))
    assert clear > 0.95
    assert 0.5 < borderline < clear, "a top that only just clears the cutoff is not a confident call"
    assert out_of_reach < 0.05, "a cutoff above anything the curve reaches is not a hit"


def test_the_profile_refits_the_curve_around_the_pinned_top():
    """The third weight reads a drop in log-likelihood, so that likelihood has to be the largest one reachable
    with the top on the cutoff, as tcplfit2's ``toplikelihood`` fits it. Holding the other parameters where the
    unconstrained fit left them understates it and the hit call then reads a drop that is not there."""
    from mantispy.tl._dose import _Fit, _log_likelihood, _profile_at_top, _winning_model

    conc = np.array([0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0])
    resp = np.array([0.0, 0.2, 0.1, 0.4, 0.7, 0.9, 0.6, 1.2])
    log_dose = np.log10(conc)
    fit = _winning_model(log_dose, resp)

    frozen = _Fit(fit.name, fit.with_top_at(log_dose, 1.0), 0.0)
    profile = _profile_at_top(fit, log_dose, resp, 1.0)
    assert profile > _log_likelihood(resp - frozen.predict(log_dose), fit.log_scale), "the rest has to be refitted"
    assert profile <= fit.log_likelihood, "and pinning the top cannot beat fitting it"


def test_a_curve_that_plateaus_is_read_by_the_logistic_and_one_still_rising_by_the_line():
    """tcplfit2 carries ten models so that a curve without a plateau still gets called.

    mantispy carries two, and picks between them by AIC.
    """
    conc = np.array([0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0])
    rng = np.random.default_rng(0)
    plateauing = 1.0 / (1 + 10 ** ((np.log10(3.0) - np.log10(conc)) * 1.5)) + rng.normal(0, 0.03, len(conc))
    still_rising = np.array([0.0, 0.2, 0.1, 0.4, 0.7, 0.9, 0.6, 1.2])

    assert _one_compound(conc, plateauing, 0.2)["hitcall_model"] == "logistic"
    assert _one_compound(conc, still_rising, 0.2)["hitcall_model"] == "linear"


def test_dose_features_names_the_features_that_move_and_leaves_the_rest_NaN(phenotypes):
    mt.tl.dose_features(phenotypes)
    table = phenotypes.uns["mantispy"]["dose_features"]
    grows = table[table["compound"] == "grows"].set_index("feature")

    assert grows.loc[["Cells_AreaShape_f0", "Cells_AreaShape_f1", "Cells_AreaShape_f2"], "bmd"].notna().all()
    assert grows.loc["Cells_AreaShape_f7", "bmd"] != grows.loc["Cells_AreaShape_f7", "bmd"], "a flat feature has no bmd"
    assert table[table["compound"] == "quiet"]["bmd"].isna().all()


def test_the_benchmark_dose_brackets_the_concentration_the_response_crosses_the_cutoff_at(phenotypes):
    """f0 rises to 8 MADs with an EC50 of 1, so it passes 3 MADs between the third and fourth concentration."""
    mt.tl.dose_features(phenotypes)
    table = phenotypes.uns["mantispy"]["dose_features"].set_index(["compound", "feature"])
    row = table.loc[("grows", "Cells_AreaShape_f0")]

    assert 0.398 <= row["bmd"] <= 2.52
    assert 6.0 <= row["max_z"] <= 11.0
    assert row["direction"] == 1.0
    assert row["spearman"] > 0.8
    assert row["qvalue"] < 0.01
    assert table.loc[("grows", "Cells_AreaShape_f7"), "qvalue"] > 0.05


def test_a_feature_with_no_spread_among_the_controls_is_left_out(phenotypes):
    mt.tl.dose_features(phenotypes)
    table = phenotypes.uns["mantispy"]["dose_features"]
    assert "Cells_AreaShape_f8" not in set(table["feature"])
    assert len(set(table["feature"])) == phenotypes.n_vars - 1


def test_dose_features_says_what_to_run_when_the_controls_are_unmarked(phenotypes):
    del phenotypes.obs["Metadata_Control"]
    with pytest.raises(KeyError, match="annotate_controls"):
        mt.tl.dose_features(phenotypes)


def test_dose_features_needs_more_than_one_control_row_to_set_a_scale(phenotypes):
    """One control well gives a baseline but no spread, so every response would be infinitely many MADs."""
    control = phenotypes.obs["Metadata_Control"].to_numpy(dtype=bool)
    phenotypes.obs["Metadata_Control"] = control & (np.cumsum(control) == 1)
    with pytest.raises(ValueError, match="at least two control rows"):
        mt.tl.dose_features(phenotypes)


def test_dose_direction_reads_a_growing_phenotype_as_one_direction(phenotypes):
    mt.tl.dose_direction(phenotypes)
    table = phenotypes.uns["mantispy"]["dose_direction"]
    grows = table[table["compound"] == "grows"].sort_values("dose")

    top = grows.tail(2)
    assert (top["split_half_cosine"] > 0.5).all(), "the direction reproduces across plates"
    assert (top["cosine_to_top"] > 0.8).all(), "and it is the direction the top concentration points in"
    assert grows["amplitude"].iloc[-1] > 3 * grows["amplitude_null"].iloc[-1]


def test_dose_direction_separates_a_turning_phenotype_from_noise(phenotypes):
    """The point of the table: a reproducible direction that disagrees with the top concentration's."""
    mt.tl.dose_direction(phenotypes)
    table = phenotypes.uns["mantispy"]["dose_direction"]

    def rotated(compound):
        block = table[table["compound"] == compound]
        return block[(block["split_half_cosine"] > 0.5) & (block["cosine_to_top"] < 0.4)]

    assert len(rotated("turns")) >= 1, "the low-dose phenotype is real and points elsewhere"
    assert len(rotated("grows")) == 0, "every reproducible concentration agrees with the top one"
    quiet = table[table["compound"] == "quiet"]
    assert (quiet["split_half_cosine"] < 0.5).all(), "noise has no direction to reproduce"


def test_the_vectorized_spearman_matches_scipy():
    from scipy.stats import spearmanr

    from mantispy.tl._dose import _spearman_against

    rng = np.random.default_rng(1)
    dose = np.repeat(np.log10(np.geomspace(0.1, 100, 5)), 4)
    block = rng.normal(size=(dose.size, 6))
    block[:, 0] += dose
    rho, pvalue = _spearman_against(dose, block)
    for column in range(block.shape[1]):
        expected = spearmanr(dose, block[:, column])
        assert rho[column] == pytest.approx(expected.statistic)
        assert pvalue[column] == pytest.approx(expected.pvalue)


def test_the_new_tables_survive_a_round_trip(phenotypes, tmp_path):
    mt.tl.dose_features(phenotypes)
    mt.tl.dose_direction(phenotypes)
    path = tmp_path / "phenotypes.h5ad"
    phenotypes.write_h5ad(path)

    import anndata as ad

    reloaded = ad.read_h5ad(path)
    for key in ("dose_features", "dose_direction"):
        pd.testing.assert_frame_equal(reloaded.uns["mantispy"][key], phenotypes.uns["mantispy"][key])


def test_a_compound_with_one_concentration_is_left_out_of_the_direction_table(phenotypes):
    """One concentration says nothing about how a response changes with concentration."""
    single = phenotypes.obs["Metadata_Compound"] == "quiet"
    phenotypes.obs.loc[single, "Metadata_Concentration"] = 1.0
    mt.tl.dose_direction(phenotypes)
    table = phenotypes.uns["mantispy"]["dose_direction"]
    assert "quiet" not in set(table["compound"])
    assert {"grows", "turns"} <= set(table["compound"])


def test_the_amplitude_floor_follows_the_plates_the_concentration_sits_on():
    """Controls drawn without regard to the layout give a floor that does not apply to the concentration.

    Here each plate's controls sit to one side of every feature, which is what a plate effect looks like. A
    concentration with a well on each plate has those offsets cancel; one with both wells on a single plate does
    not, and its floor is the offset. A floor taken from any group of the right size would report the same
    number for both.
    """
    rng = np.random.default_rng(0)
    rows = []
    for plate, offset in (("P1", 2.0), ("P2", -2.0)):
        for _ in range(8):
            rows.append({"plate": plate, "compound": "DMSO", "dose": 0.0, "offset": offset})
        for dose in np.geomspace(0.1, 100.0, 4):
            # "spread" doses one well per plate; "stacked" puts both of its wells on P1.
            rows.append({"plate": plate, "compound": "spread", "dose": dose, "offset": 0.0})
            rows.append({"plate": "P1", "compound": "stacked", "dose": dose, "offset": 0.0})
    frame = pd.DataFrame(rows)
    frame["Metadata_Plate"] = frame["plate"]
    frame["Metadata_Compound"] = frame["compound"]
    frame["Metadata_Concentration"] = frame["dose"]
    frame["Metadata_Control"] = frame["compound"] == "DMSO"
    frame["Metadata_Well"] = [f"{chr(65 + i // 24)}{i % 24 + 1:02d}" for i in range(len(frame))]
    for feature in range(4):
        frame[f"Cells_AreaShape_f{feature}"] = rng.normal(0.0, 0.1, len(frame)) + frame["offset"]
    adata = from_dataframe(frame.drop(columns=["plate", "compound", "dose", "offset"]), resolution="well")

    mt.tl.dose_direction(adata)
    table = adata.uns["mantispy"]["dose_direction"].set_index("compound")
    assert table.loc["spread", "amplitude_null"].max() < table.loc["stacked", "amplitude_null"].min()


def test_the_window_is_a_run_not_a_scatter(phenotypes):
    """Regression: one lucky concentration used to open the window several steps below the real onset.

    `grows` is silent at the two lowest concentrations. Making its second concentration look reproducible on its
    own must not pull the window down to it, because the concentration above is still silent.
    """
    mt.tl.dose_direction(phenotypes)
    table = phenotypes.uns["mantispy"]["dose_direction"]
    grows = table[table["compound"] == "grows"].sort_values("dose").reset_index(drop=True)
    labels = list(grows["phase"])

    quiet = [index for index, phase in enumerate(labels) if phase == "silent"]
    working = [index for index, phase in enumerate(labels) if phase in ("responding", "saturated")]
    assert quiet and working, labels
    assert max(quiet) < min(working), f"the window has a hole in it: {labels}"
    assert working[-1] == len(labels) - 1, "the window runs to the top concentration"


def test_a_concentration_that_lost_its_cells_is_cytotoxic_whatever_else_it_did(phenotypes):
    top = phenotypes.obs["Metadata_Concentration"] == phenotypes.obs["Metadata_Concentration"].max()
    phenotypes.obs.loc[top, "Metadata_CellCount"] = 10.0
    mt.tl.dose_direction(phenotypes)
    table = phenotypes.uns["mantispy"]["dose_direction"]
    highest = table.sort_values("dose").groupby("compound").tail(1)
    assert (highest["phase"] == "cytotoxic").all()
    assert (highest["viability"] < 0.2).all()


def test_the_phase_reaches_obs_so_the_window_can_be_subset(phenotypes):
    mt.tl.dose_direction(phenotypes)
    phases = phenotypes.obs["dose_direction_phase"]
    assert set(phases.cat.categories) == set(mt.tl._dose.DOSE_PHASES)
    assert phases[phenotypes.obs["Metadata_Control"].to_numpy(dtype=bool)].isna().all(), "controls are in no phase"

    window = phenotypes[phases == "responding"]
    assert window.n_obs > 0
    assert not window.obs["Metadata_Control"].to_numpy(dtype=bool).any()


def test_dose_trajectory_puts_every_compound_on_one_relative_axis(phenotypes):
    mt.tl.dose_direction(phenotypes)
    paths = mt.tl.dose_trajectory(phenotypes, n_positions=3)

    assert paths.n_vars == 3 * (phenotypes.n_vars - 1), "one column per feature and position, minus the flat one"
    assert list(paths.var["position"].unique()) == [0.0, 0.5, 1.0]
    assert paths.var["feature"].nunique() == phenotypes.n_vars - 1
    assert set(paths.obs["Metadata_Compound"]) <= {"grows", "turns"}, "a compound with no window is left out"
    assert (paths.obs["n_doses"] >= 2).all()
    assert (paths.obs["window_low"] < paths.obs["window_high"]).all()


def test_a_trajectory_needs_at_least_two_points(phenotypes):
    with pytest.raises(ValueError, match="at least two"):
        mt.tl.dose_trajectory(phenotypes, n_positions=1)


def test_a_turning_compound_and_a_growing_one_do_not_match_on_their_paths(phenotypes):
    """The reason to compare paths at all: `turns` and `grows` move different features in a different order."""
    mt.tl.dose_direction(phenotypes)
    paths = mt.tl.dose_trajectory(phenotypes, phases=("responding", "saturated"))
    frame = pd.DataFrame(np.asarray(paths.X), index=list(paths.obs["Metadata_Compound"]))
    assert {"grows", "turns"} <= set(frame.index)
    assert frame.loc["grows"].corr(frame.loc["turns"]) < 0.5


def test_the_fast_median_matches_numpy():
    """It exists only to avoid numpy's masked-array path on short blocks, so it has to agree with it exactly."""
    from mantispy.tl._dose import _nanmedian

    rng = np.random.default_rng(0)
    for rows in (1, 2, 3, 4, 8, 9):
        block = rng.normal(size=(rows, 12))
        block[rng.random(block.shape) < 0.3] = np.nan
        block[:, 0] = np.nan  # a feature measured nowhere
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # numpy warns on the all-NaN column; this does not
            expected = np.nanmedian(block, axis=0)
        np.testing.assert_allclose(_nanmedian(block), expected, equal_nan=True)


def test_viability_is_read_against_each_plate_not_the_whole_screen(phenotypes):
    """Plates are seeded apart, so a dense plate would otherwise read as one whose treated wells are dying."""
    dense = phenotypes.obs["Metadata_Plate"] == "P1"
    phenotypes.obs.loc[dense, "Metadata_CellCount"] = 1000.0
    mt.tl.dose_direction(phenotypes)
    table = phenotypes.uns["mantispy"]["dose_direction"]

    assert (table["phase"] != "cytotoxic").all(), "a ten-fold difference between plates is not cell loss"
    assert table["viability"].between(0.8, 1.2).all()
