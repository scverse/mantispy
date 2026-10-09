"""Dose-response trends and curve fits."""

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
    # Sigmoid, not linear in log10(dose): without a plateau the logistic has no EC50 and fit_ok would refuse it.
    response[active] += four_parameter_logistic(np.log10(dose[active]), 0.0, 10.0, 0.0, 1.5)
    adata.obs["hits_row_distance"] = response
    return adata


def test_a_missing_response_column_says_what_to_run(dosed):
    with pytest.raises(KeyError, match="mt.tl.hit_calling"):
        mt.tl.dose_response(dosed, response="not_computed")


@pytest.mark.slow
def test_fit_ok_refuses_a_curve_that_only_the_optimiser_believes():
    """Four parameters converge on almost any six points, so convergence is not a verdict."""
    from mantispy.tl._dose import _fit_curve

    rng = np.random.default_rng(0)
    log_dose = np.log10(np.array([0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]))

    converged = sum(np.isfinite(_fit_curve(log_dose, rng.normal(0, 1, 6), 0.8)[0]) for _ in range(60))
    accepted = sum(_fit_curve(log_dose, rng.normal(0, 1, 6), 0.8)[5] for _ in range(60))
    assert converged >= 55, "the optimiser really does succeed on noise; that is the point"
    assert accepted <= 10, "fit_ok must not"

    truth = four_parameter_logistic(log_dose, 0.0, 10.0, 0.0, 1.5)
    ec50, _, _, _, r_squared, ok = _fit_curve(log_dose, truth + rng.normal(0, 0.2, 6), 0.8)
    assert ok
    assert r_squared > 0.99
    assert ec50 == pytest.approx(1.0, rel=0.3)


@pytest.mark.slow
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


def _dosed_wells(conc, resp, controls=()):
    n, m = len(conc), len(controls)
    frame = pd.DataFrame(
        {
            "Metadata_Plate": "P1",
            "Metadata_Well": [f"{chr(65 + index // 24)}{index % 24 + 1:02d}" for index in range(n + m)],
            "Metadata_Compound": ["c"] * n + ["DMSO"] * m,
            "Metadata_Concentration": np.concatenate([np.asarray(conc, dtype=float), np.zeros(m)]),
            "Metadata_Control": [False] * n + [True] * m,
            "Metadata_Control_Type": ["treatment"] * n + ["negcon"] * m,
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

    The controls that fitted the centroid and covariance sit closer to the centroid they placed, so their spread is narrower than the held-out half's.
    Pooling them shrinks the MAD, which is the whole cutoff, and every curve then clears a bar that is too low.
    """
    conc = np.array([0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0])
    resp = np.array([0.0, 0.2, 0.1, 0.4, 0.7, 0.9, 0.6, 1.2])
    fitted, held_out = np.linspace(-0.1, 0.1, 12), np.linspace(-1.0, 1.0, 12)

    adata = _dosed_wells(conc, resp, controls=np.concatenate([fitted, held_out]))
    adata.obs["hits_reference_held_out"] = np.arange(adata.n_obs) >= adata.n_obs - len(held_out)

    mt.tl.dose_response(adata, min_doses=4)
    honest = float(adata.uns["mantispy"]["dose_response"].set_index("compound").loc["c", "hitcall"])

    del adata.obs["hits_reference_held_out"]
    mt.tl.dose_response(adata, min_doses=4)
    pooled = float(adata.uns["mantispy"]["dose_response"].set_index("compound").loc["c", "hitcall"])

    assert honest < 0.5 < pooled, "the narrow half of the controls must not set the bar the curve clears"


def test_dose_features_needs_more_than_one_control_row_to_set_a_scale(phenotypes):
    """One control well gives a baseline but no spread, so every response would be infinitely many MADs."""
    control = phenotypes.obs["Metadata_Control"].to_numpy(dtype=bool)
    keep = control & (np.cumsum(control) == 1)
    phenotypes.obs["Metadata_Control"] = keep
    phenotypes.obs["Metadata_Control_Type"] = np.where(keep, "negcon", "treatment")
    with pytest.raises(ValueError, match="at least two control rows"):
        mt.tl.dose_features(phenotypes)


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

    Here each plate's controls sit to one side of every feature, which is what a plate effect looks like.
    A concentration with a well on each plate has those offsets cancel; one with both wells on a single plate does not, and its floor is the offset.
    A floor taken from any group of the right size would report the same number for both.
    """
    rng = np.random.default_rng(0)
    rows = []
    for plate, offset in (("P1", 2.0), ("P2", -2.0)):
        for _ in range(8):
            rows.append({"plate": plate, "compound": "DMSO", "dose": 0.0, "offset": offset})
        for dose in np.geomspace(0.1, 100.0, 4):
            rows.append({"plate": plate, "compound": "spread", "dose": dose, "offset": 0.0})
            rows.append({"plate": "P1", "compound": "stacked", "dose": dose, "offset": 0.0})
    frame = pd.DataFrame(rows)
    frame["Metadata_Plate"] = frame["plate"]
    frame["Metadata_Compound"] = frame["compound"]
    frame["Metadata_Concentration"] = frame["dose"]
    frame["Metadata_Control"] = frame["compound"] == "DMSO"
    frame["Metadata_Control_Type"] = np.where(frame["Metadata_Control"], "negcon", "treatment")
    frame["Metadata_Well"] = [f"{chr(65 + i // 24)}{i % 24 + 1:02d}" for i in range(len(frame))]
    for feature in range(4):
        frame[f"Cells_AreaShape_f{feature}"] = rng.normal(0.0, 0.1, len(frame)) + frame["offset"]
    adata = from_dataframe(frame.drop(columns=["plate", "compound", "dose", "offset"]), resolution="well")

    mt.tl.dose_direction(adata)
    table = adata.uns["mantispy"]["dose_direction"].set_index("compound")
    assert table.loc["spread", "amplitude_null"].max() < table.loc["stacked", "amplitude_null"].min()


def test_a_trajectory_is_an_object_io_accepts(tmp_path, phenotypes):
    """Same defect as #103: var held only `feature` and `position`, so the stamped result failed validation on the nine annotation columns the schema requires."""
    mt.tl.dose_direction(phenotypes)
    paths = mt.tl.dose_trajectory(phenotypes, n_positions=3)

    report = mt.io.validate(paths)
    assert report.ok, str(report)
    # The other annotation columns describe a CellProfiler measurement a trajectory is not, so they stay empty.
    assert paths.var["feature"].nunique() == phenotypes.n_vars - 1
    for column in ("object", "feature_group", "channel", "scale", "angle", "gray_levels", "radial_bin", "params"):
        assert paths.var[column].isna().all(), column
    assert paths.var["is_feature"].all()

    written = tmp_path / "trajectory.h5ad"
    mt.io.write(paths, written)
    loaded = mt.io.read(written)
    assert list(loaded.var_names) == list(paths.var_names)
    assert list(loaded.var.columns) == list(paths.var.columns)
    # Both columns the Returns clause promises survive the writer, not just the schema's own.
    assert list(loaded.var["position"]) == list(paths.var["position"])
    assert list(loaded.var["feature"]) == list(paths.var["feature"])


def test_a_trajectory_needs_at_least_two_points(phenotypes):
    with pytest.raises(ValueError, match="at least two"):
        mt.tl.dose_trajectory(phenotypes, n_positions=1)


def test_a_position_is_named_the_same_whatever_the_grid_holds(phenotypes):
    """The suffix width was chosen from the grid, so the endpoints shared by every grid -- 0.0 and 1.0 -- were named @0.00 at 101 positions and @0.000 at 102, and no two runs of one screen lined up.
    The width no longer depends on how many positions were asked for."""
    narrow = mt.tl.dose_trajectory(phenotypes, n_positions=101)
    wide = mt.tl.dose_trajectory(phenotypes, n_positions=102)

    def endpoint_names(paths):
        at_zero = np.asarray(paths.var["position"]) == 0.0
        return {name.split("@", 1)[1] for name in paths.var_names[at_zero]}

    assert endpoint_names(narrow) == endpoint_names(wide)
