"""OASIS pilot, the dose-response dataset. Network-dependent, so marked."""

import numpy as np
import pytest

import mantispy as mt


@pytest.fixture(scope="module")
def oasis():
    return mt.ds.oasis_pilot()


@pytest.mark.network
def test_every_well_gets_a_plate_map_row(oasis):
    """The two batches that name compounds in 'Compound Name' leave it blank for DMSO and EMPTY wells.

    Reading that column alone left 576 wells, every one of them a control or an empty, looking
    unannotated, and the controls with them.
    """
    assert oasis.shape == (4604, 99)
    assert mt.io.validate(oasis).ok, mt.io.validate(oasis).errors
    assert int(oasis.obs["Metadata_Compound"].isna().sum()) == 0
    assert int(oasis.obs["Metadata_Control"].sum()) == 511


@pytest.mark.network
def test_the_dose_range_survives_the_join(oasis):
    """The point of this dataset: compounds carried over enough concentrations to fit a curve."""
    obs = oasis.obs
    treated = obs[~obs["Metadata_Control"] & (obs["Metadata_Compound"].astype(str) != "EMPTY")]
    doses = treated.groupby(["Metadata_CellLine", "Metadata_Compound"], observed=True)[
        "Metadata_Concentration"
    ].nunique()
    assert set(obs["Metadata_CellLine"].dropna().unique()) == {"U2OS", "HepaRG"}, "one spelling per line"
    for line in ("U2OS", "HepaRG"):
        assert int((doses[line] >= 6).sum()) == 28, f"{line} should dose 28 compounds six times or more"
    assert np.isfinite(treated["Metadata_Concentration"].to_numpy(dtype=float)).all()


@pytest.mark.network
def test_one_concentration_written_twice_is_one_dose(oasis):
    """The plate maps disagree on precision: one writes 0.0152416 uM and another writes 0.015."""
    obs = oasis.obs
    raw = obs["Metadata_ConcentrationRecorded"].to_numpy(dtype=float)
    aligned = obs["Metadata_Concentration"].to_numpy(dtype=float)
    dosed = raw > 0

    assert len(np.unique(aligned[dosed])) < len(np.unique(raw[dosed])), "nothing collapsed"
    # The dose a well was meant to get is one that was recorded, never an average of two spellings.
    assert set(np.unique(aligned[dosed])) <= set(np.unique(raw[dosed]))
    # A coarser spelling reads as the finer level it rounds to, so no well moves to a different rung.
    moved = aligned[dosed] != raw[dosed]
    assert (np.abs(aligned[dosed][moved] / raw[dosed][moved] - 1) < 0.5).all()


@pytest.mark.network
def test_every_dosed_compound_lands_on_the_ladder_it_was_plated_on(oasis):
    """Ten three-fold steps, or the eight two-fold ones the assay-development plates used."""
    obs = oasis.obs
    treated = obs[~obs["Metadata_Control"] & (obs["Metadata_Compound"].astype(str) != "EMPTY")]
    treated = treated[treated["Metadata_ConcentrationRecorded"].to_numpy(dtype=float) > 0]

    raw = treated.groupby("Metadata_Compound", observed=True)["Metadata_ConcentrationRecorded"].nunique()
    aligned = treated.groupby("Metadata_Compound", observed=True)["Metadata_Concentration"].nunique()
    assert set(raw[raw > 1]) - {8} != set(), "the raw column should carry the split ladders"
    assert set(aligned) <= {1, 8, 10}, f"off-ladder compounds: {aligned[~aligned.isin([1, 8, 10])].to_dict()}"
    assert int((aligned == 10).sum()) == 28


@pytest.mark.network
def test_replicates_of_one_treatment_land_in_one_group(oasis):
    """Metadata_Perturbation is named by the aligned dose, so the two spellings do not halve a group."""
    treated = oasis.obs[~oasis.obs["Metadata_Control"]]
    sizes = treated.groupby(["Metadata_CellLine", "Metadata_Perturbation"], observed=True).size()
    # HepaRG is eight plates, so its treatments carry eight wells each; U2OS is two and genuinely thin.
    heparg = sizes.loc["HepaRG"]
    assert heparg.median() >= 6, "the eight-plate line should keep its replicates together"
    assert int((heparg < 3).sum()) < len(heparg) // 5


@pytest.mark.network
def test_annotate_false_leaves_the_profiles_alone(oasis):
    raw = mt.ds.oasis_pilot(annotate=False)
    assert raw.shape == oasis.shape
    assert "Metadata_Compound" not in raw.obs
