"""OASIS pilot, the dose-response dataset. Network-dependent, so marked."""

import numpy as np
import pytest

import mantispy as mt
from mantispy.ds._datasets import _DOSE_TOLERANCE


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
    """Two plate maps write the dose to three decimals and the rest to four, so 3.704 and 3.7037 are one level."""
    obs = oasis.obs
    raw = obs["Metadata_Concentration"].to_numpy(dtype=float)
    aligned = obs["Metadata_ConcentrationNominal"].to_numpy(dtype=float)
    dosed = raw > 0

    levels = np.unique(aligned[dosed])
    gaps = np.diff(levels) / levels[:-1]
    assert (gaps >= _DOSE_TOLERANCE).all(), "two levels within the tolerance are one dose written twice"
    assert len(levels) < len(np.unique(raw[dosed])), "nothing collapsed"
    # The dose a well was meant to get is one it was recorded at, never an average of two.
    assert set(levels) <= set(np.unique(raw[dosed]))
    assert np.max(np.abs(aligned[dosed] / raw[dosed] - 1)) < _DOSE_TOLERANCE, "no well moves past the tolerance"


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
