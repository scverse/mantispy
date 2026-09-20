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
def test_annotate_false_leaves_the_profiles_alone(oasis):
    raw = mt.ds.oasis_pilot(annotate=False)
    assert raw.shape == oasis.shape
    assert "Metadata_Compound" not in raw.obs
