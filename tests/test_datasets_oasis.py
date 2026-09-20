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
    """Two plate maps write the dose to three decimals and the rest to four, so 3.704 and 3.7037 are one level."""
    obs = oasis.obs
    raw = obs["Metadata_Concentration"].to_numpy(dtype=float)
    aligned = obs["Metadata_ConcentrationRounded"].to_numpy(dtype=float)
    usable = np.isfinite(raw) & (raw > 0)

    levels = np.unique(aligned[usable])
    close = [(a, b) for a, b in zip(levels, levels[1:], strict=False) if b / a - 1 < 0.01]
    assert not close, "two levels within 1% of each other are one dose written twice"
    assert len(levels) < len(np.unique(raw[usable]))
    # The dose a well was meant to get is one it was recorded at, never an average of two.
    assert set(levels) <= set(np.unique(raw[usable]))
    assert np.max(np.abs(aligned[usable] / raw[usable] - 1)) < 0.01, "no well moves more than the tolerance"

    # Replicates of one treatment land in one group, which is what hit_calling and tl.map count.
    treated = obs[~obs["Metadata_Control"]]
    sizes = treated["Metadata_Perturbation"].astype(str).value_counts()
    assert int((sizes < 3).sum()) < len(sizes) // 4, "most treatments keep three or more wells"


@pytest.mark.network
def test_annotate_false_leaves_the_profiles_alone(oasis):
    raw = mt.ds.oasis_pilot(annotate=False)
    assert raw.shape == oasis.shape
    assert "Metadata_Compound" not in raw.obs
