"""Per-guide activity scored against a reference control class.

Fabricated profiles: reference guides at the origin, active guides shifted away, so the one-sided
p is small for the active ones and the reference guides stay unscored.
"""

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

import mantispy as mt


@pytest.fixture
def adata():
    rng = np.random.default_rng(0)
    ref = rng.normal(0, 1, (200, 20))
    active = rng.normal(0, 1, (30, 20)) + 5.0  # a clear shift from the reference centre
    quiet = rng.normal(0, 1, (30, 20))  # like the reference, no shift
    x = np.vstack([ref, active, quiet]).astype("float32")
    genes = ["intergenic"] * 200 + ["geneA"] * 30 + ["geneB"] * 30
    obs = pd.DataFrame({"Metadata_Gene": genes}, index=[f"g{i}" for i in range(260)])
    return AnnData(x, obs=obs)


def test_reference_guides_are_unscored_and_active_guides_score_low(adata):
    mt.tl.guide_activity(adata, reference="intergenic")
    p = adata.obs["guide_activity"]
    assert p[adata.obs["Metadata_Gene"] == "intergenic"].isna().all()  # reference sets the scale, not scored
    active = p[adata.obs["Metadata_Gene"] == "geneA"]
    quiet = p[adata.obs["Metadata_Gene"] == "geneB"]
    assert active.max() < 0.05  # a 5-sigma shift is unambiguously active
    assert quiet.median() > 0.2  # guides like the reference are not


def test_a_one_sided_p_in_the_unit_interval_feeds_aggregate_guides(adata):
    adata.obs.loc[adata.obs["Metadata_Gene"] == "geneB", "Metadata_Gene"] = "nontargeting"
    mt.tl.guide_activity(adata, reference="intergenic")
    scored = adata.obs["guide_activity"].dropna()
    assert scored.between(0.0, 1.0).all()
    # the activity p is a valid score for aggregate_guides against the other control class
    adata.obs["Metadata_sgRNA"] = adata.obs_names
    mt.tl.aggregate_guides(
        adata, score="guide_activity", guide="Metadata_sgRNA", gene="Metadata_Gene", control="nontargeting"
    )
    table = adata.uns["mantispy"]["gene_aggregation"]
    assert bool(table.loc[table["group"] == "geneA", "is_hit"].iloc[0])


def test_an_absent_reference_is_an_error(adata):
    with pytest.raises(ValueError, match="reference='missing' is absent"):
        mt.tl.guide_activity(adata, reference="missing")


def test_all_constant_features_cannot_be_scored():
    x = np.ones((50, 8), dtype="float32")
    obs = pd.DataFrame({"Metadata_Gene": ["intergenic"] * 25 + ["geneA"] * 25})
    with pytest.raises(ValueError, match="every feature is constant"):
        mt.tl.guide_activity(AnnData(x, obs=obs), reference="intergenic")


def test_many_guides_at_the_pvalue_floor_warn():
    """Few reference guides make a coarse floor that most guides pile onto, so hit counts go degenerate (audit B3)."""
    rng = np.random.default_rng(0)
    ref = rng.normal(0, 1, (4, 10))  # only four reference guides, so the floor is 1/5
    active = rng.normal(0, 1, (40, 10)) + 8.0  # far beyond any reference guide, so all hit the floor
    x = np.vstack([ref, active]).astype("float32")
    obs = pd.DataFrame({"Metadata_Gene": ["intergenic"] * 4 + ["geneA"] * 40})
    with pytest.warns(UserWarning, match="p-value floor"):
        mt.tl.guide_activity(AnnData(x, obs=obs), reference="intergenic")


def test_a_quarter_of_guides_at_the_floor_warns():
    """The warning must fire in the Altair case, where about half the guides sit on the floor, not only above it (audit item 4)."""
    rng = np.random.default_rng(0)
    ref = rng.normal(0, 1, (10, 10))  # floor is 1/11
    extreme = rng.normal(0, 1, (10, 10)) + 8.0  # beyond every reference guide -> on the floor
    quiet = rng.normal(0, 1, (20, 10))  # like the reference -> not on the floor
    x = np.vstack([ref, extreme, quiet]).astype("float32")  # 10 of 30 scored guides (a third) on the floor
    obs = pd.DataFrame({"Metadata_Gene": ["intergenic"] * 10 + ["geneA"] * 30})
    with pytest.warns(UserWarning, match="p-value floor"):
        mt.tl.guide_activity(AnnData(x, obs=obs), reference="intergenic")
