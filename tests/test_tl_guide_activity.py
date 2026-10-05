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
    assert bool(table.loc[table["gene"] == "geneA", "is_hit"].iloc[0])


def test_an_absent_reference_is_an_error(adata):
    with pytest.raises(ValueError, match="reference='missing' is absent"):
        mt.tl.guide_activity(adata, reference="missing")


def test_all_constant_features_cannot_be_scored():
    x = np.ones((50, 8), dtype="float32")
    obs = pd.DataFrame({"Metadata_Gene": ["intergenic"] * 25 + ["geneA"] * 25})
    with pytest.raises(ValueError, match="every feature is constant"):
        mt.tl.guide_activity(AnnData(x, obs=obs), reference="intergenic")
