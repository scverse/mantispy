"""BBBC021, the classic MOA benchmark. Network-dependent, so marked."""

import numpy as np
import pytest

import mantispy as mt
from mantispy._core._utils import as_frame


@pytest.fixture(scope="module")
def bbbc021():
    return mt.ds.bbbc021()


@pytest.mark.network
def test_loads_annotated_and_valid(bbbc021):
    assert bbbc021.shape == (632, 473)
    assert mt.io.validate(bbbc021).ok, mt.io.validate(bbbc021).errors
    assert set(bbbc021.obs.columns) == {
        "Metadata_Plate",
        "Metadata_Well",
        "Metadata_Compound",
        "Metadata_Concentration",
        "Metadata_MOA",
        "Metadata_Control",
        "Metadata_Perturbation",
    }
    # The mode= shorthands of mt.tl.map read this column, so it must exist here too.
    assert bbbc021.obs["Metadata_Perturbation"].nunique() == 104
    assert bbbc021.obs["Metadata_Compound"].nunique() == 39
    assert int(bbbc021.obs["Metadata_Control"].sum()) == 330
    assert bbbc021.var["is_feature"].all()


def _treatment_consensus(bbbc021, sphere: bool):
    """Normalize, select features, optionally sphere, then one profile per treatment."""
    adata = bbbc021.copy()
    mt.pp.normalize(adata, method="mad_robustize", by="Metadata_Plate", reference="negcon")
    mt.pp.feature_select(adata)
    adata = mt.pp.subset_features(adata)
    if sphere:
        mt.pp.sphere(adata, method="ZCA-cor", reference="negcon")

    treated = adata[~adata.obs["Metadata_Control"].to_numpy()].copy()
    obs = as_frame(treated.obs)
    obs["Metadata_Treatment"] = obs["Metadata_Compound"].astype(str) + "@" + obs["Metadata_Concentration"].astype(str)
    return mt.tl.aggregate(treated, by=("Metadata_Treatment",), min_cells=0)


def _not_same_compound_accuracy(consensus) -> float:
    """Ljosa 2013's NSC rule: the nearest neighbour of a different compound must share the MOA."""
    mt.tl.similarity(consensus, metric="cosine")
    similarity = np.asarray(consensus.obsp["similarity"]).copy()
    np.fill_diagonal(similarity, -np.inf)
    compound = consensus.obs["Metadata_Compound"].astype(str).to_numpy()
    moa = consensus.obs["Metadata_MOA"].astype(str).to_numpy()
    similarity[compound[:, None] == compound[None, :]] = -np.inf
    return float((moa[similarity.argmax(axis=1)] == moa).mean())


@pytest.mark.network
@pytest.mark.slow
def test_the_recipe_reproduces_the_moa_benchmark(bbbc021):
    """The published benchmark's shape, and MOA retrieval far above chance."""
    consensus = _treatment_consensus(bbbc021, sphere=False)
    assert consensus.n_obs == 103  # the 103 treatments of Ljosa et al. 2013
    assert consensus.obs["Metadata_MOA"].nunique() == 12

    accuracy = _not_same_compound_accuracy(consensus)
    assert accuracy > 0.55, f"not-same-compound MOA accuracy {accuracy:.1%}, chance is ~8%"

    mt.tl.map(
        consensus,
        pos_sameby=["Metadata_MOA"],
        pos_diffby=["Metadata_Compound"],
        neg_diffby=["Metadata_MOA"],
        null_size=1000,
    )
    table = consensus.uns["mantispy"]["map"]
    assert table["below_corrected_p"].sum() >= 4
    best = table.nlargest(2, "mean_average_precision")["Metadata_MOA"].tolist()
    assert "Kinase inhibitors" in best


@pytest.mark.network
@pytest.mark.slow
def test_sphering_hurts_this_dataset(bbbc021):
    """With 330 DMSO wells against 346 features, sphering is underdetermined and amplifies
    noise. pycytominer gives the same result, so the loss comes from the method itself.
    This is why evaluate_correction compares corrections instead of applying a fixed
    recipe."""
    without = _not_same_compound_accuracy(_treatment_consensus(bbbc021, sphere=False))
    with pytest.warns(UserWarning, match="fewer rows than features"):
        sphered = _treatment_consensus(bbbc021, sphere=True)
    assert _not_same_compound_accuracy(sphered) < without
