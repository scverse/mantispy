"""Rohban 2017 and the PKI dose series. Network-dependent, so marked.

These keep the package from being tuned to BBBC021 alone. The assertions cover the two
things BBBC021 lacks, cell counts and real doses.
"""

import numpy as np
import pytest

import mantispy as mt


@pytest.fixture(scope="module")
def rohban():
    return mt.ds.rohban()


@pytest.fixture(scope="module")
def pki():
    return mt.ds.pki()


@pytest.mark.network
def test_rohban_loads_with_genes_controls_and_counts(rohban):
    assert rohban.shape == (1918, 3634)
    assert mt.io.validate(rohban).ok, mt.io.validate(rohban).errors
    assert rohban.obs["Metadata_Perturbation"].nunique() == 194
    # The control ORFs only: the untreated EMPTY wells were never transfected.
    assert int(rohban.obs["Metadata_Control"].sum()) == 120
    assert set(rohban.obs.loc[rohban.obs["Metadata_Control"], "Metadata_gene_name"]) == {
        "Luciferase",
        "LacZ",
        "eGFP",
    }
    assert rohban.obs["Metadata_CellCount"].between(1, 1e5).all()


@pytest.mark.network
def test_pki_loads_with_a_dose_series(pki):
    assert pki.shape == (3072, 5857)
    assert mt.io.validate(pki).ok, mt.io.validate(pki).errors
    treated = pki.obs[~pki.obs["Metadata_Control"].to_numpy()]
    assert treated["Metadata_Compound"].nunique() == 15
    assert sorted(treated["Metadata_Concentration"].unique()) == [0.004, 0.01, 0.04, 0.2, 0.4, 1.0, 2.0]
    # Every DMSO well is one perturbation group, not one group per empty platemap row.
    assert (pki.obs.loc[pki.obs["Metadata_Control"], "Metadata_Perturbation"] == "DMSO").all()
    assert pki.obs["Metadata_CellCount"].between(1, 1e5).all()


@pytest.mark.network
def test_a_single_plate_can_be_loaded(rohban):
    one = mt.ds.rohban(plates=["41744"])
    assert one.obs["Metadata_Plate"].nunique() == 1
    assert one.n_obs < rohban.n_obs
    with pytest.raises(KeyError, match="no plate"):
        mt.ds.rohban(plates=["nope"])


@pytest.mark.network
@pytest.mark.slow
def test_higher_doses_move_further_from_the_controls(pki):
    """Profile magnitude grows with dose along PKI's real dose series."""
    adata = pki.copy()
    mt.pp.normalize(adata, method="mad_robustize", by="Metadata_Plate", reference="negcon")
    mt.pp.feature_select(adata, na_cutoff=0.0)
    adata = mt.pp.subset_features(adata)

    treated = adata[~adata.obs["Metadata_Control"].to_numpy()].copy()
    X = np.asarray(treated.X, dtype=float)
    magnitude = np.linalg.norm(np.nan_to_num(X), axis=1)
    dose = treated.obs["Metadata_Concentration"].to_numpy(dtype=float)

    low, high = magnitude[dose <= 0.04], magnitude[dose >= 0.4]
    assert high.mean() > low.mean(), f"{high.mean():.1f} at high dose vs {low.mean():.1f} at low"
