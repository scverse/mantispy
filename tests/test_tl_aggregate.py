import numpy as np
import pytest

import mantispy as mt


@pytest.fixture
def adata(cells):
    return cells


def test_aggregating_profiles_counts_their_cells_not_their_rows(adata):
    """Regression test for #63: a plate of 24 wells of 15 cells holds 360 cells, not 24."""
    wells = mt.tl.aggregate(adata, min_cells=0)
    plates = mt.tl.aggregate(wells, by=("Metadata_Plate",), min_cells=0)
    assert (plates.obs["Metadata_CellCount"] == 360).all()
    # min_cells is still a number of cells.
    assert mt.tl.aggregate(wells, by=("Metadata_Plate",), min_cells=361).n_obs == 0


def test_the_site_count_follows_the_grouping(adata):
    """A per-site row counts one field of view, a per-well row every field that contributed cells."""
    assert "Metadata_SiteCount" not in mt.tl.aggregate(adata, min_cells=0).obs
    adata.obs["Metadata_Site"] = np.tile(["1", "2"], adata.n_obs // 2)
    sites = mt.tl.aggregate(adata, by=("Metadata_Plate", "Metadata_Well", "Metadata_Site"), min_cells=0)
    wells = mt.tl.aggregate(adata, min_cells=0)
    assert (sites.obs["Metadata_SiteCount"] == 1).all()
    assert (wells.obs["Metadata_SiteCount"] == 2).all()
    assert sites.obs["Metadata_CellCount"].sum() == wells.obs["Metadata_CellCount"].sum() == adata.n_obs

    plates = mt.tl.aggregate(sites, by=("Metadata_Plate",), min_cells=0)
    assert (plates.obs["Metadata_SiteCount"] == 48).all()
    # A consensus counts replicates, and a constant site count is not carried onto it.
    consensus = mt.tl.consensus(wells, method="median")
    assert "Metadata_SiteCount" not in consensus.obs
    # Nor is a replicate count carried onto a coarser grouping of consensus profiles.
    assert "Metadata_ReplicateCount" not in mt.tl.aggregate(consensus, by=("Metadata_Control",), min_cells=0).obs


def test_a_constant_column_survives_whatever_its_name(adata):
    """§13.1: a column constant within every group is preserved; the Metadata_ prefix is not the filter."""
    adata.obs["batch_label"] = "runA"  # constant everywhere, no Metadata_ prefix
    adata.obs["cell_quality"] = np.arange(adata.n_obs)  # varies within every well
    wells = mt.tl.aggregate(adata, min_cells=0)
    assert "batch_label" in wells.obs and (wells.obs["batch_label"] == "runA").all()
    assert "cell_quality" not in wells.obs  # varying columns are dropped, not silently copied


def test_aggregate_warns_when_min_cells_wipes_a_plates_controls():
    """min_cells dropping every negcon well of a plate leaves later reference= empty (audit A3)."""
    import anndata as ad
    import pandas as pd

    rng = np.random.default_rng(0)
    # P1's negcon wells hold 2 cells each; everything else holds 10, so min_cells=5 drops only P1's negcon.
    spec = [
        ("P1", "A01", "negcon", 2),
        ("P1", "A02", "negcon", 2),
        ("P1", "B01", "treatment", 10),
        ("P2", "A01", "negcon", 10),
        ("P2", "B01", "treatment", 10),
    ]
    rows = [(plate, well, control) for plate, well, control, n in spec for _ in range(n)]
    obs = pd.DataFrame(rows, columns=["Metadata_Plate", "Metadata_Well", "Metadata_Control_Type"])
    obs["Metadata_Control"] = obs["Metadata_Control_Type"] != "treatment"
    obs.index = obs.index.astype(str)
    adata = ad.AnnData(X=rng.normal(size=(len(obs), 5)).astype(np.float32), obs=obs)
    adata.var_names = [f"Cells_AreaShape_F{index}" for index in range(5)]
    mt.io.stamp(adata, resolution="object", feature_kind="measurement")

    with pytest.warns(UserWarning, match="negcon on plate P1"):
        mt.tl.aggregate(adata, min_cells=5)


def _grouped_elsewhere(adata):
    """Well profiles whose perturbation identity lives in a non-default column with fewer groups."""
    wells = mt.tl.aggregate(adata, min_cells=0)
    wells.obs["Alt_Perturbation"] = np.where(np.arange(wells.n_obs) % 2 == 0, "even", "odd")
    return wells


def test_consensus_defaults_to_the_recorded_perturbation_column(adata):
    wells = _grouped_elsewhere(adata)
    mt.io.stamp(wells, perturbation={"column": "Alt_Perturbation"})
    assert set(mt.tl.consensus(wells, method="median").obs["Alt_Perturbation"]) == {"even", "odd"}
    explicit = mt.tl.consensus(wells, by="Metadata_Perturbation", method="median")
    assert "Alt_Perturbation" not in explicit.obs
    assert explicit.n_obs != 2


def test_consensus_without_a_record_keeps_the_default_column(adata):
    wells = mt.tl.aggregate(adata, min_cells=0)
    wells.uns["mantispy"].pop("perturbation", None)
    out = mt.tl.consensus(wells, method="median")
    assert set(out.obs["Metadata_Perturbation"]) == set(wells.obs["Metadata_Perturbation"])


def test_aggregate_writes_no_aggregated_from(adata):
    result = mt.tl.aggregate(adata, by=("Metadata_Plate", "Metadata_Well"))
    assert "aggregated_from" not in result.uns["mantispy"]
