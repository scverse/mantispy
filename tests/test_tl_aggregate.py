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


def test_store_membership_records_source_identity_and_round_trips(adata, tmp_path):
    """§13.4: opt-in membership references stable identity (ImageID + ObjectNumber), not row positions."""
    wells = mt.tl.aggregate(adata, min_cells=0, store_membership=True)
    membership = wells.uns["mantispy"]["membership"]
    assert membership["source_resolution"] == "object"
    assert "Metadata_ImageID" in membership["identity_columns"]
    assert "Metadata_ObjectNumber" in membership["identity_columns"]
    members = membership["members"]
    assert len(members) == adata.n_obs  # every source cell is accounted for
    assert set(members["Metadata_AggregateRow"]) == set(wells.obs_names)
    # default is off
    assert "membership" not in mt.tl.aggregate(adata, min_cells=0).uns["mantispy"]
    # survives an h5ad round-trip
    path = tmp_path / "wells.h5ad"
    wells.write_h5ad(path)
    import anndata as ad

    back = ad.read_h5ad(path).uns["mantispy"]["membership"]
    assert len(back["members"]) == adata.n_obs


def test_aggregate_inherits_and_extends_history(adata):
    """§14.3: an aggregate carries the source history and appends its own level-changing record."""
    from mantispy._core.provenance import read_history

    mt.pp.normalize(adata)  # one in-place op → one history record on the source
    wells = mt.tl.aggregate(adata, min_cells=0)
    history = read_history(wells)
    operations = [record["operation"] for record in history]
    assert "normalize" in operations  # inherited from the source
    assert operations[-1] == "aggregate"
    assert history[-1]["source"]["to_resolution"] == "well"
    # the source object keeps its own, shorter history
    assert "aggregate" not in [record["operation"] for record in read_history(adata)]
