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
