import numpy as np
import pytest

import mantispy as mt
from mantispy._core.schema import validate


@pytest.fixture
def adata(cells):
    return cells


def test_shape_var_and_schema(adata):
    wells = mt.tl.aggregate(adata, min_cells=0)
    assert wells.n_obs == 48  # 2 plates x 24 wells
    assert list(wells.var.columns) == list(adata.var.columns)
    assert wells.X.dtype == np.float32
    assert validate(wells).ok, validate(wells).errors


@pytest.mark.parametrize("func", ["median", "mean"])
def test_values_match_a_manual_reduction(adata, func):
    wells = mt.tl.aggregate(adata, func=func, min_cells=0)
    reduce = np.nanmedian if func == "median" else np.nanmean
    rows = ((adata.obs["Metadata_Plate"] == "Plate01") & (adata.obs["Metadata_Well"] == "A01")).to_numpy()
    target = ((wells.obs["Metadata_Plate"] == "Plate01") & (wells.obs["Metadata_Well"] == "A01")).to_numpy()
    np.testing.assert_allclose(wells.X[target][0], reduce(adata.X[rows], axis=0), rtol=1e-5)


def test_cell_count_and_min_cells(adata):
    wells = mt.tl.aggregate(adata, min_cells=0)
    assert (wells.obs["Metadata_CellCount"] == 15).all()
    assert mt.tl.aggregate(adata, min_cells=16).n_obs == 0


def test_aggregating_profiles_counts_their_cells_not_their_rows(adata):
    """Regression test for #63: a plate of 24 wells of 15 cells holds 360 cells, not 24."""
    wells = mt.tl.aggregate(adata, min_cells=0)
    plates = mt.tl.aggregate(wells, by=("Metadata_Plate",), min_cells=0)
    assert (plates.obs["Metadata_CellCount"] == 360).all()
    # min_cells is still a number of cells.
    assert mt.tl.aggregate(wells, by=("Metadata_Plate",), min_cells=361).n_obs == 0


def test_the_count_columns_can_be_named(adata):
    adata.obs["Metadata_Site"] = np.tile(["1", "2"], adata.n_obs // 2)
    wells = mt.tl.aggregate(adata, min_cells=0, count_key="n_cells", site_key="n_fields")
    assert (wells.obs["n_cells"] == 15).all()
    assert (wells.obs["n_fields"] == 2).all()
    assert "Metadata_CellCount" not in wells.obs

    plates = mt.tl.aggregate(wells, by=("Metadata_Plate",), min_cells=0, count_key="n_cells", site_key="n_fields")
    assert (plates.obs["n_cells"] == 360).all()
    assert (plates.obs["n_fields"] == 48).all()


def test_an_unknown_count_keeps_its_group(adata):
    wells = mt.tl.aggregate(adata, min_cells=0)
    wells.obs["Metadata_CellCount"] = np.where(np.arange(wells.n_obs) == 0, np.nan, 15.0)
    plates = mt.tl.aggregate(wells, by=("Metadata_Plate",), min_cells=10)
    assert plates.n_obs == 2
    assert np.isnan(plates.obs["Metadata_CellCount"]).sum() == 1


def test_the_site_count_follows_the_grouping(adata):
    """A per-site row counts one field of view, a per-well row every field that held a cell."""
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


def test_constant_metadata_carried_and_varying_dropped(adata):
    adata.obs["Metadata_Varies"] = np.arange(adata.n_obs)
    wells = mt.tl.aggregate(adata, min_cells=0)
    assert "Metadata_Perturbation" in wells.obs
    assert "Metadata_Control" in wells.obs
    assert "Metadata_Varies" not in wells.obs


def test_resolution_and_provenance(adata):
    wells = mt.tl.aggregate(adata, min_cells=0)
    assert wells.uns["mantispy"]["resolution"] == "well"
    assert wells.uns["mantispy"]["aggregated_from"]["n_obs"] == adata.n_obs

    perturbations = mt.tl.aggregate(adata, by=("Metadata_Perturbation",), min_cells=0)
    assert perturbations.uns["mantispy"]["resolution"] == "perturbation"
    assert perturbations.n_obs == adata.obs["Metadata_Perturbation"].nunique()


def test_a_grouping_finer_than_the_well_stays_at_well_resolution(adata):
    """A site is a subdivision of a well, so a per-site profile is still per-well or finer.
    Stamping it "perturbation" would stop validate requiring the plate and well columns it holds."""
    adata.obs["Metadata_Site"] = np.tile(["1", "2"], adata.n_obs // 2)
    sites = mt.tl.aggregate(adata, by=("Metadata_Plate", "Metadata_Well", "Metadata_Site"), min_cells=0)

    assert sites.n_obs == 96  # 2 plates x 24 wells x 2 sites
    assert sites.uns["mantispy"]["resolution"] == "well"

    stripped = sites.copy()
    del stripped.obs["Metadata_Plate"]
    assert not validate(stripped).ok


def test_channels_are_inherited_but_cell_level_provenance_is_not(adata):
    """Carrying the input's image_table or params forward would misdescribe the result."""
    wells = mt.tl.aggregate(adata, min_cells=0)
    assert wells.uns["mantispy"]["channels"] == adata.uns["mantispy"]["channels"]
    assert "truth" in wells.uns["mantispy"]
    assert "image_table" not in wells.uns["mantispy"]


def test_nan_is_skipped_not_propagated(adata):
    values = adata.X.copy()
    values[0, 0] = np.nan
    adata.X = values
    wells = mt.tl.aggregate(adata, min_cells=0)
    assert np.isfinite(wells.X).all()


def test_metadata_stays_categorical(adata):
    """Categorical metadata keeps memory low at scale and avoids anndata's warnings."""
    import pandas as pd

    wells = mt.tl.aggregate(adata, min_cells=0)
    for column in ("Metadata_Plate", "Metadata_Well", "Metadata_Perturbation"):
        assert isinstance(wells.obs[column].dtype, pd.CategoricalDtype), column
