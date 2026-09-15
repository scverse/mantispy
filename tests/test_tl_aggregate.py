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
