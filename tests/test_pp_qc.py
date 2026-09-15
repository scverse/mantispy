import numpy as np
import pandas as pd
import pytest

import mantispy as mt
from mantispy._core.features import load_blocklist
from mantispy.ds import synthetic_plate
from mantispy.io._profiles import from_dataframe


@pytest.fixture
def adata():
    return synthetic_plate(n_wells=8, n_cells=20, n_features=15, nan_fraction=0.02, seed=0)


def test_metrics_are_written_and_match_numpy(adata):
    mt.pp.calculate_qc_metrics(adata)
    missing = np.isnan(adata.X)
    np.testing.assert_array_equal(adata.obs["qc_n_nan_features"].to_numpy(), missing.sum(axis=1))
    np.testing.assert_allclose(adata.obs["qc_nan_fraction"].to_numpy(), missing.mean(axis=1), rtol=1e-6)
    np.testing.assert_allclose(adata.var["qc_variance"].to_numpy(), np.nanvar(adata.X, axis=0), rtol=1e-5)
    np.testing.assert_array_equal(adata.var["qc_n_nan"].to_numpy(), missing.sum(axis=0))


def test_partial_nan_does_not_fail_every_cell(adata):
    """Requiring zero missing features would fail ~26% of cells here and ~100% on a real
    2000-feature table, so filter_cells would empty the dataset."""
    mt.pp.calculate_qc_metrics(adata)
    assert (adata.obs["qc_n_nan_features"] > 0).mean() > 0.15
    assert adata.obs["qc_pass"].mean() > 0.95


def test_max_nan_fraction_is_honoured(adata):
    mt.pp.calculate_qc_metrics(adata, max_nan_fraction=0.0)
    assert adata.obs["qc_pass"].mean() < 0.9


def test_zero_area_mad_does_not_flag_the_whole_plate(adata):
    """A quantised Area column gives MAD 0 and z = +inf; without posinf handling
    nan_to_num leaves 1.8e308 and every cell is called an outlier."""
    area = next(name for name in adata.var_names if name.endswith("AreaShape_Area"))
    values = adata.X.copy()
    values[:, adata.var_names.get_loc(area)] = 1.0
    values[0, adata.var_names.get_loc(area)] = 99.0
    adata.X = values
    mt.pp.calculate_qc_metrics(adata)
    assert not adata.obs["qc_area_outlier"].all()


def test_border_flag_needs_coordinates_and_shape(adata):
    mt.pp.calculate_qc_metrics(adata)
    assert not adata.obs["qc_is_border"].any()

    adata.obs["Metadata_Center_X"] = np.linspace(0, 1000, adata.n_obs)
    adata.obs["Metadata_Center_Y"] = 500.0
    mt.pp.calculate_qc_metrics(adata, image_shape=(1024, 1024), border_margin=50)
    assert adata.obs["qc_is_border"].iloc[0]
    assert not adata.obs["qc_is_border"].iloc[adata.n_obs // 2]


def test_filter_cells_drops_failures_and_small_wells(adata):
    mt.pp.calculate_qc_metrics(adata)
    passing = int(adata.obs["qc_pass"].sum())
    filtered = mt.pp.filter_cells(adata, min_cells_per_well=0, copy=True)
    assert filtered.n_obs == passing

    assert mt.pp.filter_cells(adata, min_cells_per_well=21, qc_pass=False, copy=True).n_obs == 0


def test_filter_cells_without_metrics_says_what_to_run(adata):
    with pytest.raises(KeyError, match="calculate_qc_metrics"):
        mt.pp.filter_cells(adata)


def test_filter_features_drops_all_nan_constant_and_blocklisted(adata):
    values = adata.X.copy()
    values[:, 0] = np.nan  # all missing
    values[:, 1] = 1.0  # zero variance
    adata.X = values
    all_nan, constant, blocked = adata.var_names[0], adata.var_names[1], adata.var_names[3]

    mt.pp.filter_features(adata, min_variance=1e-8, blocklist=[blocked])

    assert set(adata.var_names).isdisjoint({all_nan, constant, blocked})
    assert adata.n_vars == 12


def test_inplace_and_copy_semantics(adata):
    mt.pp.calculate_qc_metrics(adata)
    before = adata.n_obs
    assert mt.pp.filter_cells(adata, min_cells_per_well=0, copy=True) is not adata
    assert adata.n_obs == before
    assert mt.pp.filter_cells(adata, min_cells_per_well=0) is None
    assert adata.n_obs <= before


def test_params_are_recorded(adata):
    mt.pp.calculate_qc_metrics(adata, border_margin=7)
    assert adata.uns["mantispy"]["params"]["calculate_qc_metrics"]["border_margin"] == 7


#: The parser infers its channel vocabulary from the column set it is given, so a handful
#: of names parse differently from a full screen's worth. Naming the channels explicitly
#: keeps this test independent of fixture size; without it the rename does nothing on six
#: columns and the test passes trivially.
CHANNELS = ("AGP", "DNA", "ER", "Mito", "RNA")


def test_the_blocklist_still_matches_after_standardizing_the_names():
    """standardize_feature_names rewrites channel-bearing names into a form no blocklist
    entry matches (Nuclei_Correlation_Manders_AGP_DNA becomes ..._AGP|DNA), and all 55
    bundled entries are renamed this way. The blocklist must still drop them.
    """
    blocked = sorted(load_blocklist("default"))[:4]
    names = [*blocked, "Cells_AreaShape_Area", "Nuclei_AreaShape_Area"]
    frame = pd.DataFrame(
        {
            "Metadata_Plate": ["P1", "P1"],
            "Metadata_Well": ["A01", "A02"],
            **{name: [1.0, 2.0] for name in names},
        }
    )
    adata = from_dataframe(frame, channels=CHANNELS)

    before = adata.copy()
    mt.pp.filter_features(before, blocklist="default", drop_nan=False)
    assert before.n_vars == adata.n_vars - len(blocked)

    renamed = mt.pp.standardize_feature_names(adata, copy=True)
    assert list(renamed.var_names) != list(adata.var_names), "nothing was renamed; the test proves nothing"

    after = renamed.copy()
    mt.pp.filter_features(after, blocklist="default", drop_nan=False)
    assert after.n_vars == renamed.n_vars - len(blocked)

    # pp.feature_select applies the blocklist through a separate code path, and
    # "blocklist" is in DEFAULT_OPERATIONS.
    selected = renamed.copy()
    mt.pp.feature_select(selected, operations=("blocklist",))
    assert int(selected.var["selected"].sum()) == renamed.n_vars - len(blocked)
