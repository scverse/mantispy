import logging

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
    passing = int((~adata.obs["qc_fail"]).sum())
    filtered = mt.pp.filter_cells(adata, min_cells_per_well=0, copy=True)
    assert filtered.n_obs == passing

    assert mt.pp.filter_cells(adata, min_cells_per_well=21, require_qc=False, copy=True).n_obs == 0


def test_filter_cells_does_not_copy_when_nothing_is_dropped(adata):
    """With no failing cell and no well floor the subset would copy the whole matrix only to return the same rows, so filter_cells leaves the object, and its matrix, untouched."""
    mt.pp.calculate_qc_metrics(adata)
    adata.obs["qc_fail"] = False
    x_before, n_before = adata.X, adata.n_obs
    assert mt.pp.filter_cells(adata, min_cells_per_well=0, require_qc=True) is None
    assert adata.n_obs == n_before
    assert adata.X is x_before


def test_filter_cells_without_metrics_says_what_to_run(adata):
    with pytest.raises(KeyError, match="calculate_qc_metrics"):
        mt.pp.filter_cells(adata)


def test_qc_metrics_are_opt_in(adata):
    """qc_variance and qc_n_unique are a per-feature pass filter_cells never reads, so the default qc_metrics leave them out."""
    mt.pp.calculate_qc_metrics(adata)
    assert "qc_n_nan" in adata.var
    assert "qc_variance" not in adata.var
    assert "qc_n_unique" not in adata.var

    mt.pp.calculate_qc_metrics(adata, qc_metrics=("variance", "n_unique"))
    assert "qc_variance" in adata.var
    assert "qc_n_unique" in adata.var


def test_feature_metrics_match_a_direct_computation(adata):
    """Opt-in values equal computing them straight from the matrix."""
    X = adata.X.astype(np.float64)
    expected_variance = np.nanvar(X, axis=0)
    expected_unique = np.array([len(np.unique(column[~np.isnan(column)])) for column in X.T], dtype=np.int32)

    mt.pp.calculate_qc_metrics(adata, qc_metrics=("variance", "n_unique"))
    np.testing.assert_allclose(adata.var["qc_variance"].to_numpy(), expected_variance, rtol=1e-5)
    np.testing.assert_array_equal(adata.var["qc_n_unique"].to_numpy(), expected_unique)


def test_qc_metrics_choose_columns_and_qc_fail(adata):
    """Only the requested metrics are written, and qc_fail ORs only their flags."""
    mt.pp.calculate_qc_metrics(adata, qc_metrics=["border"], image_shape=(100, 100))
    assert "qc_is_border" in adata.obs
    assert "qc_nan_fraction" not in adata.obs
    assert "qc_area_outlier" not in adata.obs
    np.testing.assert_array_equal(adata.obs["qc_fail"], adata.obs["qc_is_border"])


def test_qc_metrics_rejects_unknown_names(adata):
    with pytest.raises(ValueError, match="unknown"):
        mt.pp.calculate_qc_metrics(adata, qc_metrics=["varaince"])


def test_filter_cells_works_without_feature_metrics(adata):
    """A default (feature-metric-free) qc run still drives filter_cells: qc_fail is unchanged."""
    mt.pp.calculate_qc_metrics(adata)
    passing = int((~adata.obs["qc_fail"]).sum())
    assert mt.pp.filter_cells(adata, min_cells_per_well=0, copy=True).n_obs == passing


def test_filter_features_drops_all_nan_constant_and_blocklisted(adata):
    values = adata.X.copy()
    values[:, 0] = np.nan
    values[:, 1] = 1.0
    adata.X = values
    all_nan, constant, blocked = adata.var_names[0], adata.var_names[1], adata.var_names[3]

    mt.pp.filter_features(adata, min_variance=1e-8, blocklist=[blocked])

    assert set(adata.var_names).isdisjoint({all_nan, constant, blocked})
    assert adata.n_vars == 12


#: Explicit because the parser infers channels from the column set, and on six names the rename would then do nothing.
CHANNELS = ("AGP", "DNA", "ER", "Mito", "RNA")


def test_the_blocklist_still_matches_after_standardizing_the_names():
    """standardize_feature_names rewrites channel-bearing names into a form no blocklist entry matches (Nuclei_Correlation_Manders_AGP_DNA becomes ..._AGP|DNA), and all 55 bundled entries are renamed this way.
    The blocklist must still drop them.
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

    # pp.feature_select applies the blocklist through a separate code path.
    selected = renamed.copy()
    mt.pp.feature_select(selected, operations=("blocklist",))
    assert int(selected.var["selected"].sum()) == renamed.n_vars - len(blocked)


def _area_cells(nuclei_first: bool):
    """40 cells with one huge nucleus, with the two Area columns in either ``var`` order."""
    import anndata as ad

    from mantispy._core.features import parse_feature_names
    from mantispy._core.schema import stamp

    names = ["Cells_AreaShape_Area", "Nuclei_AreaShape_Area"]
    if nuclei_first:
        names = names[::-1]
    values = np.random.default_rng(0).normal(500.0, 10.0, (40, 2)).astype(np.float32)
    values[7, names.index("Nuclei_AreaShape_Area")] = 5000.0
    obs = pd.DataFrame(
        {"Metadata_Plate": "P1", "Metadata_Well": [f"A{index % 8 + 1:02d}" for index in range(40)]},
        index=[str(index) for index in range(40)],
    )
    adata = ad.AnnData(X=values, obs=obs, var=parse_feature_names(names))
    stamp(adata, resolution="object")
    return adata


def test_the_area_outlier_flag_does_not_depend_on_var_column_order():
    """Matching every compartment's AreaShape_Area and taking the first made qc_area_outlier, and so qc_fail, depend on column order: Cells-first flagged 0 of 40 cells and Nuclei-first flagged 1, on the same data and with no warning."""
    flagged = []
    for nuclei_first in (False, True):
        adata = _area_cells(nuclei_first)
        mt.pp.calculate_qc_metrics(adata)
        flagged.append(int(adata.obs["qc_area_outlier"].sum()))
    assert flagged == [1, 1]


def _no_area_object():
    """A 20-cell Intensity/Texture export whose names parse but measure no area."""
    import anndata as ad

    from mantispy._core.features import parse_feature_names

    names = ["Cells_Intensity_MeanIntensity_DNA", "Cells_Texture_Contrast_DNA"]
    var = parse_feature_names(names)
    assert var["feature"].notna().all(), "the annotation is parsed; it simply measured no area"
    values = np.random.default_rng(0).normal(500.0, 10.0, (20, 2)).astype(np.float32)
    obs = pd.DataFrame(
        {"Metadata_Plate": "P1", "Metadata_Well": [f"A{index % 4 + 1:02d}" for index in range(20)]},
        index=[str(index) for index in range(20)],
    )
    return ad.AnnData(X=values, obs=obs, var=var)


def test_qc_metrics_skips_the_area_check_when_no_area_is_found(caplog):
    """No area column: qc_area_outlier is skipped (left all False) with a warning, the other metrics still run."""
    adata = _no_area_object()
    with caplog.at_level(logging.INFO, logger="scverse.mantispy"):
        mt.pp.calculate_qc_metrics(adata)
    assert "qc_area_outlier is all False" in caplog.text
    assert not adata.obs["qc_area_outlier"].any()
    assert "qc_fail" in adata.obs and "qc_nan_fraction" in adata.obs


def test_qc_metrics_area_feature_names_the_column_or_rejects_an_unknown_one(caplog):
    """area_feature= names the column to score on a table whose names do not parse as an area."""
    adata = _no_area_object()
    with caplog.at_level(logging.WARNING, logger="scverse.mantispy"):
        mt.pp.calculate_qc_metrics(adata, area_feature="Cells_Intensity_MeanIntensity_DNA")
    assert "qc_area_outlier is all False" not in caplog.text

    with pytest.raises(KeyError, match="area_feature"):
        mt.pp.calculate_qc_metrics(adata, area_feature="not_a_column")
