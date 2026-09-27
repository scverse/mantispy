"""Image-, well- and cell-level QC, and feature name standardization."""

import pytest

import mantispy as mt
from mantispy.ds import synthetic_plate


@pytest.fixture
def imaged():
    return synthetic_plate(
        n_plates=2, n_wells=24, n_cells=10, n_features=10, n_images_per_well=2, n_bad_images=6, seed=0
    )


# --- image QC --------------------------------------------------------------


@pytest.mark.parametrize("method", ["mad", "knn"])
def test_image_qc_recovers_the_injected_bad_images(imaged, method):
    mt.pp.image_qc(imaged, method=method, channel="DNA")
    result = imaged.uns["mantispy"]["image_qc"]
    truth = set(imaged.uns["mantispy"]["truth"]["bad_images"])

    # The score must separate the degraded images completely, wherever the default
    # threshold falls.
    scores = result["qc_image_score"]
    assert scores.loc[sorted(truth)].min() > scores.drop(index=sorted(truth)).max()

    flagged = set(result.index[~result["qc_image_pass"]])
    assert truth <= flagged, f"missed {sorted(truth - flagged)}"
    assert len(flagged - truth) <= 2, f"false positives: {sorted(flagged - truth)}"


def test_image_qc_broadcasts_and_filters(imaged):
    mt.pp.image_qc(imaged)
    bad = imaged.uns["mantispy"]["truth"]["bad_images"]
    assert not imaged.obs.loc[imaged.obs["Metadata_ImageNumber"].isin(bad), "qc_image_pass"].any()

    n_failing = int((~imaged.obs["qc_image_pass"]).sum())
    assert mt.pp.filter_images(imaged, copy=True).n_obs == imaged.n_obs - n_failing


def test_image_qc_refuses_to_pool_plates_silently(imaged):
    """Pooling would flag every image on a dim plate and miss a blurred one on a bright plate."""
    table = imaged.uns["mantispy"]["image_table"].drop(columns=["Metadata_Plate"])
    imaged.uns["mantispy"]["image_table"] = table
    with pytest.raises(KeyError, match="by=None to pool"):
        mt.pp.image_qc(imaged)
    mt.pp.image_qc(imaged, by=None)  # explicit opt-in works


def test_image_qc_reports_a_missing_table_and_unknown_metrics(imaged):
    with pytest.raises(KeyError, match="Available"):
        mt.pp.image_qc(imaged, metrics=("NoSuchMetric",))
    del imaged.uns["mantispy"]["image_table"]
    with pytest.raises(KeyError, match="read_profiles"):
        mt.pp.image_qc(imaged)


def test_image_qc_warns_about_unmatched_cells(imaged):
    imaged.obs["Metadata_ImageNumber"] = imaged.obs["Metadata_ImageNumber"] + 10_000
    with pytest.warns(UserWarning, match="not present in the image table"):
        mt.pp.image_qc(imaged)
    assert imaged.obs["qc_image_pass"].all()


# --- well QC ---------------------------------------------------------------


def test_well_qc_criteria(cells):
    mt.pp.well_qc(cells, min_cells=16)
    assert not cells.obs["qc_well_pass"].any()

    mt.pp.well_qc(cells, min_cells=1, max_nan_fraction=1.0)
    assert cells.uns["mantispy"]["well_qc"]["qc_well_pass"].all()

    mt.pp.well_qc(cells, min_cells=1, max_control_cv=0.0)
    table = cells.uns["mantispy"]["well_qc"]
    assert not table.loc[table["control_cv"].notna(), "qc_well_pass"].any()
    assert table["control_cv"].isna().any()  # non-control wells have no CV


# --- feature names ---------------------------------------------------------


def test_standardize_keeps_originals_and_is_idempotent(cells):
    before = list(cells.var_names)
    mt.pp.standardize_feature_names(cells)
    assert cells.var["original_name"].tolist() == before
    mt.pp.standardize_feature_names(cells)
    assert cells.var["original_name"].tolist() == before


def test_standardize_refuses_a_collision_and_an_unknown_target(cells):
    with pytest.raises(ValueError, match="target must be"):
        mt.pp.standardize_feature_names(cells, target="nonsense")


def test_a_failed_image_qc_call_leaves_no_verdict_behind(imaged):
    """The uns write happened before the broadcast could raise, so a call that failed still
    left a verdict in uns that was never applied to a single cell, and pl.image_qc plotted it."""
    imaged.obs = imaged.obs.drop(columns="Metadata_ImageNumber")
    with pytest.raises(KeyError, match="Metadata_ImageNumber"):
        mt.pp.image_qc(imaged)
    assert "image_qc" not in imaged.uns["mantispy"]
