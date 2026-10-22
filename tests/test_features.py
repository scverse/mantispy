import pandas as pd
import pytest

from mantispy._core.features import parse_feature_names

CHANNELS = ["DNA", "ER", "RNA", "AGP", "Mito"]


def _row(name, channels=CHANNELS):
    return parse_feature_names([name], channels=channels).loc[name]


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("Cells_AreaShape_Area", {"object": "Cells", "feature_group": "AreaShape", "feature": "Area", "channel": None}),
        (
            "Cells_Intensity_MeanIntensity_DNA",
            {"object": "Cells", "feature_group": "Intensity", "feature": "MeanIntensity", "channel": "DNA"},
        ),
        (
            "Nuclei_Texture_Contrast_ER_3_00_256",
            {
                "object": "Nuclei",
                "feature_group": "Texture",
                "feature": "Contrast",
                "channel": "ER",
                "scale": 3.0,
                "angle": 0.0,
                "gray_levels": 256.0,
            },
        ),
        (
            "Nuclei_Texture_Contrast_ER_3",
            {"feature": "Contrast", "channel": "ER", "scale": 3.0, "angle": None, "gray_levels": None},
        ),
        (
            "Cells_RadialDistribution_FracAtD_Mito_1of4",
            {"feature_group": "RadialDistribution", "feature": "FracAtD", "channel": "Mito", "radial_bin": "1of4"},
        ),
        (
            "Cells_Granularity_3_AGP",
            {"feature_group": "Granularity", "feature": "Granularity", "channel": "AGP", "scale": 3.0},
        ),
        (
            "Cells_Correlation_Correlation_DNA_ER",
            {"feature_group": "Correlation", "feature": "Correlation", "channel": "DNA|ER"},
        ),
        (
            "Cytoplasm_Neighbors_NumberOfNeighbors_Adjacent",
            {"object": "Cytoplasm", "feature_group": "Neighbors", "feature": "NumberOfNeighbors_Adjacent"},
        ),
    ],
)
def test_parses_feature_names(name, expected):
    row = _row(name)
    assert row["is_feature"]
    for key, value in expected.items():
        actual = row[key]
        if value is None:
            assert pd.isna(actual), f"{key}: expected NA, got {actual!r}"
        else:
            assert actual == value, f"{key}: expected {value!r}, got {actual!r}"


def test_cp_measure_channel_and_group_parsed():
    rows = parse_feature_names(["Intensity_MeanIntensity__DNA", "AreaShape_Area"], convention="cp_measure")
    assert rows.loc["Intensity_MeanIntensity__DNA", "feature_group"] == "Intensity"
    assert rows.loc["Intensity_MeanIntensity__DNA", "channel"] == "DNA"
    assert rows.loc["AreaShape_Area", "is_feature"]


def test_cp_measure_colocalization_pair_channels():
    row = parse_feature_names(["Correlation_Correlation__DNA__ER"], convention="cp_measure").iloc[0]
    assert row["channel"] == "DNA|ER"


def test_cp_measure_params_and_bare_names():
    rows = parse_feature_names(["Texture_Contrast_3_0_256__ER", "Zernike_2_0", "Area"], convention="cp_measure")
    assert rows.loc["Texture_Contrast_3_0_256__ER", "feature"] == "Contrast"
    assert rows.loc["Texture_Contrast_3_0_256__ER", "scale"] == 3
    assert rows.loc["Zernike_2_0", "params"] == "2_0"
    assert rows.loc["Area", "is_feature"]
    assert pd.isna(rows.loc["Area", "feature_group"])


def test_cp_measure_convention_is_detected_from_double_underscore():
    assert parse_feature_names(["Intensity_MeanIntensity__DNA"]).iloc[0]["channel"] == "DNA"


def test_cellprofiler_names_are_unchanged_by_the_explicit_convention():
    name = "Cells_Intensity_MeanIntensity_DNA"
    pd.testing.assert_frame_equal(
        parse_feature_names([name], channels=CHANNELS),
        parse_feature_names([name], channels=CHANNELS, convention="cellprofiler"),
    )


@pytest.mark.parametrize(
    "name",
    [
        "ImageNumber",
        "ObjectNumber",
        "Nuclei_ObjectNumber",
        "Metadata_Plate",
        "Cells_Number_Object_Number",
        "Cells_Location_Center_X",
        "Cells_AreaShape_Center_X",
        "Nuclei_AreaShape_BoundingBoxMaximum_Y",
        "Cells_Parent_Nuclei",
        "Cells_Children_Cytoplasm_Count",
        "Image_Count_Cells",
        "Image_FileName_DNA",
        "Image_ExecutionTime_05IdentifyPrimaryObjects",
    ],
)
def test_non_features_flagged(name):
    assert not _row(name)["is_feature"]
