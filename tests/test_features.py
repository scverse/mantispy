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


def test_the_grammar_is_chosen_from_the_whole_list():
    """A file is written by one tool, so one cp_measure name settles how the rest are read, and a CellProfiler name in that file is a mixed file rather than a name to guess at."""
    parsed = parse_feature_names(["cell_0/max/sizeshapeSolidity", "Cells_AreaShape_Area"], channels=CHANNELS)

    assert parsed.loc["cell_0/max/sizeshapeSolidity", "feature_group"] == "sizeshape"
    assert not parsed.loc["Cells_AreaShape_Area", "is_feature"]


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
