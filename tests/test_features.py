import pandas as pd
import pytest

from mantispy._core.features import COLUMNS, load_blocklist, parse_feature_names

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


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        (
            "cell_0/max/sizeshapeSolidity",
            {"object": "cell", "feature_group": "sizeshape", "feature": "Solidity", "channel": "0"},
        ),
        (
            "cell_0/max/ferretMaxFeretDiameter",
            {"object": "cell", "feature_group": "ferret", "feature": "MaxFeretDiameter", "channel": "0"},
        ),
        (
            "nuclei_3/max/radial_zernikesRadialDistribution_ZernikeMagnitude_9_7",
            {
                "object": "nuclei",
                "feature_group": "radial_zernikes",
                "feature": "RadialDistribution_ZernikeMagnitude",
                "channel": "3",
                "params": "9_7",
            },
        ),
        (
            "cell_4/max/textureContrast_3_03_256",
            {
                "object": "cell",
                "feature_group": "texture",
                "feature": "Contrast",
                "channel": "4",
                "scale": 3.0,
                "angle": 3.0,
                "gray_levels": 256.0,
            },
        ),
        (
            "nuclei_1/max/radial_distributionRadialDistribution_FracAtD_4of4",
            {"feature_group": "radial_distribution", "feature": "RadialDistribution_FracAtD", "radial_bin": "4of4"},
        ),
        # A group separated from its feature by an underscore rather than glued to it in camel case.
        ("cell_2/max/sizeshape_Solidity", {"feature_group": "sizeshape", "feature": "Solidity", "channel": "2"}),
    ],
)
def test_parses_cp_measure_names(name, expected):
    """cp_measure separates its tokens with slashes and glues the group to the feature in camel case, which the
    CellProfiler parser read as the `0/max/intensityIntensity` group of nothing in particular, with no channel."""
    row = _row(name)
    assert row["is_feature"]
    for key, value in expected.items():
        assert row[key] == value, f"{key}: expected {value!r}, got {row[key]!r}"


def test_a_cellprofiler_name_is_never_read_as_cp_measure():
    """The two conventions are told apart by the slash, which CellProfiler never emits."""
    row = _row("Cells_Intensity_MeanIntensity_DNA")
    assert (row["object"], row["feature_group"], row["channel"]) == ("Cells", "Intensity", "DNA")


def test_the_grammar_is_chosen_from_the_whole_list():
    """A file is written by one tool, so one cp_measure name settles how the rest are read, and a CellProfiler
    name in that file is a mixed file rather than a name to guess at."""
    parsed = parse_feature_names(["cell_0/max/sizeshapeSolidity", "Cells_AreaShape_Area"], channels=CHANNELS)

    assert parsed.loc["cell_0/max/sizeshapeSolidity", "feature_group"] == "sizeshape"
    assert not parsed.loc["Cells_AreaShape_Area", "is_feature"]


def test_zernike_orders_stay_distinct():
    """Keeping only the first numeric token collapses Zernike_2_0 and Zernike_2_2."""
    names = ["Cells_AreaShape_Zernike_2_0", "Cells_AreaShape_Zernike_2_2"]
    parsed = parse_feature_names(names, channels=CHANNELS)
    assert parsed["params"].tolist() == ["2_0", "2_2"]


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


def test_the_bounding_box_area_is_still_a_feature():
    """Where an object sits is not a feature; how much of the image its bounding box covers is."""
    assert _row("Cells_AreaShape_BoundingBoxArea")["is_feature"]


def test_unprefixed_names_have_na_object():
    row = _row("AreaShape_Area")
    assert pd.isna(row["object"])
    assert row["feature_group"] == "AreaShape"
    assert row["is_feature"]


def test_channel_inference_without_explicit_channels():
    names = [
        "Cells_Intensity_MeanIntensity_DNA",
        "Cells_Intensity_MaxIntensity_DNA",
        "Cells_Intensity_MeanIntensity_ER",
        "Cells_Intensity_MaxIntensity_ER",
    ]
    assert set(parse_feature_names(names)["channel"]) == {"DNA", "ER"}


@pytest.mark.parametrize(
    ("channels", "name", "expected_channel"),
    [
        (["Hoechst", "GFP"], "Cells_Intensity_MeanIntensity_Hoechst", "Hoechst"),
        (["Hoechst", "GFP"], "Cells_Correlation_Correlation_Hoechst_GFP", "Hoechst|GFP"),
        (["w1"], "Cells_Intensity_MeanIntensity_w1", "w1"),
        (["Ch1", "Ch2", "Ch3"], "Nuclei_Texture_Entropy_Ch3_5_00_256", "Ch3"),
        (["DAPI"], "Cells_AreaShape_Area", None),  # channel-free feature stays channel-free
    ],
)
def test_parser_is_channel_vocabulary_agnostic(channels, name, expected_channel):
    """CellProfiler grammar is the default; the channel names are not baked in."""
    row = _row(name, channels=channels)
    assert row["is_feature"]
    if expected_channel is None:
        assert pd.isna(row["channel"])
    else:
        assert row["channel"] == expected_channel


def test_unknown_group_is_still_treated_as_a_feature():
    """cp_measure and new CellProfiler modules emit groups the parser does not know."""
    row = _row("Cells_SomeNewModule_SomeStatistic_GFP", channels=["GFP"])
    assert row["is_feature"]
    assert row["feature_group"] == "SomeNewModule"
    assert row["channel"] == "GFP"


def test_frame_shape_and_dtypes():
    names = ["Cells_AreaShape_Area", "Nuclei_AreaShape_Area"]
    parsed = parse_feature_names(names, channels=CHANNELS)
    assert list(parsed.index) == names
    assert list(parsed.columns) == COLUMNS
    # category dtype keeps all-missing columns writable to h5ad
    assert str(parsed["channel"].dtype) == "category"
    assert parsed["is_feature"].dtype == bool


def test_blocklist_loads():
    blocklist = load_blocklist()
    assert len(blocklist) == 55
    assert all(isinstance(name, str) for name in blocklist)
    assert any("Manders" in name for name in blocklist)
    assert not any(name.startswith("#") for name in blocklist)


def test_blocklist_rejects_unknown_name():
    with pytest.raises(ValueError, match="unknown blocklist"):
        load_blocklist("nope")
