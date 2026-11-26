from pathlib import Path

import pandas as pd
import pytest

from mantispy._core.features import parse_feature_names

CHANNELS = ["DNA", "ER", "RNA", "AGP", "Mito"]
EXPORTS = Path(__file__).parent / "data" / "cellprofiler"


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


def test_channel_names_with_underscores_parse_as_a_whole_suffix():
    """A channel whose own name holds an underscore (ERK1_ERK2, Mouse_IgG) must match as one suffix."""
    channels = [*CHANNELS, "ERK1_ERK2", "Mouse_IgG"]
    rows = parse_feature_names(
        ["Cells_Intensity_MeanIntensity_ERK1_ERK2", "Cells_Intensity_MeanIntensity_Mouse_IgG"],
        channels=channels,
    )
    erk = rows.loc["Cells_Intensity_MeanIntensity_ERK1_ERK2"]
    assert erk["channel"] == "ERK1_ERK2" and erk["feature"] == "MeanIntensity"
    assert rows.loc["Cells_Intensity_MeanIntensity_Mouse_IgG", "channel"] == "Mouse_IgG"


def test_inference_keeps_a_known_stain_beside_an_underscored_channel():
    """Without channels=, a known stain (DNA) is still inferred even next to an underscored channel name."""
    rows = parse_feature_names(["Cells_Intensity_MeanIntensity_ERK1_ERK2", "Cells_Intensity_MeanIntensity_DNA"])
    assert rows.loc["Cells_Intensity_MeanIntensity_DNA", "channel"] == "DNA"


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


@pytest.mark.parametrize(
    "name",
    ["Nuclei_Neighbors_FirstClosestObjectNumber_Expanded", "Nuclei_Neighbors_SecondClosestObjectNumber_2"],
)
def test_neighbour_object_numbers_are_not_features(name):
    """MeasureObjectNeighbors' closest-object number identifies another object, as spec 18 keeps out of X."""
    assert not _row(name)["is_feature"]


@pytest.mark.parametrize(
    "name",
    [
        "Cells_Mean_Mitochondria_Location_Center_X",
        "Nuclei_Mean_Nucleoli_AreaShape_Center_Y",
        "Nuclei_Mean_Nucleoli_AreaShape_BoundingBoxMinimum_X",
        "Nuclei_Mean_Nucleoli_Number_Object_Number",
        "Cells_Mean_Mitochondria_Neighbors_FirstClosestObjectNumber_Expanded",
        "Cells_Mean_Mitochondria_Distance_Centroid_Cells",
        "Cells_Mean_BarcodeFoci_Children_Foci_NonCellEdge_Count",
        "Cells_Mean_Foci_NonCellEdge_Parent_BarcodeFoci",
    ],
)
def test_a_per_parent_mean_follows_the_childs_non_feature_rule(name):
    """RelateObjects' <Parent>_Mean_<Child>_<column> of a location or an object number is no measurement either."""
    assert not _row(name)["is_feature"]


def test_a_per_parent_mean_of_a_measurement_stays_on_the_parent():
    row = _row("Nuclei_Mean_Nucleoli_AreaShape_Area")
    assert row["is_feature"]
    assert (row["object"], row["feature_group"]) == ("Nuclei", "Mean")


def test_a_leading_mean_is_a_per_image_statistic_read_like_median():
    """ExportToSpreadsheet's per-image Mean_<Object>_* parses as Median_ and StDev_ do, not as a per-parent mean."""
    parsed = parse_feature_names(["Mean_Cells_AreaShape_Area", "Median_Cells_AreaShape_Area"])
    assert list(parsed["object"]) == ["Mean", "Median"]
    assert list(parsed["feature_group"]) == ["Cells", "Cells"]
    assert parsed["is_feature"].all()


@pytest.mark.parametrize("name", ["Nucleoli_Distance_Centroid_Nuclei", "Mitochondria_Distance_Minimum_Cells"])
def test_distances_to_the_parent_are_not_features(name):
    """RelateObjects' distance from a child to its parent places the child, so it stays out of X."""
    assert not _row(name)["is_feature"]


@pytest.mark.parametrize(
    "name", ["Foci_Barcode_BarcodeCalled", "Foci_Barcode_MatchedTo_ID", "Foci_Barcode_MatchedTo_Score"]
)
def test_barcode_calls_are_not_features(name):
    """CallBarcodes' output identifies the object's perturbation; it is not morphology."""
    assert not _row(name)["is_feature"]


@pytest.mark.parametrize("name", ["Channel_OrigDNA", "Image_Channel_OrigDNA"])
def test_image_channel_indices_are_not_features(name):
    assert not _row(name)["is_feature"]


def _export_columns(export, tables):
    """The columns of the real export tables `tables`, each prefixed with its object as the reader prefixes them."""
    columns = []
    for file, obj in tables.items():
        header = pd.read_csv(EXPORTS / export / f"{file}.csv", nrows=0).columns
        columns += [f"{obj}_{column}" for column in header if column not in {"ImageNumber", "ObjectNumber"}]
    return columns


def test_neighbour_ids_and_relate_objects_positions_stay_out_of_x_on_a_real_export():
    """The BBBC022 tutorial export: closest-object numbers, distances to the parent and their per-parent means."""
    tables = {"MyExpt_Nuclei": "Nuclei", "MyExpt_Cells": "Cells", "MyExpt_Mitochondria": "Mitochondria"}
    parsed = parse_feature_names(_export_columns("advanced", tables))
    features = set(parsed.index[parsed["is_feature"]])
    assert {"Nuclei_Neighbors_FirstClosestDistance_Expanded", "Nuclei_Mean_Nucleoli_AreaShape_Area"} <= features
    assert parsed.loc["Nuclei_Mean_Nucleoli_AreaShape_Area", "object"] == "Nuclei"
    tokens = ("ClosestObjectNumber", "Distance_Centroid", "Distance_Minimum", "_Location_", "_Center_", "Object_Number")
    assert not [name for name in features if any(token in name for token in tokens)]


def test_an_object_name_with_an_underscore_is_read_whole():
    """PERISCOPE's Foci_NonCellEdge was read as object Foci with the unknown group NonCellEdge."""
    names = [
        "Foci_NonCellEdge_Number_Object_Number",
        "Foci_NonCellEdge_Location_Center_X",
        "Foci_NonCellEdge_Parent_BarcodeFoci",
        "Foci_NonCellEdge_AreaShape_Area",
        "Foci_AreaShape_Area",
    ]
    parsed = parse_feature_names(names, channels=CHANNELS)
    assert set(parsed["object"].iloc[:4]) == {"Foci_NonCellEdge"}
    assert list(parsed["is_feature"]) == [False, False, False, True, True]
    assert parsed.loc["Foci_NonCellEdge_AreaShape_Area", "feature_group"] == "AreaShape"
    assert parsed.loc["Foci_AreaShape_Area", "object"] == "Foci"


def test_an_object_name_with_an_underscore_is_read_whole_on_a_real_export():
    """PERISCOPE's Foci_NonCellEdge table holds locations and links only, which were read as features of Foci."""
    parsed = parse_feature_names(_export_columns("periscope_site21", {"Foci_NonCellEdge": "Foci_NonCellEdge"}))
    assert set(parsed["object"]) == {"Foci_NonCellEdge"}
    assert not parsed["is_feature"].any()


@pytest.mark.parametrize("angle", ["00", "01", "02", "03"])
def test_cp_measure_haralick_correlation_is_texture_like_cellprofiler(angle):
    """cp_measure writes Haralick Correlation as Correlation_<scale>_<angle>_<levels>, the colocalization group's name."""
    cpm = parse_feature_names([f"Correlation_3_{angle}_256__DNA"], convention="cp_measure").iloc[0]
    cp = _row(f"Cells_Texture_Correlation_DNA_3_{angle}_256")
    for key in ("feature_group", "feature", "channel", "scale", "angle", "gray_levels", "params"):
        assert cpm[key] == cp[key], key


@pytest.mark.parametrize(
    ("name", "feature"),
    [
        ("Correlation_Pearson__DNA__RNA", "Pearson"),
        ("Correlation_Slope__DNA__RNA", "Slope"),
        ("Correlation_Manders_1__DNA__RNA", "Manders"),
        ("Correlation_RWC_2__DNA__RNA", "RWC"),
        ("Correlation_Costes_1__DNA__RNA", "Costes"),
    ],
)
def test_cp_measure_colocalization_stays_in_the_correlation_group(name, feature):
    row = parse_feature_names([name], convention="cp_measure").iloc[0]
    assert (row["feature_group"], row["feature"], row["channel"]) == ("Correlation", feature, "DNA|RNA")
