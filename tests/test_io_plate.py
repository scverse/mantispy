from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest
import spatialdata as sd
from _testdata import BATCH, CHANNELS, EXPORT_OBJECTS, EXPORT_SHAPE, FIELDS, OVERLAY_PLATE, PLATE, PREFIX
from skimage.segmentation import find_boundaries

import mantispy as mt
from mantispy.io._gallery import _fov_offsets, _labels_from_outlines, _parse_well
from mantispy.io._plate import Layout


@pytest.fixture
def sdata(gallery: Path) -> sd.SpatialData:
    return mt.io.read_plate(gallery, PLATE, batch=BATCH, profile="test")


def test_images_cover_every_downloaded_well(sdata: sd.SpatialData) -> None:
    assert set(sdata.images) == {f"{PLATE}_{w}_s{s}_image" for w in ("A01", "B02") for s in (1, 2)}
    image = sdata[f"{PLATE}_A01_s1_image"]["scale0"]["image"]
    # the URL_Illum* column names a correction image, not a channel
    assert list(image.coords["c"].to_numpy()) == list(CHANNELS)


def test_labels_exist_only_where_cellprofiler_ran(sdata: sd.SpatialData) -> None:
    objects = ("nuclei", "cells", "cytoplasm")
    assert set(sdata.labels) == {f"{PLATE}_A01_s{s}_{o}" for s in (1, 2) for o in objects}


@pytest.mark.parametrize("plate", [PLATE, OVERLAY_PLATE], ids=["outlines", "colour-overlay"])
def test_labels_carry_cellprofiler_object_numbers(gallery: Path, plate: str) -> None:
    """The second plate draws its outlines in colour over the image, and its cell image carries the nuclei too."""
    sdata = mt.io.read_plate(gallery, plate, batch=BATCH, profile="test")

    cells = sdata[f"{plate}_A01_s1_cells"].to_numpy()
    nuclei = sdata[f"{plate}_A01_s1_nuclei"].to_numpy()
    assert set(np.unique(cells)) == set(np.unique(nuclei)) == {0, 1, 2}
    assert (cells == 1).sum() > (nuclei == 1).sum()
    assert ((nuclei == 1) & (cells == 1)).sum() == (nuclei == 1).sum()


def test_cytoplasm_is_cells_minus_nuclei(sdata: sd.SpatialData) -> None:
    cells = sdata[f"{PLATE}_A01_s1_cells"].to_numpy()
    nuclei = sdata[f"{PLATE}_A01_s1_nuclei"].to_numpy()
    cytoplasm = sdata[f"{PLATE}_A01_s1_cytoplasm"].to_numpy()

    assert np.array_equal(cytoplasm, np.where(nuclei > 0, 0, cells))
    assert cytoplasm.any()


def test_tables_annotate_their_elements(sdata: sd.SpatialData) -> None:
    cells, wells = sdata.tables["cells"], sdata.tables["wells"]

    assert cells.n_obs == 4
    assert set(cells.obs["region"]) == {f"{PLATE}_A01_s{s}_cells" for s in (1, 2)}
    assert wells.n_obs == 384
    assert set(wells.obs["region"]) == {f"{PLATE}_wells"}
    assert len(sdata.shapes[f"{PLATE}_wells"]) == 384


def test_stage_coordinates_lay_the_fields_out_in_three_frames(sdata: sd.SpatialData) -> None:
    assert set(sdata.coordinate_systems) >= {PLATE, f"{PLATE}_A01", f"{PLATE}_A01_s1"}

    extent = sd.get_extent(sdata[f"{PLATE}_B02_s1_image"], coordinate_system=PLATE)
    assert extent["y"][0] == pytest.approx(4500.0)
    assert extent["x"][0] == pytest.approx(4500.0)


def test_without_stage_coordinates_each_field_sits_in_its_own_frame(unlocated_gallery: Path) -> None:
    """Nothing places the fields relative to each other, and the reader must degrade rather than invent one."""
    sdata = mt.io.read_plate(unlocated_gallery, PLATE, batch=BATCH, profile="test")

    assert set(sdata.coordinate_systems) == {f"{PLATE}_{w}_s{s}" for w in ("A01", "B02") for s in (1, 2)}
    assert not sdata.shapes
    assert sdata.tables["wells"].n_obs == 384


@pytest.mark.parametrize(
    ("kwargs", "images", "tables"),
    [
        ({"wells": ["B02"], "profile": "test"}, {f"{PLATE}_B02_s{s}_image" for s in (1, 2)}, {"wells"}),
        ({"profile": None}, {f"{PLATE}_{w}_s{s}_image" for w in ("A01", "B02") for s in (1, 2)}, {"cells"}),
    ],
    ids=["one-well", "no-profile"],
)
def test_reading_part_of_a_plate(gallery: Path, kwargs: dict[str, Any], images: set[str], tables: set[str]) -> None:
    sdata = mt.io.read_plate(gallery, PLATE, batch=BATCH, **kwargs)

    assert set(sdata.images) == images
    assert set(sdata.tables) == tables


def test_two_plates_concatenate(sdata: sd.SpatialData, gallery: Path) -> None:
    """Element names carry the plate barcode, which is what lets two plates merge without renaming."""
    other = mt.io.read_plate(gallery, OVERLAY_PLATE, batch=BATCH, profile="test")
    assert not set(sdata.images) & set(other.images)

    merged = sd.concatenate([sdata, other], concatenate_tables=True)

    assert len(merged.images) == 8
    assert merged.tables["wells"].n_obs == 768
    assert merged.tables["cells"].n_obs == 8
    assert set(merged.coordinate_systems) >= {PLATE, OVERLAY_PLATE}


@pytest.mark.parametrize(("well", "expected"), [("A01", (0, 0)), ("B02", (1, 1)), ("P24", (15, 23)), ("AA01", (26, 0))])
def test_parse_well(well: str, expected: tuple[int, int]) -> None:
    assert _parse_well(well) == expected


def test_parse_well_rejects_nonsense() -> None:
    with pytest.raises(ValueError, match="not a well name"):
        _parse_well("well-1")


def _positions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "well": ["A01", "A01", "A01", "A01", "B02"],
            "x": [-5e-4, 5e-4, -5e-4, 5e-4, -5e-4],
            "y": [5e-4, 5e-4, -5e-4, -5e-4, 5e-4],
        }
    )


@pytest.mark.parametrize(
    ("columns", "expected"),
    [
        # y is flipped, and the top-left field of a well is the origin all wells share
        (["well_y", "well_x"], {"well_y": [0.0, 0.0, 1000.0, 1000.0, 0.0], "well_x": [0.0, 1000.0, 0.0, 1000.0, 0.0]}),
        (
            ["plate_y", "plate_x"],
            {"plate_y": [0.0, 0.0, 1000.0, 1000.0, 4500.0], "plate_x": [0.0, 1000.0, 0.0, 1000.0, 4500.0]},
        ),
    ],
    ids=["within-a-well", "across-the-plate"],
)
def test_fov_offsets(columns: list[str], expected: dict[str, list[float]]) -> None:
    offsets = _fov_offsets(_positions(), pixel_size=1e-6, plate_format=384)

    pd.testing.assert_frame_equal(offsets[columns], pd.DataFrame(expected))


def test_fov_offsets_without_a_plate_format() -> None:
    assert "plate_y" not in _fov_offsets(_positions(), pixel_size=1e-6, plate_format=None)


def test_fov_offsets_reject_wells_off_the_plate() -> None:
    with pytest.raises(ValueError, match="outside a 96-well plate"):
        _fov_offsets(_positions().assign(well="Z99"), pixel_size=1e-6, plate_format=96)


def _two_objects(split: bool) -> np.ndarray:
    """Two objects sharing a boundary, or one object where that boundary never closed."""
    truth = np.zeros((40, 60), np.uint32)
    truth[5:35, 5:30] = 7
    truth[5:35, 30:55] = 3 if split else 7
    return truth


def _one_object() -> np.ndarray:
    truth = np.zeros((40, 60), np.uint32)
    truth[5:35, 5:30] = 7
    return truth


def _centres(numbers: list[int], x: list[float], y: list[float], area: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {"ObjectNumber": numbers, "Location_Center_X": x, "Location_Center_Y": y, "AreaShape_Area": area}
    )


def test_labels_from_outlines_recovers_touching_objects() -> None:
    """Outlines are one pixel wide and shared between touching objects, so the interiors must come apart."""
    truth = _two_objects(split=True)
    centres = _centres([7, 3], [17.0, 42.0], [20.0, 20.0], [750.0, 750.0])

    labels = _labels_from_outlines(find_boundaries(truth, mode="inner"), centres)

    assert set(np.unique(labels)) == {0, 3, 7}
    for number in (3, 7):
        assert ((labels == number) & (truth == number)).sum() / (truth == number).sum() > 0.98


@pytest.mark.parametrize(
    ("truth", "centres", "expected"),
    [
        # no centroid falls in the second component, so nothing claims it
        (_two_objects(split=True), _centres([7], [17.0], [20.0], [750.0]), {0, 7}),
        # object 3's outline never closed, so its centroid sits in the background
        (_one_object(), _centres([7, 3], [17.0, 45.0], [20.0, 20.0], [750.0, 600.0]), {0, 7}),
        # the boundary between the two objects is missing, so one component holds both centroids
        (_two_objects(split=False), _centres([1, 2], [17.0, 42.0], [20.0, 20.0], [750.0, 750.0]), {0}),
        # the measured area is nothing like the component, so the component cannot be the object
        (_one_object(), _centres([7], [17.0], [20.0], [1.0]), {0}),
    ],
    ids=["unclaimed-component", "centroid-in-background", "merged-objects", "area-mismatch"],
)
def test_labels_from_outlines_only_takes_unambiguous_components(
    truth: np.ndarray, centres: pd.DataFrame, expected: set[int]
) -> None:
    labels = _labels_from_outlines(find_boundaries(truth, mode="inner"), centres)

    assert set(np.unique(labels)) == expected


def test_labels_from_outlines_area_check_can_be_turned_off() -> None:
    outlines = find_boundaries(_one_object(), mode="inner")
    centres = _centres([7], [17.0], [20.0], [1.0])

    assert _labels_from_outlines(outlines, centres, area_column=None).any()


def test_labels_from_outlines_with_no_objects() -> None:
    labels = _labels_from_outlines(np.zeros((8, 8), np.uint8), _centres([], [], [], []))

    assert labels.shape == (8, 8)
    assert not labels.any()


def test_labels_from_outlines_rejects_a_stack() -> None:
    with pytest.raises(ValueError, match="2D outline image"):
        _labels_from_outlines(np.zeros((2, 10, 10)), _centres([1], [1.0], [1.0], [1.0]))


def test_export_reads_images_labels_and_the_table(export: Path) -> None:
    sdata = mt.io.read_plate(export)

    assert set(sdata.images) == {f"{field}_image" for field in FIELDS}
    assert set(sdata.labels) == {f"{field}__{obj}" for field in FIELDS for obj in EXPORT_OBJECTS}
    # the module exports no stage coordinates, so each field is its own frame
    assert set(sdata.coordinate_systems) == set(FIELDS)
    assert sdata.tables["cells"].shape == (4, 3)
    assert list(sdata.images[f"{FIELDS[0]}_image"].coords["c"].values) == list(CHANNELS)


def test_every_element_of_a_field_shares_that_fields_frame(export: Path) -> None:
    """An image and the labels from one field are the same pixel grid, which is what makes overlay work."""
    named = {
        name for _, name, _ in mt.io.read_plate(export).filter_by_coordinate_system(FIELDS[0]).gen_spatial_elements()
    }

    assert named == {f"{FIELDS[0]}_image", f"{FIELDS[0]}__Nuclei", f"{FIELDS[0]}__Cells"}


def test_export_rows_join_onto_the_label_arrays(export: Path) -> None:
    sdata = mt.io.read_plate(export)
    table = sdata.tables["cells"]

    for name in (f"{field}__Cells" for field in FIELDS):
        in_array = set(np.unique(np.asarray(sdata.labels[name])).tolist()) - {0}
        in_table = set(table.obs.loc[table.obs["region_key"] == name, "label_id"].tolist())
        assert in_array == in_table, name
    # Nuclei elements are read but no row annotates them, so `region` must not claim them
    regions = table.uns["spatialdata_attrs"]["region"]
    assert set(regions) == {f"{field}__Cells" for field in FIELDS}


def test_a_failed_field_is_left_out_along_with_its_rows(make_export) -> None:
    """A cycle that failed wrote no arrays, so the reader must skip them rather than open a file that is gone."""
    sdata = mt.io.read_plate(make_export(failed=FIELDS[1]))

    assert set(sdata.images) == {f"{FIELDS[0]}_image"}
    assert set(sdata.labels) == {f"{FIELDS[0]}__{obj}" for obj in EXPORT_OBJECTS}
    table = sdata.tables["cells"]
    assert table.n_obs == 2
    assert set(table.uns["spatialdata_attrs"]["region"]) <= set(sdata.labels)


@pytest.mark.parametrize("label_dtype", ["int8", "int16", "int32"])
def test_labels_arrive_as_one_type_whatever_width_cellprofiler_used(make_export, label_dtype: str) -> None:
    """CellProfiler narrows its label arrays to fit the object count, so two fields of one plate can disagree."""
    sdata = mt.io.read_plate(make_export(label_dtype=label_dtype))

    assert sdata.labels[f"{FIELDS[0]}__Cells"].dtype == np.uint32


@pytest.mark.parametrize("lazy", [True, False])
def test_lazy_and_eager_reads_give_the_same_pixels(export: Path, lazy: bool) -> None:
    sdata = mt.io.read_plate(export, lazy=lazy)

    assert np.asarray(sdata.images[f"{FIELDS[0]}_image"]).shape == (len(CHANNELS), *EXPORT_SHAPE)
    assert np.unique(np.asarray(sdata.labels[f"{FIELDS[0]}__Cells"])).tolist() == [0, 1, 2]


def test_the_object_writes_to_zarr(export: Path, tmp_path: Path) -> None:
    """The manifest travels in the table's `uns`, so anything unwritable in it makes the whole object
    unwritable."""
    sdata = mt.io.read_plate(export)

    sdata.write(tmp_path / "plate.zarr")
    back = sd.read_zarr(tmp_path / "plate.zarr")

    assert set(back.labels) == set(sdata.labels)
    assert np.array_equal(np.asarray(back.tables["cells"].X), np.asarray(sdata.tables["cells"].X))
    assert "cellprofiler_mapping" in back.tables["cells"].uns


@pytest.mark.parametrize(
    ("drop", "match"),
    [
        ("manifest", "ExportForSpatialData writes the manifest"),
        ("region_key", "region_key"),
    ],
)
def test_a_table_the_reader_cannot_use_says_which_module_writes_one(export: Path, drop: str, match: str) -> None:
    import anndata as ad

    path = export / "tables" / f"{PREFIX}.h5ad"
    adata = ad.read_h5ad(path)
    if drop == "manifest":
        del adata.uns["cellprofiler_mapping"]["elements"]
    else:
        del cast("pd.DataFrame", adata.obs)["region_key"]
        del adata.uns["spatialdata_attrs"]
    adata.write_h5ad(path)

    with pytest.raises(ValueError, match=match):
        mt.io.read_plate(export)


def test_an_export_root_with_one_plate_needs_no_plate(export: Path) -> None:
    assert set(mt.io.read_plate(export.parent).images) == set(mt.io.read_plate(export).images)


def test_an_export_root_with_several_plates_needs_one_named(make_export) -> None:
    make_export(plate="BR00000001")
    root = make_export(plate="BR00000002").parent

    with pytest.raises(ValueError, match="holds several plates"):
        mt.io.read_plate(root)
    assert mt.io.read_plate(root, "BR00000002").images


@pytest.mark.parametrize(
    ("kwargs", "error", "match"),
    [
        ({"plate": "nope"}, FileNotFoundError, "no plate 'nope'"),
        ({"batch": "2020_01_01_TEST"}, ValueError, "do not apply to a CellProfiler export"),
    ],
)
def test_export_arguments_that_do_not_apply(export: Path, kwargs: dict, error, match: str) -> None:
    with pytest.raises(error, match=match):
        mt.io.read_plate(export.parent, **kwargs)


def test_a_gallery_source_needs_a_batch_and_a_plate(gallery: Path) -> None:
    with pytest.raises(ValueError, match="needs both a plate and a batch"):
        mt.io.read_plate(gallery, PLATE)


def test_a_directory_that_is_neither_layout_says_so(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="neither a Cell Painting Gallery source"):
        mt.io.read_plate(tmp_path)


@pytest.mark.parametrize("layout", ["gallery", "cellprofiler"])
def test_the_layout_can_be_stated_outright(gallery: Path, export: Path, layout: Layout) -> None:
    kwargs: dict[str, Any] = {"batch": BATCH, "plate": PLATE, "profile": "test"} if layout == "gallery" else {}
    path = gallery if layout == "gallery" else export

    assert mt.io.read_plate(path, layout=layout, **kwargs).images
