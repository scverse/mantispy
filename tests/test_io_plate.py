from __future__ import annotations

import os
from pathlib import Path
from typing import cast

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import spatialdata as sd
from _testdata import BATCH, CELL_PAINTING_CHANNELS, EXPORT_SHAPE, FIELDS, OVERLAY_PLATE, PLATE, PREFIX
from skimage.segmentation import find_boundaries

import mantispy as mt
from mantispy.io._gallery import _fov_offsets, _labels_from_outlines, _parse_well


@pytest.fixture
def sdata(gallery: Path) -> sd.SpatialData:
    return mt.io.read_plate(gallery, PLATE, batch=BATCH, profile="test")


def test_without_stage_coordinates_each_field_sits_in_its_own_frame(unlocated_gallery: Path) -> None:
    """Without stage coordinates each field keeps its own frame; the reader does not invent a layout."""
    sdata = mt.io.read_plate(unlocated_gallery, PLATE, batch=BATCH, profile="test")

    assert set(sdata.coordinate_systems) == {f"{PLATE}_{w}_s{s}" for w in ("A01", "B02") for s in (1, 2)}
    assert not sdata.shapes
    assert sdata.tables["wells"].n_obs == 384


def test_a_plate_with_no_images_at_all_raises(gallery: Path) -> None:
    for path in (gallery / "images" / BATCH).rglob("*.tiff"):
        path.unlink()

    with pytest.raises(FileNotFoundError, match="no images for well"):
        mt.io.read_plate(gallery, PLATE, batch=BATCH, wells=["A01"], profile=None)


def test_two_plates_concatenate(sdata: sd.SpatialData, gallery: Path) -> None:
    """Element names carry the plate barcode, so two plates merge without renaming."""
    other = mt.io.read_plate(gallery, OVERLAY_PLATE, batch=BATCH, profile="test")
    assert not set(sdata.images) & set(other.images)

    merged = sd.concatenate([sdata, other], concatenate_tables=True)

    assert len(merged.images) == 8
    assert merged.tables["wells"].n_obs == 768
    assert merged.tables["cells"].n_obs == 8
    assert set(merged.coordinate_systems) >= {PLATE, OVERLAY_PLATE}


def _positions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "well": ["A01", "A01", "A01", "A01", "B02"],
            "x": [-5e-4, 5e-4, -5e-4, 5e-4, -5e-4],
            "y": [5e-4, 5e-4, -5e-4, -5e-4, 5e-4],
        }
    )


def test_fov_offsets_reject_what_they_cannot_place() -> None:
    assert "plate_y" not in _fov_offsets(_positions(), pixel_size=1e-6, plate_format=None)
    with pytest.raises(ValueError, match="outside a 96-well plate"):
        _fov_offsets(_positions().assign(well="Z99"), pixel_size=1e-6, plate_format=96)
    with pytest.raises(ValueError, match="not a well name"):
        _parse_well("well-1")


def _one_object() -> np.ndarray:
    truth = np.zeros((40, 60), np.uint32)
    truth[5:35, 5:30] = 7
    return truth


def _centres(numbers: list[int], x: list[float], y: list[float], area: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {"ObjectNumber": numbers, "Location_Center_X": x, "Location_Center_Y": y, "AreaShape_Area": area}
    )


def test_labels_from_outlines_degenerate_input() -> None:
    outlines = find_boundaries(_one_object(), mode="inner")
    assert _labels_from_outlines(outlines, _centres([7], [17.0], [20.0], [1.0]), area_column=None).any()

    empty = _labels_from_outlines(np.zeros((8, 8), np.uint8), _centres([], [], [], []))
    assert empty.shape == (8, 8)
    assert not empty.any()

    with pytest.raises(ValueError, match="2D outline image"):
        _labels_from_outlines(np.zeros((2, 10, 10)), _centres([1], [1.0], [1.0], [1.0]))


def _outline_dirs(root: Path, plate: str) -> list[Path]:
    return sorted((root / "workspace/analysis" / BATCH / plate / "analysis").glob("*/outlines"))


def test_outlines_the_reader_cannot_name_warn(gallery: Path) -> None:
    """Outline images present but unmatched are a broken read, not a plate without segmentations."""
    for directory in _outline_dirs(gallery, PLATE):
        for path in sorted(directory.glob("A01_s*_outlines.png")):
            path.rename(path.with_name(path.name.replace("A01_s", "A01_site")))

    with pytest.warns(UserWarning, match="could not name"):
        sdata = mt.io.read_plate(gallery, PLATE, batch=BATCH, profile="test")

    assert not sdata.labels


@pytest.mark.parametrize("label_dtype", ["int8", "int32"])
@pytest.mark.parametrize("lazy", [True, False])
def test_arrays_read_back_the_same_whatever_the_width_or_the_backing(make_export, label_dtype: str, lazy: bool) -> None:
    """CellProfiler narrows its label arrays to fit the object count, so two fields of one plate can disagree."""
    sdata = mt.io.read_plate(make_export(label_dtype=label_dtype), lazy=lazy)

    assert sdata.labels[f"{FIELDS[0]}__Cells"].dtype == np.uint32
    assert np.asarray(sdata.images[f"{FIELDS[0]}_image"]).shape == (len(CELL_PAINTING_CHANNELS), *EXPORT_SHAPE)
    assert np.unique(np.asarray(sdata.labels[f"{FIELDS[0]}__Cells"])).tolist() == [0, 1, 2]


def test_a_lazy_read_holds_no_file_descriptor(export: Path) -> None:
    """Each lazy element kept its own descriptor for as long as the object lived, and one 384-well plate holds about ten thousand elements, which exhausts the process limit."""
    if not Path("/dev/fd").is_dir():
        pytest.skip("counting open descriptors needs /dev/fd")
    before = len(os.listdir("/dev/fd"))

    sdata = mt.io.read_plate(export, lazy=True)

    assert len(os.listdir("/dev/fd")) <= before, "a lazy read must not keep the HDF5 files open"
    assert np.asarray(sdata.images[f"{FIELDS[0]}_image"]).shape == (len(CELL_PAINTING_CHANNELS), *EXPORT_SHAPE)


@pytest.mark.parametrize(
    ("drop", "match"),
    [("manifest", "ExportForSpatialData writes the manifest"), ("region", "region")],
)
def test_a_table_the_reader_cannot_use_says_which_module_writes_one(export: Path, drop: str, match: str) -> None:
    path = export / "tables" / f"{PREFIX}.h5ad"
    adata = ad.read_h5ad(path)
    if drop == "manifest":
        del adata.uns["cellprofiler_mapping"]["elements"]
    else:
        del cast("pd.DataFrame", adata.obs)["region"]
        del adata.uns["spatialdata_attrs"]
    adata.write_h5ad(path)

    with pytest.raises(ValueError, match=match):
        mt.io.read_plate(export)


def test_the_export_table_satisfies_the_schema_and_links_by_object_number(export: Path) -> None:
    """The cells table meets schema 2.0 and links to its primary-object labels by region + ObjectNumber (§18, §19)."""
    cells = mt.io.read_plate(export).tables["cells"]

    mt.io.validate(cells, raise_on_error=True)
    assert cells.uns["mantispy"]["resolution"] == "object"
    assert (cells.var["feature_kind"] == "measurement").all()
    assert {"region", "Metadata_ObjectNumber", "Metadata_ImageID"} <= set(cells.obs.columns)
    assert "label_id" not in cells.obs.columns

    attrs = cells.uns[sd.models.TableModel.ATTRS_KEY]
    assert attrs["region_key"] == "region"
    assert attrs["instance_key"] == "Metadata_ObjectNumber"


def test_region_and_object_number_address_a_real_label_instance(export: Path) -> None:
    """Every row's region + Metadata_ObjectNumber is an instance value present in that label element."""
    sdata = mt.io.read_plate(export)
    cells = sdata.tables["cells"]

    region = f"{FIELDS[0]}__Cells"
    numbers = cells.obs.loc[cells.obs["region"] == region, "Metadata_ObjectNumber"].to_numpy()
    present = np.unique(np.asarray(sdata.labels[region]))
    assert set(numbers.tolist()) == {1, 2}
    assert set(numbers.tolist()) <= set(present.tolist())


def test_an_export_root_is_read_through_its_plate_folders(export: Path, make_export) -> None:
    assert set(mt.io.read_plate(export.parent).images) == set(mt.io.read_plate(export).images)

    root = make_export(plate="BR00000002").parent
    with pytest.raises(ValueError, match="holds several plates"):
        mt.io.read_plate(root)
    assert mt.io.read_plate(root, "BR00000002").images


@pytest.mark.parametrize(
    ("kwargs", "error", "match"),
    [
        ({"plate": "nope"}, FileNotFoundError, "no plate 'nope'"),
        ({"batch": BATCH}, ValueError, "do not apply to a CellProfiler export"),
    ],
)
def test_export_arguments_that_do_not_apply(export: Path, kwargs: dict, error, match: str) -> None:
    with pytest.raises(error, match=match):
        mt.io.read_plate(export.parent, **kwargs)


def test_a_layout_that_cannot_be_read(gallery: Path, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="needs both a plate and a batch"):
        mt.io.read_plate(gallery, PLATE)
    blank = tmp_path / "blank"
    blank.mkdir()
    with pytest.raises(ValueError, match="neither a Cell Painting Gallery source"):
        mt.io.read_plate(blank)
