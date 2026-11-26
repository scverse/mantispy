from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
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
from mantispy._core.frames import as_frame
from mantispy.io._gallery import _fov_offsets, _labels_from_outlines, _parse_well


@pytest.fixture
def sdata(gallery: Path) -> sd.SpatialData:
    return mt.io.read_cellprofiler_gallery(gallery, batch=BATCH, plate=PLATE, profile="test")


def test_without_stage_coordinates_each_field_sits_in_its_own_frame(unlocated_gallery: Path) -> None:
    """Without stage coordinates each field keeps its own frame; the reader does not invent a layout."""
    sdata = mt.io.read_cellprofiler_gallery(unlocated_gallery, batch=BATCH, plate=PLATE, profile="test")

    assert set(sdata.coordinate_systems) == {f"{PLATE}_{w}_s{s}" for w in ("A01", "B02") for s in (1, 2)}
    assert not sdata.shapes
    assert sdata.tables["wells"].n_obs == 384


def test_a_plate_with_no_images_at_all_raises(gallery: Path) -> None:
    for path in (gallery / "images" / BATCH).rglob("*.tiff"):
        path.unlink()

    with pytest.raises(FileNotFoundError, match="no images for well"):
        mt.io.read_cellprofiler_gallery(gallery, batch=BATCH, plate=PLATE, wells=["A01"], profile=None)


def test_two_plates_concatenate(sdata: sd.SpatialData, gallery: Path) -> None:
    """Element names carry the plate barcode, so two plates merge without renaming."""
    other = mt.io.read_cellprofiler_gallery(gallery, batch=BATCH, plate=OVERLAY_PLATE, profile="test")
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


def test_outlines_the_reader_cannot_name_warn(gallery: Path, caplog) -> None:
    """Outline images present but unmatched are a broken read, not a plate without segmentations."""
    for directory in _outline_dirs(gallery, PLATE):
        for path in sorted(directory.glob("A01_s*_outlines.png")):
            path.rename(path.with_name(path.name.replace("A01_s", "A01_site")))

    with caplog.at_level(logging.WARNING, logger="scverse.mantispy"):
        sdata = mt.io.read_cellprofiler_gallery(gallery, batch=BATCH, plate=PLATE, profile="test")
    assert "could not name" in caplog.text

    assert not sdata.labels


@pytest.mark.parametrize("label_dtype", ["int8", "int32"])
@pytest.mark.parametrize("lazy", [True, False])
def test_arrays_read_back_the_same_whatever_the_width_or_the_backing(make_export, label_dtype: str, lazy: bool) -> None:
    """CellProfiler narrows its label arrays to fit the object count, so two fields of one plate can disagree."""
    sdata = mt.io.read_cellprofiler_zarr(make_export(label_dtype=label_dtype), lazy=lazy)

    assert sdata.labels[f"{FIELDS[0]}__Cells"].dtype == np.uint32
    assert np.asarray(sdata.images[f"{FIELDS[0]}_image"]).shape == (len(CELL_PAINTING_CHANNELS), *EXPORT_SHAPE)
    assert np.unique(np.asarray(sdata.labels[f"{FIELDS[0]}__Cells"])).tolist() == [0, 1, 2]


def test_a_lazy_read_holds_no_file_descriptor(export: Path) -> None:
    """Each lazy element kept its own descriptor for as long as the object lived, and one 384-well plate holds about ten thousand elements, which exhausts the process limit."""
    if not Path("/dev/fd").is_dir():
        pytest.skip("counting open descriptors needs /dev/fd")
    before = len(os.listdir("/dev/fd"))

    sdata = mt.io.read_cellprofiler_zarr(export, lazy=True)

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
        mt.io.read_cellprofiler_zarr(export)


def test_the_export_table_satisfies_the_schema_and_links_by_object_number(export: Path) -> None:
    """The cells table meets schema 2.0 and links to its primary-object labels by region + ObjectNumber (§18, §19)."""
    cells = mt.io.read_cellprofiler_zarr(export).tables["cells"]

    mt.io.validate(cells, raise_on_error=True)
    assert cells.uns["mantispy"]["resolution"] == "object"
    assert (cells.var["feature_kind"] == "measurement").all()
    assert {"region", "Metadata_ObjectNumber", "Metadata_ImageID"} <= set(cells.obs.columns)
    assert "label_id" not in cells.obs.columns

    attrs = cells.uns[sd.models.TableModel.ATTRS_KEY]
    assert attrs["region_key"] == "region"
    assert attrs["instance_key"] == "Metadata_ObjectNumber"


def test_a_table_in_the_modules_own_layout_links_every_row_to_its_own_instance(export: Path) -> None:
    """ExportForSpatialData writes region_key (the label element), label_id (the instance) and region (the object set).

    The rows are reversed, so linking by row order would attach every measurement to another object.
    """
    path = export / "tables" / f"{PREFIX}.h5ad"
    adata = ad.read_h5ad(path)
    obs = cast("pd.DataFrame", adata.obs)
    truth = {
        (str(region), int(number)): float(value)
        for region, number, value in zip(
            obs["region"], obs["Metadata_ObjectNumber"], np.asarray(adata.X)[:, 0], strict=True
        )
    }
    layout = obs.rename(columns={"region": "region_key", "Metadata_ObjectNumber": "label_id"})
    layout["region"] = "Cells"
    adata = ad.AnnData(
        X=np.asarray(adata.X)[::-1].copy(),
        obs=layout.iloc[::-1].copy(),
        var=cast("pd.DataFrame", adata.var),
        uns=adata.uns,
    )
    adata.uns["spatialdata_attrs"] = {
        "region": sorted(set(layout["region_key"])),
        "region_key": "region_key",
        "instance_key": "label_id",
    }
    adata.write_h5ad(path)

    cells = mt.io.read_cellprofiler_zarr(export).tables["cells"]
    mt.io.validate(cells, raise_on_error=True)
    read = {
        (str(region), int(number)): float(value)
        for region, number, value in zip(
            cells.obs["region"], cells.obs["Metadata_ObjectNumber"], np.asarray(cells.X)[:, 0], strict=True
        )
    }
    assert read == truth
    assert set(cells.obs["Metadata_ObjectType"]) == {"Cells"}


def test_region_and_object_number_address_a_real_label_instance(export: Path) -> None:
    """Every row's region + Metadata_ObjectNumber is an instance value present in that label element."""
    sdata = mt.io.read_cellprofiler_zarr(export)
    cells = sdata.tables["cells"]

    region = f"{FIELDS[0]}__Cells"
    obs = as_frame(cells.obs)
    numbers = obs.loc[obs["region"] == region, "Metadata_ObjectNumber"].to_numpy()
    present = np.unique(np.asarray(sdata.labels[region]))
    assert set(numbers.tolist()) == {1, 2}
    assert set(numbers.tolist()) <= set(present.tolist())


def test_an_export_root_is_read_through_its_plate_folders(export: Path, make_export) -> None:
    assert set(mt.io.read_cellprofiler_zarr(export.parent).images) == set(mt.io.read_cellprofiler_zarr(export).images)

    root = make_export(plate="BR00000002").parent
    with pytest.raises(ValueError, match="holds several plates"):
        mt.io.read_cellprofiler_zarr(root)
    assert mt.io.read_cellprofiler_zarr(root, plate="BR00000002").images


def test_an_unknown_plate_of_an_export_root_raises(export: Path) -> None:
    with pytest.raises(FileNotFoundError, match="no plate 'nope'"):
        mt.io.read_cellprofiler_zarr(export.parent, plate="nope")


def test_each_reader_points_at_the_other_layout(gallery: Path, export: Path, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="read_cellprofiler_gallery"):
        mt.io.read_cellprofiler_zarr(gallery)
    with pytest.raises(ValueError, match="read_cellprofiler_zarr"):
        mt.io.read_cellprofiler_gallery(export, batch=BATCH, plate=PLATE)
    blank = tmp_path / "blank"
    blank.mkdir()
    with pytest.raises(ValueError, match="not an ExportForSpatialData export"):
        mt.io.read_cellprofiler_zarr(blank)
    with pytest.raises(ValueError, match="not a Cell Painting Gallery source"):
        mt.io.read_cellprofiler_gallery(blank, batch=BATCH, plate=PLATE)


def test_read_cellprofiler_reads_a_gallery_plate_from_its_csv_exports_without_images(unlocated_gallery: Path) -> None:
    """Every cell of every analysed site, as the CSVs hold them; reading through the masks loaded each image and dropped cells."""
    shutil.rmtree(unlocated_gallery / "images")
    sites = sorted((unlocated_gallery / "workspace/analysis" / BATCH / PLATE / "analysis").iterdir())
    cells = pd.concat([pd.read_csv(site / "Cells.csv") for site in sites], ignore_index=True)

    adata = mt.io.read_cellprofiler(unlocated_gallery)
    assert adata.n_obs == len(cells)
    np.testing.assert_array_equal(
        np.asarray(adata[:, "Cells_AreaShape_Area"].X).ravel(), cells["AreaShape_Area"].to_numpy()
    )
    assert set(adata.obs["Metadata_Site"]) == {1, 2}
    assert adata.obs_names[0].startswith(f"{sites[0].name}:")
    mt.io.validate(adata, raise_on_error=True)
    assert mt.io.read_cellprofiler(unlocated_gallery, aggregate="count").n_obs == len(cells)


def test_read_cellprofiler_profiles_the_cell_table_of_an_export(export: Path, make_export) -> None:
    """read_cellprofiler returns the cell table the zarr reader reads, as one AnnData, from a plate folder or its root."""
    cases = {
        export: mt.io.read_cellprofiler_zarr(export),
        export.parent: mt.io.read_cellprofiler_zarr(export),
    }
    for path, sdata in cases.items():
        adata = mt.io.read_cellprofiler(path)
        expected = sdata.tables["cells"]
        assert isinstance(adata, ad.AnnData)
        assert list(adata.obs_names) == list(expected.obs_names)
        assert list(adata.var_names) == list(expected.var_names)
        np.testing.assert_array_equal(np.asarray(adata.X), np.asarray(expected.X))
        assert "spatialdata_attrs" not in adata.uns
        assert adata.uns["mantispy"]["params"]["read_cellprofiler"]["path"] == str(path)
        mt.io.validate(adata, raise_on_error=True)

    with pytest.raises(ValueError, match="aggregate= needs one table per object"):
        mt.io.read_cellprofiler(export, aggregate="mean")
    with pytest.raises(ValueError, match="holds one table, of Cells"):
        mt.io.read_cellprofiler(export, root="Nuclei")
    with pytest.raises(ValueError, match="reads on its own"):
        mt.io.read_cellprofiler([export, export])
    make_export(plate="BR00000002")
    with pytest.raises(ValueError, match="pass one plate folder"):
        mt.io.read_cellprofiler(export.parent)


def test_read_cellprofiler_of_a_gallery_of_several_plates_names_them(gallery: Path) -> None:
    with pytest.raises(ValueError, match=r"holds 2 plates.*read_cellprofiler_gallery"):
        mt.io.read_cellprofiler(gallery)


def test_image_root_finds_the_arrays_of_a_moved_export(export: Path, tmp_path_factory: pytest.TempPathFactory) -> None:
    """The table stays behind while its arrays move, so the manifest's relative paths resolve under image_root."""
    expected = mt.io.read_cellprofiler_zarr(export, lazy=False)
    moved = tmp_path_factory.mktemp("arrays")
    for name in ("images", "labels"):
        shutil.move(export / name, moved / name)

    with pytest.raises(FileNotFoundError):
        mt.io.read_cellprofiler_zarr(export)
    sdata = mt.io.read_cellprofiler_zarr(export, lazy=False, image_root=moved)

    assert set(sdata.images) == set(expected.images)
    assert set(sdata.labels) == set(expected.labels)
    for name in expected.labels:
        np.testing.assert_array_equal(np.asarray(sdata.labels[name]), np.asarray(expected.labels[name]))
    assert sdata.tables["cells"].n_obs == expected.tables["cells"].n_obs


def test_image_root_finds_gallery_images_kept_apart(
    sdata: sd.SpatialData, gallery: Path, tmp_path_factory: pytest.TempPathFactory
) -> None:
    moved = tmp_path_factory.mktemp("images") / "images"
    shutil.move(gallery / "images", moved)

    with pytest.raises(FileNotFoundError, match="no images for well"):
        mt.io.read_cellprofiler_gallery(gallery, batch=BATCH, plate=PLATE, profile="test")
    read = mt.io.read_cellprofiler_gallery(gallery, batch=BATCH, plate=PLATE, profile="test", image_root=moved)

    assert set(read.images) == set(sdata.images)
    assert set(read.labels) == set(sdata.labels)
    assert read.tables["cells"].n_obs == sdata.tables["cells"].n_obs


def test_without_spatialdata_mantispy_imports_and_the_readers_name_the_extra(export: Path) -> None:
    """Importing mantispy must not need spatialdata, and the readers that do say which extra installs it."""
    script = f"""
import importlib.abc
import sys


class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] == "spatialdata":
            raise ImportError(f"blocked {{name}}")


sys.meta_path.insert(0, Block())
import mantispy as mt

readers = {{
    mt.io.read_cellprofiler: {{}},
    mt.io.read_cellprofiler_zarr: {{}},
    mt.io.read_cellprofiler_gallery: {{"batch": "b", "plate": "p"}},
}}
for reader, kwargs in readers.items():
    try:
        reader({str(export)!r}, **kwargs)
    except ImportError as error:
        assert "pip install 'mantispy[images]'" in str(error), error
    else:
        raise AssertionError(f"{{reader.__name__}} read without spatialdata")
assert "spatialdata" not in sys.modules
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
