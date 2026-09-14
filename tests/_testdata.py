from __future__ import annotations

import gzip
from pathlib import Path

import anndata as ad
import h5py
import imageio.v3 as iio
import numpy as np
import pandas as pd
from skimage.segmentation import find_boundaries

BATCH = "2020_01_01_TEST"
PLATES = ("BR00000001", "BR00000002")
PLATE, OVERLAY_PLATE = PLATES
CELL_PAINTING_CHANNELS = ("DNA", "RNA")
SHAPE = (32, 32)
PIXEL_SIZE = 1e-6


def _truth(kind: str) -> np.ndarray:
    """Two objects; nuclei sit inside cells, as CellProfiler grows them."""
    labels = np.zeros(SHAPE, np.uint32)
    if kind == "Cells":
        labels[3:16, 3:29] = 1
        labels[17:29, 3:29] = 2
    else:
        labels[7:13, 10:22] = 1
        labels[20:26, 10:22] = 2
    return labels


def _write_site(directory: Path, well: str, site: int, *, overlay: bool) -> None:
    (directory / "outlines").mkdir(parents=True)
    for kind, name in (("Cells", "cell"), ("Nuclei", "nuclei")):
        truth = _truth(kind)
        path = directory / "outlines" / f"{well}_s{site}--{name}_outlines"
        if not overlay:
            iio.imwrite(path.with_suffix(".png"), (find_boundaries(truth, mode="inner") * 255).astype(np.uint8))
            continue
        # the outlines drawn over the greyscale image, and the cell image carries the nuclei outlines too
        image = np.full((*SHAPE, 3), 40, np.uint8)
        image[find_boundaries(truth, mode="inner")] = (0, 255, 255) if kind == "Cells" else (0, 128, 0)
        if kind == "Cells":
            image[find_boundaries(_truth("Nuclei"), mode="inner")] = (0, 128, 0)
        iio.imwrite(path.with_suffix(".tiff"), image)

    for kind in ("Cells", "Nuclei"):
        truth = _truth(kind)
        pd.DataFrame(
            {
                "ImageNumber": [1, 1],
                "ObjectNumber": [1, 2],
                "AreaShape_Center_X": [16.0, 16.0],
                "AreaShape_Center_Y": [float(np.argwhere(truth == n)[:, 0].mean()) for n in (1, 2)],
                "AreaShape_Area": [float((truth == n).sum()) for n in (1, 2)],
                "Intensity_MeanIntensity_DNA": [0.5, 0.25],
            }
        ).to_csv(directory / f"{kind}.csv", index=False)


def write_plate(root: Path, plate: str, *, overlay: bool, located: bool) -> None:
    folder = f"{plate}__2020-01-01T00_00_00-Measurement1"
    images = root / "images" / BATCH / "images" / folder / "Images"
    images.mkdir(parents=True)
    rows = []
    for well, (row, column) in (("A01", (1, 1)), ("B02", (2, 2))):
        for site, (x, y) in enumerate([(-8e-6, 8e-6), (8e-6, -8e-6)], start=1):
            urls = {}
            for index, channel in enumerate(CELL_PAINTING_CHANNELS, start=1):
                name = f"r{row:02d}c{column:02d}f{site:02d}p01-ch{index}.tiff"
                iio.imwrite(images / name, np.full(SHAPE, index * 100, np.uint16))
                urls[f"URL_Orig{channel}"] = f"s3://bucket/acc/source_1/images/{BATCH}/images/{folder}/Images/{name}"
            stage = (
                {
                    "Metadata_PositionX": x,
                    "Metadata_PositionY": y,
                    "Metadata_ImageResolutionX": PIXEL_SIZE,
                    "Metadata_ImageResolutionY": PIXEL_SIZE,
                }
                if located
                else {}
            )
            rows.append(
                {
                    **urls,
                    # an illumination-correction column, which must not be read as a channel
                    "URL_IllumDNA": "ignored",
                    "Metadata_Plate": plate,
                    "Metadata_Well": well,
                    "Metadata_Site": site,
                    **stage,
                    "Metadata_ImageSizeX": SHAPE[1],
                    "Metadata_ImageSizeY": SHAPE[0],
                }
            )
    load_data = root / "workspace/load_data_csv" / BATCH / plate
    load_data.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(load_data / "load_data.csv", index=False)

    analysis = root / "workspace/analysis" / BATCH / plate / "analysis"
    for site in (1, 2):
        _write_site(analysis / f"{plate}-A01-{site}", "A01", site, overlay=overlay)

    profiles = root / "workspace/profiles" / BATCH / plate
    profiles.mkdir(parents=True)
    plate_map = pd.DataFrame(
        {
            "Metadata_Plate": [plate] * 384,
            "Metadata_Well": [f"{chr(ord('A') + r)}{c + 1:02d}" for r in range(16) for c in range(24)],
            "Cells_AreaShape_Area": np.arange(384, dtype=float),
        }
    )
    with gzip.open(profiles / f"{plate}_test.csv.gz", "wt") as handle:
        plate_map.to_csv(handle, index=False)


PREFIX = "Testrun"
EXPORT_OBJECTS = ("Nuclei", "Cells")
EXPORT_SHAPE = (8, 16)
FIELDS = ("A01_01_img1", "A01_02_img2")
ELEMENT_COLUMNS = (
    "sample_key",
    "image_number",
    "element_type",
    "element_name",
    "path",
    "shape",
    "element_dtype",
    "region_key_value",
    "status",
    "error",
)


def _write_h5(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("data", data=array, compression="gzip", compression_opts=1)


def _export_labels(n: int, dtype: str) -> np.ndarray:
    labels = np.zeros(EXPORT_SHAPE, dtype)
    for i in range(n):
        labels[2 * i + 1, 1 : 4 + i] = i + 1
    return labels


def build_export(tmp_path: Path, *, plate: str = PLATE, failed: str | None = None, label_dtype: str = "int8") -> Path:
    """Build one plate folder as ExportForSpatialData writes it, manifest and all.

    `failed` names a field whose arrays are not written, the way a cycle that raised leaves them, so its
    manifest rows carry a status of `failed` and an error.
    """
    folder = tmp_path / f"{PREFIX}_export" / plate
    rows, obs = [], []
    for number, field in enumerate(FIELDS, start=1):
        broken = field == failed
        error = "RuntimeError: the cycle failed" if broken else ""
        if not broken:
            _write_h5(
                folder / "images" / f"{field}.h5", np.full((len(CELL_PAINTING_CHANNELS), *EXPORT_SHAPE), 0.5, "float32")
            )
        rows.append(
            {
                "sample_key": field,
                "image_number": number,
                "element_type": "image",
                "element_name": "",
                "path": f"images/{field}.h5",
                "shape": "" if broken else f"{len(CELL_PAINTING_CHANNELS)},{EXPORT_SHAPE[0]},{EXPORT_SHAPE[1]}",
                "element_dtype": "" if broken else "float32",
                "region_key_value": "",
                "status": "failed" if broken else "ok",
                "error": error,
            }
        )
        for obj in EXPORT_OBJECTS:
            if not broken:
                _write_h5(folder / "labels" / field / f"{obj}.h5", _export_labels(2, label_dtype))
            rows.append(
                {
                    "sample_key": field,
                    "image_number": number,
                    "element_type": "labels",
                    "element_name": obj,
                    "path": f"labels/{field}/{obj}.h5",
                    "shape": "" if broken else f"{EXPORT_SHAPE[0]},{EXPORT_SHAPE[1]}",
                    "element_dtype": "" if broken else label_dtype,
                    "region_key_value": f"{field}__{obj}",
                    "status": "failed" if broken else "ok",
                    "error": error,
                }
            )
        obs += [{"region_key": f"{field}__Cells", "label_id": i + 1, "ImageNumber": number} for i in range(2)]

    adata = ad.AnnData(
        np.arange(len(obs) * 3, dtype="float32").reshape(len(obs), 3),
        obs=pd.DataFrame(obs).astype({"label_id": "int32", "ImageNumber": "int32"}),
        var=pd.DataFrame(index=["Cells__AreaShape_Area", "Cells__Intensity_MeanIntensity_DNA", "Nuclei__A"]),
    )
    adata.obs_names = [f"{str(row['region_key']).rsplit('__', 1)[0]}_{row['label_id']}" for row in obs]
    adata.uns["cellprofiler_mapping"] = {
        "elements": pd.DataFrame(rows, columns=list(ELEMENT_COLUMNS)),
        "image_channels": pd.DataFrame(
            {"channel": list(CELL_PAINTING_CHANNELS), "stack_index": range(len(CELL_PAINTING_CHANNELS))}
        ),
    }
    adata.uns["spatialdata_attrs"] = {
        "region": sorted(set(adata.obs["region_key"])),
        "region_key": "region_key",
        "instance_key": "label_id",
    }
    (folder / "tables").mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(folder / "tables" / f"{PREFIX}.h5ad")
    return folder
