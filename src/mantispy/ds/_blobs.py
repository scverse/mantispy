from __future__ import annotations

from typing import TYPE_CHECKING

import anndata as ad
import numpy as np
import pandas as pd

from mantispy._core._reduce import get_matrix
from mantispy._core.features import parse_feature_names
from mantispy._core.frames import as_frame
from mantispy._core.schema import stamp

if TYPE_CHECKING:
    import numpy.typing as npt
    from spatialdata import SpatialData

CELL_PAINTING_CHANNELS = ("DNA", "RNA", "AGP", "ER", "Mito")
PIXEL_SIZE_METRES = 1e-6
WELL_PITCH_METRES = 9.0e-3
WELL_RADIUS_METRES = 1.65e-3

_COMPARTMENTS = ("Cells", "Nuclei", "Cytoplasm")


def _wells(n_wells: int) -> list[str]:
    return [f"{chr(ord('A') + index // 12)}{index % 12 + 1:02d}" for index in range(n_wells)]


def _draw_field(
    shape: tuple[int, int], n_cells: int, rng: np.random.Generator
) -> tuple[npt.NDArray[np.uint32], npt.NDArray[np.uint32]]:
    """Place `n_cells` non-overlapping round cells, each with a concentric nucleus."""
    cells = np.zeros(shape, np.uint32)
    nuclei = np.zeros(shape, np.uint32)
    grid_y, grid_x = np.mgrid[: shape[0], : shape[1]]
    number = 0
    for _ in range(20 * n_cells):
        if number == n_cells:
            break
        radius = rng.integers(5, 9)
        y = rng.integers(radius, shape[0] - radius)
        x = rng.integers(radius, shape[1] - radius)
        disc = (grid_y - y) ** 2 + (grid_x - x) ** 2 <= radius**2
        if cells[disc].any():
            continue
        number += 1
        cells[disc] = number
        nuclei[(grid_y - y) ** 2 + (grid_x - x) ** 2 <= (radius / 2) ** 2] = number
    return cells, nuclei


def _draw_image(
    cells: npt.NDArray[np.uint32], nuclei: npt.NDArray[np.uint32], rng: np.random.Generator
) -> npt.NDArray[np.float32]:
    """Stain the objects: DNA in the nucleus, the rest in the cytoplasm, over a noisy background."""
    cytoplasm = (cells > 0) & (nuclei == 0)
    image = rng.normal(0.05, 0.01, (len(CELL_PAINTING_CHANNELS), *cells.shape)).astype(np.float32)
    for index, channel in enumerate(CELL_PAINTING_CHANNELS):
        brightness = 0.9 if channel == "DNA" else 0.3
        image[index][nuclei > 0] += brightness
        image[index][cytoplasm] += 0.1 if channel == "DNA" else 0.6 * rng.uniform(0.5, 1.0)
    return image.clip(0, None)


def _measure(image: npt.NDArray, masks: dict[str, npt.NDArray], numbers: npt.NDArray) -> pd.DataFrame:
    """Per-object features under CellProfiler names."""
    from scipy import ndimage as ndi

    columns = {}
    for compartment, mask in masks.items():
        columns[f"{compartment}_AreaShape_Area"] = np.bincount(mask.ravel(), minlength=numbers.max() + 1)[numbers]
        for index, channel in enumerate(CELL_PAINTING_CHANNELS):
            means = ndi.mean(image[index], labels=mask, index=numbers)
            columns[f"{compartment}_Intensity_MeanIntensity_{channel}"] = np.nan_to_num(means)
    dna, rna = columns["Cells_Intensity_MeanIntensity_DNA"], columns["Cells_Intensity_MeanIntensity_RNA"]
    columns["Cells_Correlation_Correlation_DNA_RNA"] = dna * rna
    return pd.DataFrame(columns, dtype=np.float32)


def blobs(
    *,
    n_wells: int = 4,
    n_sites: int = 2,
    n_cells: int = 12,
    shape: tuple[int, int] = (64, 64),
    plate: str = "BLOBS01",
    seed: int = 0,
) -> SpatialData:
    """A small synthetic Cell Painting plate, laid out like one :func:`mantispy.io.read_plate` returns.

    Round cells with a concentric nucleus are stained in the five Cell Painting channels and measured.
    The result is complete without a download: Images, Labels for the three compartments, well Shapes, and ``cells`` and ``wells`` Tables whose features carry CellProfiler names.

    Every element sits in three coordinate systems, ``{plate}_{well}_s{site}``, ``{plate}_{well}`` and ``{plate}``, laying the fields of a well out as a mosaic and the wells as a plate map.

    Args:
        n_wells: Wells to simulate, filled across the rows of a 96-well plate from ``A01``.
        n_sites: Fields of view per well, laid out in a square mosaic.
        n_cells: Cells per field, or as many as fit without overlapping.
        shape: Pixel height and width of one field.
        plate: Plate barcode, which every element name is prefixed with.
        seed: Seed of the random generator.
            Two calls with one seed give the same plate.

    Returns:
        The plate.

    Examples:
        >>> import mantispy as mt
        >>> sdata = mt.ds.blobs()  # doctest: +SKIP
        >>> sdata.tables["cells"].var["channel"].value_counts()  # doctest: +SKIP
    """
    from geopandas import GeoDataFrame
    from shapely import Point
    from spatialdata import SpatialData
    from spatialdata.models import Image2DModel, Labels2DModel, ShapesModel, TableModel
    from spatialdata.transformations import Identity, Translation

    rng = np.random.default_rng(seed)
    wells = _wells(n_wells)
    columns = int(np.ceil(np.sqrt(n_sites)))

    images, labels, frames, obs = {}, {}, [], []
    for well in wells:
        row, column = ord(well[0]) - ord("A"), int(well[1:]) - 1
        for site in range(1, n_sites + 1):
            field = f"{plate}_{well}_s{site}"
            cells, nuclei = _draw_field(shape, n_cells, rng)
            masks = {"cells": cells, "nuclei": nuclei, "cytoplasm": np.where(nuclei > 0, 0, cells)}
            image = _draw_image(cells, nuclei, rng)

            offset = np.array([(site - 1) // columns, (site - 1) % columns]) * np.array(shape)
            plate_offset = offset + np.array([row, column]) * (WELL_PITCH_METRES / PIXEL_SIZE_METRES)
            transformations = {
                field: Identity(),
                f"{plate}_{well}": Translation(offset.astype(float), axes=("y", "x")),
                plate: Translation(plate_offset, axes=("y", "x")),
            }
            images[f"{field}_image"] = Image2DModel.parse(
                image, dims=("c", "y", "x"), c_coords=list(CELL_PAINTING_CHANNELS), transformations=transformations
            )
            for name, mask in masks.items():
                labels[f"{field}_{name}"] = Labels2DModel.parse(mask, dims=("y", "x"), transformations=transformations)

            numbers = np.unique(cells)[1:]
            frame = _measure(image, {c: masks[c.lower()] for c in _COMPARTMENTS}, numbers)
            frames.append(frame)
            obs.append(
                pd.DataFrame(
                    {
                        "Metadata_Plate": plate,
                        "Metadata_Well": well,
                        "Metadata_Site": site,
                        "Metadata_ObjectNumber": numbers.astype(np.int32),
                        "region": f"{field}_cells",
                    }
                )
            )

    cells_obs = pd.concat(obs, ignore_index=True).astype(
        {"Metadata_Plate": "category", "Metadata_Well": "category", "region": "category"}
    )
    cells_obs.index = pd.Index(cells_obs["region"].astype(str) + ":" + cells_obs["Metadata_ObjectNumber"].astype(str))
    var = parse_feature_names(list(frames[0].columns), channels=CELL_PAINTING_CHANNELS)
    table = ad.AnnData(pd.concat(frames, ignore_index=True).to_numpy(np.float32), obs=cells_obs, var=var)
    stamp(table, resolution="cell")
    tables = {
        "cells": TableModel.parse(
            table,
            region=sorted(cells_obs["region"].cat.categories),
            region_key="region",
            instance_key="Metadata_ObjectNumber",
        ),
        "wells": _well_table(table, plate=plate, wells=wells),
    }

    points = [
        Point(column * WELL_PITCH_METRES / PIXEL_SIZE_METRES, row * WELL_PITCH_METRES / PIXEL_SIZE_METRES)
        for row in range(8)
        for column in range(12)
    ]
    shapes = GeoDataFrame({"radius": WELL_RADIUS_METRES / PIXEL_SIZE_METRES}, geometry=points, index=np.arange(96))
    return SpatialData(
        images=images,
        labels=labels,
        shapes={f"{plate}_wells": ShapesModel.parse(shapes, transformations={plate: Identity()})},
        tables=tables,
    )


def _well_table(cells: ad.AnnData, *, plate: str, wells: list[str]) -> ad.AnnData:
    from spatialdata.models import TableModel

    frame = pd.DataFrame(get_matrix(cells), columns=cells.var_names)
    frame["Metadata_Well"] = cells.obs["Metadata_Well"].to_numpy()
    means = frame.groupby("Metadata_Well", observed=True).mean().reindex(wells)
    obs = pd.DataFrame(
        {
            "Metadata_Plate": pd.Categorical([plate] * len(wells)),
            "Metadata_Well": pd.Categorical(wells),
            "well_index": [(ord(w[0]) - ord("A")) * 12 + int(w[1:]) - 1 for w in wells],
            "region": pd.Categorical([f"{plate}_wells"] * len(wells)),
        },
        index=pd.Index([f"{plate}:{well}" for well in wells]),
    )
    adata = ad.AnnData(means.to_numpy(np.float32), obs=obs, var=as_frame(cells.var).copy())
    stamp(adata, resolution="well")
    return TableModel.parse(adata, region=f"{plate}_wells", region_key="region", instance_key="well_index")
