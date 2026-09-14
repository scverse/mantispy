from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple, cast

import anndata as ad
import numpy as np
import pandas as pd

from mantispy.io._profiles import read_profiles

if TYPE_CHECKING:
    import numpy.typing as npt
    from geopandas import GeoDataFrame
    from spatialdata import SpatialData


class PlateFormat(NamedTuple):
    n_rows: int
    n_columns: int
    well_pitch_metres: float


PLATE_FORMATS: Mapping[int, PlateFormat] = {
    96: PlateFormat(8, 12, 9.0e-3),
    384: PlateFormat(16, 24, 4.5e-3),
    1536: PlateFormat(32, 48, 2.25e-3),
}

WELL_RADIUS_METRES: float = 1.65e-3

_WELL = re.compile(r"([A-Za-z]+)(\d+)")


def _parse_well(well: str) -> tuple[int, int]:
    """Split a well name into its zero-based row and column.

    Rows count like spreadsheet columns: ``A`` is 0, ``Z`` is 25, ``AA`` is 26, as 1536-well plates are named.

    Args:
        well: A well name such as ``A01``, ``P24`` or ``AF48``.

    Returns:
        The zero-based row and column.

    Raises:
        ValueError: The name is not letters followed by digits.
    """
    match = _WELL.fullmatch(well.strip())
    if match is None:
        msg = f"not a well name: {well!r}"
        raise ValueError(msg)
    letters, digits = match.groups()
    row = 0
    for letter in letters.upper():
        row = row * 26 + ord(letter) - ord("A") + 1
    return row - 1, int(digits) - 1


def _fov_offsets(positions: pd.DataFrame, *, pixel_size: float, plate_format: int | None = 384) -> pd.DataFrame:
    """Pixel offsets of the top-left corner of each field of view, within its well and within the plate.

    ``y`` points down, unlike in the stage coordinates, so the offsets serve directly as ``(y, x)`` translations of an image whose row index grows downwards.
    The origin is the top-left corner of a well's top-left field, shared across wells, which lays every well of a plate out in one frame.

    Args:
        positions: One row per field of view, with columns ``well``, and ``x`` and ``y`` holding the stage coordinates of the field centre in metres, relative to the centre of its well and with ``y`` pointing up.
        pixel_size: Size of a pixel in metres, as ``Metadata_ImageResolutionX`` records it.
        plate_format: Number of wells on the plate, used to place the wells on their nominal grid.
            This assumes the standard well pitch of that format rather than anything measured; pass ``None`` to leave the plate offsets out.

    Returns:
        The offsets in pixels, indexed like `positions`, with columns ``well_y`` and ``well_x``, and ``plate_y`` and ``plate_x`` unless `plate_format` is ``None``.

    Raises:
        ValueError: The plate format is unknown, or a well falls outside a plate of that format.
    """
    x = positions["x"].to_numpy(float) / pixel_size
    y = positions["y"].to_numpy(float) / pixel_size
    offsets = pd.DataFrame({"well_y": y.max() - y, "well_x": x - x.min()}, index=positions.index)
    if plate_format is not None:
        if plate_format not in PLATE_FORMATS:
            msg = f"unknown plate format {plate_format}; known: {sorted(PLATE_FORMATS)}"
            raise ValueError(msg)
        layout = PLATE_FORMATS[plate_format]
        grid = np.array([_parse_well(well) for well in positions["well"]])
        if (grid < 0).any() or (grid >= [layout.n_rows, layout.n_columns]).any():
            msg = f"wells fall outside a {plate_format}-well plate"
            raise ValueError(msg)
        offsets[["plate_y", "plate_x"]] = offsets[["well_y", "well_x"]].to_numpy() + grid * (
            layout.well_pitch_metres / pixel_size
        )
    return offsets


def _labels_from_outlines(
    outlines: npt.ArrayLike,
    centres: pd.DataFrame,
    *,
    x_column: str = "Location_Center_X",
    y_column: str = "Location_Center_Y",
    object_column: str = "ObjectNumber",
    area_column: str | None = "AreaShape_Area",
    max_area_ratio: float = 1.25,
) -> npt.NDArray[np.uint32]:
    """Turn a CellProfiler outline image back into a label image carrying the CellProfiler object numbers.

    The gallery publishes outlines, not masks.
    Outlines are one pixel wide and shared between touching objects, so filling them and labelling connected components separates the interiors.
    Each component takes the object number of the centroid inside it; components with no centroid are dropped.
    Growing the boundary back one pixel reproduces the CellProfiler areas to under a percent.

    A component is accepted only when exactly one centroid falls in it and its area is within `max_area_ratio` of the area CellProfiler measured.
    This rejects the two failures of an unclosed outline: two objects merging into one component, and a centroid landing in the background or in a fragment.
    Objects whose component was rejected are absent, so compare the label count against the centroid count.

    Args:
        outlines: A 2D outline image, anything above zero being outline.
        centres: One row per object, as CellProfiler wrote it, with the centroid and object number columns below.
        x_column: Column of `centres` holding the centroid column index.
        y_column: Column of `centres` holding the centroid row index.
        object_column: Column of `centres` holding the object number, which becomes the label value.
        area_column: Column of `centres` holding the area CellProfiler measured, used to reject components that cannot be the object.
            Pass ``None``, or leave the column out of `centres`, to skip that check.
        max_area_ratio: How far a component's area may differ from the measured area, either way, and still be accepted.
            Reconstruction is exact to a fraction of a percent when it works, so the default is tight.

    Returns:
        A label image the shape of `outlines`, zero outside objects.

    Raises:
        ValueError: `outlines` is not 2D.
    """
    from scipy import ndimage as ndi
    from skimage.segmentation import expand_labels

    mask = np.asarray(outlines) > 0
    if mask.ndim != 2:
        msg = f"expected a 2D outline image, got shape {mask.shape}"
        raise ValueError(msg)
    if centres.empty:
        return np.zeros(mask.shape, np.uint32)
    components, n_components = ndi.label(ndi.binary_fill_holes(~mask) & ~mask)

    rows = np.rint(centres[y_column].to_numpy(float)).astype(int).clip(0, mask.shape[0] - 1)
    columns = np.rint(centres[x_column].to_numpy(float)).astype(int).clip(0, mask.shape[1] - 1)
    numbers = centres[object_column].to_numpy()
    areas = centres[area_column].to_numpy(float) if area_column is not None and area_column in centres else None
    hit = components[rows, columns]
    if areas is not None:
        # An interior is always smaller than the object, so only the upper bound can reject anything.
        sizes = np.bincount(components.ravel(), minlength=n_components + 1)
        hit = np.where(sizes[hit] > max_area_ratio * areas, 0, hit)
    claims = np.bincount(hit, minlength=n_components + 1)
    hit = np.where(claims[hit] > 1, 0, hit)

    lookup = np.zeros(n_components + 1, dtype=np.uint32)
    lookup[hit] = numbers
    lookup[0] = 0
    labels = expand_labels(lookup[components], distance=1).astype(np.uint32)

    if areas is not None:
        grown = np.bincount(labels.ravel(), minlength=int(numbers.max()) + 1)[numbers] / areas
        rejected = numbers[(grown > max_area_ratio) | (grown < 1 / max_area_ratio)]
        labels[np.isin(labels, rejected)] = 0
    return labels


def _load_data(root: Path, batch: str, plate: str) -> pd.DataFrame:
    frame = pd.read_csv(root / "workspace/load_data_csv" / batch / plate / "load_data.csv")
    if "Metadata_Well" not in frame.columns:
        msg = "load_data.csv does not name the well of each field"
        raise ValueError(msg)
    return frame.set_index(["Metadata_Well", "Metadata_Site"]).rename_axis(["well", "site"]).sort_index()


def _pixel_size(load_data: pd.DataFrame) -> float:
    sizes = np.unique(load_data[["Metadata_ImageResolutionX", "Metadata_ImageResolutionY"]].to_numpy().round(12))
    if sizes.size != 1:
        msg = f"the plate mixes pixel sizes: {sizes}"
        raise ValueError(msg)
    return float(sizes.item())


def _channels(load_data: pd.DataFrame) -> tuple[str, list[str]]:
    """The column prefix the source records images under, and the channel names below it."""
    for prefix in ("URL_Orig", "FileName_Orig", "URL_", "FileName_"):
        names = [c.removeprefix(prefix) for c in load_data.columns if c.startswith(prefix)]
        names = [name for name in names if not name.startswith("Illum")]
        if names:
            return prefix, names
    msg = "load_data.csv does not name the image of each channel"
    raise ValueError(msg)


def _image_path(root: Path, batch: str, row: pd.Series, prefix: str, channel: str) -> Path:
    """Locate a channel of one field under `root`.

    Sources record images either as an ``URL_*`` S3 URI or as a ``FileName_*``/``PathName_*`` pair pointing at wherever the images sat when CellProfiler ran.
    Both end in the gallery ``<batch>/images/<acquisition>/Images/`` layout, and that suffix is what is matched.
    """
    if prefix.startswith("URL_"):
        location = str(row[f"{prefix}{channel}"])
    else:
        directory = row[f"PathName_{prefix.removeprefix('FileName_')}{channel}"]
        location = f"{directory}/{row[f'{prefix}{channel}']}"
    marker = f"/{batch}/images/"
    if marker not in location:
        msg = f"image location does not follow the gallery layout: {location}"
        raise ValueError(msg)
    return root / "images" / location[location.index(marker) + 1 :]


def _read_fov(root: Path, batch: str, row: pd.Series, prefix: str, channels: Sequence[str]) -> npt.NDArray:
    import imageio.v3 as iio

    return np.stack([iio.imread(_image_path(root, batch, row, prefix, channel)) for channel in channels])


def _site_dir(root: Path, batch: str, plate: str, well: str, site: int) -> Path:
    return root / "workspace/analysis" / batch / plate / "analysis" / f"{plate}-{well}-{site}"


def _outline_file(directory: Path, well: str, site: int, kind: str) -> Path | None:
    """Find one outline image, whatever the source called it.

    Seen across the gallery: ``outlines/A01_s1--cell_outlines.png``, ``outlines/a01_1--cell_outlines.png``, and ``A01_s1_cell_outlines.tiff`` in a directory named after the plate.
    """
    for stem in (
        f"{well}_s{site}--{kind}_outlines",
        f"{well}_{site}--{kind}_outlines",
        f"{well}_s{site}_{kind}_outlines",
    ):
        if matches := sorted(directory.glob(f"{stem}.*")) + sorted(directory.glob(f"*/{stem}.*")):
            return matches[0]
    return None


def _outline_candidates(image: npt.NDArray) -> list[npt.NDArray[np.bool_]]:
    """The ways an outline image might encode its outlines, best guess first.

    Most sources publish the outlines alone.
    Some draw them in colour over the greyscale image, where the outline is whatever is not grey.
    One image can carry two object types in two colours, so each colour is offered separately and the caller keeps whichever reconstructs the most objects.
    """
    if image.ndim == 2:
        return [image > 0]
    if image.ndim != 3 or image.shape[-1] not in (3, 4):
        return []
    channels = image[..., :3].astype(np.int16)
    coloured = (channels[..., 0] != channels[..., 1]) | (channels[..., 1] != channels[..., 2])
    colours = np.unique(channels[coloured].reshape(-1, 3), axis=0)
    if len(colours) < 2:
        return [coloured]
    return [coloured, *((channels == colour).all(axis=-1) for colour in colours)]


def _centre_columns(objects: pd.DataFrame) -> tuple[str, str]:
    for prefix in ("Location_Center", "AreaShape_Center"):
        if {f"{prefix}_X", f"{prefix}_Y"}.issubset(objects.columns):
            return f"{prefix}_X", f"{prefix}_Y"
    msg = "object table has neither Location_Center nor AreaShape_Center columns"
    raise ValueError(msg)


def _site_labels(directory: Path, well: str, site: int) -> dict[str, npt.NDArray[np.uint32]]:
    import imageio.v3 as iio

    masks = {}
    for name, kind, csv in (("nuclei", "nuclei", "Nuclei"), ("cells", "cell", "Cells")):
        path = _outline_file(directory, well, site, kind)
        if path is None or not (directory / f"{csv}.csv").exists():
            return {}
        candidates = _outline_candidates(np.squeeze(iio.imread(path)))
        if not candidates:
            return {}
        objects = pd.read_csv(directory / f"{csv}.csv")
        x_column, y_column = _centre_columns(objects)
        reconstructed = [
            _labels_from_outlines(candidate, objects, x_column=x_column, y_column=y_column) for candidate in candidates
        ]
        masks[name] = max(reconstructed, key=lambda labels: len(np.unique(labels)))
    masks["cytoplasm"] = np.where(masks["nuclei"] > 0, 0, masks["cells"])
    return masks


def _well_shapes(load_data: pd.DataFrame, *, pixel_size: float, plate_format: int, system: str) -> GeoDataFrame:
    from geopandas import GeoDataFrame
    from shapely import Point
    from spatialdata.models import ShapesModel
    from spatialdata.transformations import Identity

    x = load_data["Metadata_PositionX"].to_numpy(float) / pixel_size
    y = load_data["Metadata_PositionY"].to_numpy(float) / pixel_size
    height, width = int(load_data["Metadata_ImageSizeY"].iloc[0]), int(load_data["Metadata_ImageSizeX"].iloc[0])
    centre = (y.max() + height / 2, -x.min() + width / 2)
    layout = PLATE_FORMATS[plate_format]
    n_wells = layout.n_rows * layout.n_columns
    pitch = layout.well_pitch_metres
    rows, columns = np.divmod(np.arange(n_wells), layout.n_columns)
    points = [
        Point(centre[1] + c * pitch / pixel_size, centre[0] + r * pitch / pixel_size)
        for r, c in zip(rows, columns, strict=True)
    ]
    frame = GeoDataFrame({"radius": WELL_RADIUS_METRES / pixel_size}, geometry=points, index=np.arange(n_wells))
    return ShapesModel.parse(frame, transformations={system: Identity()})


def _well_table(path: Path, *, region: str | None, plate_format: int) -> ad.AnnData:
    from spatialdata import sanitize_table
    from spatialdata.models import TableModel

    adata = read_profiles(path, index_columns=("Plate", "Well"))
    # some sources annotate wells with column names SpatialData rejects, "Common Name" among them
    sanitize_table(adata)
    if region is None:
        return TableModel.parse(adata)
    grid = np.array([_parse_well(well) for well in adata.obs["Well"]])
    adata.obs["well_index"] = grid[:, 0] * PLATE_FORMATS[plate_format].n_columns + grid[:, 1]
    adata.obs["region"] = pd.Categorical([region] * adata.n_obs)
    return TableModel.parse(adata, region=region, region_key="region", instance_key="well_index")


def _cell_table(files: Sequence[Path], masks: Mapping[str, npt.NDArray]) -> ad.AnnData:
    from spatialdata import sanitize_table
    from spatialdata.models import TableModel

    adata = read_profiles(files, metadata_columns=("ImageNumber", "ObjectNumber"), path_columns={"Metadata_Key": 1})
    obs = cast("pd.DataFrame", adata.obs)
    keys = obs.pop("Key").str.rsplit("-", n=2, expand=True)
    plates, wells, sites = (keys[i].astype(str) for i in range(3))
    obs["Well"] = pd.Categorical(wells)
    obs["Site"] = sites.astype(int)
    obs["region"] = pd.Categorical(plates + "_" + wells + "_s" + sites + "_cells")
    adata.obs_names = (obs["region"].astype(str) + ":" + obs["ObjectNumber"].astype(str)).tolist()

    numbers = adata.obs["ObjectNumber"].to_numpy()
    keep = np.zeros(adata.n_obs, bool)
    for region, mask in masks.items():
        rows = (adata.obs["region"] == region).to_numpy()
        keep[rows] = np.isin(numbers[rows], np.unique(mask))
    adata = adata[keep].copy()
    adata.obs["region"] = adata.obs["region"].cat.remove_unused_categories()
    sanitize_table(adata)
    return TableModel.parse(
        adata, region=sorted(adata.obs["region"].cat.categories), region_key="region", instance_key="ObjectNumber"
    )


def read_gallery_plate(
    root: Path | str,
    batch: str,
    plate: str,
    *,
    wells: Sequence[str] | None = None,
    plane: int | None = None,
    profile: str | Path | None = "normalized_feature_select_negcon_batch",
    plate_format: int = 384,
) -> SpatialData:
    """Read one plate of a Cell Painting Gallery source.

    See :func:`mantispy.io.read_plate`, which dispatches here, for what the result holds.

    Args:
        root: Directory holding the ``images/`` and ``workspace/`` trees of one source of one accession.
        batch: Batch name, the directory below ``images/`` and ``workspace/analysis/``.
        plate: Plate barcode.
        wells: Wells to read images and labels for.
        plane: Which ``Metadata_PlaneID`` to read where a source imaged a z stack.
        profile: Variant of the well-level profile, or a path to it, or ``None``.
        plate_format: Number of wells on the plate.

    Returns:
        The plate, with Images, Labels, Shapes and the ``wells`` and ``cells`` Tables.
    """
    from spatialdata import SpatialData
    from spatialdata.models import Image2DModel, Labels2DModel
    from spatialdata.transformations import Identity, Translation

    root = Path(root)
    load_data = _load_data(root, batch, plate)
    if load_data.index.duplicated().any():
        planes = sorted(load_data["Metadata_PlaneID"].unique()) if "Metadata_PlaneID" in load_data else []
        if plane is None:
            msg = f"{plate} has several rows per field; pass plane= to choose one of {planes}"
            raise ValueError(msg)
        load_data = load_data[load_data["Metadata_PlaneID"] == plane]
    prefix, channels = _channels(load_data)
    located = {
        "Metadata_PositionX",
        "Metadata_PositionY",
        "Metadata_ImageResolutionX",
        "Metadata_ImageResolutionY",
    }.issubset(load_data.columns)
    if located:
        pixel_size = _pixel_size(load_data)
        offsets = _fov_offsets(
            pd.DataFrame(
                {
                    "well": load_data.index.get_level_values("well"),
                    "x": load_data["Metadata_PositionX"].to_numpy(float),
                    "y": load_data["Metadata_PositionY"].to_numpy(float),
                },
                index=load_data.index,
            ),
            pixel_size=pixel_size,
            plate_format=plate_format,
        )

    if wells is None:
        first = load_data.groupby(level="well").head(1)
        wells = [
            str(key[0])
            for key, row in first.iterrows()
            if isinstance(key, tuple) and _image_path(root, batch, row, prefix, channels[0]).exists()
        ]

    images, labels, masks, analysed = {}, {}, {}, []
    for well in wells:
        for site in sorted(load_data.loc[well].index):
            fov = f"{plate}_{well}_s{site}"
            transformations: dict[str, Identity | Translation] = {fov: Identity()}
            if located:
                offset = offsets.loc[(well, site)]
                transformations[f"{plate}_{well}"] = Translation([offset["well_y"], offset["well_x"]], axes=("y", "x"))
                transformations[plate] = Translation([offset["plate_y"], offset["plate_x"]], axes=("y", "x"))
            images[f"{fov}_image"] = Image2DModel.parse(
                _read_fov(root, batch, load_data.loc[(well, site)], prefix, channels),
                dims=("c", "y", "x"),
                c_coords=channels,
                transformations=transformations,
                scale_factors=[2, 2],
            )
            directory = _site_dir(root, batch, plate, well, site)
            site_labels = _site_labels(directory, well, site) if directory.is_dir() else {}
            if not site_labels:
                continue
            analysed.append(directory / "Cells.csv")
            for name, mask in site_labels.items():
                masks[f"{fov}_{name}"] = mask
                labels[f"{fov}_{name}"] = Labels2DModel.parse(mask, dims=("y", "x"), transformations=transformations)

    tables, shapes = {}, {}
    if analysed:
        tables["cells"] = _cell_table(analysed, {k: v for k, v in masks.items() if k.endswith("_cells")})
    if profile is not None:
        path = (
            Path(profile)
            if isinstance(profile, Path)
            else root / "workspace/profiles" / batch / plate / f"{plate}_{profile}.csv.gz"
        )
        tables["wells"] = _well_table(path, region=f"{plate}_wells" if located else None, plate_format=plate_format)
        if located:
            shapes[f"{plate}_wells"] = _well_shapes(
                load_data, pixel_size=pixel_size, plate_format=plate_format, system=plate
            )
    return SpatialData(images=images, labels=labels, shapes=shapes, tables=tables)
