"""Build the Aconcagua guide-by-well profiles and single-cell subset from the Brieflow public outputs.

    python scripts/build_aconcagua.py --out build/ --work build/aconcagua-work --workers 8

Streams the OPS Explorer per-cell table from ``s3://ops-explorer-public`` (anonymous, ``us-west-2``), keeps
the pinned 500-gene subset at the pinned per-(unit, well, class) keep probabilities in
``scripts/data/aconcagua/``, then converts the brieflow ``cp_emulator`` feature names to mantispy's
CellProfiler-style grammar and, per plate, thins mitotic cells back to their natural share, normalizes and
aggregates to guide-by-well profiles. It writes the two files the registry rehosts:

* ``aconcagua_guide_well.h5ad`` -- 84,125 guide x well profiles over 1,483 features (``mt.ds.aconcagua``);
* ``aconcagua_cells.h5ad`` -- the raw single cells of eight genes and the non-targeting controls (``mt.ds.aconcagua_cells``).

The cells file keeps every extracted cell of the eight interpretation genes (RRM1, DTL, SKA2, HAUS6, ITGAV,
TLN1, PSMB7, SF3B3) and, drawn independently in each part with the generator reseeded per part, a tenth of the
non-targeting controls. The mitotic cells are left uncapped (not thinned to their natural share as the pooled
profiles are), because the cell-cycle analysis the file feeds needs the mitotic cells; the per-part draw keeps
the natural mitotic fraction. This reproduces how the file was first built, so its rows match the hosted file.

Prints each file's size and sha256 (to pin in the registry) and a content fingerprint; a rebuild is checked
against the fingerprint, not the sha256, because anndata or h5py releases can write the same object as
different bytes. Extraction streams every row group of the per-cell table and takes about 1.5 h on four cores;
it is resumable (one part per row group) and the per-cell draw is deterministic, so a rerun keeps the same cells.

The 500-gene subset and its keep probabilities were planned once from the screen-wide metadata (brieflow's
published gene tables plus a screen-wide mitotic index) and are pinned here as small inputs rather than
recomputed, so the exploratory planning code is not part of this script.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.fs as pfs
import pyarrow.parquet as pq
from anndata import AnnData

import mantispy as mt

DATA = Path(__file__).parent / "data" / "aconcagua"
BUCKET = "ops-explorer-public"
CELL_DATA = f"{BUCKET}/aconcagua/aconcagua/cell_data.parquet"
REGION = "us-west-2"

CHANNELS = ("DAPI", "TUBULIN", "GH2AX", "PHALLOIDIN")
POSITIVE_CONTROLS = ("H2AX", "ECT2", "TUBB", "CFL1")
PLATES = range(1, 9)
# The eight genes the single-cell file keeps, beside a sample of the non-targeting controls.
CELLS_GENES = ("RRM1", "DTL", "SKA2", "HAUS6", "ITGAV", "TLN1", "PSMB7", "SF3B3")
# The single-cell file keeps every cell of the eight genes and, independently in each part (the generator
# reseeded per part), this fraction of the non-targeting controls, matching how the file was first built.
CELLS_CONTROL_FRACTION = 0.10
CELLS_SEED = 0

# Coordinates and bounding boxes are positions, not phenotypes (94 columns); they stay out of the subset.
COORD = re.compile(
    r"^(nucleus|cell|cytoplasm)_((DAPI|TUBULIN|GH2AX|PHALLOIDIN)_)?"
    r"(bounds_\d|centroid_[rc]|[ij]|max_location_[rc]|center_mass_[rc]|(min|max)_feret_[rc][01])$"
)

OBJECTS = {"cell": "Cells", "nucleus": "Nuclei", "cytoplasm": "Cytoplasm"}
# Per-channel intensity statistics, as CellProfiler's MeasureObjectIntensity names them.
INTENSITY = {
    "mean": "MeanIntensity",
    "std": "StdIntensity",
    "max": "MaxIntensity",
    "min": "MinIntensity",
    "median": "MedianIntensity",
    "mad": "MADIntensity",
    "lower_quartile": "LowerQuartileIntensity",
    "upper_quartile": "UpperQuartileIntensity",
    "integrated": "IntegratedIntensity",
    "mass_displacement": "MassDisplacement",
    "mean_edge": "MeanIntensityEdge",
    "std_edge": "StdIntensityEdge",
    "max_edge": "MaxIntensityEdge",
    "min_edge": "MinIntensityEdge",
    "integrated_edge": "IntegratedIntensityEdge",
}
# Haralick indices follow mahotas order, which is CellProfiler's (brieflow CP_EMULATOR_FEATURES.md).
HARALICK = (
    "AngularSecondMoment",
    "Contrast",
    "Correlation",
    "Variance",
    "InverseDifferenceMoment",
    "SumAverage",
    "SumVariance",
    "SumEntropy",
    "Entropy",
    "DifferenceVariance",
    "DifferenceEntropy",
    "InfoMeas1",
    "InfoMeas2",
)
RADIAL = {"frac_at_d": "FracAtD", "mean_frac": "MeanFrac", "radial_cv": "RadialCV"}
CORRELATION = {
    "correlation": "Correlation",
    "manders": "Manders",
    "overlap": "Overlap",
    "rwc": "RWC",
    "K": "K",
    "lstsq_slope": "Slope",
}
SHAPE = {
    "area": "Area",
    "perimeter": "Perimeter",
    "convex_area": "ConvexArea",
    "form_factor": "FormFactor",
    "solidity": "Solidity",
    "extent": "Extent",
    "euler_number": "EulerNumber",
    "eccentricity": "Eccentricity",
    "major_axis": "MajorAxisLength",
    "minor_axis": "MinorAxisLength",
    "orientation": "Orientation",
    "compactness": "Compactness",
    "max_radius": "MaximumRadius",
    "median_radius": "MedianRadius",
    "mean_radius": "MeanRadius",
    "min_feret_diameter": "MinFeretDiameter",
    "max_feret_diameter": "MaxFeretDiameter",
}
NEIGHBORS = {
    "number_neighbors_1": "NumberOfNeighbors_1",
    "percent_touching_1": "PercentTouching_1",
    "first_neighbor_distance": "FirstClosestDistance_1",
    "second_neighbor_distance": "SecondClosestDistance_1",
    "angle_between_neighbors": "AngleBetweenNeighbors_1",
}
CH = "|".join(CHANNELS)
QC_COLUMNS = {
    "distance": "Metadata_BarcodeDistance",
    "mapped_n_barcodes": "Metadata_MappedBarcodes",
    "fov_distance_phenotype": "Metadata_FovDistancePhenotype",
    "fov_distance_sbs": "Metadata_FovDistanceSBS",
    "cell_stage_confidence": "Metadata_ClassConfidence",
    "i_phenotype": "Metadata_TileY",
    "j_phenotype": "Metadata_TileX",
    "y": "Metadata_Y",
    "x": "Metadata_X",
}


def _rename(column: str) -> str | None:
    """CellProfiler-style name for a brieflow feature column, or None for a non-feature."""
    match = re.match(r"^(cell|nucleus|cytoplasm)_(.+)$", column)
    if not match:
        return None
    obj, rest = OBJECTS[match.group(1)], match.group(2)
    if m := re.fullmatch(rf"({CH})_(.+)", rest):
        ch, stat = m.groups()
        if stat in INTENSITY:
            return f"{obj}_Intensity_{INTENSITY[stat]}_{ch}"
        if m2 := re.fullmatch(r"haralick_(\d+)_(\d+)", stat):
            # Direction-averaged at one distance, so the name carries the scale but no direction.
            return f"{obj}_Texture_{HARALICK[int(m2.group(2))]}_{ch}_{m2.group(1)}"
        if m2 := re.fullmatch(r"(frac_at_d|mean_frac|radial_cv)_(\d)", stat):
            return f"{obj}_RadialDistribution_{RADIAL[m2.group(1)]}_{ch}_{int(m2.group(2)) + 1}of4"
        if m2 := re.fullmatch(r"pftas_(\d+)", stat):
            return f"{obj}_PFTAS_Stat{m2.group(1)}_{ch}"
        if m2 := re.fullmatch(r"weighted_hu_moments_(\d)", stat):
            return f"{obj}_Intensity_WeightedHuMoment{m2.group(1)}_{ch}"
        if m2 := re.fullmatch(r"foci_(count|area)", stat):
            return f"{obj}_Foci_{m2.group(1).capitalize()}_{ch}"
        return None
    if m := re.fullmatch(rf"(correlation|manders|overlap|rwc|K|lstsq_slope)_({CH})_({CH})", rest):
        return f"{obj}_Correlation_{CORRELATION[m.group(1)]}_{m.group(2)}_{m.group(3)}"
    if rest in SHAPE:
        return f"{obj}_AreaShape_{SHAPE[rest]}"
    if m := re.fullmatch(r"hu_moments_(\d)", rest):
        return f"{obj}_AreaShape_HuMoment_{m.group(1)}"
    if m := re.fullmatch(r"zernike_(\d+)_(\d+)", rest):
        return f"{obj}_AreaShape_Zernike_{m.group(1)}_{m.group(2)}"
    if rest in NEIGHBORS:
        return f"{obj}_Neighbors_{NEIGHBORS[rest]}"
    return None


def _convert(frame: pd.DataFrame, tiers: pd.Series) -> AnnData:
    """Brieflow cell table to a mantispy cell-level AnnData, with the guide in ``Metadata_sgRNA``."""
    features = {c: n for c in frame.columns if (n := _rename(c)) is not None}
    control = frame["is_control"].to_numpy(dtype=bool)
    gene = frame["perturbation_id"].astype(str)
    obs = pd.DataFrame(
        {
            "Metadata_Plate": frame["plate"].astype(str).to_numpy(),
            "Metadata_Well": frame["well"].astype(str).to_numpy(),
            "Metadata_Site": frame["tile"].to_numpy(),
            "Metadata_ObjectNumber": frame["cell_phenotype"].to_numpy(),
            "Metadata_sgRNA": frame["barcode"].astype(str).to_numpy(),
            "Metadata_Gene": np.where(control, "nontargeting", gene),
            "Metadata_Perturbation": np.where(control, "nontargeting", gene),
            "Metadata_Control_Type": np.where(
                control, "negcon", np.where(gene.isin(POSITIVE_CONTROLS), "poscon", "treatment")
            ),
            "Metadata_Class": frame["class"].astype(str).to_numpy(),
            # Genes outside the planned subset (in a complete-tile block) are "other".
            "Metadata_Tier": np.where(control, "control", gene.map(tiers).fillna("other").to_numpy()),
        }
    )
    obs["Metadata_Control"] = obs["Metadata_Control_Type"] != "treatment"
    # Quality-control metadata: barcode mapping, classifier confidence and positions in the tile and the well.
    for source, name in QC_COLUMNS.items():
        if source in frame:
            obs[name] = frame[source].to_numpy()
    table = pd.concat([obs, frame[list(features)].rename(columns=features).astype(np.float32)], axis=1)
    # Through mantispy's own reader, so the result carries the schema stamp and the parsed var annotation.
    return mt.io.from_dataframe(table, channels=list(CHANNELS), resolution="object")


def _uniform(*arrays: np.ndarray) -> np.ndarray:
    """Deterministic per-cell uniform draw from its identity, so a rerun keeps the same cells."""
    h = np.zeros(len(arrays[0]), dtype=np.uint64)
    for a in arrays:
        h = (h ^ np.asarray(a, dtype=np.uint64)) * np.uint64(0x9E3779B97F4A7C15)
        h ^= h >> np.uint64(29)
    return (h >> np.uint64(11)).astype(np.float64) / float(1 << 53)


def extract(parts: Path, keep_prob: pd.DataFrame, workers: int) -> None:
    """Stream the per-cell table row group by row group, keep the planned subset, write one part each."""
    parts.mkdir(parents=True, exist_ok=True)
    fs = pfs.S3FileSystem(anonymous=True, region=REGION)
    schema = pq.ParquetFile(fs.open_input_file(CELL_DATA)).schema_arrow
    cols = [c for c in schema.names if not COORD.match(c)]
    prob = {
        (u, p, w, c): v
        for u, p, w, c, v in keep_prob[["unit", "plate", "well", "class", "p_keep"]].itertuples(index=False)
    }

    def one(i: int) -> tuple[int, str, int]:
        dest = parts / f"rg{i:03d}.parquet"
        if dest.exists():
            return i, "skip", 0
        table = pq.ParquetFile(fs.open_input_file(CELL_DATA), pre_buffer=True).read_row_group(i, columns=cols)
        meta = table.select(
            ["plate", "well", "perturbation_id", "barcode", "is_control", "class", "tile", "cell_phenotype"]
        ).to_pandas()
        unit = np.where(meta.is_control, meta.barcode.astype(str), meta.perturbation_id.astype(str))
        p = np.array(
            [
                prob.get(k, 0.0)
                for k in zip(unit, meta.plate, meta.well.astype(str), meta["class"].astype(str), strict=True)
            ]
        )
        u = _uniform(
            meta.plate.to_numpy(), pd.factorize(meta.well)[0], meta.tile.to_numpy(), meta.cell_phenotype.to_numpy()
        )
        table = table.filter(pa.array(u < p))
        # float32 halves the size; the features are image measurements with far less precision than that.
        table = table.cast(
            pa.schema(
                [
                    pa.field(f.name, pa.float32())
                    if pa.types.is_floating(f.type) and f.name not in ("distance",)
                    else f
                    for f in table.schema
                ]
            )
        )
        pq.write_table(table, str(dest) + ".tmp", compression="zstd")
        os.replace(str(dest) + ".tmp", dest)
        return i, "done", table.num_rows

    n = pq.ParquetFile(fs.open_input_file(CELL_DATA)).metadata.num_row_groups
    start, kept = time.time(), 0
    with ThreadPoolExecutor(workers) as pool:
        for k, (i, status, rows) in enumerate(pool.map(one, range(n)), 1):
            kept += rows
            print(f"[{time.time() - start:6.0f}s] rg{i:03d} {status} {rows} rows | {k}/{n} | kept {kept}", flush=True)


def _parts_of_plate(parts: Path, plate: int) -> list[Path]:
    """Parts whose footer statistics say they hold this plate (row groups are sorted by plate)."""
    keep = []
    for path in sorted(parts.glob("rg*.parquet")):
        meta = pq.ParquetFile(path).metadata
        column = meta.schema.names.index("plate")
        stats = [meta.row_group(i).column(column).statistics for i in range(meta.num_row_groups)]
        if any(s is not None and s.min <= plate <= s.max for s in stats):
            keep.append(path)
    return keep


def process_plate(parts: Path, keep_prob: pd.DataFrame, tiers: pd.Series, plate: int) -> AnnData:
    """One plate: thin mitotic cells, convert, normalize, aggregate to guide x well profiles."""
    frame = pd.concat(
        [pq.read_table(p, filters=[("plate", "=", plate)]).to_pandas() for p in _parts_of_plate(parts, plate)],
        ignore_index=True,
    )
    # Mitotic cells were extracted uncapped. Keep each with the probability the interphase cells of its guide and
    # well were kept with, so a pooled profile carries the guide's natural mitotic share.
    inter = keep_prob[(keep_prob["class"] == "Interphase") & (keep_prob.plate == plate)].set_index(["unit", "well"])[
        "p_keep"
    ]
    unit = np.where(frame.is_control, frame.barcode.astype(str), frame.perturbation_id.astype(str))
    p_inter = inter.reindex(pd.MultiIndex.from_arrays([unit, frame.well.astype(str)])).fillna(1.0).to_numpy()
    mitotic = (frame["class"] == "Mitotic").to_numpy()
    rng = np.random.default_rng(plate)
    frame = frame[~mitotic | (rng.random(len(frame)) < p_inter)].reset_index(drop=True)
    # Control guides were capped at 20 cells per guide per well, targeting genes at 40 per gene per well (about 10
    # per guide). A profile's noise falls with its cell count, so halve the controls to match; otherwise every
    # targeting guide looks active merely for being noisier than the controls.
    frame = frame[~frame.is_control.to_numpy() | (rng.random(len(frame)) < 0.5)].reset_index(drop=True)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        cells = _convert(frame, tiers)
    del frame
    cells.obs["Metadata_Mitotic"] = (cells.obs["Metadata_Class"] == "Mitotic").astype(np.float32)

    mt.pp.normalize(cells, by=["Metadata_Plate", "Metadata_Well"], reference="negcon")
    keys = ["Metadata_Plate", "Metadata_Well", "Metadata_sgRNA"]
    fraction = cells.obs.groupby(keys, observed=True)["Metadata_Mitotic"].mean().rename("Metadata_MitoticFraction")
    profiles = mt.tl.aggregate(cells, by=tuple(keys), func="median", min_cells=5)
    profiles.obs = profiles.obs.join(fraction, on=keys)
    profiles.var["degenerate_scale"] = cells.var["degenerate_scale"].to_numpy()
    print(
        f"plate {plate}: {cells.n_obs} cells -> {profiles.n_obs} guide x well profiles; "
        f"mitotic {cells.obs.Metadata_Mitotic.mean():.3f}; degenerate features {int(cells.var.degenerate_scale.sum())}",
        flush=True,
    )
    return profiles


def fingerprint(adata: AnnData) -> str:
    """sha256 over ``X``, ``obs``, ``var`` and what ``uns`` says the object is, however a file encodes them."""
    store = adata.uns["mantispy"]
    # h5ad reads a list back as an array, so both are hashed as lists.
    identity = [np.asarray(store.get(key)).tolist() for key in ("resolution", "channels", "dataset")]
    digest = hashlib.sha256(repr(identity).encode())
    digest.update(np.ascontiguousarray(adata.X, dtype=np.float32))
    for frame in (adata.obs, adata.var):
        digest.update("\t".join(frame.columns).encode())
        digest.update(pd.util.hash_pandas_object(frame, index=True).to_numpy())
    return digest.hexdigest()


def _report(path: Path, adata: AnnData) -> None:
    """Print a built file's size, sha256 and content fingerprint, as ``build_dataset.py`` does."""
    with path.open("rb") as file:
        sha256 = hashlib.file_digest(file, "sha256").hexdigest()
    print(f"{path}  {adata.n_obs} x {adata.n_vars}  {path.stat().st_size / 1e6:.1f} MB")
    print(f"sha256       {sha256}")
    print(f"fingerprint  {fingerprint(adata)}")


def build_cells(parts: Path, tiers: pd.Series) -> AnnData:
    """Raw single cells of the eight genes and a per-part 10% sample of the non-targeting controls.

    Reproduces the originally hand-built cells file: every extracted cell of the eight genes and, independently in
    each part (the generator reseeded per part), a tenth of the non-targeting controls. The mitotic cells are left
    uncapped -- no mitotic thinning and no normalization is applied here, unlike the pooled profiles -- because the
    cell-cycle analysis the file feeds needs the mitotic cells.
    """
    kept = []
    for path in sorted(parts.glob("rg*.parquet")):
        table = pq.read_table(path).to_pandas()
        draw = np.random.default_rng(CELLS_SEED).random(len(table))
        keep_ctrl = table["is_control"].to_numpy() & (draw < CELLS_CONTROL_FRACTION)
        keep = table["perturbation_id"].astype(str).isin(CELLS_GENES).to_numpy() | keep_ctrl
        kept.append(table[keep])
    frame = pd.concat(kept, ignore_index=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        return _convert(frame, tiers)


def main() -> None:
    """Extract the subset, build both files and print each one's size, sha256 and fingerprint."""
    parser = argparse.ArgumentParser(description="Build the Aconcagua files from the Brieflow public outputs.")
    parser.add_argument("--out", type=Path, default=Path("build"), help="directory to write the two h5ads into")
    parser.add_argument(
        "--work",
        type=Path,
        default=None,
        help="scratch directory for the extracted parts (default: <out>/aconcagua-work)",
    )
    parser.add_argument("--workers", type=int, default=4, help="threads for the streaming extraction")
    args = parser.parse_args()
    work = args.work or args.out / "aconcagua-work"
    args.out.mkdir(parents=True, exist_ok=True)

    keep_prob = pd.read_parquet(DATA / "keep_prob.parquet")
    tiers = pd.read_csv(DATA / "genes.csv").set_index("gene")["set"]

    extract(work / "parts", keep_prob, args.workers)

    profiles = [process_plate(work / "parts", keep_prob, tiers, plate) for plate in PLATES]

    # Guide x well: carry degenerate_scale as the OR over plates (a feature degenerate on any plate is flagged).
    degenerate = np.any([a.var["degenerate_scale"].to_numpy() for a in profiles], axis=0)
    for a in profiles:
        del a.var["degenerate_scale"]
    guide_well = ad.concat(profiles, merge="same", uns_merge="first")
    guide_well.var["degenerate_scale"] = degenerate
    guide_well.obs_names = [str(i) for i in range(guide_well.n_obs)]

    cells = build_cells(work / "parts", tiers)
    cells.obs_names = [str(i) for i in range(cells.n_obs)]

    guide_well_path = args.out / "aconcagua_guide_well.h5ad"
    cells_path = args.out / "aconcagua_cells.h5ad"
    mt.io.write(guide_well, guide_well_path)
    mt.io.write(cells, cells_path)
    _report(guide_well_path, guide_well)
    _report(cells_path, cells)


if __name__ == "__main__":
    main()
