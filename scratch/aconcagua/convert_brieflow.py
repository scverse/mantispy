"""Convert brieflow (cp_emulator) cell tables to mantispy's CellProfiler-style naming.

Standalone on purpose: it lives outside the package while the mapping settles.

    python -I convert_brieflow.py PARTS_DIR GENES_CSV OUT.h5ad
"""

import re
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from mantispy.io._profiles import from_dataframe

CHANNELS = ("DAPI", "TUBULIN", "GH2AX", "PHALLOIDIN")
POSITIVE_CONTROLS = ("H2AX", "ECT2", "TUBB", "CFL1")
OBJECTS = {"cell": "Cells", "nucleus": "Nuclei", "cytoplasm": "Cytoplasm"}

# Per-channel intensity statistics, as CellProfiler's MeasureObjectIntensity names them.
INTENSITY = {
    "mean": "MeanIntensity", "std": "StdIntensity", "max": "MaxIntensity", "min": "MinIntensity",
    "median": "MedianIntensity", "mad": "MADIntensity", "lower_quartile": "LowerQuartileIntensity",
    "upper_quartile": "UpperQuartileIntensity", "integrated": "IntegratedIntensity",
    "mass_displacement": "MassDisplacement", "mean_edge": "MeanIntensityEdge", "std_edge": "StdIntensityEdge",
    "max_edge": "MaxIntensityEdge", "min_edge": "MinIntensityEdge", "integrated_edge": "IntegratedIntensityEdge",
}
# Haralick indices follow mahotas order, which is CellProfiler's (brieflow CP_EMULATOR_FEATURES.md).
HARALICK = (
    "AngularSecondMoment", "Contrast", "Correlation", "Variance", "InverseDifferenceMoment", "SumAverage",
    "SumVariance", "SumEntropy", "Entropy", "DifferenceVariance", "DifferenceEntropy", "InfoMeas1", "InfoMeas2",
)
RADIAL = {"frac_at_d": "FracAtD", "mean_frac": "MeanFrac", "radial_cv": "RadialCV"}
CORRELATION = {"correlation": "Correlation", "manders": "Manders", "overlap": "Overlap", "rwc": "RWC", "K": "K", "lstsq_slope": "Slope"}
SHAPE = {
    "area": "Area", "perimeter": "Perimeter", "convex_area": "ConvexArea", "form_factor": "FormFactor",
    "solidity": "Solidity", "extent": "Extent", "euler_number": "EulerNumber", "eccentricity": "Eccentricity",
    "major_axis": "MajorAxisLength", "minor_axis": "MinorAxisLength", "orientation": "Orientation",
    "compactness": "Compactness", "max_radius": "MaximumRadius", "median_radius": "MedianRadius",
    "mean_radius": "MeanRadius", "min_feret_diameter": "MinFeretDiameter", "max_feret_diameter": "MaxFeretDiameter",
}
NEIGHBORS = {
    "number_neighbors_1": "NumberOfNeighbors_1", "percent_touching_1": "PercentTouching_1",
    "first_neighbor_distance": "FirstClosestDistance_1", "second_neighbor_distance": "SecondClosestDistance_1",
    "angle_between_neighbors": "AngleBetweenNeighbors_1",
}
CH = "|".join(CHANNELS)
QC_COLUMNS = {
    "distance": "Metadata_BarcodeDistance", "mapped_n_barcodes": "Metadata_MappedBarcodes",
    "fov_distance_phenotype": "Metadata_FovDistancePhenotype", "fov_distance_sbs": "Metadata_FovDistanceSBS",
    "cell_stage_confidence": "Metadata_ClassConfidence", "i_phenotype": "Metadata_TileY", "j_phenotype": "Metadata_TileX",
    "y": "Metadata_Y", "x": "Metadata_X",
}


def rename(column: str) -> str | None:
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


def convert(frame: pd.DataFrame, tiers: pd.Series) -> ad.AnnData:
    mapping = {c: rename(c) for c in frame.columns}
    features = {c: n for c, n in mapping.items() if n is not None}
    control = frame["is_control"].to_numpy(dtype=bool)
    gene = frame["perturbation_id"].astype(str)
    obs = pd.DataFrame({
        "Metadata_Plate": frame["plate"].astype(str).to_numpy(),
        "Metadata_Well": frame["well"].astype(str).to_numpy(),
        "Metadata_Site": frame["tile"].to_numpy(),
        "Metadata_ObjectNumber": frame["cell_phenotype"].to_numpy(),
        "Metadata_Guide": frame["barcode"].astype(str).to_numpy(),
        "Metadata_Gene": np.where(control, "nontargeting", gene),
        "Metadata_Perturbation": np.where(control, "nontargeting", gene),
        "Metadata_Control_Type": np.where(control, "negcon", np.where(gene.isin(POSITIVE_CONTROLS), "poscon", "trt")),
        "Metadata_Class": frame["class"].astype(str).to_numpy(),
        # Genes outside the planned subset (in a complete-tile block) are "other".
        "Metadata_Tier": np.where(control, "control", gene.map(tiers).fillna("other").to_numpy()),
    })
    obs["Metadata_Control"] = obs["Metadata_Control_Type"] != "trt"
    # Quality-control metadata: barcode mapping, classifier confidence and positions in the tile and the well.
    for source, name in QC_COLUMNS.items():
        if source in frame:
            obs[name] = frame[source].to_numpy()
    table = pd.concat([obs, frame[list(features)].rename(columns=features).astype(np.float32)], axis=1)
    # Through mantispy's own reader, so the result carries the schema stamp and the parsed var annotation.
    return from_dataframe(table, channels=list(CHANNELS), resolution="object")


def main() -> None:
    parts, genes_csv, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    genes = pd.read_csv(genes_csv)
    tiers = genes.set_index("gene")["set"]
    frame = pd.concat([pq.read_table(p).to_pandas() for p in sorted(parts.glob("rg*.parquet"))], ignore_index=True)
    adata = convert(frame, tiers)
    adata.write_h5ad(out)
    print(adata)


if __name__ == "__main__":
    main()
