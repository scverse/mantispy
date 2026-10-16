"""Stream cell_data.parquet row group by row group, keep the planned subset, write one part per row group (resumable)."""
import os, re, sys, time, numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq, pyarrow.fs as pfs
from concurrent.futures import ThreadPoolExecutor
out, workers = sys.argv[1], int(sys.argv[2])
only = [int(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3 else None
PATH = "ops-explorer-public/aconcagua/aconcagua/cell_data.parquet"
fs = pfs.S3FileSystem(anonymous=True, region="us-west-2")
schema = pq.ParquetFile(fs.open_input_file(PATH)).schema_arrow
# Coordinates and bounding boxes are positions, not phenotypes (94 columns); they stay out of the subset.
coord = re.compile(r"^(nucleus|cell|cytoplasm)_((DAPI|TUBULIN|GH2AX|PHALLOIDIN)_)?(bounds_\d|centroid_[rc]|[ij]|max_location_[rc]|center_mass_[rc]|(min|max)_feret_[rc][01])$")
cols = [c for c in schema.names if not coord.match(c)]
kp = pd.read_parquet(f"{out}/keep_prob.parquet")
prob = {(u, p, w, c): v for u, p, w, c, v in kp[["unit", "plate", "well", "class", "p_keep"]].itertuples(index=False)}

def uniform(*arrays):
    # Deterministic per-cell uniform draw from its identity, so a rerun keeps the same cells.
    h = np.zeros(len(arrays[0]), dtype=np.uint64)
    for a in arrays:
        h = (h ^ np.asarray(a, dtype=np.uint64)) * np.uint64(0x9E3779B97F4A7C15)
        h ^= h >> np.uint64(29)
    return (h >> np.uint64(11)).astype(np.float64) / float(1 << 53)

def one(i):
    dest = f"{out}/parts/rg{i:03d}.parquet"
    if os.path.exists(dest):
        return i, "skip", 0
    t = pq.ParquetFile(fs.open_input_file(PATH), pre_buffer=True).read_row_group(i, columns=cols)
    meta = t.select(["plate", "well", "perturbation_id", "barcode", "is_control", "class", "tile", "cell_phenotype"]).to_pandas()
    unit = np.where(meta.is_control, meta.barcode.astype(str), meta.perturbation_id.astype(str))
    p = np.array([prob.get(k, 0.0) for k in zip(unit, meta.plate, meta.well.astype(str), meta["class"].astype(str))])
    u = uniform(meta.plate.to_numpy(), pd.factorize(meta.well)[0], meta.tile.to_numpy(), meta.cell_phenotype.to_numpy())
    t = t.filter(pa.array(u < p))
    # float32 halves the size; the features are image measurements with far less precision than that.
    t = t.cast(pa.schema([pa.field(f.name, pa.float32()) if pa.types.is_floating(f.type) and f.name not in ("distance",) else f for f in t.schema]))
    pq.write_table(t, dest + ".tmp", compression="zstd")
    os.replace(dest + ".tmp", dest)
    return i, "done", t.num_rows

n = pq.ParquetFile(fs.open_input_file(PATH)).metadata.num_row_groups
todo = only if only is not None else list(range(n))
T = time.time(); kept = 0
with ThreadPoolExecutor(workers) as ex:
    for k, (i, status, rows) in enumerate(ex.map(one, todo), 1):
        kept += rows
        print(f"[{time.time()-T:6.0f}s] rg{i:03d} {status} {rows} rows | {k}/{len(todo)} | kept {kept}", flush=True)
print("FINISHED", flush=True)
