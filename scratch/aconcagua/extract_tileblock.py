"""Every cell, of every gene, in 100 consecutive tiles of plate 1 well A1 (tile positions in the well are not published)."""
import re, pyarrow as pa, pyarrow.parquet as pq, pyarrow.fs as pfs, pyarrow.compute as pc
PATH = "ops-explorer-public/aconcagua/aconcagua/cell_data.parquet"
fs = pfs.S3FileSystem(anonymous=True, region="us-west-2")
open_ = lambda: pq.ParquetFile(fs.open_input_file(PATH), pre_buffer=True)
md = open_().metadata; names = open_().schema_arrow.names
ip, iw = names.index("plate"), names.index("well")
st = lambda i, j: md.row_group(i).column(j).statistics
rgs = [i for i in range(md.num_row_groups) if st(i, ip).min <= 1 <= st(i, ip).max and st(i, iw).min <= "A1" <= st(i, iw).max]
tiles = pa.array(range(600, 700), type=pa.int64())  # middle of the 0-1280 scan
coord = re.compile(r"^(nucleus|cell|cytoplasm)_((DAPI|TUBULIN|GH2AX|PHALLOIDIN)_)?(bounds_\d|centroid_[rc]|[ij]|max_location_[rc]|center_mass_[rc]|(min|max)_feret_[rc][01])$")
cols = [c for c in names if not coord.match(c)]
keep = []
for i in rgs:
    head = open_().read_row_group(i, columns=["plate", "well", "tile"])
    mask = pc.and_(pc.and_(pc.equal(head["plate"], 1), pc.equal(head["well"], "A1")), pc.is_in(head["tile"], tiles))
    n = pc.sum(mask).as_py() or 0
    if n:
        t = open_().read_row_group(i, columns=cols).filter(mask)
        keep.append(t.cast(pa.schema([pa.field(x.name, pa.float32()) if pa.types.is_floating(x.type) and x.name != "distance" else x for x in t.schema])))
    print(i, n, flush=True)
t = pa.concat_tables(keep)
pq.write_table(t, "acon/tileblock/rg_block.parquet", compression="zstd")
print("tiles", len(set(t["tile"].to_pylist())), "cells", t.num_rows, "genes", len(set(t["perturbation_id"].to_pylist())), flush=True)
