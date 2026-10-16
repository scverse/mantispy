"""One plate of the Aconcagua subset: thin mitotic cells, convert, normalize, aggregate to guide x well.

    python -I process_plate.py SUBSET_DIR PLATE
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).parent))
import mantispy as mt  # noqa: E402
from convert_brieflow import convert  # noqa: E402

subset, plate = Path(sys.argv[1]), int(sys.argv[2])


def parts_of_plate(parts: Path, plate: int) -> list[Path]:
    # Row groups are sorted by plate, so the footer statistics say which parts hold this plate.
    keep = []
    for path in sorted(parts.glob("rg*.parquet")):
        meta = pq.ParquetFile(path).metadata
        column = meta.schema.names.index("plate")
        stats = [meta.row_group(i).column(column).statistics for i in range(meta.num_row_groups)]
        if any(s is not None and s.min <= plate <= s.max for s in stats):
            keep.append(path)
    return keep


frame = pd.concat([pq.read_table(p, filters=[("plate", "=", plate)]).to_pandas() for p in parts_of_plate(subset / "parts", plate)],
                  ignore_index=True)

# Mitotic cells were extracted uncapped. Keep each with the probability the interphase cells of its guide and
# well were kept with, so a pooled profile carries the guide's natural mitotic share.
kp = pd.read_parquet(subset / "keep_prob.parquet")
kp = kp[(kp["class"] == "Interphase") & (kp.plate == plate)].set_index(["unit", "well"])["p_keep"]
unit = np.where(frame.is_control, frame.barcode.astype(str), frame.perturbation_id.astype(str))
p_inter = kp.reindex(pd.MultiIndex.from_arrays([unit, frame.well.astype(str)])).fillna(1.0).to_numpy()
mitotic = (frame["class"] == "Mitotic").to_numpy()
rng = np.random.default_rng(plate)
frame = frame[~mitotic | (rng.random(len(frame)) < p_inter)].reset_index(drop=True)
# Control guides were capped at 20 cells per guide per well, targeting genes at 40 per gene per well (about 10
# per guide). A profile's noise falls with its cell count, so halve the controls to match; otherwise every
# targeting guide looks active merely for being noisier than the controls.
frame = frame[~frame.is_control.to_numpy() | (rng.random(len(frame)) < 0.5)].reset_index(drop=True)

tiers = pd.read_csv(subset / "genes.csv").set_index("gene")["set"]
with warnings.catch_warnings():
    warnings.simplefilter("ignore", FutureWarning)
    cells = convert(frame, tiers)
del frame
cells.obs["Metadata_Mitotic"] = (cells.obs["Metadata_Class"] == "Mitotic").astype(np.float32)
if len(sys.argv) > 3 and sys.argv[3] == "cells":
    # The raw, unnormalized cells behind this plate's profiles, for the single-cell case studies.
    cells.write_h5ad(subset / f"cells_plate{plate}.h5ad")

mt.pp.normalize(cells, by=["Metadata_Plate", "Metadata_Well"], reference="negcon")
keys = ["Metadata_Plate", "Metadata_Well", "Metadata_Guide"]
fraction = cells.obs.groupby(keys, observed=True)["Metadata_Mitotic"].mean().rename("Metadata_MitoticFraction")
profiles = mt.tl.aggregate(cells, by=tuple(keys), func="median", min_cells=5)
profiles.obs = profiles.obs.join(fraction, on=keys)
profiles.var["degenerate_scale"] = cells.var["degenerate_scale"].to_numpy()
profiles.write_h5ad(subset / f"profiles_plate{plate}.h5ad")
print(f"plate {plate}: {cells.n_obs} cells -> {profiles.n_obs} guide x well profiles; "
      f"mitotic {cells.obs.Metadata_Mitotic.mean():.3f}; degenerate features {int(cells.var.degenerate_scale.sum())}")
