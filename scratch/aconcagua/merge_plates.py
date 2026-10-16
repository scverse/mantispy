"""Concatenate the per-plate guide x well profiles into acon/aconcagua_guide_well.h5ad (the file mt.ds.aconcagua will host)."""
import anndata as ad
import numpy as np

plates = [ad.read_h5ad(f"acon/subset_v3/profiles_plate{p}.h5ad") for p in range(1, 9)]
deg = np.any([a.var["degenerate_scale"].to_numpy() for a in plates], axis=0)
for a in plates:
    del a.var["degenerate_scale"]
full = ad.concat(plates, merge="same", uns_merge="first")
full.var["degenerate_scale"] = deg
full.obs_names = [str(i) for i in range(full.n_obs)]
full.write_h5ad("acon/aconcagua_guide_well.h5ad")
print(full.shape)
