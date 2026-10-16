"""Tutorial analysis, steps 1-3: QC, feature selection, batch check."""
import time, warnings
import anndata as ad, numpy as np, pandas as pd, scanpy as sc, mantispy as mt
from scipy.stats import mannwhitneyu
warnings.filterwarnings("ignore"); mt.settings.verbosity = 1
T = time.time(); lap = lambda s: print(f"[{time.time()-T:5.0f}s] {s}", flush=True)
w = ad.read_h5ad("acon/aconcagua_guide_well.h5ad"); lap(f"loaded {w.shape}")
o = w.obs; ntc = (o.Metadata_Control_Type == "negcon").to_numpy()

# 1. QC: positive controls move their expected channel (guide x well level, against control guide x wells).
checks = [("H2AX", "Nuclei_Intensity_MeanIntensity_GH2AX", "down"), ("TUBB", "Cells_Intensity_MeanIntensity_TUBULIN", "down"),
          ("CFL1", "Cells_Intensity_MeanIntensity_PHALLOIDIN", "altered"), ("ECT2", "Nuclei_Intensity_IntegratedIntensity_DAPI", "up")]
for gene, feat, expect in checks:
    g = (o.Metadata_Gene == gene).to_numpy(); x = w[:, feat].X.ravel()
    print(f"  {gene:5s} {feat:44s} expect {expect:7s} median {np.median(x[g]):+.2f} vs NTC {np.median(x[ntc]):+.2f}  p={mannwhitneyu(x[g], x[ntc]).pvalue:.1e}  (n={g.sum()})")
for gene in ["ECT2"]:
    g = (o.Metadata_Gene == gene).to_numpy()
    for f in ["Nuclei_AreaShape_Area", "Cells_AreaShape_Area", "Cells_Intensity_IntegratedIntensity_DAPI", "Cells_Neighbors_NumberOfNeighbors_1"]:
        x = w[:, f].X.ravel(); print(f"  ECT2  {f:44s} median {np.median(x[g]):+.2f} vs NTC {np.median(x[ntc]):+.2f}  p={mannwhitneyu(x[g], x[ntc]).pvalue:.1e}")

# 2. Feature selection: drop degenerate, NaN-heavy, near-constant and correlated features.
mt.pp.feature_select(w)
sel = w.var.selected.to_numpy()
print("  selected", int(sel.sum()), "of", w.n_vars, "| by group:", w.var[sel].feature_group.value_counts().to_dict())
print("  degenerate kept:", int((w.var.degenerate_scale & w.var.selected).sum()))
w = mt.pp.subset_features(w); lap("feature selection")

# 3. Batch: does plate (or well) still explain control variation after per-well normalization?
c = w[ntc].copy(); c.X = np.nan_to_num(c.X); sc.pp.pca(c, n_comps=20)
pcs = c.obsm["X_pca"]
def r2(labels):
    labels = pd.Series(labels).astype(str).to_numpy(); out = []
    for k in range(10):
        y = pcs[:, k]; means = pd.Series(y).groupby(labels).transform("mean").to_numpy()
        out.append(1 - ((y - means) ** 2).sum() / ((y - y.mean()) ** 2).sum())
    return np.array(out)
rng = np.random.default_rng(0)
pl, wl = c.obs.Metadata_Plate.to_numpy(), (c.obs.Metadata_Plate.astype(str) + c.obs.Metadata_Well.astype(str)).to_numpy()
print("  variance explained in control PCs 1-10 (R2):")
print("    by plate     ", np.round(r2(pl), 3), "| shuffled", np.round(r2(rng.permutation(pl)), 3).max())
print("    by plate+well", np.round(r2(wl), 3), "| shuffled", np.round(r2(rng.permutation(wl)), 3).max())
print("    by guide     ", np.round(r2(c.obs.Metadata_Guide), 3), "| shuffled", np.round(r2(rng.permutation(c.obs.Metadata_Guide.to_numpy())), 3).max())
print("  PC variance ratio 1-10:", np.round(c.uns["pca"]["variance_ratio"][:10], 3))
w.write_h5ad("acon/aconcagua_selected.h5ad"); lap("saved")
