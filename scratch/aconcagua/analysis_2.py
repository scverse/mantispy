"""Steps 4-6: guide consensus, guide_activity -> aggregate_guides, calibration and tier recovery."""
import time, warnings
import anndata as ad, numpy as np, pandas as pd, scanpy as sc, mantispy as mt
warnings.filterwarnings("ignore"); mt.settings.verbosity = 1
T = time.time(); lap = lambda s: print(f"[{time.time()-T:5.0f}s] {s}", flush=True)
w = ad.read_h5ad("acon/aconcagua_selected.h5ad")
g = mt.tl.aggregate(w, by=("Metadata_Guide",), func="median", min_cells=1)
lap(f"guide consensus {g.shape}; obs {list(g.obs.columns)[:12]}")
o = g.obs
# Three disjoint control sets: scale reference, null pool, pseudo-genes (12 groups of 4).
ntc_guides = np.sort(o.loc[o.Metadata_Control_Type == "negcon", "Metadata_Guide"].astype(str).unique())
perm = np.random.default_rng(0).permutation(ntc_guides)
role = dict.fromkeys(perm[:100], "ntc_scale") | dict.fromkeys(perm[100:200], "nontargeting")
role |= {gd: f"PSEUDO{i // 4:02d}" for i, gd in enumerate(perm[200:248])}
label = o.Metadata_Guide.astype(str).map(role)
o["Analysis_Group"] = np.where(label.notna(), label, o.Metadata_Gene.astype(str))
o = o[o.Analysis_Group.notna()]
g = g[g.obs.Analysis_Group.isin(set(o.Analysis_Group))].copy() if False else g[~o.index.isin([])].copy()
g.obs["Analysis_Group"] = o.Analysis_Group.reindex(g.obs_names).to_numpy()
g = g[g.obs.Analysis_Group.notna() & (g.obs.Analysis_Group != "nan")].copy()
tier = dict(zip(g.obs.Metadata_Gene.astype(str), g.obs.Metadata_Tier.astype(str)))
print("  guides", g.n_obs, "| groups:", g.obs.Analysis_Group.str.startswith("PSEUDO").sum(), "pseudo guides,",
      (g.obs.Analysis_Group == "ntc_scale").sum(), "scale,", (g.obs.Analysis_Group == "nontargeting").sum(), "null")

def run(name, use_rep=None):
    a = g.copy()
    mt.tl.guide_activity(a, reference="ntc_scale", group="Analysis_Group", use_rep=use_rep)
    mt.tl.aggregate_guides(a, score="guide_activity", guide="Metadata_Guide", gene="Analysis_Group", control="nontargeting")
    r = pd.DataFrame(a.uns["mantispy"]["gene_aggregation"]).set_index("gene")
    r["tier"] = [("pseudo" if k.startswith("PSEUDO") else tier.get(k, "?")) for k in r.index]
    s = r.groupby("tier").agg(n=("is_hit", "size"), called=("is_hit", "sum"), p05=("pvalue", lambda p: (p < 0.05).mean()))
    print(f"\n== {name}")
    print(s.assign(rate=(s.called / s.n).round(3)).to_string())
    for k in ["H2AX", "TUBB", "CFL1", "ECT2", "OR4F29", "OTOP1", "AGAP9"]:
        if k in r.index: print(f"   {k:7s} q={r.loc[k,'qvalue']:.3g} hit={bool(r.loc[k,'is_hit'])} ({r.loc[k,'tier']})")
    return r

ra = run("all selected features")
x = np.nan_to_num(g.X); scale = g.obs.Analysis_Group == "ntc_scale"
pca = sc.pp.pca(x[scale.to_numpy()], n_comps=50, return_info=True)
g.obsm["X_pca_ctrl"] = (x - x[scale.to_numpy()].mean(0)) @ pca[1].T
rp = run("50 PCs fitted on the scale controls", use_rep="X_pca_ctrl")
ra.to_csv("acon/gene_calls_features.csv"); rp.to_csv("acon/gene_calls_pca.csv")
lap("done")
