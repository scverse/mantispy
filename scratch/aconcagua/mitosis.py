"""Step 4: how the mitotic fraction should enter the pooled analysis. Three variants, one calibrated pipeline."""
import warnings
import anndata as ad, numpy as np, pandas as pd, scanpy as sc, mantispy as mt
warnings.filterwarnings("ignore"); mt.settings.verbosity = 0
UNEXP = mt.io.unexpressed_genes("depmap/OmicsExpressionProteinCodingGenesTPMLogp1.csv", "ACH-001086")

def call(w):
    """Guide consensus -> 50 PCs on all guides -> guide_activity -> aggregate_guides; two nulls."""
    g = mt.tl.aggregate(w, by=("Metadata_Guide",), func="median", min_cells=1); o = g.obs
    ntc = np.sort(o.loc[o.Metadata_Control_Type == "negcon", "Metadata_Guide"].astype(str).unique()); perm = np.random.default_rng(0).permutation(ntc)
    role = dict.fromkeys(perm[:100], "ntc_scale") | dict.fromkeys(perm[100:200], "nontargeting") | {x: f"PSEUDO{i//4:02d}" for i, x in enumerate(perm[200:248])}
    lab = o.Metadata_Guide.astype(str).map(role); o["Analysis_Group"] = np.where(lab.notna(), lab, o.Metadata_Gene.astype(str))
    g.X = np.nan_to_num(g.X); sc.pp.pca(g, n_comps=50)
    mt.tl.guide_activity(g, reference="ntc_scale", group="Analysis_Group", use_rep="X_pca")
    mt.tl.aggregate_guides(g, score="guide_activity", guide="Metadata_Guide", gene="Analysis_Group", control="nontargeting")
    r = pd.DataFrame(g.uns["mantispy"]["gene_aggregation"]).set_index("gene")
    tier = dict(zip(o.Metadata_Gene.astype(str), o.Metadata_Tier.astype(str)))
    r["tier"] = ["pseudo" if k.startswith("PSEUDO") else tier.get(k, "?") for k in r.index]
    genes = r[r.tier != "pseudo"]; null = sorted(set(genes.index) & UNEXP)
    e = ad.AnnData(obs=pd.DataFrame({"gene": genes.index, "score": genes.statistic.to_numpy()}, index=genes.index))
    mt.tl.empirical_fdr(e, control_genes=null, score="score", group="gene", alpha=0.05, criterion="q")
    ef = pd.DataFrame(e.uns["mantispy"]["empirical_fdr"]).set_index("group")
    r["unexp_hit"] = ef.is_hit.reindex(r.index); r["unexpressed"] = r.index.isin(null)
    return r

pooled = ad.read_h5ad("acon/aconcagua_selected.h5ad"); features = pooled.var_names
results = {"pooled": call(pooled)}

# Regress out: slope of each feature on mitotic fraction, fitted on control guide x wells only.
reg = pooled.copy(); f = reg.obs.Metadata_MitoticFraction.to_numpy(np.float64); ctrl = (reg.obs.Metadata_Control_Type == "negcon").to_numpy()
X = np.asarray(reg.X, dtype=np.float64); fc = f[ctrl] - f[ctrl].mean()
slope = np.nansum(fc[:, None] * (X[ctrl] - np.nanmean(X[ctrl], 0)), 0) / (fc ** 2).sum()
reg.X = (X - np.outer(f - f[ctrl].mean(), slope)).astype(np.float32)
top = pd.Series(np.abs(slope), index=features).nlargest(5)
print("largest mitotic-fraction slopes (z per unit fraction):", {k: round(v, 1) for k, v in top.items()})
results["regressed"] = call(reg)

# Interphase only: rebuild guide x well profiles from the interphase cells.
parts = []
for p in range(1, 9):
    c = ad.read_h5ad(f"acon/subset_v3/cells_plate{p}.h5ad")
    c = c[c.obs.Metadata_Class == "Interphase"].copy()
    mt.pp.normalize(c, by=["Metadata_Plate", "Metadata_Well"], reference="negcon")
    parts.append(mt.tl.aggregate(c, by=("Metadata_Plate", "Metadata_Well", "Metadata_Guide"), func="median", min_cells=5)[:, features].copy())
    del c
inter = ad.concat(parts, merge="same", uns_merge="first"); inter.obs_names = [str(i) for i in range(inter.n_obs)]
results["interphase"] = call(inter)

mi = pd.read_csv("acon/mitotic_index_genes.csv", index_col=0)
rows = []
for name, r in results.items():
    t = r[(r.tier != "pseudo") & ~r.unexpressed]
    row = {"variant": name, "pseudo_called": int(r[r.tier == "pseudo"].is_hit.sum())}
    for tr in ["positive_control", "strong", "weak", "random"]:
        row[f"{tr} vsNTC"] = round(t[t.tier == tr].is_hit.mean(), 3); row[f"{tr} beyond"] = round(t[t.tier == tr].unexp_hit.astype(float).mean(), 3)
    j = t.join(mi.z.rename("mi_z"), how="inner")
    row["rho(stat, MI z)"] = round(j.statistic.corr(j.mi_z, method="spearman"), 3)
    rows.append(row)
print(pd.DataFrame(rows).set_index("variant").T.to_string())
arrest = mi[(mi.q < 0.05) & (mi.z > 0)].index
for name, r in results.items():
    a = r.reindex(arrest).dropna(subset=["is_hit"]); a = a[a.tier != "pseudo"]
    print(f"{name:10s} MI-arrest genes in subset: {len(a)}, called vs NTC {a.is_hit.mean():.2f}, beyond cutting {a.unexp_hit.astype(float).mean():.2f}")
pd.concat({k: v for k, v in results.items()}, names=["variant"]).to_csv("acon/mitosis_variants.csv")
