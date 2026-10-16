"""Hit calling on 50 PCs fitted on all guides; per-guide calibration; two nulls (non-targeting, unexpressed)."""
import warnings
import anndata as ad, numpy as np, pandas as pd, scanpy as sc, mantispy as mt
from scipy.stats import kstest
warnings.filterwarnings("ignore"); mt.settings.verbosity = 1
w = ad.read_h5ad("acon/aconcagua_selected.h5ad")
g = mt.tl.aggregate(w, by=("Metadata_Guide",), func="median", min_cells=1)
o = g.obs
ntc = np.sort(o.loc[o.Metadata_Control_Type == "negcon", "Metadata_Guide"].astype(str).unique()); perm = np.random.default_rng(0).permutation(ntc)
role = dict.fromkeys(perm[:100], "ntc_scale") | dict.fromkeys(perm[100:200], "nontargeting") | {x: f"PSEUDO{i//4:02d}" for i, x in enumerate(perm[200:248])}
lab = o.Metadata_Guide.astype(str).map(role); o["Analysis_Group"] = np.where(lab.notna(), lab, o.Metadata_Gene.astype(str))
g.X = np.nan_to_num(g.X); sc.pp.pca(g, n_comps=50)  # unsupervised, every guide; the reference is 4 % of the fit
mt.tl.guide_activity(g, reference="ntc_scale", group="Analysis_Group", use_rep="X_pca")
p = g.obs.guide_activity; grp = o.Analysis_Group
for name, m in [("null-pool guides", grp == "nontargeting"), ("pseudo guides", grp.str.startswith("PSEUDO"))]:
    print(f"{name}: KS vs uniform p={kstest(p[m], 'uniform').pvalue:.2f}, share p<0.05 {(p[m] < 0.05).mean():.3f}, share p==1 {(p[m] == 1).mean():.3f}")
mt.tl.aggregate_guides(g, score="guide_activity", guide="Metadata_Guide", gene="Analysis_Group", control="nontargeting")
r = pd.DataFrame(g.uns["mantispy"]["gene_aggregation"]).set_index("gene")
tier = dict(zip(o.Metadata_Gene.astype(str), o.Metadata_Tier.astype(str)))
r["tier"] = ["pseudo" if k.startswith("PSEUDO") else tier.get(k, "?") for k in r.index]
un = mt.io.unexpressed_genes("depmap/OmicsExpressionProteinCodingGenesTPMLogp1.csv", "ACH-001086")
genes = r[r.tier != "pseudo"]; null = sorted(set(genes.index) & un)
e = ad.AnnData(obs=pd.DataFrame({"gene": genes.index, "score": genes.statistic.to_numpy()}, index=genes.index))
mt.tl.empirical_fdr(e, control_genes=null, score="score", group="gene", alpha=0.05, criterion="q")
ef = pd.DataFrame(e.uns["mantispy"]["empirical_fdr"]).set_index("group")
r["unexp_q"] = ef.qvalue; r["unexp_hit"] = ef.is_hit; r["unexpressed"] = r.index.isin(null)
print(f"\npseudo-genes called vs NTC: {int(r[r.tier=='pseudo'].is_hit.sum())}/{(r.tier=='pseudo').sum()}; unexpressed genes called vs NTC: {r.loc[null,'is_hit'].mean():.2f} ({len(null)})")
t = r[(r.tier != "pseudo") & ~r.unexpressed]
print(t.groupby("tier").agg(n=("is_hit", "size"), vs_NTC=("is_hit", "mean"), vs_unexpressed=("unexp_hit", "mean")).round(3).to_string())
for k in ["H2AX", "TUBB", "CFL1", "ECT2", "OR4F29", "AGAP9"]:
    print(f"  {k:7s} vs NTC q={r.loc[k,'qvalue']:.3g} | vs unexpressed q={r.loc[k,'unexp_q']:.3g}")
r.to_csv("acon/gene_calls_two_nulls.csv"); g.write_h5ad("acon/aconcagua_guides_scored.h5ad")
