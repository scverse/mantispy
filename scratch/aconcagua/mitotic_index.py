"""Screen-wide mitotic index per gene, calibrated against random groups of non-targeting guides."""
import numpy as np, pandas as pd
from scipy.stats import false_discovery_control
m = pd.read_parquet("acon/meta.parquet", columns=["perturbation_id", "barcode", "is_control", "class"])
m["mitotic"] = m["class"] == "Mitotic"
g = m.groupby("barcode", observed=True).agg(gene=("perturbation_id", "first"), ctrl=("is_control", "first"), n=("mitotic", "size"), k=("mitotic", "sum"))
g = g[g.n >= 200]  # a guide with fewer cells gives a too-noisy proportion
g["logit"] = np.log((g.k + 0.5) / (g.n - g.k + 0.5))
ntc = g[g.ctrl]
centre, scale = ntc.logit.median(), 1.4826 * (ntc.logit - ntc.logit.median()).abs().median()
g["z"] = (g.logit - centre) / scale
genes = g[~g.ctrl].groupby("gene", observed=True).agg(z=("z", "mean"), guides=("z", "size"), cells=("n", "sum"), mi=("k", "sum"))
genes["mi"] = genes.mi / genes.cells
rng = np.random.default_rng(0); nz = ntc.assign(z=(ntc.logit - centre) / scale).z.to_numpy()
null = {k: np.array([rng.choice(nz, k, replace=False).mean() for _ in range(20000)]) for k in genes.guides.unique()}
genes["p"] = [((np.abs(null[k]) >= abs(z)).sum() + 1) / (len(null[k]) + 1) for z, k in zip(genes.z, genes.guides)]
genes["q"] = false_discovery_control(genes.p)
ntc_mi = ntc.k.sum() / ntc.n.sum()
genes.sort_values("z", ascending=False).to_csv("acon/mitotic_index_genes.csv")
print(f"guides kept {len(g)} ({g.ctrl.sum()} NTC); NTC mitotic index {ntc_mi:.3f}; genes {len(genes)}")
print(f"q<0.05: up {((genes.q<0.05)&(genes.z>0)).sum()}, down {((genes.q<0.05)&(genes.z<0)).sum()}")
# Calibration: pseudo-genes of 4 NTC guides scored the same way should come out null.
pseudo = np.array([rng.choice(nz, 4, replace=False).mean() for _ in range(2000)])
pp = np.array([((np.abs(null[4]) >= abs(z)).sum() + 1) / (len(null[4]) + 1) for z in pseudo])
print(f"pseudo-gene p<0.05: {(pp<0.05).mean():.3f} (expect 0.05)")
print("top arrest (MI up):"); print(genes.head(15)[["mi", "z", "q", "guides"]].round(4).to_string())
print("top MI down:"); print(genes.tail(8)[["mi", "z", "q", "guides"]].round(4).to_string())
bench = {"spindle/kinetochore": "NDC80 NUF2 SPC24 SPC25 KNL1 MIS12 KIF11 TPX2 PLK1 AURKA", "APC/C": "ANAPC1 ANAPC2 ANAPC4 ANAPC5 CDC20 CDC27",
         "CPC": "AURKB INCENP BIRC5 CDCA8", "replication": "MCM2 MCM3 MCM4 MCM5 MCM6 MCM7 GINS1 GINS2 GINS3 GINS4", "ribosome": "RPL5 RPL11 RPS6 RPS19", "CDK1/CCNB1": "CDK1 CCNB1"}
for k, v in bench.items():
    s = genes.reindex(v.split()).dropna()
    print(f"{k:20s} median MI {s.mi.median():.3f} (NTC {ntc_mi:.3f}), median z {s.z.median():+.1f}, q<0.05 {int((s.q<0.05).sum())}/{len(s)}")
