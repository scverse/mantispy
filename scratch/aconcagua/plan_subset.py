"""Pick the 387-gene subset and per-(unit, well, class) keep probabilities from the screen-wide metadata."""
import sys, numpy as np, pandas as pd
out = sys.argv[1]
m = pd.read_parquet(f"{out}/../meta.parquet", columns=["plate", "well", "perturbation_id", "barcode", "is_control", "class"])
genes = [g for g in m.perturbation_id.cat.categories if not g.startswith("nontargeting")]
import anndata as ad
# Known hits come from evidence independent of our own feature analysis: brieflow's published interphase
# gene table (largest |z| over its 76 features) and our screen-wide mitotic index.
inter = ad.read_h5ad(f"{out}/../brief/interphase.h5ad")
effect = pd.Series(np.nanmax(np.abs(np.asarray(inter.X)), axis=1), index=inter.obs_names)
ntc_max = effect[effect.index.str.startswith("nontargeting")].max()
effect = effect[~effect.index.str.startswith("nontargeting")].sort_values(ascending=False)
mitotic = pd.read_csv(f"{out}/../mitotic_index_genes.csv", index_col=0)
positive = ["H2AX", "ECT2", "TUBB", "CFL1"]
# Diversity: walk each ranking and take at most two genes per brieflow functional cluster.
inter_cluster = inter.obs["cluster_group_1"].astype(str)
mito_cluster = ad.read_h5ad(f"{out}/../brief/mitotic.h5ad").obs["cluster_group_1"].astype(str)

def diverse(ranked, k, clusters, taken, per_cluster=2):
    picked, seen = [], {}
    for gene in ranked:
        if gene in taken or gene in picked:
            continue
        c = clusters.get(gene, gene)
        if seen.get(c, 0) < per_cluster:
            picked.append(gene)
            seen[c] = seen.get(c, 0) + 1
        if len(picked) == k:
            break
    return picked

strong = diverse(effect.index, 40, inter_cluster, set(positive))
arrest = mitotic[(mitotic.q < 0.05) & (mitotic.z > 0)].sort_values("z", ascending=False).index
strong += diverse(arrest, 10, mito_cluster, set(positive) | set(strong))
# Weak: genes just beyond the strongest non-targeting guide, i.e. real but small.
weak = diverse(effect[effect > ntc_max].index[::-1], 50, inter_cluster, set(positive) | set(strong))
chosen = positive + strong + weak
rest = [g for g in genes if g not in chosen]
background = list(np.random.default_rng(0).choice(rest, 500 - len(chosen), replace=False))
bench = chosen
labels = ["positive_control"] * len(positive) + ["strong"] * len(strong) + ["weak"] * len(weak)
pd.DataFrame({"gene": bench + background, "set": labels + ["random"] * len(background)}).to_csv(f"{out}/genes.csv", index=False)
s = m[m.perturbation_id.isin(set(bench) | set(background)) | m.is_control]
unit = np.where(s.is_control, s.barcode.astype(str), s.perturbation_id.astype(str))
n = s.groupby([unit, s.plate.values, s.well.astype(str).values, s["class"].astype(str).values]).size().rename("n").reset_index()
n.columns = ["unit", "plate", "well", "class", "n"]
ctrl = n.unit.isin(set(s.loc[s.is_control, "barcode"].astype(str)))
cap = np.where(n["class"] == "Mitotic", np.inf, np.where(ctrl, 20, 40))
n["p_keep"] = np.minimum(1.0, cap / n.n)
n.to_parquet(f"{out}/keep_prob.parquet")
print(len(bench), "known (incl. 4 positive controls) +", len(background), "random genes; expected cells", int((n.n * n.p_keep).sum()))
