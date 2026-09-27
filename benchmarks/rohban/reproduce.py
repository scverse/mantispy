"""Reproduce the computational results of Rohban et al. 2017 (eLife 6:e24060) with mantispy.

Rohban et al. profiled U2OS cells overexpressing single ORFs by Cell Painting and reported, among
other things, that ~50% of QC-passing constructs are phenotypically active, that average-linkage
clustering on 1-Pearson recovers pathway co-clusters (Hippo/YAP, RAS-RAF-MEK-ERK, NF-kB/TRAF2),
that "highly correlated" gene pairs sit above a Pearson of ~0.43, and that those top pairs are
enriched for known protein-protein interactions.

This is the v2 reproduction: it runs at the paper's CONSTRUCT level (the loader now sets
``Metadata_Perturbation`` to the ORF construct, ~323 of them) and uses mantispy's own interpretation
layer instead of hand-rolled scipy/scanpy, namely ``mt.tl.consensus``, ``mt.tl.percent_replicating``,
``mt.tl.cluster``, ``mt.tl.ora`` (against ``mt.ds.gene_sets``) and ``mt.tl.network_enrichment``
(against ``mt.ds.interactions``). It grades each mantispy number against the published one (targets
G1..G11), ends with explicit asserts, and writes ``REPRODUCTION.md`` with an old-vs-new-vs-paper table.

Two data facts the loader already resolves (see the docstring of ``mt.ds.rohban``):
  1. The normalization reference is the UNTREATED (EMPTY) wells, not the ORF transfection controls.
  2. ``Metadata_Perturbation`` is the construct and ``Metadata_Gene`` the gene (194 of them), so the
     active call and clustering run per construct and the enrichment groups by gene.

Run from the repo root inside the project venv:

    python benchmarks/rohban/reproduce.py
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc

import mantispy as mt

warnings.filterwarnings("ignore")
RNG_SEED = 0
HERE = Path(__file__).resolve().parent

# The distance the paper cut its average-linkage tree at (Methods; chosen by a stability sweep).
PAPER_CUT = 0.522
# RAS-RAF-MEK-ERK cascade genes to look for co-clustering (the subset present in these pilot plates).
RAS_CASCADE = {"KRAS", "HRAS", "NRAS", "BRAF", "RAF1", "MAP2K1", "MAP2K3", "MAP2K4", "MAPK1", "MAPK3", "SOS1"}
# NF-kB / TRAF2 module: TRAF2 is the paper's cluster-11 anchor; the rest are its fallbacks if inactive.
NFKB_GENES = ["TRAF2", "RELA", "NFKB1", "REL", "CHUK", "IKBKB", "RELB", "NFKB2", "TRAF6", "MAP3K7"]

# The v1 (gene-level) reproduction's numbers, for the old-vs-new column of the report.
OLD = {
    "G1": "74.2% (percent_replicating); 22.6% (literal)",
    "G2": "141 / 190",
    "G3": "implemented exactly",
    "G4": "26",
    "G5": "YAP1 & WWTR1 co-clustered",
    "G6": "KRAS, MAP2K1, MAP2K4",
    "G7": "mean r=-0.252 (4th pct)",
    "G8": "10 / 26 (per-cluster BH)",
    "G9": "10.6% vs 9.1%, p=0.0079 (BioGRID)",
    "G10": "0.411",
    "G11": "n/a",
}

results: dict[str, dict] = {}


def banner(text: str) -> None:
    print(f"\n{'=' * 78}\n{text}\n{'=' * 78}")


def row(gid, quantity, published, new, passed, note):
    verdict = "PASS" if passed is True else ("FAIL" if passed is False else "FLAG")
    results[gid] = dict(quantity=quantity, published=published, old=OLD[gid], new=new, verdict=verdict, note=note)
    print(f"  {gid:4s} {verdict:4s} {quantity}: pub={published} | new={new}")


# =============================================================================================
# Step 0-1: load + mark the untreated reference (the loader already sets the construct/gene split)
# =============================================================================================
banner("Step 0-1: load rohban well profiles at construct level")
adata = mt.ds.rohban()
adata.obs["is_untreated"] = (adata.obs["Metadata_Perturbation_Type"].astype(str) == "untreated").to_numpy()
n_constructs = int(adata.obs.loc[~adata.obs["is_untreated"] & ~adata.obs["Metadata_Control"], "Metadata_Perturbation"].nunique())
n_genes = int(adata.obs["Metadata_Gene"].astype(str).nunique())
print(f"loaded {adata.n_obs} wells x {adata.n_vars} features")
print(f"  screened ORF constructs (Metadata_Perturbation): {n_constructs}; genes (Metadata_Gene): {n_genes}")
print(f"  untreated (EMPTY) wells: {int(adata.obs['is_untreated'].sum())}; control wells: {int(adata.obs['Metadata_Control'].sum())}")

# =============================================================================================
# Step 2: per-plate MAD normalization to the untreated (EMPTY) wells
# =============================================================================================
banner("Step 2: normalize per plate to the untreated (EMPTY) wells")
mt.pp.normalize(adata, method="mad_robustize", by="Metadata_Plate", reference="is_untreated")
print(f"  degenerate (unscalable) features flagged: {int(adata.var['degenerate_scale'].sum())}")

# =============================================================================================
# Step 3: feature selection (pycytominer defaults; approximates the paper's MAD=0 + redundancy pruning)
# =============================================================================================
banner("Step 3: feature selection")
mt.pp.feature_select(
    adata,
    operations=("drop_degenerate", "variance_threshold", "correlation_threshold", "drop_na_columns", "blocklist"),
)
adata = mt.pp.subset_features(adata)
adata.X = np.nan_to_num(np.asarray(adata.X, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
print(f"  features kept: {adata.n_vars}")

# =============================================================================================
# Step 4: PCA to 99% variance (paper: 158 PCs; the shipped augmented profiles are lower-rank)
# =============================================================================================
banner("Step 4: PCA to 99% variance")
n_try = int(min(adata.n_vars - 1, adata.n_obs - 1, 400))
sc.pp.pca(adata, n_comps=n_try, svd_solver="arpack", random_state=RNG_SEED)
cum = np.cumsum(adata.uns["pca"]["variance_ratio"])
n_pcs = min(int(np.searchsorted(cum, 0.99) + 1), n_try)
adata.obsm["X_pca"] = adata.obsm["X_pca"][:, :n_pcs].copy()
print(f"  PCs for >=99% variance: {n_pcs} (cum var {cum[n_pcs - 1]:.4f}); paper: 158 (data here is lower-rank)")

# The screened set: the ORF constructs, without the untreated EMPTY wells or the transfection controls.
screen = adata[~adata.obs["is_untreated"].to_numpy() & ~adata.obs["Metadata_Control"].to_numpy()].copy()

# =============================================================================================
# Step 5: active-construct call via mt.tl.percent_replicating [G1-G3]
# =============================================================================================
banner("Step 5: active-construct call [G1-G3]")
mt.tl.percent_replicating(
    screen, groupby="Metadata_Perturbation", metric="pearson", quantile=0.95, use_rep="X_pca", seed=RNG_SEED
)
pr = screen.uns["mantispy"]["percent_replicating"]
n_active = int(pr["is_replicating"].sum())
frac_active = float(pr["is_replicating"].mean())
print(f"  mt.tl.percent_replicating (95th-pct non-replicate null, X_pca): "
      f"active {n_active}/{len(pr)} = {100 * frac_active:.1f}% of constructs")

# =============================================================================================
# Step 6: one consensus profile per construct via mt.tl.consensus (modz)
# =============================================================================================
banner("Step 6: modz consensus per construct")
cons = mt.tl.consensus(screen, by="Metadata_Perturbation", method="modz", correlation="spearman", min_replicates=2)
cons.X = np.nan_to_num(np.asarray(cons.X, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
print(f"  consensus profiles: {cons.n_obs} constructs x {cons.n_vars} features")

# =============================================================================================
# Step 7: hierarchical clustering via mt.tl.cluster + Pearson similarity [G4]
# =============================================================================================
banner("Step 7: average-linkage clustering (1-Pearson) + Pearson similarity [G4]")
mt.tl.cluster(cons, use_rep=None, method="hierarchical", linkage="average", metric="correlation", distance_cut=PAPER_CUT)
mt.tl.similarity(cons, metric="pearson", use_rep=None)

lab = cons.obs["cluster"].astype(str)
gene = cons.obs["Metadata_Gene"].astype(str)
sizes = lab.value_counts()
n_clusters_ge2 = int((sizes >= 2).sum())
genes_per_cluster = gene.groupby(lab, observed=True).nunique()
multigene = set(genes_per_cluster[genes_per_cluster >= 2].index)
n_multigene = len(multigene)

# The mantispy-native default cut (silhouette sweep), reported alongside the paper's fixed cut.
auto = cons.copy()
mt.tl.cluster(auto, use_rep=None, method="hierarchical", linkage="average", metric="correlation")
n_auto = int(auto.obs["cluster"].nunique())
print(f"  cut {PAPER_CUT}: {n_clusters_ge2} clusters with >=2 constructs, {n_multigene} multi-gene (paper: 25, 22)")
print(f"  auto-cut (silhouette): {n_auto} clusters at height {auto.uns['mantispy']['cluster']['distance_cut']:.3f}")

# =============================================================================================
# Step 8: pathway biology from the cluster labels + similarity [G5-G7]
# =============================================================================================
banner("Step 8: pathway co-clusters + anti-correlation [G5-G7]")
S = np.asarray(cons.obsp["similarity"], dtype=np.float64)
pos = {name: i for i, name in enumerate(cons.obs_names)}
lab_by_name = lab.to_dict()


def clusters_of(g: str) -> set[str]:
    return set(lab[gene == g])


def members(cluster: str) -> list[int]:
    return [pos[name] for name in cons.obs_names if lab_by_name[name] == cluster]


def mean_between(a: list[int], b: list[int]) -> float:
    return float(np.mean(S[np.ix_(a, b)])) if a and b else float("nan")


# G5: Hippo/YAP -- a YAP1 construct and a WWTR1 construct share a cluster.
yap_clusters, taz_clusters = clusters_of("YAP1"), clusters_of("WWTR1")
hippo_shared = yap_clusters & taz_clusters
hippo_co = bool(hippo_shared)
yap_cluster = sorted(hippo_shared)[0] if hippo_shared else (sorted(yap_clusters)[0] if yap_clusters else None)
print(f"  YAP1 clusters={sorted(yap_clusters)}, WWTR1 clusters={sorted(taz_clusters)} -> Hippo co-cluster: {hippo_co}")

# G6: RAS-RAF-MEK-ERK -- a cluster holding >=2 distinct cascade genes.
cascade_present = sorted(g for g in RAS_CASCADE if (gene == g).any())
cascade_by_cluster: dict[str, set[str]] = {}
for g in cascade_present:
    for c in clusters_of(g):
        cascade_by_cluster.setdefault(c, set()).add(g)
ras_group = max((genes for genes in cascade_by_cluster.values() if len(genes) >= 2), key=len, default=set())
ras_co = bool(ras_group)
print(f"  RAS cascade present={cascade_present}; >=2 co-clustered: {ras_co}; group={sorted(ras_group)}")

# G7: the NF-kB/TRAF2 cluster anti-correlates with the YAP cluster.
nfkb_gene = next((g for g in NFKB_GENES if (gene == g).any() and clusters_of(g)), None)
nfkb_clusters = clusters_of(nfkb_gene) if nfkb_gene else set()
# Pick the NF-kB cluster that is not the YAP cluster, so the anti-correlation is between two modules.
nfkb_cluster = next((c for c in sorted(nfkb_clusters) if c != yap_cluster), None)
yap_nfkb = mean_between(members(yap_cluster), members(nfkb_cluster)) if (yap_cluster and nfkb_cluster) else float("nan")
multi = sorted(multigene)
inter = np.array(
    [mean_between(members(a), members(b)) for i, a in enumerate(multi) for b in multi[i + 1:]],
    dtype=np.float64,
)
inter = inter[np.isfinite(inter)]
pctile = float((inter < yap_nfkb).mean() * 100) if inter.size else float("nan")
anti_corr = bool(np.isfinite(yap_nfkb) and yap_nfkb < 0 and pctile <= 25)
print(f"  YAP cluster {yap_cluster} vs NF-kB/{nfkb_gene} cluster {nfkb_cluster}: mean Pearson={yap_nfkb:.3f} "
      f"(more negative than {100 - pctile:.0f}% of inter-cluster means); anti-corr: {anti_corr}")

