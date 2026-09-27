# Rohban 2017 reproduction with mantispy

Computational reproduction of Rohban et al. 2017 (eLife 6:e24060), *Systematic morphological profiling of human gene and allele function via Cell Painting*, starting from the well-level augmented CellProfiler profiles shipped by `mt.ds.rohban()` (five pilot plates of `cpg0017-rohban-pathways`; 1918 wells, 190 screened genes). Produced by `benchmarks/rohban/reproduce.py`.

**Pipeline:** normalize per plate to the untreated (EMPTY) wells -> feature select (751 features) -> PCA (36 PCs, >=99% variance) -> active call (literal Rohban criterion + `mt.tl.percent_replicating` + `mt.tl.hit_calling`) -> modz consensus per gene -> Pearson similarity -> average-linkage clustering (1-Pearson, cut 0.522) -> GO Fisher enrichment + BioGRID PPI enrichment.

| ID | Quantity | Published | mantispy | Agreement | Note |
|----|----------|-----------|----------|-----------|------|
| G1 | active fraction | 50% (110/220) | 74.2% (percent_replicating); 22.6% (literal Rohban) | FAIL | paper's 50% is bracketed by the two faithful nulls [23%, 74%]; mantispy's matched-null percent_replicating over-calls, the literal per-pair criterion under-calls on the pilot subset |
| G2 | active count | 110 | 141 (percent_replicating) / 43 (literal) of 190 | FAIL | only 5 pilot plates ship (190 genes vs 220); grade the fraction (G1) |
| G3 | active criterion reproduced | median rep Pearson > 95th-pct non-rep | implemented exactly | PASS | literal_rohban_activity() reproduces C5; mt.tl.percent_replicating uses the modern matched-median null |
| G4 | # clusters (>=2 constructs) | 25 | 26 | PASS | average linkage, 1-Pearson on selected-feature consensus, cut 0.522 |
| G5 | Hippo/YAP co-cluster | YAP1+WWTR1 (cluster 20) | YAP1 & WWTR1 in cluster 1: True | PASS |  |
| G6 | RAS-RAF-MEK-ERK co-cluster | >=2 cascade genes | ['KRAS', 'MAP2K1', 'MAP2K4'] | PASS | gene-level |
| G7 | NF-kB(TRAF2) vs YAP anti-corr | strong negative | mean r=-0.252 (4th pct) | PASS | cluster 11 vs 20; among the most negative inter-cluster means |
| G8 | GO/complex-enriched clusters | 19/22 | 10/26 | FAIL | one-tailed Fisher, BH per cluster; GO-BP + CORUM + Reactome (Enrichr) |
| G9 | BioGRID PPI enrichment | 9% vs 5%, p=0.04 | 10.6% vs 9.1%, p=0.00786 | PASS | Fisher one-sided |
| G10 | correlation threshold | Pearson 0.43 | 0.411 | PASS | top-5% / 95th-pct non-replicate cut on well-level correlations (~top 5% of pairs) |
| G11 | NF-kB/YAP GSEA | BH p=2e-8 | n/a | FLAG | needs external L1000 signatures; out of core scope |

## Verdict

- **Reproduced:** the clustering biology and the correlation scale. Average-linkage clustering on 1-Pearson (cut 0.522) recovers 26 multi-construct clusters (paper 25), the YAP1+WWTR1 Hippo co-cluster (G5), RAS-RAF-MEK-ERK co-clustering (KRAS, MAP2K1, MAP2K4) (G6), and the NF-kB/TRAF2 vs YAP anti-correlation (mean Pearson -0.25, G7). The top-5% non-replicate correlation cut lands at 0.41, matching the paper's 0.43 (G10), and top-correlated pairs are enriched for BioGRID interactions (11% vs 9%, p=0.0079, G9).

- **Partially reproduced:** GO/complex enrichment (G8) covers 10/26 multi-gene clusters vs the paper's 19/22. Our clustering of all 190 screened genes produces more, smaller clusters (many gene pairs) than the paper's clustering of its 110 active genes, and small clusters clear a per-cluster FDR less often.

- **Did not reproduce exactly:** the 50% active-fraction headline (G1/G2). The paper's value is *bracketed* by the two faithful mantispy nulls on this 5-plate pilot: the literal Rohban per-pair criterion under-calls at 23% (per-gene median replicate Pearson ~0.20, about half the paper's implied ~0.41), while `mt.tl.percent_replicating`'s modern matched-median null (the Way et al. 'percent replicating' standard) over-calls at 74%. Both are valid activity tests; they differ in how the non-replicate null is built.

- **Capability gaps confirmed (for maintainers):** (1) no clustering/dendrogram primitive - scipy `linkage`/`fcluster` used; (2) no PCA in `pp` - `scanpy.pp.pca` used, and 99% variance here is only ~36 PCs (the augmented profiles are far more redundant than the paper's feature table), so the PCA space over-compresses the between-gene structure and clustering runs on the selected-feature profiles instead; (3) `gene_sets` has no GO/KEGG (GO-BP supplied via Enrichr); (4) no PPI/BioGRID primitive - raw `scipy.stats.fisher_exact`; (5) the loader emits `Metadata_gene_name` but the knowledge functions default `gene_key="Metadata_Gene"`, so `gene_key` must be passed explicitly; (6) no median-polish plate detrending (paper step 4).

- **Where mantispy improves on the original:** one seeded, deterministic AnnData script replaces the original's multi-file R/knitr + MySQL setup; `tl.consensus` (modz) is more robust to a single bad replicate than a plate median; `tl.percent_replicating` and `tl.hit_calling` offer calibrated activity nulls; `tl.pathway_coherence` tests within-pathway profile similarity with a permutation null, a stronger claim than cluster-then-Fisher. Note that `hit_calling` here is conservative (0/190 at 36 PCs), not the over-caller the plan expected.

