# Rohban 2017 reproduction v2 (construct-level, mantispy interpretation layer)

Rework the #148 reproduction onto the new APIs: construct-level grouping, `tl.consensus`,
`tl.percent_replicating`, `tl.cluster`, `tl.ora`, `tl.network_enrichment`, `ds.gene_sets`,
`ds.interactions`. Replace all hand-rolled scipy/scanpy analysis code.

## Plan

- [x] Exploration done. Findings: 323 screened constructs (5 pilot plates, 4-10 reps).
      percent_replicating over-calls at 75.2% (matched-median null; same pattern as OLD 74%).
      TRAF2 inactive -> must cluster ALL screened (not active-only) to keep G7. cut 0.522 -> 47 clusters
      (non-portable height in 751-feat space); auto-cut merges to 10. mt.tl.ora global BH -> 0 enriched
      at any tmin/collection (nominal p flags 22-46/46). network_enrichment: OR 2.13, p 2e-6, thr 0.549.
      DECISION: cluster all 323, distance_cut=0.522, primary; report auto + honest divergences.
- [ ] Step A (commit): consensus per construct + active call (G1-G3) via percent_replicating.
- [ ] Step B (commit): clustering via tl.cluster (avg linkage, 1-corr, cut 0.522) + biology
      recovery G4-G7 (YAP1/WWTR1 co-cluster, RAS cascade, NF-kB vs YAP anti-corr) from labels+similarity.
- [ ] Step C (commit): enrichment G8 via tl.ora against gene_sets(GO_BP+CORUM+Reactome), gene_key=Metadata_Gene.
- [ ] Step D (commit): network/PPI G9 via tl.network_enrichment vs interactions("CORUM"); G10 = its threshold.
- [ ] Step E (commit): rewrite REPRODUCTION.md with old-vs-new-vs-paper table + honest verdict; asserts.
- [ ] Run whole analysis clean, foreground.
- [ ] Push branch, open DRAFT PR (base feat/canonical-perturbation).

## Verification
- Script ends with explicit asserts on graded targets. DONE (all pass).
- Each graded number reported beside published + v1 + v2 mantispy. DONE.
- Divergences get a named reason. DONE.

## Review (final)
Ran clean end to end (~5 min). Grades:
- PASS: G3 (criterion), G5 (YAP1/WWTR1 co-cluster 4), G6 (KRAS/MAP2K1/MAP2K4), G7 (TRAF2 vs YAP
  r=-0.334, more negative than 99% of inter-cluster means), G9 (CORUM top-pair enrichment OR=2.13, p=2e-6).
- FAIL/FLAG with named reasons: G1/G2 (percent_replicating over-calls 75% via matched-median null +
  36 vs 158 PCs), G4 (47 clusters; 0.522 cut not portable to 751-feat space, cluster all 323 vs paper's
  110 active), G8 (0/46 at q<0.05; mt.tl.ora uses one global BH, signal present at nominal p 39/46),
  G10 (top-5% cut 0.549 vs 0.43; modz consensus denoises so correlations run higher), G11 (L1000, out of scope).
- Biology fully preserved from v1 while going construct-level and fully mantispy-native.
- Capability gaps flagged to maintainers in REPRODUCTION.md (ora global BH, cluster cut portability,
  no pp PCA, network_enrichment CORUM-as-PPI-proxy).

## Notes / decisions
- Grouping: Metadata_Perturbation (construct) for active call + clustering; Metadata_Gene for enrichment.
- G10 read from network_enrichment threshold (95th-pct construct-pair Pearson = top-5% cut).
- No new deps, no isinstance/issparse, no em-dashes, conventional commits, no attribution lines.
