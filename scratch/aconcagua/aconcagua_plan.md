# Aconcagua OPS tutorial: agreed plan

## Decisions
| Topic | Decision |
|---|---|
| Deliverable | New "Optical pooled screens" tutorial under genetic screens, backed by `mt.ds.aconcagua` |
| Hosted data | Guide x well profiles, all features, float32, about 600 MB, uncompressed h5ad |
| Converter | Standalone script in the scratchpad for now (not in the package) |
| Primary hit calling | `guide_activity` then `aggregate_guides` (Stouffer); `hit_calling` on PCA and mAP as sanity checks |
| Cell cycle | Pool all cells per guide; mitotic fraction kept as a covariate |
| Batch | Normalize per plate+well on controls; sphere (TVN) only if controls still separate by plate or well |
| brieflow comparison | One short section near the end, framed as a pseudoreplication lesson |

## Subset (extraction running: acon/subset_v3)
- 250 non-targeting guides; 4 positive controls (H2AX, ECT2, TUBB, CFL1).
- 50 strong hits: top of brieflow's interphase effect sizes plus top mitotic-arrest genes, at most 2 per brieflow cluster (24 clusters).
- 50 weak hits: just above the strongest non-targeting guide, at most 2 per cluster (38 clusters). Includes likely-unexpressed OR4F29, OTOP1, AGAP9 as built-in negatives.
- 396 random library genes (seed 0).
- Cells: interphase capped at 40 per gene per well and 20 per control guide per well; all mitotic cells kept (about 1.2 M cells).

## Pipeline
1. **Convert** (scratch script): parts to a cell-level AnnData.
   - obs: Metadata_Plate, _Well, _Site (tile), _ObjectNumber, _Guide (barcode), _Gene (perturbation_id), _Perturbation (controls collapsed to "nontargeting"), _Control_Type (negcon / poscon / trt), _Class, _Tier.
   - Features renamed to mantispy's grammar; the 94 coordinate columns are already excluded.
2. **Restore natural cell-cycle proportions before pooling.** Mitotic cells were kept uncapped, so thin them with the same keep probability as the interphase cells of their guide and well. Otherwise pooled profiles would carry about 3x the natural mitotic share.
3. **Normalize:** `mad_robustize`, by plate+well, reference negcon. Done per plate to fit in 15 GB.
4. **Aggregate:** median of cells into guide x well rows, `min_cells=5`, recording cell count and mitotic fraction. This file is `mt.ds.aconcagua`.

## Tutorial outline
1. **Design:** guides, wells, controls; why the well is the replicate unit.
2. **QC:** cells per guide x well; positive controls move in their expected channel.
3. **Feature selection:** `feature_select` + `subset_features`, with the degenerate-feature lesson (zero-MAD foci and neighbour counts).
4. **Batch check:** do controls separate by plate or well? Sphere only if they do.
5. **Hit calling:** `guide_activity` against half the controls, then `aggregate_guides` with a null drawn from the other half. The two halves must differ, otherwise the calibration is circular.
6. **Calibration:** pseudo-genes of 4 held-out control guides; false-positive rate near 5 %.
7. **Recovery:** hit rate for strong vs weak vs random genes; how the unexpressed weak genes behave.
8. **Cell cycle:** which hits track mitotic fraction (screen-wide mitotic index already computed: 305 up, 218 down).
9. **Interpretation:** which channel x compartment moved, for example RRM1/DTL (γH2AX), SKA/HAUS (tubulin, arrest), ITGAV/TLN1 (actin).
10. **Sanity checks:** `hit_calling` on PCA and mAP agree with the primary call.
11. **Comparison:** brieflow's per-feature calls (17 % of control guides significant in interphase) vs a well-aware test.

## Success criteria
- Pseudo-gene false-positive rate at most about 5 %; no pseudo-gene survives FDR.
- All 4 positive controls called, each in its expected channel.
- Strong tier mostly called (≥ 80 %); weak tier recovered well above random genes; unexpressed weak genes not called.
- Tutorial runs within the docs CI budget.

## Open items
- Hosting `mt.ds.aconcagua` needs an upload to scverse-exampledata (maintainer access).
- Memory for cell-level normalization: process plate by plate.
