# Genetic screens

Genetic screens perturb genes, by CRISPR knockout or by ORF overexpression, so each gene is measured through one or several reagents.
These tutorials aggregate reagents to genes and read the screen as clusters of genes that share a phenotype.

```{toctree}
:maxdepth: 1

genetics/interpreting_a_screen
genetics/hit_calling_methods
```

## Worked examples

- The {doc}`CRISPR case study <../case_studies/crispr>` recovers protein complexes from the JUMP CRISPR plates. It also holds the chromosome-arm diagnostic, which checks whether genes on the same arm look alike for position rather than biology.
- The {doc}`gene aggregation case study <../case_studies/gene_aggregation>` calls gene hits in a pooled screen from per-guide scores.
- The {doc}`jump_crispr dataset <../datasets/jump_crispr>` is the arrayed CRISPR screen the first of these uses.

## Guides and genes

`resolution` is `object`, `well` or `aggregate`.
There is no guide level, by design: a guide is a value in a column, not a kind of row.

- In an arrayed screen such as JUMP, each well carries one reagent per gene, so rows that look like guides are wells.
- To go from guides to genes, collapse the replicate profiles of a gene with {func}`~mantispy.tl.consensus` or {func}`~mantispy.tl.aggregate`.
- In a pooled screen where a gene is targeted by many guides, score each guide with {func}`~mantispy.tl.guide_activity` and combine them with {func}`~mantispy.tl.aggregate_guides`. The `cp_posh` dataset is the example.
- In a pooled screen many cells carry a guide but show no phenotype. {func}`~mantispy.tl.perturbation_score` scores each cell by how clearly a classifier tells it from the non-targeting cells, and gives each gene a ROC AUC. By default it keeps each guide's cells whole within a cross-validation fold (`replicate="Metadata_sgRNA"`), so a phenotype one guide carries alone does not inflate the gene; pass `replicate=None` for Brieflow's folds over cells. Pass `by="Metadata_Class"` to score interphase and mitotic cells separately. Subset on the score before aggregating to keep the responders.
