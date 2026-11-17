# Coming from Brieflow

[Brieflow](https://github.com/cheeseman-lab/brieflow) runs an optical pooled screen from raw images to clustered genes as a Snakemake workflow.
The page maps its aggregate and cluster stages to mantispy calls in the order Brieflow runs them.
Each step gives the arguments that reproduce Brieflow’s defaults and notes where mantispy differs on purpose.
The Brieflow names refer to functions under `workflow/lib` and `workflow/scripts` in the Brieflow repository.

% TODO(maintainer): add the mantispy and Brieflow benchmark comparison from the final re-benchmark here.
A benchmark comparison of mantispy and Brieflow on the same screen will be added to the page.

## Run order

  | Step               | Brieflow                                                     | mantispy                                                                                                          |
  | ------------------ | ------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------- |
  | Read               | parquet through pyarrow                                      | {func}`~mantispy.io.from_dataframe`, {func}`~mantispy.io.read_profiles`, {func}`~mantispy.io.read`                |
  | Filter             | `missing_values_filter` (drop part)                          | {func}`~mantispy.pp.feature_select`, {func}`~mantispy.pp.calculate_qc_metrics`, {func}`~mantispy.pp.filter_cells` |
  | KNN impute         | `missing_values_filter` (impute part)                        | {func}`~mantispy.pp.impute`                                                                                       |
  | Intensity filter   | `intensity_filter`                                           | {func}`~mantispy.pp.outliers`                                                                                     |
  | Per-batch scaling  | `centerscale_by_batch`, `centerscale_on_controls`            | {func}`~mantispy.pp.normalize`                                                                                    |
  | PCA                | `sklearn.decomposition.PCA` in `scripts/aggregate/align.py`  | {func}`scanpy.pp.pca`                                                                                             |
  | TVN and CORAL      | `tvn_on_controls`                                            | {func}`~mantispy.pp.tvn`                                                                                          |
  | Perturbation score | `perturbation_score`                                         | {func}`~mantispy.tl.perturbation_score`                                                                           |
  | Gene aggregation   | `aggregate`                                                  | {func}`~mantispy.tl.aggregate`                                                                                    |
  | Pseudo-genes       | `create_pseudogene_groups`                                   | none                                                                                                              |
  | Feature bootstrap  | `run_gene_bootstrap`, `apply_multiple_hypothesis_correction` | {func}`~mantispy.tl.bootstrap_features`                                                                           |
  | mAP                | none                                                         | {func}`~mantispy.tl.map`                                                                                          |
  | Cluster            | `phate_leiden_pipeline`, `run_benchmark_analysis`            | {func}`~mantispy.tl.phate`, {func}`~mantispy.tl.cluster`, {func}`~mantispy.metrics.known_relationships`           |

The code on the page uses the following imports.

```python
import numpy as np
import scanpy as sc

import mantispy as mt
```

## Read the cells

Brieflow writes one parquet table of cells per plate and cell class, with its own column names.
Rename the metadata columns to mantispy’s `Metadata_` names and build an object at object resolution with {func}`~mantispy.io.from_dataframe`.
Brieflow’s other metadata columns, the ones listed in its `metadata_cols_fp` file, go to `metadata_columns` so they land in `obs` and not among the features.

```python
df = df.rename(
    columns={
        "plate": "Metadata_Plate",
        "well": "Metadata_Well",
        "cell_barcode_0": "Metadata_sgRNA",  # Brieflow's perturbation_id_col
        "gene_symbol_0": "Metadata_Gene",  # Brieflow's perturbation_name_col
    }
)
df["Metadata_Control_Type"] = np.where(df["Metadata_Gene"].str.startswith("nontargeting"), "negcon", "treatment")
adata = mt.io.from_dataframe(df, metadata_columns=other_metadata_columns, resolution="object")
```

Brieflow’s `query_filter` and `perturbation_filter` drop cells with no gene or with a gene the configuration excludes, which is a plain subset on `adata.obs["Metadata_Gene"]`.
Feature names that don’t follow CellProfiler’s grammar are kept as features with `feature_kind="embedding"`.

A screen that doesn’t fit in memory can be written once with {func}`~mantispy.io.write` and read back with `mt.io.read(path, lazy=True)`.
{func}`~mantispy.pp.normalize`, {func}`~mantispy.pp.tvn`, {func}`~mantispy.tl.aggregate` and `sc.pp.pca(..., svd_solver="covariance_eigh")` stream a lazy object and give the same numbers as in memory.
A function that needs the whole matrix warns before it reads it.

## Drop sparse features and cells

Brieflow’s `missing_values_filter` drops features missing in at least `drop_cols_threshold` of the cells, and then cells missing at least `drop_rows_threshold` of the features.

```python
mt.pp.feature_select(adata, operations=["drop_na_columns"], na_cutoff=0.2, subset=True)
mt.pp.calculate_qc_metrics(adata, max_nan_fraction=0.01, qc_metrics=("nan",))
mt.pp.filter_cells(adata, min_cells_per_well=0)
```

Brieflow drops a feature or a cell whose missing fraction equals the threshold, and mantispy keeps it, so results differ only for values that land exactly on the threshold.
`qc_metrics=("nan",)` and `min_cells_per_well=0` turn off the border, area and cells-per-well checks, which Brieflow doesn’t run here.

## Impute missing values

Brieflow fills the remaining gaps with scikit-learn’s `KNNImputer(n_neighbors=5)`, drawing donors from a random sample of 10,000 complete cells for each batch of incomplete cells.

```python
mt.pp.impute(adata, method="knn", n_neighbors=5)
```

The default `n_donors=10000` bounds the donor search in the same way.
Brieflow imputes before it scales the features, so keep the same order to reproduce it, although the {func}`~mantispy.pp.impute` docstring recommends imputing after {func}`~mantispy.pp.normalize`.

## Drop intensity outliers

Brieflow’s `intensity_filter` fits scikit-learn’s `LocalOutlierFactor` on the mean intensity of each channel and drops the most outlying 1% of cells.
{func}`~mantispy.pp.outliers` with `method="lof"` uses the same 20 neighbours, and `key` names a boolean `var` column that selects the features to score.

```python
channels = ["DAPI", "TUBULIN", "GH2AX", "PHALLOIDIN"]
adata.var["intensity_mean"] = adata.var_names.str.endswith(tuple(f"_{channel}_mean" for channel in channels))
mt.pp.outliers(adata, method="lof", contamination=0.01, key="intensity_mean")
mt.pp.filter_cells(adata, min_cells_per_well=0)
```

The flagged cells are added to `obs["qc_fail"]`, which {func}`~mantispy.pp.filter_cells` drops.

## Scale within each plate and well

Brieflow scales the features twice, and both scalings start from the same filtered cells.
`centerscale_by_batch` standardizes every cell within its plate and well, and feeds the PCA.
`centerscale_on_controls` standardizes on the non-targeting cells within each plate and well, and feeds the feature table that the bootstrap tests.
Write the control scaling to a layer first, and then scale `X` in place.

```python
batch = ["Metadata_Plate", "Metadata_Well"]
mt.pp.normalize(adata, method="standardize", by=batch, reference="negcon", key_added="controls_scaled")
mt.pp.normalize(adata, method="standardize", by=batch, reference=None)
```

Both use the population standard deviation, as scikit-learn’s `StandardScaler` does.

## PCA

Brieflow fits `PCA(n_components=0.99)` on a random sample of 100,000 cells, keeping the fewest components that explain 99% of the variance, and projects every cell onto them.
mantispy fits the PCA on every cell instead.
With many more cells than features, `svd_solver="covariance_eigh"` decomposes the small feature covariance, so the exact fit costs little.

```python
sc.pp.pca(adata, n_comps=adata.n_vars - 1, svd_solver="covariance_eigh")
n_comps = int(np.searchsorted(np.cumsum(adata.uns["pca"]["variance_ratio"]), 0.99, side="right")) + 1
adata.obsm["X_pca"] = adata.obsm["X_pca"][:, :n_comps]
```

The cut-off line picks the same number of components as scikit-learn does for `n_components=0.99`.

## TVN and CORAL

Brieflow’s `tvn_on_controls` centres and scales the components on the controls, rotates them onto the principal components of the controls, and then aligns each plate and well with CORAL.

```python
adata.obs["Metadata_Batch"] = adata.obs["Metadata_Plate"].astype(str) + "_" + adata.obs["Metadata_Well"].astype(str)
mt.pp.tvn(adata, batch_key="Metadata_Batch", reference="negcon", use_rep="X_pca", epsilon=0.5)
```

The result is in `obsm["X_tvn"]`.
`epsilon=0.5` is the shrinkage Brieflow adds to each covariance before inverting it.

## Perturbation score

Brieflow’s `perturbation_score` fits a logistic regression per gene, on the 200 features with the largest ANOVA F statistic, against an equal-sized sample of control cells, and records each cell’s out-of-fold probability and each gene’s ROC AUC.
Brieflow skips the step by default (`skip_perturbation_score=True`).

```python
mt.tl.perturbation_score(
    adata,
    groupby="Metadata_Gene",
    reference="negcon",
    layer="controls_scaled",
    replicate=None,
    n_features=200,
    min_cells=50,
)
```

`replicate=None` draws the cross-validation folds over cells, as Brieflow does.
The mantispy default, `replicate="Metadata_sgRNA"`, keeps each guide’s cells in one fold, so a phenotype that only one guide shows doesn’t raise the gene’s AUC.
`min_cells=50` matches Brieflow’s minimum of 100, which counts the gene’s cells together with its equal-sized control sample.
Brieflow scales each gene’s sample on its own sampled controls, while mantispy reads the control-scaled layer, so Brieflow’s AUCs come out slightly higher.
Pass `by="Metadata_Class"` to score interphase and mitotic cells separately when one object holds both classes.

## Aggregate to genes

Brieflow’s `aggregate` takes the median of the TVN components over each gene’s cells.
Its `ps_probability_threshold` drops cells whose perturbation score is below a threshold first, and keeps cells with no score.

```python
responders = adata[~(adata.obs["perturbation_score"] < ps_probability_threshold)]
genes = mt.tl.aggregate(responders, by=["Metadata_Gene"], func="median", min_cells=1, use_rep="X_tvn")
genes = genes[genes.obs["Metadata_CellCount"] > min_cell_cutoff]
```

Skip the first line when Brieflow runs without the threshold.
The last line is Brieflow’s `clean_aggregate`, which keeps genes with more than `min_cell_cutoffs[cell_class]` cells.
`min_cells=1` turns off mantispy’s own minimum of 10 cells, so the cut-off is applied only once.

## Pseudo-genes

Brieflow’s `create_pseudogene_groups` groups control guides into pseudo-genes and counts how many of them the bootstrap calls, as an estimate of its false-positive rate.
mantispy has no counterpart, because the check can’t see the problems it’s meant to find.
The bootstrap null is drawn from the same control guides, so plate offsets, differences between control guides and clustering of control cells widen the null as much as they move the pseudo-genes, and the bootstrap calls fewer genes rather than more pseudo-genes.
With Aconcagua’s 250 control guides, the check could only detect a false-positive rate four to six times too high.
{func}`~mantispy.tl.hit_calling` uses pseudo-genes differently, as the null that each gene is ranked against, with `null="pseudogenes"`.

## Feature bootstrap

Brieflow’s `run_gene_bootstrap` tests each feature of each gene and guide against medians resampled from the control guides’ cells, and `apply_multiple_hypothesis_correction` adds Benjamini-Hochberg q-values.
It runs on the control-scaled features, not on the TVN components.

```python
mt.tl.bootstrap_features(
    adata,
    groupby="Metadata_Gene",
    replicate="Metadata_sgRNA",
    reference="negcon",
    control_scope="pooled",
    layer="controls_scaled",
)
```

The gene table is `adata.uns["mantispy"]["bootstrap"]` and the guide table is `adata.uns["mantispy"]["bootstrap_guides"]`.
Every feature in `adata` is tested, so subset `adata` to Brieflow’s `bootstrap_features_fp` list first if you use one.

mantispy computes the bootstrap null exactly instead of simulating it, so there’s no `num_sims`.
The p-values are the ones an infinite bootstrap converges to, with no simulation noise and no lower limit.
Brieflow replaces a p-value of zero by `1 / num_sims` before the correction, and mantispy keeps it at zero.
A null median equal to the observed median counts half in mantispy, where Brieflow counts it as below, so a feature that most cells share one value of, such as a count that’s mostly zero, isn’t called.

## Gene filters before clustering

Brieflow filters genes before clustering by their perturbation AUC (`filter_by_perturbation_auc`) and by their bootstrap results (`filter_genes_by_bootstrap`).
mantispy has no functions for the filters, because each one is a single subset of tables mantispy already writes.
Both keep the controls.

```python
auc = adata.uns["mantispy"]["perturbation_score"]
boot = adata.uns["mantispy"]["bootstrap"]
is_control = genes.obs["Metadata_Control_Type"] == "negcon"

# filter_by_perturbation_auc
genes = genes[genes.obs["Metadata_Gene"].isin(auc.loc[auc["auc"] > auc_threshold, "group"]) | is_control]

# filter_genes_by_bootstrap, filter_mode="fdr"
genes = genes[genes.obs["Metadata_Gene"].isin(boot.loc[boot["qvalue"] < fdr_threshold, "group"]) | is_control]

# filter_genes_by_bootstrap, filter_mode="both"
hits = boot.loc[(boot["qvalue"] < fdr_threshold) & (boot["effect"].abs() >= zscore_threshold), "group"]
genes = genes[genes.obs["Metadata_Gene"].isin(hits) | is_control]
```

Recompute `is_control` after each subset if you chain the filters.
The `effect` column is the gene’s median minus the median of its control pool, so it is close to Brieflow’s z-score, which is the gene’s median alone, but not identical.
For `zscore_direction="positive"` or `"negative"`, compare `boot["effect"]` with the threshold directly instead of its absolute value.

## mAP

Brieflow has no mAP step.
{func}`~mantispy.tl.map` scores whether each gene’s guide profiles retrieve each other ahead of the control profiles on the same plate.

```python
guide_profiles = mt.tl.aggregate(
    adata, by=["Metadata_Plate", "Metadata_Well", "Metadata_sgRNA"], func="median", min_cells=1, use_rep="X_tvn"
)
mt.tl.map(guide_profiles, mode="activity", label="Metadata_Gene")
```

## PHATE and Leiden

Brieflow’s `phate_leiden_pipeline` embeds the gene profiles with PHATE (`run_phate`, 10 neighbours) and clusters them with Leiden (`run_leiden_clustering`).
{func}`~mantispy.tl.phate` implements PHATE in mantispy, with the parameter names and defaults of `phate.PHATE`.

```python
genes = genes.copy()  # the filters above return views
mt.tl.phate(genes, knn=10, metric=phate_distance_metric, use_rep=None)
mt.tl.cluster(genes, method="leiden", neighbors_key="phate", resolution=leiden_resolution)
```

`use_rep=None` embeds `X`, which after aggregation holds the TVN components, as in Brieflow.
The embedding is in `obsm["X_phate"]` and the clusters are in `obs["cluster"]`.

mantispy clusters on a different graph on purpose.
Brieflow runs Leiden on PHATE’s diffusion operator, a random walk whose rows are normalized one at a time, averaged with its transpose to make it symmetric.
{func}`~mantispy.tl.phate` stores PHATE’s symmetric affinity, the kernel before that normalization, in the layout {func}`scanpy.pp.neighbors` uses, and Leiden clusters on that graph.
The affinity is symmetric to begin with, so it needs no averaging.
Leiden also runs through {func}`scanpy.tl.leiden` with two iterations, where Brieflow iterates until the partition stops changing.

Brieflow’s `calculate_potential_to_nontargeting` measures each gene’s distance to the controls in PHATE’s diffusion potential, and mantispy deliberately has no counterpart.
The diffusion averages each row over its neighbours in the graph, so a row’s potential depends on every other row, and the distance of a gene and the distance of a control aren’t comparable draws.
On Aconcagua’s published gene profiles, a permutation test of distance to the controls in the potential called 14% to 17% of pseudo-genes made only of control guides at p < 0.05, where a calibrated test calls about 5%, and the same test on the principal components called at most about 2%.
To test genes against the controls, run {func}`~mantispy.tl.bootstrap_features` on the features rather than on the potential.
To test whole gene profiles against the controls, run {func}`~mantispy.tl.hit_calling` with `null="pseudogenes"`, which ranks each gene among pseudo-genes made of control guides.

## Benchmark against CORUM

Brieflow’s `run_benchmark_analysis` measures how well the clusters recover CORUM complexes and other gene sets.
{func}`~mantispy.metrics.known_relationships` measures a related but different quantity, the share of CORUM gene pairs whose profiles fall in the most similar or least similar 5% of all pairs.

```python
mt.metrics.known_relationships(genes, mt.ds.corum(), label_key="Metadata_Gene", n_permutations=100)
```

It scores the gene profiles and doesn’t depend on the clustering, so its number isn’t comparable to Brieflow’s cluster precision and recall.
`n_permutations` gives the recall expected by chance, which is the baseline to read the result against.

## Out of scope

mantispy starts from measured cells, so it leaves out the following parts of Brieflow.

- The image stages, which are preprocessing, SBS base calling, segmentation and feature extraction, merging the phenotype and SBS data, and montages.
- The per-cell classifier and the labelling of cells as interphase or mitotic. mantispy reads the label from `obs["Metadata_Class"]`.
- Cluster annotation with an LLM through mozzarellm.
- The sweep over Leiden resolutions. Call {func}`~mantispy.tl.cluster` once per resolution instead.
