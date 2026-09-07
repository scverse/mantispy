# mantispy Release Plan: 0.1.0 to 0.5.0

Scope rule for this plan: every release up to and including 0.5.0 is **AnnData only**. Images, segmentation masks, SpatialData assembly, coordinate systems, cell galleries and the spatial mode of the CellProfiler plugin all start with 0.6.0. Image-level QC in this window uses the CellProfiler `Image.csv` measurements (MeasureImageQuality), never pixels.

Fixed decisions carried over: `import mantispy as ms`; `Metadata_` obs grammar; CellProfiler-style feature names (cp_measure format as recommended canonical form); numba reimplementation of pycytominer operations with pycytominer as dev dependency for equivalence tests; copairs and decoupler as hard dependencies; harmonypy for Harmony; pertpy deferred (no pertpy imports anywhere before 0.6); OPS columns reserved in the schema but unused; scverse cookiecutter, BSD-3, scverse core as target.

Ordering logic: each release is a coherent, teachable slice with its own tutorials. 0.1 establishes the data contract and the aggregate workflow end to end (read, QC basics, normalize, aggregate). 0.2 makes QC and feature selection complete so profiles are trustworthy. 0.3 adds batch correction and the evaluation metrics needed to judge it. 0.4 adds hit calling, perturbation statistics and enrichment, which need trustworthy corrected profiles first. 0.5 adds single-cell heterogeneity, scale and ecosystem integration, and freezes the API and schema before the SpatialData expansion.

---

## Release overview

| release | theme | headline capabilities | tutorials |
|---|---|---|---|
| 0.1.0 | Foundation | schema 0.1, CellProfiler CSV reader, feature parser, basic QC, normalization, aggregation, plate plots, BBBC021 | 01, 02 |
| 0.2.0 | Trustworthy profiles | full feature selection, image and cell QC, outlier detection, diagnostic plots | 03, 04 |
| 0.3.0 | Batch correction and evaluation | sphering, Harmony, plate-position correction, confounder regression, copairs mAP, percent replicating, grit, scib-style metrics | 05, 06 |
| 0.4.0 | Hits, perturbations, enrichment | Mahalanobis and KS hit calling, e-distance, effect sizes, consensus, dose response, NN MOA classification, decoupler feature sets and MOA enrichment, JUMP annotation | 07, 08, 09 |
| 0.5.0 | Heterogeneity, scale, ecosystem | cluster composition, cell cycle phase, Chatterjee selection, subpopulation hits, backed and chunked IO, SQLite, CytoTable and JUMP readers, CellProfiler table-only export plugin, benchmark suite, schema 1.0, API freeze | 10, 11, 12 |

---

## 0.1.0 Foundation

Goal: a user can go from a CellProfiler ExportToSpreadsheet directory to normalized well-level profiles in ten lines, and the on-disk contract is fixed.

### Modules and functions

`ms._core` (private)
- `schema.py`: required and reserved column constants, `SCHEMA_VERSION = "0.1"`, `validate(adata) -> ValidationReport`.
- `features.py`: `parse_feature_names(names, channels) -> DataFrame` (object, feature_group, feature, channel, scale, angle, gray_levels, radial_bin, is_feature); default blocklist loader.
- `plate.py`: well name normalization (`A1`, `a01` to `A01`), row and column derivation, plate format detection (96, 384, 1536).
- `_numba.py`: grouped median, grouped MAD, grouped mean and count kernels (parallel over groups and features, float32).
- `_utils.py`: `inplace`/`copy` handling, params recording into `uns["mantispy"]["params"]`, logging.

`ms.io`
- `read_cellprofiler(path, primary_object="Cells", objects=None, channels=None, platemap=None, strict_one_to_one=True, chunksize=None) -> AnnData`: CSV directory reader, parent-child join, object prefixing, var parsing, `Image_*` metadata into obs, `Image_ImageQuality_*` kept in `uns["mantispy"]["image_table"]`.
- `read_profiles(path) -> AnnData`: pycytominer-style profile CSV or parquet at any resolution.
- `from_dataframe(df) -> AnnData`, `read(path)`, `write(adata, path)` (h5ad default, zarr by suffix), both stamping and checking `schema_version`.
- `validate(adata)` re-exported.

`ms.pp`
- `calculate_qc_metrics(adata, image_shape=None, border_margin=10)`: obs `qc_n_nan_features`, `qc_is_border`, `qc_area_outlier`, `qc_pass`; var `qc_n_nan`, `qc_variance`, `qc_n_unique`.
- `filter_cells(adata, min_cells_per_well=50, qc_pass=True)`.
- `filter_features(adata, drop_nan=True, min_variance=0.0, blocklist="default")`.
- `normalize(adata, method="mad_robustize" | "standardize" | "robustize", by="Metadata_Plate", reference=None | "negcon" | obs bool column, epsilon=1e-6, key_added=None)`. Numba grouped kernels. `layers["raw"]` written on first call unless `keep_raw=False`.

`ms.tl`
- `aggregate(adata, by=("Metadata_Plate", "Metadata_Well"), func="median", min_cells=10) -> AnnData`: built on `sc.get.aggregate`; carries constant `Metadata_` columns, `Metadata_CellCount`, sets `uns["mantispy"]["resolution"]` and `aggregated_from`.

`ms.get`
- `features(adata, object=None, feature_group=None, channel=None) -> list[str]`
- `to_dataframe(adata, layer=None) -> DataFrame`
- `controls(adata, kind="negcon") -> ndarray[bool]`
- `obs_df`, `var_df` re-exports from scanpy.

`ms.pl`
- `plate(adata, color, plate=None, agg="median", ax=None)`: 96/384/1536 well grid heatmap, multi-plate grid.
- `cell_counts(adata, groupby="Metadata_Plate")`.
- `feature_distributions(adata, features, groupby="Metadata_Plate", layer_before="raw", kind="ecdf" | "kde")`: before and after normalization.
- `nan_matrix(adata)`.
- `qc(adata)`: summary dashboard of the 0.1 QC metrics.

`ms.datasets`
- `bbbc021() -> AnnData` (CellProfiler features, `Metadata_MOA`, `Metadata_Compound`, `Metadata_Concentration`), pooch cached.
- `synthetic_plate(n_wells=384, n_cells=200, n_features=600, effects=..., seed=0) -> AnnData`: generator with known ground truth, used by tests and docs.

### Dependencies added
anndata, scanpy, numpy, numba, pandas, scipy, matplotlib, pooch. Dev: pytest, pytest-mpl, pycytominer, pre-commit, ruff.

### Tests
- Synthetic ExportToSpreadsheet fixture (Image.csv plus three object CSVs) including a non-1:1 case that must raise.
- Parser table with several hundred real names from CellProfiler 2.2, 3.1, 4.2 and cp_measure.
- Equivalence vs pycytominer: `normalize` (all three methods, with and without `samples` query), `aggregate` (mean, median).
- Schema validate accepts fixtures and rejects each single mutation.

### Tutorials
- **01 From CellProfiler to AnnData**: reading, what lands where (obs, var, uns), inspecting parsed var, validating, writing h5ad.
- **02 First profiles**: basic QC, normalize on negative controls per plate, aggregate to wells, plate heatmaps, export to a pycytominer DataFrame.

### Exit criteria
BBBC021 well profiles match pycytominer `aggregate` plus `normalize(method="mad_robustize")` within rtol 1e-6. Tutorials 01 and 02 run in CI. `spec/schema-0.1.json` published.

---

## 0.2.0 Trustworthy profiles

Goal: complete QC at cell, well and image level, and complete feature selection, so that everything downstream runs on clean data.

### Modules and functions

`ms.pp`
- `feature_select(adata, operations=("variance_threshold", "correlation_threshold", "blocklist", "drop_na_columns", "drop_outliers", "noise_removal"), variance_freq_cut=0.05, variance_unique_cut=0.01, corr_threshold=0.9, corr_method="pearson", outlier_cutoff=500, noise_removal_perturb_groups="Metadata_Perturbation", noise_removal_stdev_cutoff=..., key_added="selected")`: boolean var column, pycytominer semantics and defaults, chunked float32 correlation in numba.
- `subset_features(adata, key="selected") -> AnnData`.
- `image_qc(adata, metrics=("FocusScore", "PowerLogLogSlope", "PercentMaximal", "PercentMinimal", "Saturation"), channel="DNA", method="mad" | "knn", threshold="auto")`: reads `uns["mantispy"]["image_table"]`; `mad` flags per-plate MAD outliers per metric; `knn` is the scmorph-style kNN dissimilarity on the standardized image-QC matrix. Writes `uns["mantispy"]["image_qc"]` and broadcasts `qc_image_pass` to obs.
- `filter_images(adata)`: drops cells from flagged images.
- `outliers(adata, method="ecod" | "isolation_forest" | "mad", contamination=0.01, key_added="qc_outlier")`: native ECOD (ECDF tail probabilities, numba), isolation forest via scikit-learn, MAD on selected features. scmorph `filter_outliers` parity.
- `well_qc(adata, min_cells=50, max_cv_controls=...)`: per-well flags (`qc_well_pass`) from cell count, control coefficient of variation, NaN fraction.
- `standardize_feature_names(adata, target="cp_measure")`: opt-in rename of legacy CellProfiler names, original kept in `var["original_name"]`.

`ms.pl`
- `plate_effects(adata, feature=None, use_rep="X_pca")`: row and column marginals per plate.
- `image_qc(adata, plate=None)`: per-image metrics with flags, optional plate grid at site resolution.
- `feature_correlation(adata, groupby="feature_group", key="selected")`: clustered heatmap with group and channel color bars, before and after selection.
- `feature_groups(adata, groupby=None)`: counts or effect sizes per feature_group by channel.
- `control_drift(adata, reference="negcon", groupby="Metadata_Plate")`: controls projected on control PCs across plates.
- `outliers(adata)`: outlier score distributions and fraction per plate.
- `well_qc(adata)`: plate grid of well flags.

### Dependencies added
scikit-learn (isolation forest, PCA utilities), seaborn (ECDF and KDE convenience).

### Tests
- Equivalence vs pycytominer for every `feature_select` operation, including order of greedy correlation removal.
- ECOD against pyod on the synthetic plate (pyod as test-only dependency).
- Image QC recovers injected blurred images in the synthetic fixture.

### Tutorials
- **03 Feature selection**: the operations, their effect on redundancy, correlation heatmaps, blocklist provenance.
- **04 Quality control at three levels**: image QC from MeasureImageQuality, well QC, cell outliers with ECOD versus isolation forest, plate effect diagnostics.

### Exit criteria
Full pycytominer `feature_select` equivalence. Injected artifacts in the synthetic plate are recovered with the default thresholds. Tutorials 03 and 04 in CI.

---

## 0.3.0 Batch correction and evaluation

Goal: implement the correction steps of the Broad recipe (sphering, Harmony) plus plate-level corrections, and the metrics needed to judge them, with copairs mAP as the central readout.

### Modules and functions

`ms.pp`
- `sphere(adata, method="ZCA-cor" | "ZCA" | "PCA" | "PCA-cor", reference="negcon", epsilon=1e-6, by=None, key_added=None)`: whitening fitted on controls (TVN), applied to all. numpy linalg, float64 internally, float32 output.
- `harmony(adata, key="Metadata_Batch", basis="X_pca", key_added="X_harmony", **harmonypy_kwargs)`.
- `correct_plate_position(adata, method="median_polish", by="Metadata_Plate", reference="negcon" | None, max_iter=10)`: removes row and column effects per plate per feature; alternative `method="loess"` later.
- `regress_out(adata, keys=("Metadata_CellCount",), by="Metadata_Plate")`: linear confounder regression per plate (cell count, density, later cell cycle fraction), numba least squares per feature.
- `pca`, `neighbors`: not implemented, documented as scanpy calls.

`ms.tl`
- `map(adata, pos_sameby, pos_diffby, neg_sameby, neg_diffby, mode="activity" | "consistency", reference="negcon", use_rep=None, null_size=10000, seed=0, key_added="map")`: copairs wrapper; per-group mAP, p-value, q-value in `uns["mantispy"]["map"]` and in obs at well or perturbation resolution.
- `percent_replicating(adata, groupby="Metadata_Perturbation", null_size=10000, quantile=0.95)`: replicate correlation versus permutation null, cytominer-eval semantics.
- `grit(adata, groupby="Metadata_Perturbation", reference="negcon")`: grit z-scores per perturbation.
- `similarity(adata, metric="cosine" | "pearson", use_rep=None, key_added="similarity")`: `obsp["similarity"]` at aggregated resolution.

`ms.metrics` (new module)
- `silhouette_batch(adata, label_key, batch_key, use_rep)`, `silhouette_label(...)`, `lisi(adata, key, use_rep, perplexity=30)` (iLISI and cLISI), `pc_regression(adata, key, use_rep)`, `batch_variance_explained(adata, keys, use_rep)`: scib-style metrics reimplemented natively (no scib dependency), returning a DataFrame; `evaluate_correction(adata, reps=("X_pca", "X_sphered", "X_harmony"), label_key="Metadata_Perturbation", batch_key="Metadata_Batch")` combining them with mAP into one table.

`ms.pl`
- `map(adata, key="map")`: mAP versus q-value, fraction active, null overlay.
- `replicate_correlation(adata, key="percent_replicating")`: replicate versus null distributions.
- `batch_variance(adata, keys, use_rep)`: variance explained per PC per factor, before and after.
- `silhouette(adata, ...)`, `metrics(table)`: bar chart of `evaluate_correction`.
- `similarity(adata, key="similarity", row_colors="Metadata_MOA")`: clustered heatmap.
- `embedding_compare(adata, reps, color)`: side-by-side UMAPs of representations via `sc.pl.embedding`.

### Dependencies added
copairs, harmonypy.

### Tests
- Sphering equivalence vs pycytominer `Spherize` for all four methods.
- Median polish recovers injected row and column gradients on the synthetic plate.
- `tl.map` equals copairs called directly on `ms.get.to_dataframe`.
- Metrics against scib on a small fixture (scib as test-only dependency).

### Tutorials
- **05 Batch correction**: sphering, Harmony, plate-position correction, when to use which, following Arévalo et al. 2024; `evaluate_correction` to pick.
- **06 Profile strength with mAP**: activity and consistency with copairs, percent replicating and grit for comparison, interpreting nulls and q-values.

### Exit criteria
Full Broad recipe reproducible on BBBC021: aggregate, normalize, feature_select, sphere, harmony, map. Correction metrics reproduce scib values on the fixture. Tutorials 05 and 06 in CI.

---

## 0.4.0 Hits, perturbations and enrichment

Goal: everything a screen analyst asks after profiles are clean: which perturbations are active, how strongly, along which features, whether the effect is dose dependent, and which mechanism the neighborhood suggests.

### Modules and functions

`ms.tl`
- `hit_calling(adata, groupby="Metadata_Perturbation", reference="negcon", method="mahalanobis" | "tstat" | "euclidean", use_rep="X_pca", n_permutations=1000, key_added="hits")`: scmorph parity and beyond. At cell resolution: PCA on controls, Mahalanobis to control medoid with control covariance, KS statistic of treated versus control distance distributions, permutation p-values. At well resolution: distance of replicate profiles to control profiles with replicate-aware permutations. Writes `uns["mantispy"]["hits"]`, and `obs["hit_distance"]`, `obs["hit_pvalue"]`, `obs["hit_qvalue"]` at aggregated resolutions.
- `effect_size(adata, groupby, reference="negcon", shrink=True)`: per feature and perturbation Cohen's d, robust z, Kruskal p-value; empirical Bayes shrinkage of d toward the group mean for low replicate counts. `varm["effect_<groupby>"]`, `var["kruskal_pvalue"]`.
- `edistance(adata, groupby, reference=None, n_permutations=1000, use_rep="X_pca")`: energy distance between groups (pairwise or versus control) with permutation E-test; numba pairwise distances in chunks. `obsp["edistance"]` at perturbation resolution, `uns["mantispy"]["edistance"]`.
- `wasserstein_features(adata, groupby, reference="negcon")`: per feature Wasserstein-1 versus control from sorted samples, numba. `varm["wasserstein_<groupby>"]`.
- `consensus(adata, by="Metadata_Perturbation", method="modz" | "median", min_replicates=2) -> AnnData`: modified z-score weighted consensus, pycytominer `modz` equivalence.
- `dose_response(adata, perturbation_key="Metadata_Compound", dose_key="Metadata_Concentration", response="hit_distance" | "map", fit="hill" | "spearman")`: Spearman trend and four-parameter logistic fit where at least four doses exist. `uns["mantispy"]["dose_response"]`.
- `nn_moa_classify(adata, moa_key="Metadata_MOA", metric="cosine", scheme="nsc" | "nscb", batch_key="Metadata_Batch")`: leave-one-out nearest neighbor with not-same-compound and not-same-compound-or-batch restrictions (Ljosa et al. 2013). `obs["moa_predicted"]`, accuracy and confusion in `uns`.
- `feature_sets(adata, by=("feature_group", "channel")) -> DataFrame`: decoupler net from parsed var (also `object`, `feature_group`, `channel`, `group_by_channel`, user supplied).
- `enrich(adata, net="group_by_channel", method="ulm" | "mlm" | "ora", key_added=None)`: decoupler wrapper, `obsm["score_<net>"]`, `obsm["padj_<net>"]`.
- `rank_features(adata, groupby, method="wilcoxon")`: `sc.tl.rank_genes_groups` wrapper with parsed feature annotation in the result table.
- `rank_sets(adata, groupby, net)`: `dc.tl.rankby_group` on enrichment scores.
- `moa_enrichment(adata, moa_key="Metadata_MOA", similarity_key="similarity", method="ora", top_k=20)`: over-representation of MOAs among each perturbation's nearest neighbors.
- `annotate_controls(adata, negcon=("DMSO",), poscon=None, rules=None)`: `Metadata_Control`, JUMP-Target-2 positive controls recognized.
- `annotate_jump(adata, key="Metadata_JCP2022")`: native loader of JUMP `compound.csv`, `orf.csv`, `crispr.csv` (pooch cached, frozen snapshot shipped for offline CI): `Metadata_InChIKey`, `Metadata_Gene`, target annotations.

`ms.pl`
- `hits(adata, key="hits", color="Metadata_MOA", label_top=20)`: distance versus q-value.
- `effect_sizes(adata, perturbation, top=30)`: bars colored by feature_group and channel.
- `dose_response(adata, compound)`.
- `distance_heatmap(adata, key="edistance", row_colors="Metadata_MOA")`.
- `sets_heatmap(adata, groupby, net)`: enrichment scores per cluster or perturbation.
- `moa_enrichment(adata, perturbation)`.
- `moa_confusion(adata)`: confusion matrix from `nn_moa_classify`.
- `perturbation_umap(adata, color="Metadata_MOA")`: `sc.pl.umap` wrapper with MOA palette handling and legend placement.
- `feature_volcano(adata, perturbation)`: effect size versus Kruskal p-value per feature.

### Dependencies added
decoupler.

### Tests
- `hit_calling` versus scmorph `aggregate_mahalanobis` plus `get_ks` on the rohban2017 minimal dataset (scmorph as test-only dependency).
- `consensus(method="modz")` versus pycytominer.
- `edistance` versus a reference numpy implementation and against scPerturb published values on a toy set.
- `nn_moa_classify` reproduces the BBBC021 NSC accuracy range reported by Ljosa et al. 2013 for mean profiles.
- decoupler round trip: `enrich` output identical to calling `dc.mt.ulm` directly.

### Tutorials
- **07 Hit calling**: aggregate-level and single-cell Mahalanobis hits, e-distance, effect sizes with shrinkage, feature volcano, comparing with mAP.
- **08 Mechanism of action**: similarity, nearest-neighbor MOA classification on BBBC021, MOA enrichment among neighbors, feature-set enrichment to explain clusters.
- **09 Dose response and consensus**: consensus signatures with modz, dose-response fits, concentration consistency.

### Exit criteria
BBBC021 NSC MOA accuracy within the published range. scmorph hit-calling parity on rohban2017. Tutorials 07 to 09 in CI.

---

## 0.5.0 Heterogeneity, scale and ecosystem

Goal: single-cell heterogeneity tools that justify the single-cell resolution, JUMP-scale performance, remaining readers, the table-only CellProfiler plugin, and a frozen 1.0 schema and API so 0.6 can add SpatialData without breaking users.

### Modules and functions

`ms.tl`
- `cluster_composition(adata, cluster_key="leiden", by=("Metadata_Plate", "Metadata_Well"), reference="negcon", test="chi2" | "dirichlet") -> AnnData`: per-well cell-state fractions as a new AnnData (obs wells, var clusters), with enrichment versus controls. Feeds `aggregate`-style downstream tools.
- `cell_cycle_phase(adata, dna_feature="Nuclei_Intensity_IntegratedIntensity_DNA", method="gmm", key_added="Metadata_CellCyclePhase")`: G1, S, G2M from integrated DNA intensity with a per-plate two-component Gaussian mixture on log intensity plus S in between; fractions per well written for use as a covariate in `regress_out`.
- `subpopulation_hits(adata, cluster_key, groupby, reference="negcon")`: per-cluster KS and Wasserstein of perturbation versus control cells, finds hits invisible at the aggregate level.
- `neighbors_local_density(adata, k=15, by="Metadata_ImageNumber")`: local cell density from centroids within a field of view via squidpy `spatial_neighbors` (squidpy already a scverse core dependency); `obs["Metadata_LocalDensity"]` for `regress_out`.

`ms.pp`
- `feature_select_chatterjee(adata, threshold=0.5, key_added="selected_chatterjee")`: adapted Chatterjee rank correlation (Lin and Han 2023), numba, scmorph parity.
- `downsample(adata, n_per_well=500, stratify="Metadata_Perturbation", seed=0)`: stratified subsampling for exploration at scale.

`ms.io`
- `read_cellprofiler_sqlite(path, **kwargs)`: ExportToDatabase SQLite, `TableNumber` aware.
- `read_cytotable(path)` behind `mantispy[cytotable]`.
- `read_jump(path, plates=None, sources=None)`: JUMP parquet layout with `Metadata_Source`, `Metadata_Batch`.
- `read_cellprofiler(..., backed=True)`: chunked CSV to zarr on read, returns backed AnnData; all `pp` functions accept backed objects when `by=` is given and stream group by group.
- `read_h5ad(path, backed="r")` documented path for large tables.

`ms.pl`
- `cluster_composition(adata_comp, plate=None)`: stacked bars per well or plate grid per cluster.
- `cell_cycle(adata)`: DNA intensity histogram with fitted phases per plate.
- `ridge(adata, features, groupby)` and `cumulative_density(adata, features, groupby)`: scmorph plotting parity.
- `subpopulation_hits(adata)`.
- `density(adata, feature, groupby)`: local density versus feature.

`ms.datasets`
- `jump_target2(plates=2) -> AnnData`: JUMP-Target-2 subset for cross-batch tutorials.
- `rohban2017_minimal()`: for scmorph comparisons.

CellProfiler export plugin (separate repo `scverse/mantispy-cellprofiler`, released alongside 0.5.0)
- `ExportToAnnData` module: table-only mode, writes Level 1 h5ad directly from a CellProfiler pipeline, including parsed var and `uns["mantispy"]`. Spatial mode arrives with 0.7.

Benchmark suite (`benchmarks/`, asv or pytest-benchmark, not in CI)
- `normalize`, `feature_select`, `sphere`, `hit_calling`, `map` on 1 M and 5 M cells by 4000 features; targets: within 2x of a numpy lower bound, at least 10x faster than pycytominer on the same operations, peak memory below 2x the size of `X`.

Schema and API
- `SCHEMA_VERSION = "1.0"`; migration `ms.io.migrate(adata)` from 0.x; all public signatures marked stable; deprecation policy documented (two minor releases).

### Dependencies added
squidpy (spatial neighbors within a field of view; no image use), scikit-learn already present for GMM. Extras: `cytotable`.

### Tests
- `feature_select_chatterjee` versus scmorph `select_features`.
- Backed mode produces identical results to in-memory on the synthetic plate.
- SQLite and CSV readers produce identical AnnData from the same pipeline output.
- Plugin output passes `ms.io.validate` and equals `read_cellprofiler` on the same pipeline.

### Tutorials
- **10 Single-cell heterogeneity**: clustering with scanpy, cluster composition per well, cell cycle phase, subpopulation hits, density as a confounder.
- **11 Scaling to JUMP**: backed reads, streaming normalization per plate, Harmony across batches on JUMP-Target-2, mAP across sources, benchmark numbers.
- **12 Exporting directly from CellProfiler**: the ExportToAnnData module, validating the output, continuing in mantispy.
- **00 Overview and canonical workflow** (written last, placed first in the docs): the ten-call recipe from read to MOA enrichment, linking to all other tutorials.

### Exit criteria
JUMP-Target-2 subset processed end to end with backed IO within the benchmark targets. Plugin round trip identical to reader. Schema 1.0 frozen; docs complete; core-package application submitted to scverse.

---

## Function catalog by release

| module | 0.1.0 | 0.2.0 | 0.3.0 | 0.4.0 | 0.5.0 |
|---|---|---|---|---|---|
| ms.io | read_cellprofiler, read_profiles, from_dataframe, read, write, validate | | | | read_cellprofiler_sqlite, read_cytotable, read_jump, backed reads, migrate |
| ms.pp | calculate_qc_metrics, filter_cells, filter_features, normalize | feature_select, subset_features, image_qc, filter_images, outliers, well_qc, standardize_feature_names | sphere, harmony, correct_plate_position, regress_out | | feature_select_chatterjee, downsample |
| ms.tl | aggregate | | map, percent_replicating, grit, similarity | hit_calling, effect_size, edistance, wasserstein_features, consensus, dose_response, nn_moa_classify, feature_sets, enrich, rank_features, rank_sets, moa_enrichment, annotate_controls, annotate_jump | cluster_composition, cell_cycle_phase, subpopulation_hits, neighbors_local_density |
| ms.metrics | | | silhouette_batch, silhouette_label, lisi, pc_regression, batch_variance_explained, evaluate_correction | | |
| ms.pl | plate, cell_counts, feature_distributions, nan_matrix, qc | plate_effects, image_qc, feature_correlation, feature_groups, control_drift, outliers, well_qc | map, replicate_correlation, batch_variance, silhouette, metrics, similarity, embedding_compare | hits, effect_sizes, dose_response, distance_heatmap, sets_heatmap, moa_enrichment, moa_confusion, perturbation_umap, feature_volcano | cluster_composition, cell_cycle, ridge, cumulative_density, subpopulation_hits, density |
| ms.get | features, to_dataframe, controls, obs_df, var_df | | | | |
| ms.datasets | bbbc021, synthetic_plate | | | | jump_target2, rohban2017_minimal |

## scmorph parity map

| scmorph | mantispy | release |
|---|---|---|
| io.read_cellprofiler_csv, read_cellprofiler_batches, read_sql, make_AnnData, split_feature_names | io.read_cellprofiler, io.from_dataframe, io.read_cellprofiler_sqlite, _core.features.parse_feature_names | 0.1, 0.5 |
| qc.filter_outliers (ECOD) | pp.outliers | 0.2 |
| qc.qc_images_by_dissimilarity, qc.qc_images, qc.read_image_qc | pp.image_qc(method="knn") | 0.2 |
| qc.count_cells_per_group | pp.well_qc, pl.cell_counts | 0.1, 0.2 |
| pp.drop_na, pp.scale, pp.scale_by_batch | pp.filter_features, pp.normalize | 0.1 |
| pp.remove_batch_effects (scone) | pp.sphere, pp.harmony, pp.correct_plate_position | 0.3 |
| pp.select_features (Chatterjee), pp.corr | pp.feature_select_chatterjee, pl.feature_correlation | 0.5, 0.2 |
| pp.kruskal_test, pp.kruskal_filter | tl.effect_size (Kruskal column), pp.feature_select(operations=("confounder",)) added in 0.4 | 0.4 |
| pp.aggregate, aggregate_pc, aggregate_ttest, tstat_distance | tl.aggregate, tl.hit_calling(method="tstat") | 0.1, 0.4 |
| pp.aggregate_mahalanobis, tl.get_ks | tl.hit_calling(method="mahalanobis") | 0.4 |
| pp.pca, neighbors, umap | scanpy directly | n/a |
| tl.slingshot, test_common_trajectory, test_differential_* | deliberately out of scope | n/a |
| pl.pca, pl.umap, pl.cumulative_density, pl.ridge_plot | scanpy, pl.cumulative_density, pl.ridge | 0.5 |
| datasets.rohban2017* | datasets.rohban2017_minimal | 0.5 |

## Deliberately excluded before 0.6

Images, labels, SpatialData, coordinate systems, cell galleries, pixel-based blur metrics, pertpy wrappers (distances, Augur, Mixscape, metadata fetchers for PubChem, ChEMBL, DepMap), foundation model embeddings, OPS barcode handling, trajectory inference.

## Outlook after 0.5.0

- 0.6.0: SpatialData assembly from CellProfiler output with images and masks, element naming grammar, per-plate coordinate systems, `PlateView`, `pl.cells`, `pl.site`, `pl.plate_layout`, pixel-based image QC.
- 0.7.0: CellProfiler plugin spatial mode; cp_measure re-measurement from masks; pertpy extra.
- 0.8.0: OPS: populate reserved barcode columns, multi-round image elements, gene-level aggregation, Mixscape-style assignment QC.

## Cross-cutting workstreams (every release)

- Docs: API reference with a "Stores" section per function; tutorials executed in CI with the synthetic plate or cached downloads; changelog per release.
- Equivalence tests: every reimplemented operation asserts equality with its reference (pycytominer, scmorph, copairs, decoupler, scib) as test-only dependencies.
- Performance: numba kernels grouped by `by=`, float32 storage, chunked correlation and pairwise distances, streaming over groups for backed objects.
- Statistics: permutation nulls and FDR for every hit metric, effect sizes always reported alongside p-values.
- Community: schema review with CytoData, pycytominer and CytoTable maintainers before 0.1.0 and before the 1.0 freeze in 0.5.0.
