# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog][],
and this project adheres to [Semantic Versioning][].

[keep a changelog]: https://keepachangelog.com/
[semantic versioning]: https://semver.org/

## [Unreleased]

### Added

- `mantispy.io`: `read_profiles` for profile files, CellProfiler `ExportToSpreadsheet` directories and CytoTable parquet parts; `read_plate` for a Cell Painting Gallery source or an `ExportForSpatialData` plate folder as `SpatialData`; `read_jump`, `read`, `write` and `validate`
- `mantispy.ds`: the generated `synthetic_plate` and `blobs`; `bbbc021`, `rohban`, `pki` and `jump_target2` with the annotations the analyses need; nine further Cell Painting Gallery accessions
- `mantispy.ds`: `jump_cells`, `jump_export` and `jump_plate`, the single cells, one CellProfiler export directory and the images of `BR00121438`, the plate `jump_target2` reads well profiles for; `jump_cells(selected=True)` returns only the features `var['selected']` marks
- `mantispy.pp`: quality control at cell, image and well level, normalization, feature selection, outlier detection, sphering, plate-position correction and Harmony
- `mantispy.tl`: aggregation, consensus profiles, mAP and replicate retrieval, hit calling, effect sizes, dose response, mechanism-of-action retrieval and enrichment, differential features, transport across sites and single-cell heterogeneity
- `mantispy.metrics`, `mantispy.get` and `mantispy.pl`, for judging a correction, reading results out and plotting them
- `mantispy.settings`, holding the verbosity and the cache directory the datasets download into
- `mantispy.io`: `stamp`, which puts an `AnnData` built elsewhere — a published h5ad, another pipeline's output, a matrix of learned embeddings — on the mantispy API surface
- `mantispy.metrics`: `known_relationships`, the share of annotated perturbation pairs whose similarity falls in either tail of the distribution over all pairs, and `evaluate_correction(covariates=...)`, which reports what a representation spends its variance on besides the batch and the label
- `mantispy.pp`: `tvn`, typical variation normalization with per-batch CORAL, which aligns each batch's controls onto the pooled controls
- `mantispy.ds`: `jump_lite`, the same 1,536 JUMP Target-2 wells embedded by five models and measured by `cp_measure`, and `jump_lite_targets`, the gene each compound is annotated to act on
- `mantispy.metrics`: `known_relationships(n_permutations=...)` measures chance by shuffling which perturbation each annotation row names, keeping every set's size and every perturbation's number of sets, and reports it as `null` with a `p_value`. Chance is 2 × `percentile` only when every perturbation belongs to the same number of sets
- `mantispy.pp`: `feature_select`'s `drop_degenerate`, run first by default, drops the features `normalize` flagged in `var["degenerate_scale"]` so that they no longer decide which other features are kept
- `mantispy.ds`: `jump_crispr` joins JUMP's CRISPR annotation by default, naming the gene each well targets, its control type and the chromosome arm the gene sits on, and `corum` returns CORUM's protein complexes in the shape `known_relationships` and `pathway_coherence` read
- `mantispy.pp`: `annotate_jump(kind="crispr")`, which also reads profiles that already carry `Metadata_JCP2022`, as JUMP's assembled profiles do
- `mantispy.pl`: `hits` and `feature_volcano` write how many points sit above and below the significance line, next to it
- `mantispy.pp`: `regress_out(reference=...)` fits the covariate on the reference rows, re-expresses each group at their mean and clips it to their range, so a cell count is regressed out where density varies for technical reasons only; without a reference it warns when a group never reaches the value it is re-expressed at

### Fixed

- `mantispy.io`: an `ExportToSpreadsheet` directory takes its channels from the features it measured, so a run whose images are named `OrigDNA` or `IllumDNA` no longer leaves every feature without a channel
- `mantispy.io`: CellProfiler 4's `AreaShape_Center_X/Y` and bounding-box corners are read as where an object sits rather than as features, and the centroid goes to `Metadata_Center_X/Y`; nine of the packaged datasets carried them in their profiles
- `mantispy.ds`: `jump_cells` keeps the image quality of every field of view, each under its own image number, instead of the first field's alone
- `mantispy.pl`: `feature_groups` counts features that have no channel, such as `AreaShape`, under "none"; under pandas 3 it left them out of the bars
- `mantispy.io`: `cp_measure` column names are read as `<object>_<channel>/<aggregation>/<group><Feature>` rather than through the CellProfiler grammar, which left `var['channel']` empty and split one feature group into as many as the channels and aggregations it was written with
- `mantispy.pp`: `well_qc` says it expects cell resolution, instead of counting one row per well and failing every well on a well-level object
- `mantispy.pp`: `feature_select` warns when it selects nothing, rather than leaving an empty matrix for whatever runs next; `noise_removal`'s `stdev_cutoff` is documented as an absolute threshold on the scale `normalize` left the values on
- `mantispy.tl`: `map(mode="activity")` retrieves against the controls on the query's own plate. It pooled every plate's controls, so a perturbation with no effect of its own looked more active the more controls the other plates carried
- `mantispy.tl`: `map` leaves out, with a warning, a query whose replicates have no negative pair to be ranked against, such as one on a plate without controls under `mode="activity"`. Scored, it came out at an average precision of 1 and the smallest p-value
- `mantispy.tl`: `map` warns when `null_size` is too small for the multiple-testing correction to call a group on its own
- `mantispy.ds`: `jump_lite` returns every feature set with the wells in one order, sorted by source, plate and well, where each file lists them in its own
- `mantispy.tl`: `map` draws its permutation nulls afresh on every call instead of through copairs' cache in the home directory. The cache keys a null without the seed it was drawn with, so a p-value depended on whichever earlier call had written that null
- `mantispy.tl`: `feature_signature` and `dose_trajectory` build `var` from the annotation columns every non-CellProfiler object supplies, so `io.validate` accepts their results and `io.write` writes them. Both stamped an object that carried three and one of the ten columns the schema requires, and could not be saved; `cluster_composition`, which supplied all ten by hand, now takes them from the same helper, so its empty `channel`, `radial_bin` and `params` are categoricals rather than floats
- `mantispy._core`: `stamp` supplies the annotation columns `var` does not carry, so a tool that builds a new object cannot return one `io.validate` rejects; `io.write` opts out, and still reports an annotation a caller has damaged rather than repairing it. `metrics.diagnose_testing`'s internal objects carried none of the ten
- `mantispy.pp`: `calculate_qc_metrics` treats an empty `feature` column as no column, so `qc_pass` cannot claim an area check that never ran. Any object stamped rather than parsed — a learned embedding through `io.stamp` — carried the column with nothing in it and slipped past the guard
- `mantispy.tl`: `feature_signature`'s `by` columns keep the dtype and the missing values `var` held them in, instead of the strings that name the family; grouping on `scale` no longer writes text into a float column, on `is_feature` no longer writes `"True"` into a boolean one, and a genuinely absent channel is missing rather than the string `"none"`. Naming the same column twice now raises
- `mantispy.tl`: `dose_trajectory` widens its position suffix past 101 positions, where a fixed two decimals named several positions identically and gave the object duplicate `var_names`
- `mantispy.tl`: `cluster_composition` leaves a cell the clustering did not assign out of the fractions, with a warning. Its code is -1, which counted it into the last cluster, and the label it contributed became a cluster called `nan` under pandas 2 or raised under pandas 3
