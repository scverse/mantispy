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

### Changed

- `tl.consensus` now defaults to `method="median"` (was `modz`) — matches pycytominer and is more robust; breaking change

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
