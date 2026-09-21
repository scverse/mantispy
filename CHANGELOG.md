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
