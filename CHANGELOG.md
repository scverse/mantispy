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
- `mantispy.get`, for reading features, controls and flat tables out of an object
- `mantispy.settings`, holding the verbosity and the cache directory the datasets download into
- `mantispy.pp`: quality control at cell, image and well level, normalization, feature selection, outlier detection, sphering, plate-position correction and Harmony
- `mantispy.tl`: aggregation, consensus profiles, mAP and replicate retrieval, hit calling, effect sizes, dose response, mechanism-of-action retrieval and enrichment, differential features, transport across sites and single-cell heterogeneity
- `mantispy.metrics`, for judging a batch correction
