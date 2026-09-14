# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog][],
and this project adheres to [Semantic Versioning][].

[keep a changelog]: https://keepachangelog.com/
[semantic versioning]: https://semver.org/

## [Unreleased]

### Added

- `mantispy.io.read_profiles`, reading CellProfiler well- and cell-level profiles into `AnnData`
- `mantispy.io.read_plate`, reading a Cell Painting Gallery source or a CellProfiler `ExportForSpatialData`
  plate folder into `SpatialData`
- `mantispy.ds`, with the synthetic `blobs` and `blobs_profiles` and the downloadable `cpjump1` and `lincs`
- `mantispy.settings`, holding the cache directory the datasets download into
- Basic tool, preprocessing and plotting functions
