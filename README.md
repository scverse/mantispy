<img src="https://raw.githubusercontent.com/scverse/mantispy/main/docs/_static/mantispy-logo.png" width="400" alt="mantispy">

[![Tests][badge-tests]][tests]
[![Build][badge-build]][build]
[![Coverage][badge-codecov]][codecov]
[![Documentation][badge-docs]][documentation]
[![PyPI][badge-pypi]][pypi]
[![Python Version][badge-pyversions]][pypi]
[![License][badge-license]][license]
[![pre-commit][badge-precommit]][pre-commit]

[badge-tests]: https://img.shields.io/github/actions/workflow/status/scverse/mantispy/test.yaml?branch=main&label=tests
[badge-build]: https://img.shields.io/github/actions/workflow/status/scverse/mantispy/build.yaml?branch=main&label=build
[badge-codecov]: https://codecov.io/gh/scverse/mantispy/graph/badge.svg
[badge-docs]: https://app.readthedocs.org/projects/mantispy/badge/
[badge-pypi]: https://img.shields.io/pypi/v/mantispy?include_prereleases
[badge-pyversions]: https://img.shields.io/pypi/pyversions/mantispy
[badge-license]: https://img.shields.io/badge/license-BSD--3-blue
[badge-precommit]: https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white

mantispy brings Cell Painting and other image-based profiling data into the scverse ecosystem, from reading what a
pipeline wrote through quality control, normalization and batch correction to evaluating what survived.
It is for biologists and computational scientists who analyze Cell Painting or similar image-based screens in Python.
mantispy is in alpha: the API may still change.

```python
import mantispy as mt
import scanpy as sc

cells = mt.io.read_profiles("analysis/", platemap="platemap.csv")  # an ExportToSpreadsheet directory, or profile files
mt.pp.annotate_controls(cells)  # assumes the negative control is labeled "DMSO"; see negcon= for other names
mt.pp.image_qc(cells)  # which fields were out of focus
mt.pp.normalize(cells, by="Metadata_Plate", reference="negcon")  # robust MAD against the negative controls
wells = mt.tl.aggregate(cells)  # cells -> wells

mt.pp.feature_select(wells)  # flags features in var["selected"]
wells = mt.pp.subset_features(wells)  # keeps only those
sc.pp.pca(wells, n_comps=20)  # hit calling reads distances on a PCA, as the covariance of all features is singular
mt.tl.hit_calling(wells, use_rep="X_pca")  # which perturbations moved
mt.pl.hits(wells)
```

`pp` prepares, `tl` computes, `pl` plots, `metrics` evaluates a correction, `io` reads and
writes, and `ds` downloads public screens. Everything operates on an `AnnData`, so scanpy's
PCA, neighbors, UMAP and Leiden work on the same object. `io.read_plate` reads the images
behind the profiles as `SpatialData`.

To start, [From a CellProfiler run][tutorial-1] shows how to read a screen and [Hits, effects and cell
loss][tutorial-5] how to call hits on it. Neither needs your own data, and `mt.ds.bbbc021()`
downloads a public screen.

Set `mt.settings.verbosity = 2` when you first run your own screen. Several steps drop
features or wells, and they log it at that level.

Please refer to the [documentation][], in particular the [API documentation][].

## Installation

You need to have Python 3.12 or newer installed on your system.
If you don't have Python installed, we recommend installing [uv][].

We recommend managing dependencies in project-specific virtual environments to avoid dependency conflicts.
This is most convenient using package managers such as [uv][].
Choose from the options below to install mantispy:

1. Install the latest release from [PyPI][]. The current release is an alpha pre-release, so pip needs `--pre`:

   ```bash
   pip install --pre mantispy  # (or `uv add --prerelease allow mantispy`)
   ```

1. Install the latest development version:

   ```bash
   pip install git+https://github.com/scverse/mantispy.git  # (or `uv add`)
   ```

Optional extras add features: `spatial` (reading images as `SpatialData`), `map` (copairs
retrieval metrics), `harmony` and `integration` (batch correction and its scoring), and
`interactive` (plotly plots). Install the ones you need, for example:

```bash
pip install 'mantispy[spatial,map]'
```

The [installation guide](https://mantispy.readthedocs.io/en/latest/installation.html) says what
each extra pulls in.

## Release notes

See the [releases page](https://github.com/scverse/mantispy/releases).

## Contact

For questions and help requests, you can reach out in the [scverse discourse][].
If you found a bug, please use the [issue tracker][].

## License

mantispy is released under the BSD 3-Clause license.

## Citation

> A preprint describing mantispy is in preparation. Until then, cite the
> [repository][mantispy] directly.

[uv]: https://github.com/astral-sh/uv
[scverse discourse]: https://discourse.scverse.org/
[issue tracker]: https://github.com/scverse/mantispy/issues
[tests]: https://github.com/scverse/mantispy/actions/workflows/test.yaml
[build]: https://github.com/scverse/mantispy/actions/workflows/build.yaml
[codecov]: https://codecov.io/gh/scverse/mantispy
[pre-commit]: https://github.com/pre-commit/pre-commit
[documentation]: https://mantispy.readthedocs.io
[api documentation]: https://mantispy.readthedocs.io/page/api.html
[pypi]: https://pypi.org/project/mantispy
[license]: https://github.com/scverse/mantispy/blob/main/LICENSE
[mantispy]: https://github.com/scverse/mantispy
[tutorial-1]: https://mantispy.readthedocs.io/page/tutorials/data/cellprofiler.html
[tutorial-5]: https://mantispy.readthedocs.io/page/tutorials/compounds/hits.html

## scverse

mantispy is part of the scverse® project ([website](https://scverse.org), [governance](https://scverse.org/about/roles)) and is fiscally sponsored by [NumFOCUS](https://numfocus.org/).
If you like scverse® and want to support our mission, please consider making a tax-deductible [donation](https://numfocus.org/donate-to-scverse) to help the project pay for developer time, professional services, travel, workshops, and a variety of other needs.

<div align="center">
<a href="https://numfocus.org/project/scverse">
  <img
    src="https://raw.githubusercontent.com/numfocus/templates/master/images/numfocus-logo.png"
    width="200"
  >
</a>
</div>
