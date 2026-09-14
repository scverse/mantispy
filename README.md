# mantispy

[![Tests][badge-tests]][tests]
[![Documentation][badge-docs]][documentation]

[badge-tests]: https://img.shields.io/github/actions/workflow/status/scverse/mantispy/test.yaml?branch=main
[badge-docs]: https://app.readthedocs.org/projects/mantispy/badge/

Image-based profiling on AnnData

```python
import mantispy as mt

adata = mt.io.read_profiles("BR00116991_normalized.csv.gz", index_columns=("Plate", "Well"))
sdata = mt.io.read_plate("cpg0000-jump-pilot/source_4", "BR00116991", batch="2020_11_04_CPJUMP1")
```

`mantispy.io` reads image-based profiling data into scverse structures: `read_profiles` turns the tables a
CellProfiler or pycytominer pipeline writes into an `AnnData` of observations × features, and `read_plate`
turns the images and segmentations behind them into a `SpatialData` object — from a
[Cell Painting Gallery](https://github.com/broadinstitute/cellpainting-gallery) source or from a plate folder
written by the `ExportForSpatialData` CellProfiler module.

Reading images needs the spatial stack, which is an extra:

```bash
pip install 'mantispy[spatial]'
```

`mantispy.ds` has a synthetic plate to try things on and a few real ones to download:

```python
sdata = mt.ds.blobs()  # synthetic, no download
adata = mt.ds.lincs()  # two LINCS plates from the Cell Painting Gallery
```

## Getting started

Please refer to the [documentation][],
in particular, the [API documentation][].

## Installation

You need to have Python 3.12 or newer installed on your system.
If you don't have Python installed, we recommend installing [uv][].

We recommend managing dependencies in project-specific virtual environments to avoid dependency conflicts.
This is most convenient using package managers such as [uv][].
Choose from the options below to install mantispy:

<!--
1. Add the latest release of `mantispy` from [PyPI][] to your `uv` project:

   ```bash
   uv add mantispy
   ```

1. Install the latest release into a [standard virtual environment][venv]:

   ```bash
   (after activating your venv)
   pip install mantispy
   ```

-->

1. Install the latest development version:

   ```bash
   pip install git+https://github.com/scverse/mantispy.git  # (or `uv add`)
   ```

## Release notes

See the [changelog][].

## Contact

For questions and help requests, you can reach out in the [scverse discourse][].
If you found a bug, please use the [issue tracker][].

## Citation

> t.b.a

[uv]: https://github.com/astral-sh/uv
[scverse discourse]: https://discourse.scverse.org/
[issue tracker]: https://github.com/scverse/mantispy/issues
[tests]: https://github.com/scverse/mantispy/actions/workflows/test.yaml
[documentation]: https://mantispy.readthedocs.io
[changelog]: https://mantispy.readthedocs.io/page/changelog.html
[api documentation]: https://mantispy.readthedocs.io/page/api.html
[pypi]: https://pypi.org/project/mantispy
[venv]: https://docs.python.org/3/tutorial/venv.html
