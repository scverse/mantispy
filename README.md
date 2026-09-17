# mantispy

[![Tests][badge-tests]][tests]
[![Documentation][badge-docs]][documentation]

[badge-tests]: https://img.shields.io/github/actions/workflow/status/scverse/mantispy/test.yaml?branch=main
[badge-docs]: https://app.readthedocs.org/projects/mantispy/badge/

mantispy brings Cell Painting and other image-based profiling data into the scverse ecosystem, from reading what a
pipeline wrote through quality control, normalization and batch correction to evaluating what survived.

```python
import mantispy as mt

cells = mt.io.read_profiles("analysis/", platemap="platemap.csv")  # an ExportToSpreadsheet directory, or profile files
mt.pp.annotate_controls(cells)
mt.pp.image_qc(cells)  # which fields were out of focus
mt.pp.normalize(cells, by="Metadata_Plate", reference="negcon")  # robust MAD against the negative controls
wells = mt.tl.aggregate(cells)  # cells -> wells

mt.pp.feature_select(wells)
mt.tl.hit_calling(wells)  # which perturbations moved
mt.pl.hits(wells)
```

`pp` prepares, `tl` computes, `pl` plots, `metrics` evaluates a correction, `io` reads and
writes, and `ds` downloads public screens. Everything operates on an `AnnData`, so scanpy's
PCA, neighbors, UMAP and Leiden work on the same object. `io.read_plate` reads the images
behind the profiles as `SpatialData`.

To start, [From CellProfiler to AnnData][tutorial-1] shows how to read a screen and [Hits and
effects][tutorial-5] how to call hits on it. Neither needs your own data, and `mt.ds.bbbc021()`
downloads a public screen.

Set `mt.settings.verbosity = 2` when you first run your own screen. Several steps drop
features or wells, and they log it at that level.

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

Reading images needs the `spatial` extra:

```bash
pip install 'mantispy[spatial]'
```

## Release notes

See the [changelog][].

## Contact

For questions and help requests, you can reach out in the [scverse discourse][].
If you found a bug, please use the [issue tracker][].

## Citation

> A preprint describing mantispy is in preparation. Until then, cite the
> [repository][mantispy] directly.

[uv]: https://github.com/astral-sh/uv
[scverse discourse]: https://discourse.scverse.org/
[issue tracker]: https://github.com/scverse/mantispy/issues
[tests]: https://github.com/scverse/mantispy/actions/workflows/test.yaml
[documentation]: https://mantispy.readthedocs.io
[changelog]: https://mantispy.readthedocs.io/page/changelog.html
[api documentation]: https://mantispy.readthedocs.io/page/api.html
[pypi]: https://pypi.org/project/mantispy
[venv]: https://docs.python.org/3/tutorial/venv.html
[mantispy]: https://github.com/scverse/mantispy
[tutorial-1]: https://mantispy.readthedocs.io/en/latest/tutorials/01_from_cellprofiler_to_anndata.html
[tutorial-5]: https://mantispy.readthedocs.io/en/latest/tutorials/05_hits_and_effects.html
