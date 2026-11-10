# Installation

You need Python 3.13 or newer.
If you don't have Python installed, we recommend [uv](https://github.com/astral-sh/uv).
PyPI is the supported install. Cloning the repository and running `uv sync` is for contributors, see [Contributing](contributing.md).

## Alpha release

mantispy is in alpha, and its releases are pre-releases (0.1.0aN).
While no final release exists, pip and uv both fall back to the newest pre-release, so a plain install works:

```console
uv add mantispy  # in a uv project; `uv init` creates one
pip install mantispy  # or with pip
```

Pass `--pre` (`uv pip install --pre mantispy`, or `uv add --prerelease allow mantispy` in a project) to get pre-releases of mantispy's dependencies as well, or to keep getting the newest alpha once a final release is out.

## Development version

Install the latest development version from GitHub:

```console
uv add git+https://github.com/scverse/mantispy  # in a uv project
pip install git+https://github.com/scverse/mantispy.git  # or with pip
```

## Optional dependencies

### Images and segmentations

{func}`~mantispy.io.read_plate`, {func}`~mantispy.ds.jump_plate` and {func}`~mantispy.ds.blobs` read images into `SpatialData`:

```console
uv add 'mantispy[spatial]'  # in a uv project
pip install 'mantispy[spatial]'  # or with pip
```

### Mean average precision

{func}`~mantispy.tl.map` runs copairs, an optional extra:

```console
uv add 'mantispy[map]'  # in a uv project
pip install 'mantispy[map]'  # or with pip
```

### Integration metrics

{func}`~mantispy.metrics.evaluate_integration` delegates to scib-metrics:

```console
uv add 'mantispy[integration]'  # in a uv project
pip install 'mantispy[integration]'  # or with pip
```

### Interactive plots

With plotly installed, plots in a Jupyter notebook also draw an interactive version:

```console
uv add 'mantispy[interactive]'  # in a uv project
pip install 'mantispy[interactive]'  # or with pip
```

### Tutorials

The single-cell tutorial clusters cells with `scanpy.tl.leiden(..., flavor="igraph")`, which needs igraph:

```console
uv add igraph  # in a uv project
pip install igraph  # or with pip
```

To run every tutorial and case study, install all of the above at once:

```console
uv add 'mantispy[map,integration,interactive]' igraph  # in a uv project
pip install 'mantispy[map,integration,interactive]' igraph  # or with pip
```
