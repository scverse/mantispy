# Installation

You need Python 3.12 or newer.
If you don't have Python installed, we recommend [uv](https://github.com/astral-sh/uv).

## Alpha release

mantispy is in alpha, and its releases are pre-releases (0.1.0aN).
While no final release exists, pip and uv both fall back to the newest pre-release, so a plain install works:

```console
pip install mantispy
```

Pass `--pre` (`uv pip install --pre mantispy`, or `uv add --prerelease allow mantispy` in a project) to get pre-releases of mantispy's dependencies as well, or to keep getting the newest alpha once a final release is out.

## Development version

Install the latest development version from GitHub:

```console
pip install git+https://github.com/scverse/mantispy.git
```

## Optional dependencies

### Images and segmentations

{func}`~mantispy.io.read_plate`, {func}`~mantispy.ds.jump_plate` and {func}`~mantispy.ds.blobs` read images into `SpatialData`:

```console
pip install 'mantispy[spatial]'
```

### Harmony

{func}`~mantispy.pp.harmony` needs harmonypy:

```console
pip install 'mantispy[harmony]'
```

### Mean average precision

{func}`~mantispy.tl.map` runs copairs, an optional extra:

```console
pip install 'mantispy[map]'
```

### Integration metrics

{func}`~mantispy.metrics.evaluate_integration` delegates to scib-metrics:

```console
pip install 'mantispy[integration]'
```

### Interactive plots

With plotly installed, plots in a Jupyter notebook also draw an interactive version:

```console
pip install 'mantispy[interactive]'
```

### Tutorials

The single-cell tutorial clusters cells with `scanpy.tl.leiden(..., flavor="igraph")`, which needs igraph:

```console
pip install igraph
```

To run every tutorial and case study, install all of the above at once:

```console
pip install 'mantispy[map,harmony,integration,interactive]' igraph
```
