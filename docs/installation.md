# Installation

You need Python 3.12 or newer.
If you don't have Python installed, we recommend [uv](https://github.com/astral-sh/uv).

## Alpha release

mantispy 0.1.0a2 is an alpha pre-release.
pip skips pre-releases by default, so a plain `pip install mantispy` does not select it.
Pass `--pre` to install it:

```console
pip install --pre mantispy
```

With [uv](https://github.com/astral-sh/uv) the equivalent is `uv pip install --pre mantispy`, or `uv add --prerelease allow mantispy` in a project.

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
