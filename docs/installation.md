# Installation

You need Python 3.12 or newer.
If you don't have Python installed, we recommend [uv](https://github.com/astral-sh/uv).

## Development version

mantispy is not on PyPI yet.
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

### Interactive plots

With plotly installed, plots in a Jupyter notebook also draw an interactive version:

```console
pip install 'mantispy[interactive]'
```
