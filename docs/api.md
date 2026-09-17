# API

Import as:

```python
import mantispy as mt
```

Every function that modifies an object takes `copy`. With `copy=False` (the default) it
mutates in place and returns `None`; with `copy=True` it returns a modified copy and
leaves the input alone. Each call records its arguments under
`uns["mantispy"]["params"][<function name>]`, so a finished object says how it was made.

The Stores column in the tables below lists the keys each function writes.

## Reading and writing

```{eval-rst}
.. module:: mantispy.io
.. currentmodule:: mantispy

.. autosummary::
    :toctree: generated

    io.read_profiles
    io.read_plate
    io.read_jump
    io.read
    io.write
    io.validate
```

`io.read_profiles` reads the output of a profiling pipeline, and the input type decides how. CSV, TSV or parquet
files are stacked, a CellProfiler `ExportToSpreadsheet` directory is joined across its objects, and a directory of
CytoTable parquet parts is read as single cells. `io.read_plate` reads the images and segmentations
behind the profiles into `SpatialData`, from a Cell Painting Gallery source or an `ExportForSpatialData` plate
folder, and needs the spatial extra:

```bash
pip install 'mantispy[spatial]'
```

| Function | Stores |
| --- | --- |
| `io.read_profiles` | `X`, `obs` (`Metadata_*`, and from an export directory `Metadata_Center_X`/`_Y`), `var` (parsed annotation), `uns["mantispy"]`: `schema_version`, `resolution`, `channels`, `params`, and `image_table` from an export directory |
| `io.read_plate` | returns `SpatialData`: fields of view as Images, segmentations as Labels, wells as Shapes; the `cells` and `wells` Tables of a gallery source follow the contract below |
| `io.read_jump` | as `io.read_profiles`, plus `obs`: `Metadata_JCP2022`, `Metadata_Perturbation`, `Metadata_InChIKey`, `Metadata_Control` |
| `io.write` | validates first, then writes h5ad (or zarr for a `.zarr` suffix) |

`io.validate` returns a report rather than raising, unless `raise_on_error=True`:

```{eval-rst}
.. autoclass:: mantispy._core.schema.ValidationReport
    :members:
```

## Accessors

```{eval-rst}
.. module:: mantispy.get
.. currentmodule:: mantispy

.. autosummary::
    :toctree: generated

    get.features
    get.to_dataframe
    get.controls
    get.obs_df
    get.var_df
```

Accessors do not modify the object. `get.to_dataframe` returns the flat table that
pycytominer and similar tools expect.

## Datasets

```{eval-rst}
.. module:: mantispy.ds
.. currentmodule:: mantispy

.. autosummary::
    :toctree: generated

    ds.synthetic_plate
    ds.blobs
    ds.bbbc021
    ds.rohban
    ds.pki
    ds.jump_target2
    ds.agnp
    ds.amish
    ds.chroma
    ds.jump_crispr
    ds.luad
    ds.miami
    ds.neuropainting
    ds.oasis_pilot
    ds.pooled_rare
```

By default `io.read_profiles` keeps features from the three compartments (`Cells`, `Cytoplasm`, `Nuclei`), matching
`pycytominer.infer_cp_features`. Whole-field `Image_` measurements are excluded (a JUMP plate has 1077 of them
against 3634 per-cell features); `objects=None` keeps them.

`synthetic_plate` and `blobs` are generated locally. `synthetic_plate` is a single-cell profile table with injected
artifacts for quality control to find; `blobs` is a small `SpatialData` plate of images, labels and tables. The other
datasets download once, checked against a pinned sha256, into `mt.settings.cache_dir` (set `MANTISPY_CACHE_DIR` to
change it). Four of them carry the annotations the analysis functions need:

| dataset | download | perturbations | carries |
|---|---|---|---|
| `bbbc021` | ~10 MB | 39 compounds | MOA labels, the classic retrieval benchmark |
| `rohban` | ~27 MB | 194 overexpressed genes | cell counts, ~10 replicates per gene |
| `pki` | ~71 MB | 15 kinase inhibitors x 7 doses | cell counts, MOA labels, 32-64 replicates |
| `jump_target2` | ~40 MB | 302 compounds, one shared plate map | the same plate run at two sites, so any difference between them is technical |

The other nine are further gallery accessions, normalized and feature-selected by their authors and read with the
`io.read_profiles` defaults. Use them to run a method across a range of screens.

Check anything tuned on one dataset against the others. Cutoffs that looked universal on BBBC021 turned out to be
dataset-dependent on `rohban` and `pki`.

## Settings

```{eval-rst}
.. currentmodule:: mantispy

.. autosummary::
    :toctree: generated

    settings
    settings.override
    settings.reset
```

Both settings, `verbosity` and `cache_dir`, are also read from `MANTISPY_VERBOSITY` and `MANTISPY_CACHE_DIR`, and
`with mt.settings.override(verbosity=2):` changes one for a block.

## The data contract

`X` is `float32`, one row per profile. `obs` carries `Metadata_` columns identifying
where each profile came from; `var` carries the parsed feature annotation
(`object`, `feature_group`, `feature`, `channel`, `scale`, `angle`, `gray_levels`,
`radial_bin`, `params`, `is_feature`). `uns["mantispy"]` holds the schema version, the
resolution, the channel vocabulary and provenance.

Resolution (`"cell"`, `"well"`, `"perturbation"`) is advisory. It sets which identifier
columns `io.validate` requires, and functions warn instead of raising when they get a
resolution they do not expect.

The machine-readable contract is published as `spec/schema-1.0.json`, alongside the
earlier `spec/schema-0.1.json`.

### Seeing what mantispy is doing

mantispy logs what it drops, skips and shrinks through the standard library's `logging`.
Warnings print by default; raise the verbosity to see the rest:

```python
import mantispy as mt

mt.settings.verbosity = 2  # 0 errors, 1 warnings (default), 2 info, 3 debug
```

Set it to 2 when you first run a new screen. Several functions discard features or wells,
and they log it at that level.

## Stability

The schema is frozen at 1.0. The names in `spec/schema-1.0.json` (the required and
reserved `obs` columns, the `var` annotation, the `uns["mantispy"]` keys and the result
tables) are stable, as are the public signatures in the tables above.

- Anything removed gets a `DeprecationWarning` for two minor releases first.
- `mt.io.read` migrates a file written against an older schema on read. A schema version
  this build does not know raises an error.
- `_core` and other `_`-prefixed modules are private.

`io.read_plate` and `ds.blobs` bring images and segmentations in as SpatialData on top of this contract, without
changing it.
