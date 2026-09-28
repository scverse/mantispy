# Reading and writing

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
    io.stamp
    io.validate
```

`io.read_profiles` reads the output of a profiling pipeline, and the input type decides how.
CSV, TSV or parquet files are stacked, a CellProfiler `ExportToSpreadsheet` directory is joined across its objects, and a directory of CytoTable parquet parts is read as single cells.
`io.read_plate` reads the images and segmentations behind the profiles into `SpatialData`, from a Cell Painting Gallery source {cite:p}`Weisbart_2024` or an `ExportForSpatialData` plate folder, and needs the spatial extra:

```bash
pip install 'mantispy[spatial]'
```

By default `io.read_profiles` keeps features from the three compartments (`Cells`, `Cytoplasm`, `Nuclei`), matching `pycytominer.infer_cp_features`.
Whole-field `Image_` measurements are excluded (a JUMP plate has 1077 of them against 3616 per-cell features); `objects=None` keeps them.

| Function | Stores |
| --- | --- |
| `io.read_profiles` | `X`, `obs` (`Metadata_*`, and from an export directory `Metadata_Center_X`/`_Y`), `var` (parsed annotation), `uns["mantispy"]`: `schema_version`, `resolution`, `channels`, `params`, and `image_table` from an export directory |
| `io.read_plate` | returns `SpatialData`: fields of view as Images, segmentations as Labels, wells as Shapes; the `cells` and `wells` Tables of a gallery source follow the [data contract](../api.md#the-data-contract) |
| `io.read_jump` | as `io.read_profiles`, plus `obs`: `Metadata_JCP2022`, `Metadata_Perturbation`, `Metadata_InChIKey`, `Metadata_Control` |
| `io.write` | validates first, then writes h5ad (or zarr for a `.zarr` suffix) |
| `io.stamp` | `uns["mantispy"]`: `schema_version`, `resolution`; `var` (the required annotation columns, empty where absent). Refuses an object whose `obs` lacks a column that resolution requires |

`io.validate` returns a report rather than raising, unless `raise_on_error=True`:

```{eval-rst}
.. autoclass:: mantispy._core.schema.ValidationReport
    :members:
```

## Working backed

`mt.io.read(path, backed="r")` leaves `X` on disk.
`obs` and `var` stay in memory, so metadata, QC flags and feature selection work unchanged.
Functions that rewrite `X` need `copy=True` and raise a `ValueError` saying so when it is missing.

| operation | backed |
| --- | --- |
| `pp.calculate_qc_metrics`, `pp.outliers`, `pp.feature_select`, `pp.well_qc` | yes (they write `obs`/`var`) |
| `tl.aggregate`, `tl.consensus`, `tl.map`, `tl.hit_calling`, the metrics | yes (they return new objects or tables) |
| `pp.normalize`, `pp.sphere`, `pp.correct_plate_position`, `pp.regress_out` | with `copy=True`; the result is materialized |
| writing `X` in place | no; the file is opened read-only |

Grouped reductions read one group at a time when the matrix is on disk, and the tests assert that the results are identical to the in-memory path.
For a per-plate median on 100 000 cells x 500 features, backed mode peaks at 44 MB against a 200 MB resident matrix in memory, and takes 3.5 s against 0.8 s.
That is about four times the runtime for four times less memory, so use it only on data that does not fit in memory.
