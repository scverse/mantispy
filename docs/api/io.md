# Reading and writing

```{eval-rst}
.. module:: mantispy.io
.. currentmodule:: mantispy
```

Read CellProfiler exports, profiles, plates and JUMP data into `AnnData` or `SpatialData`, and check them against the schema.

```{eval-rst}
.. autosummary::
   :toctree: generated

   io.read_cellprofiler
   io.read_cellprofiler_csv
   io.read_cellprofiler_sqlite
   io.read_profiles
   io.read_plate
   io.read_jump
   io.unexpressed_genes
   io.read
   io.write
   io.stamp
   io.from_dataframe
   io.from_embedding
   io.validate
```

```{eval-rst}
.. autoclass:: mantispy._core.schema.ValidationReport
    :members:
```
