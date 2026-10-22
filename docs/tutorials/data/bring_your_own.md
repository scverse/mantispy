# Bring your own data

This page takes a pycytominer or CellProfiler table to a stamped mantispy object.
Everything downstream reads the stamp, so an object that carries it works with every `mt.pp`, `mt.tl` and `mt.pl` function.

## What a row is

Pick the resolution that matches the table, then make sure `obs` carries the identifier columns it requires.

| `resolution` | One row is | Required `obs` columns | `grouped_by` |
| --- | --- | --- | --- |
| `object` | one segmented cell or other object | `Metadata_ImageID`, `Metadata_ObjectType`, `Metadata_ObjectNumber`, plus `Metadata_Plate` and `Metadata_Well` for a plate | defaults |
| `well` | one well | `Metadata_Plate`, `Metadata_Well` | defaults to plate and well |
| `aggregate` | one consensus profile, such as a gene or a compound | none | you give it, for example `["Metadata_Perturbation"]` |

There is no guide level.
Rows are objects, wells or aggregates, and guides are values in a column.

`Metadata_Well` is a zero-padded well name such as `A03`, not `A3`.
`mt.io.read_profiles` normalizes the spelling when it reads a table or a platemap.
An object assembled by hand has to use it already.

`Metadata_Perturbation` is the canonical perturbation column.
Compound names, gene symbols and ORF identifiers all go there, and the tools that need to know what was done to a row read it first.
Keep a more specific column such as `Metadata_Compound` or `Metadata_Gene` beside it if you want one.

## Read a table

```python
import mantispy as mt

adata = mt.io.read_profiles("profiles.parquet", platemap="platemap.csv")
```

`read_profiles` splits the measurements from the metadata and records `resolution="well"` for files.
`platemap` is a table, or a path to one, with a `Metadata_Well` column and optionally `Metadata_Plate`.
It is left-joined onto `obs`, which is how the perturbation reaches each well.
Pass `resolution="aggregate"` with `grouped_by` for a table that holds one row per gene.

## Mark the controls

```python
mt.pp.annotate_controls(adata, negcon=["DMSO"])
```

`annotate_controls` marks the negative controls, and optionally the positive ones, from the perturbation column.
Readouts such as `mt.tl.map` and `mt.tl.hit_calling` compare against these rows.

## Stamp an object that came from elsewhere

A published `h5ad`, another pipeline's output or a matrix of embeddings has no stamp yet.
`mt.io.stamp` records the resolution and the kind of feature, and rejects the object if `obs` lacks a required column.

```python
# one row per well, CellProfiler measurements
mt.io.stamp(adata, resolution="well", feature_kind="measurement")

# one row per gene
mt.io.stamp(adata, resolution="aggregate", grouped_by=["Metadata_Perturbation"], feature_kind="measurement")

# learned embeddings
adata.X = adata.X.astype("float32")
mt.io.stamp(adata, resolution="well", feature_kind="embedding")
```

`X` is stored as `float32`, so convert a `float64` matrix before stamping.

## Build one from a DataFrame

{func}`~mantispy.io.from_dataframe` turns a table that is already in memory into a stamped object.
It splits `Metadata_*` columns from features and parses CellProfiler feature names.

A slide or any other table without plates and wells needs no `Metadata_Plate` or `Metadata_Well` at object resolution.
The field of view comes from `Metadata_Site`, or from an explicit `Metadata_ImageID` if the table has one.
An object is identified by `Metadata_ImageID`, `Metadata_ObjectType` and `Metadata_ObjectNumber`.

```python
adata = mt.io.from_dataframe(df, resolution="object", object_type="cell")
```

A table from cp_measure uses different feature names.
Say so with `convention`; the default is `"cellprofiler"`.

```python
adata = mt.io.from_dataframe(df, convention="cp_measure", object_type="cell")
```

Numeric columns that match neither convention, such as proprietary or learned features, are kept as features with `feature_kind="embedding"`, never dropped.
`object_type` is also an argument of {func}`~mantispy.io.stamp`.

## Check it

```python
report = mt.io.validate(adata)
```

`validate` returns a report of what is wrong and what is only worth a warning.
Stamping already runs it, so a stamped object passes.
Run it again after you edit `obs` or `X` by hand.

See the {doc}`data model <../../data_model/index>` for every field and the {func}`~mantispy.io.stamp` reference for the arguments.
