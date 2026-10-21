# Data model

Starting from your own table? Follow {doc}`../tutorials/data/bring_your_own`.

## Quick recipe

Readers such as `mt.io.read_profiles` record these values for you.
For an object that came from somewhere else, `mt.io.stamp` records them:

```python
import mantispy as mt

# one row per well, features are CellProfiler measurements
mt.io.stamp(adata, resolution="well", feature_kind="measurement")

# one row per gene
mt.io.stamp(adata, resolution="aggregate", grouped_by=["Metadata_Perturbation"], feature_kind="measurement")

# a matrix of learned embeddings, one row per well
mt.io.stamp(adata, resolution="well", feature_kind="embedding")
```

The `object` and `well` resolutions have a default `grouped_by`, so it can be left out.
The `aggregate` resolution has none, because only you know which columns define one aggregate.
Stamping at `aggregate` without `grouped_by` raises a `ValueError` that asks for it.

Each resolution requires identifier columns in `obs`:

- `object`: `Metadata_Plate`, `Metadata_Well`, `Metadata_ImageID`, `Metadata_ObjectType` and `Metadata_ObjectNumber`.
- `well`: `Metadata_Plate` and `Metadata_Well`.
- `aggregate`: none, but `grouped_by` names the columns that define a row.

Stamping runs the full validation, so an object that lacks them is rejected before it is changed.

`Metadata_Well` is a zero-padded well name such as `A03`.
`Metadata_Perturbation` is the canonical perturbation column; keep `Metadata_Compound` or `Metadata_Gene` beside it when you want the specific name.

Embedding provenance is recorded with `stamp(provenance={"model": ..., "version": ..., "pooling": ...})` and stored in `uns["mantispy"]["embedding"]`.
`model` is required when `provenance` is given.

## Why there is a schema

Image-based profiling data arrives in many shapes: one row per segmented cell, one row per well, one row per gene.
The columns might be CellProfiler measurements, a learned embedding, or values computed from either.
Without a shared contract, every function would have to guess which of these it was given, and a wrong guess changes the answer without raising an error.

mantispy writes the answer down once, inside the `AnnData`, and every function reads it from there.
A function that needs to know what a row is fails with a clear message when the object does not say, and it never assumes.

## The four independent axes

Four questions describe any profiling object, and mantispy keeps the answer to each in its own place.
No answer is encoded in another: the resolution is never `gene`, and a column of embeddings is not a different kind of row.

| Question | Where the answer lives | Allowed values |
| --- | --- | --- |
| What does one row represent? | `uns["mantispy"]["resolution"]` and `uns["mantispy"]["grouped_by"]` | `object`, `well`, `aggregate` |
| What does one column represent? | `var["feature_kind"]` | `measurement`, `embedding`, `derived` |
| What treatment or condition produced it? | `Metadata_Compound`, `Metadata_Gene`, `Metadata_sgRNA` and related `obs` columns | the biological intervention |
| Where did it come from? | `Metadata_Plate`, `Metadata_Well`, `Metadata_ImageID`, `Metadata_ObjectNumber` | the identity and origin of the observation |

The first two axes describe the shape of the data and are stamped on the object.
The last two are ordinary `obs` columns that describe biology and provenance.

A gene-level profile shows how they separate.
It has `resolution="aggregate"` and `grouped_by=["Metadata_Gene"]` for the row structure.
The gene it targets sits in `Metadata_Gene`, which is the intervention.
The plate it was measured on sits in `Metadata_Plate`, which is the origin.

### One model, many modalities

Drug, CRISPR, ORF and OPS screens are not different schemas.
They are different annotations inside one model.

```python
# drug screen, well-level
resolution = "well"
grouped_by = ["Metadata_Plate", "Metadata_Well"]
# pooled CRISPR cells
resolution = "object"
grouped_by = []
# gene-level profiles
resolution = "aggregate"
grouped_by = ["Metadata_Gene"]
# a learned embedding (any resolution)
feature_kind = "embedding"
```

## Where the field-level detail lives

- {func}`mantispy.io.stamp` documents the arguments and the checks that run.
- The {doc}`stamp reference <stamp_reference>` explains every key of `uns["mantispy"]`: what it is, who sets it and how to choose a value.
- The {doc}`schema specification <schema_spec>` defines every field, requirement and naming rule. Section 2.6 is the source of the four axes above.
- The [tutorials](../tutorials/general) show each axis on real data.
