# The mantispy stamp (`uns["mantispy"]`)

The stamp is the dict at `adata.uns["mantispy"]`.
It is where mantispy writes down what an object is, so that every function can read the answer instead of guessing.
Readers such as `mt.io.read_profiles` and tools that return a new object write it for you; `mt.io.stamp` writes it for an object that came from somewhere else.
The concepts behind it are in {doc}`index`, and the exact contract is in the {doc}`schema specification <schema_spec>`.

## Core descriptors

The first five keys are required: an object without them is not a valid schema-2.0 object (spec section 3.5).
The others are optional and appear when the data or a tool has something to record.

| Key | What it is | Why it exists | Who sets it | Allowed values | How to choose |
| --- | --- | --- | --- | --- | --- |
| `schema_version` (required) | The schema version the object follows. | Lets a reader tell a current object from one written by an earlier mantispy. | Written by `mt.io.stamp` and by every loader and tool that stamps. | A version string; `"2.0"` is the current one. | Not a choice. It is filled in for you. |
| `schema_status` (required) | Whether the schema version is `experimental` or `stable`. | The schema is still being finished; the status says how far to trust it (spec section 3.5). | Written by `mt.io.stamp` and by every loader and tool that stamps. | `"experimental"` or `"stable"`. An object keeps the status it was stamped with. | Not a choice. It is filled in for you. |
| `resolution` (required) | What one row of the object represents. | Functions need to know whether a row is a segmented object, a well or something coarser, and a wrong guess changes the result (spec section 4). | You, through `mt.io.stamp`, or a loader. `tl.aggregate` writes the new one when it changes the level. | `"object"`, `"well"` or `"aggregate"`. | Pick the structural tier that matches a row. Re-stamping without a value keeps the recorded one; an object that records none must be given one. |
| `grouped_by` (required) | The `obs` columns that define one row. | The resolution is only a tier; this names the exact unit, such as a gene or a compound and dose (spec sections 2.5 and 4.3). | You, through `mt.io.stamp`, or a loader. `tl.aggregate` writes it for the result. | A list of `obs` column names, which must exist in `obs`. | `object` and `well` have a default. `aggregate` has none and needs the columns that define a row. |
| `history` (required) | An ordered, append-only log of what was done to the object. | Keeps portable provenance inside the object (spec section 14). | `mt.io.stamp` starts it on the first stamp. Tools append to it, and an aggregated object inherits its source's log. | A list of operation records. | Not a choice. Do not edit it by hand. |
| `channels` | The channel names used to parse the feature names. | Records the vocabulary, given or inferred, so the parse can be reproduced from the object. | Written by `mt.io.read_profiles` and by the synthetic dataset builder. `tl.aggregate` carries it over. | A list of channel names. | Pass the vocabulary to the reader when the default parse does not fit your data. |
| `dataset` | The accession or name of the dataset the object came from. | Says which published dataset a staged object is. | Written by the `mt.ds` dataset loaders. `tl.aggregate` carries it over. | A string such as an accession. | Not set by hand for your own data. |
| `image_table` | One row of image-level metadata per field of view. | Image QC and other image-level steps read it, and every object's `Metadata_ImageID` has to resolve to exactly one row (spec section 8). | Written by `mt.io.read_profiles` from the CellProfiler image table, and by the synthetic dataset builder. | A table indexed by `Metadata_ImageID`. | Keep it when you have image-level measurements. `pp.image_qc` needs it. |
| `params` | The parameters of the latest call of each tool. | A plot or downstream function reads result-specific parameters from here. | Tools, automatically. | A mapping from function name to its parameters. A second call of the same function overwrites the first. | Not a choice. Use `history` for the full record of calls. |
| `aggregated_from` | How an aggregated object was built from its source. | Records the grouping, function and size of the source. | Written by `tl.aggregate`. | A mapping with `by`, `func`, `n_obs` and `min_cells`. | Not a choice. Set by the call you make to `tl.aggregate`. |
| `consensus_from` | Reserved for the source of a consensus profile. | Part of the core key set, so `validate` does not flag it. | Not written by any function in this release. | Not defined by the schema. | Leave it out. |
| `membership` | Which source rows went into each aggregated row. | Lets a profile be traced back to its cells by stable identity, not row position (spec section 13.4). | Written by `tl.aggregate` when you pass `store_membership=True`. | A mapping with `source_resolution`, `identity_columns`, `target_column` and `members`. | Ask for it when you need the trace; it can be large, so it is off by default. |
| `truth` | The effects injected into a simulated dataset. | Lets tests and tutorials check that a method recovers a known effect. | Written by the synthetic dataset builder. `tl.aggregate` carries it over. | A mapping of the injected effects. | Not set by hand. |
| `embedding` | Where a learned embedding came from. | Records the model behind an embedding matrix. | You, through `mt.io.stamp(provenance=...)`. | A mapping with `model` (required when given), `version` and `pooling`, all strings. | Record it when the features are embeddings. `validate` warns when it lacks a string `model`. |
| `perturbation` | Which `obs` columns hold perturbation identity and control type. | Tools such as `tl.consensus` and `tl.map` read it as their default column, so they need not parse a column name (spec sections 9 and 10). | You, through `mt.io.stamp(perturbation=...)`. An object with a `Metadata_Perturbation` column records the defaults when you give none. | A mapping with `column` and optionally `control_column`; both must be columns of `obs`. | Leave it unset to use `Metadata_Perturbation` and `Metadata_Control_Type`. Set it when your data uses other columns. |

## How to make these choices

### Resolution and grouping

Ask what one row of your table is.
A segmented cell or other primary object is `object`, a physical well is `well`, and anything coarser is `aggregate`.
These are structural tiers, not biology: a gene-level profile is `aggregate` with `grouped_by=["Metadata_Gene"]`, never a resolution called `gene` (spec sections 2.5 and 4.3).

`grouped_by` names the columns whose values define a row.
For `well` it is the plate and well, and for `object` it is empty, and both are filled in by default.
For `aggregate` only you know which columns define a row, so `mt.io.stamp` asks for them and raises a `ValueError` without them.

```python
import mantispy as mt

mt.io.stamp(adata, resolution="aggregate", grouped_by=["Metadata_Gene"], feature_kind="measurement")
```

The four-axes framing that separates these answers from the feature kind, the intervention and the origin is on the {doc}`concepts page <index>`.

### Perturbation

Tools that group by perturbation, such as `tl.consensus` and `tl.map`, read `uns["mantispy"]["perturbation"]` to learn which column to use.
If `obs` has `Metadata_Perturbation`, the defaults are recorded for you.
Record the key yourself only when your perturbation lives in another column, or your control type does:

```python
mt.io.stamp(adata, perturbation={"column": "Metadata_Gene", "control_column": "Metadata_Control_Type"})
```

Both columns must exist in `obs`.

### Embeddings

A matrix of learned embeddings is stamped with `feature_kind="embedding"`.
Add `provenance` to say which model produced it:

```python
mt.io.stamp(
    adata,
    resolution="well",
    feature_kind="embedding",
    provenance={"model": "my-encoder", "version": "1.2", "pooling": "mean"},
)
```

`model` is required when `provenance` is given.

### History

`history` is append-only.
Each mantispy operation that changes data or creates a result appends a record, and running the same operation twice appends two records (spec section 14).
A new object made by aggregation inherits the source object's history and adds its own entry.
Treat it as a record to read, not a field to set.

## Result tables

Beyond the descriptors above, tools write their result tables under `uns["mantispy"]` as well, for example the output of `tl.map` or `tl.empirical_fdr`.
The keys they use form a registered set, and `validate` checks the stamp against it.
A key that is neither a core descriptor nor a registered result only produces a warning, so a custom `key_added` is allowed.
The columns of a result table are defined in section 15 of the {doc}`schema specification <schema_spec>`, and each tool's API page lists the key it writes.
