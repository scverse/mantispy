# API

Import as:

```python
import mantispy as mt
```

Every function that modifies an object takes `copy`.
With `copy=False` (the default) it mutates in place and returns `None`; with `copy=True` it returns a modified copy and leaves the input alone.
Each call records its arguments under `uns["mantispy"]["params"][<function name>]`, so a finished object says how it was made.

The Stores column on each module's page lists the keys each function writes.

```{toctree}
:maxdepth: 1

api/io
api/preprocessing
api/tools
api/metrics
api/get
api/plotting
api/datasets
api/settings
```

## Relative to scmorph

`scmorph` is the other AnnData-based morphological profiling package.
mantispy covers its functionality with two exceptions:

| scmorph | here |
| --- | --- |
| reading CellProfiler CSV output | `io.read_profiles`, on an `ExportToSpreadsheet` directory or on published tables |
| reading a CellProfiler SQLite database | not yet (planned) |
| quality control, batch correction, aggregation | `pp` and `tl.aggregate` |
| feature selection | `pp.feature_select` (pycytominer-equivalent), `pp.feature_select_chatterjee`, `pp.feature_reproducibility` |
| trajectory inference (`slingshot`, differential progression) | out of scope: a trajectory through morphology space needs an ordering the assay rarely justifies, and scanpy's `sc.tl.paga` and `sc.tl.dpt` already work on these objects |

Hit calling, effect sizes, dose response, mechanism retrieval, feature-set and pathway enrichment, cell-state composition, replicate power and cytotoxicity have no counterpart in scmorph.

## Not reimplemented here

Dimensionality reduction, neighborhood graphs, embeddings and clustering come from scanpy.
Call them directly on the same object:

```python
import scanpy as sc

sc.pp.pca(wells, n_comps=50)
sc.pp.neighbors(wells)
sc.tl.umap(wells)
```

Harmony is wrapped as `pp.harmony`.
It is the last step of the [JUMP profiling recipe](https://github.com/broadinstitute/jump-profiling-recipe) for compound and ORF profiles.
It ranked in the top three in every scenario of {cite:t}`Arevalo_2024`, as did Seurat RPCA.
It needs the optional extra:

```bash
pip install 'mantispy[harmony]'
```

It corrects an embedding rather than the features, so run `sc.pp.pca` first.
The other corrections are `pp.sphere`, `pp.correct_plate_position` and `pp.regress_out`.

## The data contract

`X` is `float32`, one row per profile.
`obs` carries `Metadata_` columns identifying where each profile came from; `var` carries the parsed feature annotation (`object`, `feature_group`, `feature`, `channel`, `scale`, `angle`, `gray_levels`, `radial_bin`, `params`, `is_feature`).
`uns["mantispy"]` holds the schema version, the resolution, the channel vocabulary and provenance.

Resolution (`"cell"`, `"well"`, `"perturbation"`) is advisory.
It sets which identifier columns `io.validate` requires, and functions warn instead of raising when they get a resolution they do not expect.

The machine-readable contract is published as `spec/schema-1.0.json`, alongside the earlier `spec/schema-0.1.json`.

## Performance

`benchmarks/` holds a suite that is not part of the test run; run it with `pytest benchmarks -s`.
Measured on a laptop, 100 000 cells by 500 features (`X` is 200 MB):

| operation | time | peak |
|---|---|---|
| `np.median` over the whole matrix (the floor) | 0.86 s | 200 MB |
| `pp.normalize`, per plate | 2.06 s | 232 MB |
| `pp.feature_select` | 0.41 s | 458 MB |
| `tl.aggregate` to 7680 wells | 0.39 s | 78 MB |
| `pp.sphere` on those wells | 0.48 s | 227 MB |
| `tl.hit_calling`, 1000 permutations | 0.21 s | 7 MB |
| `tl.map` (copairs) | 65 s | 2465 MB |

At 500 000 cells the preprocessing scales linearly: normalize 12.7 s, aggregate 2.0 s.
`tl.map` is by far the most expensive step, in both time and memory.

## Stability

The schema is frozen at 1.0.
The names in `spec/schema-1.0.json` (the required and reserved `obs` columns, the `var` annotation, the `uns["mantispy"]` keys and the result tables) are stable, as are the public signatures listed on the module pages.

- Anything removed gets a `DeprecationWarning` for two minor releases first.
- `mt.io.read` migrates a file written against an older schema on read.
  A schema version this build does not know raises an error.
- `_core` and other `_`-prefixed modules are private.

`io.read_plate` and `ds.blobs` bring images and segmentations in as SpatialData on top of this contract, without changing it.
