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

## Preprocessing

```{eval-rst}
.. module:: mantispy.pp
.. currentmodule:: mantispy

.. autosummary::
    :toctree: generated

    pp.annotate_controls
    pp.annotate_jump
    pp.find_perturbation_key
    pp.calculate_qc_metrics
    pp.filter_cells
    pp.filter_features
    pp.normalize
    pp.feature_select
    pp.subset_features
    pp.outliers
    pp.image_qc
    pp.filter_images
    pp.well_qc
    pp.standardize_feature_names
    pp.sphere
    pp.correct_plate_position
    pp.regress_out
    pp.harmony
    pp.rank_int
    pp.downsample
    pp.feature_select_chatterjee
    pp.feature_reproducibility
    pp.feature_batch_sensitivity
```

| Function | Stores |
| --- | --- |
| `pp.annotate_controls` | `obs["Metadata_Control"]`, and `obs["Metadata_Control_Type"]` when `poscon` is given |
| `pp.annotate_jump` | `obs`: `Metadata_JCP2022`, `Metadata_Perturbation`, `Metadata_InChIKey`, `Metadata_Control` |
| `pp.calculate_qc_metrics` | `obs`: `qc_n_nan_features`, `qc_nan_fraction`, `qc_is_border`, `qc_area_outlier`, `qc_pass`; `var`: `qc_n_nan`, `qc_variance`, `qc_n_unique` |
| `pp.filter_cells` | subsets `obs` in place |
| `pp.filter_features` | subsets `var` in place |
| `pp.normalize` | `X`, or `layers[key_added]`; `layers["raw"]` when `keep_raw=True` |
| `pp.feature_select` | `var[key_added]`, `uns["mantispy"]["feature_select"]` |
| `pp.outliers` | `obs[key_added]`, `obs[key_added + "_score"]` |
| `pp.image_qc` | `uns["mantispy"]["image_qc"]`, `obs["qc_image_pass"]` |
| `pp.well_qc` | `uns["mantispy"]["well_qc"]`, `obs["qc_well_pass"]` |
| `pp.standardize_feature_names` | `var_names`, `var["original_name"]` |
| `pp.sphere` | `X`, or `layers[key_added]` |
| `pp.correct_plate_position` | `X` or `layers[key_added]`, `uns["mantispy"]["plate_position"]` |
| `pp.regress_out` | `X`, or `layers[key_added]` |
| `pp.harmony` | `obsm[key_added]`; needs `mantispy[harmony]` |
| `pp.rank_int` | `X`, or `layers[key_added]` |
| `pp.downsample` | returns a new object holding the sampled rows |
| `pp.feature_select_chatterjee` | `var[key_added]`, `var["chatterjee_xi"]` |
| `pp.feature_reproducibility` | `var[key_added]`, `var[key_added + "_selected"]` |
| `pp.feature_batch_sensitivity` | `var[key_added + "_pvalue"]`, `_qvalue`, `_sensitive` |

`pp.normalize` also writes `var["degenerate_scale"]`, flagging features with no spread in
some group. Those are divided by `epsilon` rather than by zero and come back at ~1e17;
drop them before computing anything from distances.

## Tools

```{eval-rst}
.. module:: mantispy.tl
.. currentmodule:: mantispy

.. autosummary::
    :toctree: generated

    tl.aggregate
    tl.map
    tl.similarity
    tl.percent_replicating
    tl.grit
    tl.consensus
    tl.effect_size
    tl.wasserstein_features
    tl.differential_features
    tl.feature_signature
    tl.hit_calling
    tl.edistance
    tl.transport
    tl.dose_response
    tl.nn_moa_classify
    tl.moa_enrichment
    tl.feature_sets
    tl.enrich
    tl.rank_features
    tl.rank_sets
    tl.cluster_composition
    tl.subpopulation_hits
    tl.cell_cycle_phase
    tl.neighbors_local_density
    tl.replicate_saturation
    tl.cytotoxicity
    tl.gene_sets
    tl.pathway_coherence
    tl.enrich_hits
```

| Function | Stores |
| --- | --- |
| `tl.aggregate` | returns a new object: `obs` gains `Metadata_CellCount`; `uns["mantispy"]` gains `aggregated_from` and a `resolution` of `"well"` or `"perturbation"` |
| `tl.map` | `uns["mantispy"][key_added]`, `obs[key_added]`, `obs[key_added + "_qvalue"]` |
| `tl.similarity` | `obsp[key_added]` |
| `tl.percent_replicating` | `uns["mantispy"][key_added]` and `..._summary` |
| `tl.grit` | `obs[key_added]`, `uns["mantispy"][key_added]` |
| `tl.consensus` | returns a new object at `"perturbation"` resolution; `obs["Metadata_ReplicateCount"]`, `uns["mantispy"]["consensus_weights"]` |
| `tl.effect_size` | `varm[key_added]`, `uns["mantispy"][key_added]` and `..._groups` |
| `tl.wasserstein_features` | `varm[key_added]`, `uns["mantispy"][key_added]` and `..._groups` |
| `tl.hit_calling` | `uns["mantispy"][key_added]`, `obs[key_added + "_distance"]`, `obs[key_added + "_qvalue"]` |
| `tl.edistance` | `uns["mantispy"][key_added]`, or `..._pairwise` when `reference=None` |
| `tl.transport` | `uns["mantispy"][key_added]` and `..._units`, `obs[key_added + "_agreement"]` |
| `tl.dose_response` | `uns["mantispy"][key_added]` |
| `tl.nn_moa_classify` | `obs[key_added + "_predicted"]`, `uns["mantispy"][key_added]` and `..._confusion` |
| `tl.moa_enrichment` | `uns["mantispy"][key_added]` |
| `tl.feature_sets` | returns a decoupler network; stores nothing |
| `tl.enrich` | `obsm["score_<method>"]`, `obsm["padj_<method>"]` (written by decoupler) |
| `tl.rank_features` | `uns["mantispy"][key_added]` |
| `tl.rank_sets` | `uns["mantispy"][key_added]` |
| `tl.cluster_composition` | returns a new object: wells by clusters; `uns["mantispy"]["composition_test"]` |
| `tl.subpopulation_hits` | `uns["mantispy"][key_added]` |
| `tl.cell_cycle_phase` | `obs[key_added]` |
| `tl.neighbors_local_density` | `obs[key_added]` |
| `tl.replicate_saturation` | `uns["mantispy"][key_added]` |
| `tl.cytotoxicity` | `uns["mantispy"][key_added]`, `obs[key_added + "_suspect"]` |
| `tl.gene_sets` | returns a gene-set network; stores nothing |
| `tl.pathway_coherence` | `uns["mantispy"][key_added]`, sorted by coherence |
| `tl.enrich_hits` | `uns["mantispy"][key_added]` |

`tl.map` has four modes, named after the questions the field asks. `"activity"` matches the copairs
reference implementation (0.9267 against 0.9267 over 301 JUMP compounds):

| mode | question | needs |
| --- | --- | --- |
| `"activity"` | is this perturbation distinguishable from the negative controls? | controls |
| `"consistency"` | do perturbations sharing an annotation look alike, against those that do not? | `annotation_key` |
| `"replicability"` | do a perturbation's replicates retrieve each other against everything else? | — |
| `"cross_plate"` | the same, counting only replicates from a different plate | `Metadata_Plate` |

## Metrics

```{eval-rst}
.. module:: mantispy.metrics
.. currentmodule:: mantispy

.. autosummary::
    :toctree: generated

    metrics.silhouette_label
    metrics.silhouette_batch
    metrics.lisi
    metrics.pc_regression
    metrics.batch_variance_explained
    metrics.evaluate_correction
    metrics.diagnose_testing
```

Metrics do not modify the object. Each returns a tidy frame with `metric`, `representation`,
`key` and `value`, so results from several calls stack. `evaluate_correction` adds a `better`
column giving the direction of improvement for each metric, because batch mixing and
biological separation trade off against each other and neither is meaningful alone.

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

## Not reimplemented here

Dimensionality reduction, neighborhood graphs, embeddings and clustering come from scanpy.
Call them directly on the same object:

```python
import scanpy as sc

sc.pp.pca(wells, n_comps=50)
sc.pp.neighbors(wells)
sc.tl.umap(wells)
```

Harmony is wrapped as `pp.harmony`. It is the last step of the JUMP consortium's recipe and
the best performer in Arevalo et al. (2024). It needs the optional extra:

```bash
pip install 'mantispy[harmony]'
```

It corrects an embedding rather than the features, so run `sc.pp.pca` first. The other
corrections are `pp.sphere`, `pp.correct_plate_position` and `pp.regress_out`.

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

## Working backed

`mt.io.read(path, backed="r")` leaves `X` on disk. `obs` and `var` stay in memory, so
metadata, QC flags and feature selection work unchanged. Functions that rewrite `X` need
`copy=True` and raise a `ValueError` saying so when it is missing.

| operation | backed |
| --- | --- |
| `pp.calculate_qc_metrics`, `pp.outliers`, `pp.feature_select`, `pp.well_qc` | yes (they write `obs`/`var`) |
| `tl.aggregate`, `tl.consensus`, `tl.map`, `tl.hit_calling`, the metrics | yes (they return new objects or tables) |
| `pp.normalize`, `pp.sphere`, `pp.correct_plate_position`, `pp.regress_out` | with `copy=True`; the result is materialized |
| writing `X` in place | no; the file is opened read-only |

Grouped reductions read one group at a time when the matrix is on disk, and the tests assert
that the results are identical to the in-memory path. For a per-plate median on 100 000 cells
x 500 features, backed mode peaks at 44 MB against a 200 MB resident matrix in memory, and
takes 3.5 s against 0.8 s. That is about four times the runtime for four times less memory,
so use it only on data that does not fit in memory.

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
