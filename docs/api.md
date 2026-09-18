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
Writing to a layer suffixes the column, so `key_added="sphered"` flags `var["degenerate_scale_sphered"]` and one call cannot overwrite what another measured.

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
| `tl.aggregate` | returns a new object: `obs` gains `Metadata_CellCount` (`count_key`), and `Metadata_SiteCount` (`site_key`) when the fields of view are known; `uns["mantispy"]` gains `aggregated_from` and a `resolution` of `"well"` or `"perturbation"` |
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

## Plotting

```{eval-rst}
.. module:: mantispy.pl
.. currentmodule:: mantispy

.. autosummary::
    :toctree: generated

    pl.plate
    pl.cell_counts
    pl.feature_distributions
    pl.nan_matrix
    pl.qc
    pl.plate_effects
    pl.image_qc
    pl.control_drift
    pl.outliers
    pl.feature_correlation
    pl.feature_groups
    pl.feature_signature
    pl.map
    pl.replicate_correlation
    pl.hits
    pl.effect_sizes
    pl.feature_volcano
    pl.dose_response
    pl.moa_confusion
    pl.moa_enrichment
    pl.distance_heatmap
    pl.setting_agreement
    pl.transport
    pl.sets_heatmap
    pl.cluster_composition
    pl.cell_cycle
    pl.density
    pl.subpopulation_hits
    pl.replicate_saturation
    pl.cytotoxicity
    pl.pathway_coherence
    pl.batch_variance
    pl.metrics
    pl.similarity
```

There is no dedicated function for a plate map of a per-well flag, which is
`mt.pl.plate(adata, color="qc_well_pass")`, or for embeddings side by side, which is a loop
over `sc.pl.embedding`.

Plotting functions return Matplotlib axes and do not modify the object.

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
    ds.jump_cells
    ds.jump_export
    ds.jump_plate
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

`jump_cells`, `jump_export` and `jump_plate` are three layers of one plate, `BR00121438`, whose well-level profiles
`jump_target2` already reads, so a profile aggregated from the cells can be compared with the published one.
`jump_cells` marks the features feature selection keeps in `var["selected"]`, as scanpy marks `highly_variable`, and
`jump_cells(selected=True)` hands back only those, 1607 of 5857. The reduced object is kept beside the whole one, so
a notebook that wants it reads 87 MB rather than 308 MB. No step of the selection draws a random number, so the same
pinned files always give the same 1607 features.
`jump_cells` returns single cells, `jump_export` the directory one field of view was measured in, as CellProfiler
wrote it, and `jump_plate` the images and segmentations of one well as `SpatialData`. They share one download, so
asking for more than one costs little beyond the first.

The wells in `jump_cells` were chosen by cell count, not by distance. Ranking this plate's wells by distance from
the controls selects almost entirely for cytotoxicity: wells with fewer than 20 cells sit at a median distance of
2213, and wells with at least 120 cells at 25.7. Every well here holds more than 120 cells in its first field. Four
of the twelve compounds keep both of their replicate wells, so a per-well measurement can be checked against a
replicate; the other eight are the strongest movers among the wells that survived.

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
| `jump_cells` | ~1.5 GB | 12 compounds and DMSO | single cells, 24 wells x 4 fields of view of `BR00121438` |

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

## Relative to scmorph

`scmorph` is the other AnnData-based morphological profiling package. mantispy covers its
functionality with two exceptions:

| scmorph | here |
| --- | --- |
| reading CellProfiler CSV output | `io.read_profiles`, on an `ExportToSpreadsheet` directory or on published tables |
| reading a CellProfiler SQLite database | not yet (planned) |
| quality control, batch correction, aggregation | `pp` and `tl.aggregate` |
| feature selection | `pp.feature_select` (pycytominer-equivalent), `pp.feature_select_chatterjee`, `pp.feature_reproducibility` |
| trajectory inference (`slingshot`, differential progression) | out of scope: a trajectory through morphology space needs an ordering the assay rarely justifies, and scanpy's `sc.tl.paga` and `sc.tl.dpt` already work on these objects |

Hit calling, effect sizes, dose response, mechanism retrieval, feature-set and pathway
enrichment, cell-state composition, replicate power and cytotoxicity have no counterpart in
scmorph.

## Not reimplemented here

Dimensionality reduction, neighborhood graphs, embeddings and clustering come from scanpy.
Call them directly on the same object:

```python
import scanpy as sc

sc.pp.pca(wells, n_comps=50)
sc.pp.neighbors(wells)
sc.tl.umap(wells)
```

Harmony is wrapped as `pp.harmony`. It is the last step of the JUMP consortium's recipe
{cite:p}`Chandrasekaran_2024` and the best performer in {cite:t}`Arevalo_2024`. It needs the
optional extra:

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

The schema is frozen at 1.0. The names in `spec/schema-1.0.json` (the required and
reserved `obs` columns, the `var` annotation, the `uns["mantispy"]` keys and the result
tables) are stable, as are the public signatures in the tables above.

- Anything removed gets a `DeprecationWarning` for two minor releases first.
- `mt.io.read` migrates a file written against an older schema on read. A schema version
  this build does not know raises an error.
- `_core` and other `_`-prefixed modules are private.

`io.read_plate` and `ds.blobs` bring images and segmentations in as SpatialData on top of this contract, without
changing it.
