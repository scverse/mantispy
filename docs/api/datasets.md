# Datasets

{mod}`mantispy.ds` serves public screens, most from the [Cell Painting Gallery](https://github.com/broadinstitute/cellpainting-gallery) {cite:p}`Weisbart_2024` and the optical pooled screens from their authors' own hosts ({func}`~mantispy.ds.scallops_arv471` from the `Genentech/scallops-manuscript` GitHub repository, {func}`~mantispy.ds.cp_posh` from insitro's S3 bucket), each downloaded from files pinned by sha256.
The [overview](../datasets/overview.ipynb) covers the well-level screens, and five have pages of their own: where each comes from, how every column mantispy adds was derived, and what the data looks like.

```{eval-rst}
.. module:: mantispy.ds
.. currentmodule:: mantispy
```

## Screens

```{eval-rst}
.. autosummary::
    :toctree: generated

    ds.bbbc021
    ds.rohban
    ds.pki
    ds.jump_target2
    ds.jump_crispr
    ds.jump_lite
    ds.agnp
    ds.amish
    ds.chroma
    ds.luad
    ds.miami
    ds.neuropainting
    ds.oasis_pilot
    ds.pooled_rare
```

## Single cells and images

```{eval-rst}
.. autosummary::
    :toctree: generated

    ds.jump_cells
    ds.jump_export
    ds.jump_plate
    ds.scallops_arv471
    ds.cp_posh
```

`jump_cells`, `jump_export` and `jump_plate` are three layers of one plate, `BR00121438`, whose well-level profiles `jump_target2` already reads, so a profile aggregated from the cells can be compared with the published one.
`jump_cells` marks the features feature selection keeps in `var["selected"]`, as scanpy marks `highly_variable`, and `jump_cells(selected=True)` hands back only those, 1605 of 5839.
The reduced object is kept beside the whole one, so a notebook that wants it reads 87 MB rather than 308 MB.
No step of the selection draws a random number, so the same pinned files always give the same 1605 features.
`jump_cells` returns single cells, `jump_export` the directory one field of view was measured in, as CellProfiler wrote it, and `jump_plate` the images and segmentations of one well as `SpatialData`.
They share one download, so asking for more than one costs little beyond the first.

The wells in `jump_cells` were chosen by cell count, not by distance.
Ranking this plate's wells by distance from the controls selects almost entirely for cytotoxicity: wells with fewer than 20 cells sit at a median distance of 2213, and wells with at least 120 cells at 25.7.
Every well here holds more than 120 cells in its first field.
Four of the twelve compounds keep both of their replicate wells, so a per-well measurement can be checked against a replicate; the other eight are the strongest movers among the wells that survived.

## Annotations

```{eval-rst}
.. autosummary::
    :toctree: generated

    ds.corum
    ds.gene_sets
    ds.interactions
    ds.jump_lite_targets
```

## Synthetic

```{eval-rst}
.. autosummary::
    :toctree: generated

    ds.synthetic_plate
    ds.blobs
```

`synthetic_plate` and `blobs` are generated locally.
`synthetic_plate` is a single-cell profile table with injected artifacts for quality control to find; `blobs` is a small `SpatialData` plate of images, labels and tables.

## Downloads

The other datasets download once, checked against a pinned sha256, into `mt.settings.cache_dir` (set `MANTISPY_CACHE_DIR` to change it).
Nine of them carry the annotations the analysis functions need:

| dataset | download | perturbations | carries |
|---|---|---|---|
| `bbbc021` | ~10 MB | 39 compounds | MOA labels, the classic retrieval benchmark |
| `rohban` | ~27 MB | 194 overexpressed genes | cell counts, ~10 replicates per gene |
| `pki` | ~71 MB | 15 kinase inhibitors x 7 doses | cell counts, MOA labels, 32-64 replicates |
| `jump_target2` | ~0.7 GB | 302 compounds, one shared plate map | the same plate run at eleven sites, so any difference between them is technical |
| `jump_crispr` | ~180 MB | about 8,000 knocked-out genes | gene symbols, controls and chromosome arms; `corum` gives the protein complexes the genes form |
| `jump_cells` | ~1.5 GB | 12 compounds and DMSO | single cells, 24 wells x 4 fields of view of `BR00121438` |
| `jump_lite` | ~10 MB per feature set | 302 compounds, four laboratories | the same 1,536 wells under five learned embeddings and `cp_measure`, so the feature set is the only thing that changes; `jump_lite_targets` gives the gene each compound acts on |
| `scallops_arv471` | ~205 MB | 680 guides across 159 gene groups | single cells of an optical pooled CRISPR screen under the ER degrader ARV-471; gene symbols, guides and the non-targeting controls, with `CRBN`, `DDB1`, `CUL4A`, `CUL4B` and `ESR1` the known-mechanism rescuers |
| `cp_posh` | ~1.6 GB | 1,622 guides across 124 genes | single cells of a broad-morphology Cell Painting pooled CRISPR screen in A549; about 1,278 well-normalized CellStats features, gene symbols, guides and the non-targeting and intergenic controls, with `KIF18A`, the proteasome, the mitochondrial ribosome, ARP2/3 and COPI the known-mechanism genes |

The others are further gallery accessions, normalized and feature-selected by their authors and read with the `io.read_profiles` defaults.
Use them to run a method across a range of screens.

Check anything tuned on one dataset against the others.
Cutoffs that looked universal on BBBC021 turned out to be dataset-dependent on `rohban` and `pki`.

Each rehosted file is built by `python scripts/build_dataset.py <name>`, which loads the dataset from its pinned upstream files, writes it as h5ad, and prints its sha256 and a fingerprint of its contents.
A rebuild is checked against the fingerprint, since another anndata or h5py release can write the same object as different bytes.

## By application

| application | dataset | what it is | used in |
|---|---|---|---|
| compound screens | {func}`~mantispy.ds.bbbc021` | MCF-7 cells, 38 compounds with mechanism labels | [overview](../tutorials/overview.ipynb), [artifacts](../tutorials/profiles/artifacts.ipynb), [screen quality](../tutorials/profiles/screen_quality.ipynb), [mechanism of action](../tutorials/compounds/mechanism_of_action.ipynb), [which measurements moved](../tutorials/phenotypes/which_features_moved.ipynb) |
| | {func}`~mantispy.ds.pki` | kinase inhibitors over a dose series, many replicate wells | [published profiles](../tutorials/data/profiles.ipynb), [normalize and select](../tutorials/profiles/normalize_and_select.ipynb), [plate artifacts](../tutorials/profiles/artifacts.ipynb), [hits](../tutorials/compounds/hits.ipynb) |
| | {func}`~mantispy.ds.miami` | compounds in U2OS cells | |
| toxicology | {func}`~mantispy.ds.oasis_pilot` | liver toxicity, HepaRG and U2OS over ten concentrations | [concentration response](../tutorials/compounds/dose_response.ipynb) |
| | {func}`~mantispy.ds.agnp` | silver nanoparticles in Huh7 cells | |
| genetic screens | {func}`~mantispy.ds.jump_crispr` | the JUMP CRISPR knockout arm | [CRISPR knockouts](../tutorials/genetics/crispr.ipynb) |
| | {func}`~mantispy.ds.rohban` | ORF overexpression of pathway genes | [which measurements moved](../tutorials/phenotypes/which_features_moved.ipynb) |
| optical pooled screens | {func}`~mantispy.ds.scallops_arv471` | single cells of a pooled CRISPR screen under the ER degrader ARV-471 | |
| | {func}`~mantispy.ds.cp_posh` | single cells of a broad-morphology Cell Painting pooled CRISPR screen | |
| variants | {func}`~mantispy.ds.luad` | lung adenocarcinoma alleles, mutant against wild type | |
| | {func}`~mantispy.ds.pooled_rare` | rare variants in a pooled screen | |
| cell models | {func}`~mantispy.ds.neuropainting` | astrocytes and neurons | |
| | {func}`~mantispy.ds.amish` | a patient cohort at several densities and timepoints | |
| assay development | {func}`~mantispy.ds.chroma` | alternative dyes across eight channels | |
| several laboratories | {func}`~mantispy.ds.jump_target2` | one plate map run at many sites | [reproducing across laboratories](../tutorials/multisite/cross_laboratory.ipynb) |
| learned embeddings | {func}`~mantispy.ds.jump_lite` | the same wells measured by five models and `cp_measure` | [bringing your own embedding](../tutorials/data/embeddings.ipynb), [learned embeddings against CellProfiler](../tutorials/multisite/learned_embeddings.ipynb) |
| single cells and images | {func}`~mantispy.ds.jump_cells` | single cells of one JUMP plate | [quality control](../tutorials/profiles/quality_control.ipynb), [normalize and select](../tutorials/profiles/normalize_and_select.ipynb), [what the well median hides](../tutorials/single_cells/heterogeneity.ipynb), [which measurements moved](../tutorials/phenotypes/which_features_moved.ipynb) |
| | {func}`~mantispy.ds.jump_plate` | images and segmentations behind one of its wells | [images and segmentations](../tutorials/data/images.ipynb), [quality control](../tutorials/profiles/quality_control.ipynb) |
| | {func}`~mantispy.ds.jump_export` | a CellProfiler `ExportToSpreadsheet` directory | [from a CellProfiler run](../tutorials/data/cellprofiler.ipynb) |
| synthetic, for tests | {func}`~mantispy.ds.synthetic_plate` | a plate that records what was injected into it | |
| | {func}`~mantispy.ds.blobs` | a synthetic plate of images, as `SpatialData` | |

## Dataset pages

```{toctree}
:maxdepth: 1

../datasets/overview
../datasets/bbbc021
../datasets/jump_target2
../datasets/jump_cells
../datasets/jump_lite
../datasets/oasis_pilot
```
