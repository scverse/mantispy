# Datasets

{mod}`mantispy.ds` serves public screens from the
[Cell Painting Gallery](https://github.com/broadinstitute/cellpainting-gallery) {cite:p}`Weisbart_2024`, each
downloaded from files pinned by sha256.
The overview covers the well-level screens, and five have pages of their own: where each comes from, how every
column mantispy adds was derived, and what the data looks like.

Each rehosted file is built by `python scripts/build_dataset.py <name>`, which loads the dataset from its pinned
upstream files, writes it as h5ad, and prints its sha256 and a fingerprint of its contents.
A rebuild is checked against the fingerprint, since another anndata or h5py release can write the same object as
different bytes.

## By application

| application | dataset | what it is | used in |
|---|---|---|---|
| compound screens | {func}`~mantispy.ds.bbbc021` | MCF-7 cells, 38 compounds with mechanism labels | [overview](../tutorials/overview.ipynb), [artifacts](../tutorials/profiles/artifacts.ipynb), [screen quality](../tutorials/profiles/screen_quality.ipynb), [mechanism of action](../tutorials/compounds/mechanism_of_action.ipynb), [which measurements moved](../tutorials/phenotypes/which_features_moved.ipynb) |
| | {func}`~mantispy.ds.pki` | kinase inhibitors over a dose series, many replicate wells | [published profiles](../tutorials/data/profiles.ipynb), [hits](../tutorials/compounds/hits.ipynb) |
| | {func}`~mantispy.ds.miami` | compounds in U2OS cells | |
| toxicology | {func}`~mantispy.ds.oasis_pilot` | liver toxicity, HepaRG and U2OS over ten concentrations | [concentration response](../tutorials/compounds/dose_response.ipynb) |
| | {func}`~mantispy.ds.agnp` | silver nanoparticles in Huh7 cells | |
| genetic screens | {func}`~mantispy.ds.jump_crispr` | the JUMP CRISPR knockout arm | [CRISPR knockouts](../tutorials/genetics/crispr.ipynb) |
| | {func}`~mantispy.ds.rohban` | ORF overexpression of pathway genes | [which measurements moved](../tutorials/phenotypes/which_features_moved.ipynb) |
| variants | {func}`~mantispy.ds.luad` | lung adenocarcinoma alleles, mutant against wild type | |
| | {func}`~mantispy.ds.pooled_rare` | rare variants in a pooled screen | |
| cell models | {func}`~mantispy.ds.neuropainting` | astrocytes and neurons | |
| | {func}`~mantispy.ds.amish` | a patient cohort at several densities and timepoints | |
| assay development | {func}`~mantispy.ds.chroma` | alternative dyes across eight channels | |
| several laboratories | {func}`~mantispy.ds.jump_target2` | one plate map run at many sites | [reproducing across laboratories](../tutorials/multisite/cross_laboratory.ipynb) |
| learned embeddings | {func}`~mantispy.ds.jump_lite` | the same wells measured by five models and `cp_measure` | [bringing your own embedding](../tutorials/data/embeddings.ipynb), [learned embeddings against CellProfiler](../tutorials/multisite/learned_embeddings.ipynb) |
| single cells and images | {func}`~mantispy.ds.jump_cells` | single cells of one JUMP plate | [quality control](../tutorials/profiles/quality_control.ipynb), [what the well median hides](../tutorials/single_cells/heterogeneity.ipynb) |
| | {func}`~mantispy.ds.jump_plate` | images and segmentations behind one of its wells | [quality control](../tutorials/profiles/quality_control.ipynb) |
| | {func}`~mantispy.ds.jump_export` | a CellProfiler `ExportToSpreadsheet` directory | |
| synthetic | {func}`~mantispy.ds.synthetic_plate` | a plate that records what was injected into it | most pages |
| | {func}`~mantispy.ds.blobs` | a synthetic plate of images, as `SpatialData` | [images and segmentations](../tutorials/data/images.ipynb) |

```{toctree}
:maxdepth: 1

overview
bbbc021
jump_target2
jump_cells
jump_lite
oasis_pilot
```
