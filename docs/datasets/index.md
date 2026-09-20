# Datasets

{mod}`mantispy.ds` serves fifteen datasets from the
[Cell Painting Gallery](https://github.com/broadinstitute/cellpainting-gallery) {cite:p}`Weisbart_2024`, each
downloaded from files pinned by sha256.
The overview covers all fifteen, and four have pages of their own: where each comes from, how every column
mantispy adds was derived, and what the data looks like.

Each rehosted file is built by `python scripts/build_dataset.py <name>`, which loads the dataset from its pinned
upstream files, writes it as h5ad, and prints its sha256 and a fingerprint of its contents.
A rebuild is checked against the fingerprint, since another anndata or h5py release can write the same object as
different bytes.

```{toctree}
:maxdepth: 1

overview
bbbc021
jump_target2
jump_cells
oasis_pilot
```
