# Datasets

{mod}`mantispy.ds` serves fifteen datasets from the
[Cell Painting Gallery](https://github.com/broadinstitute/cellpainting-gallery) {cite:p}`Weisbart_2024`, each
downloaded from files pinned by sha256.
These pages say where each one comes from, how every column mantispy adds was derived, what the data looks like, and
how to rebuild the file that is rehosted for it.

Each file is built by `scripts/build_dataset.py`, which loads a dataset from its pinned upstream files, writes it as
h5ad, and prints its sha256 and a fingerprint of its contents.

The pages are executed when the data changes rather than on every build, since together they load about 3 GB.

```{toctree}
:maxdepth: 1

overview
bbbc021
jump_target2
jump_cells
```
