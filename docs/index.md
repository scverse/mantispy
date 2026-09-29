# mantispy - Image-based profiling in Python

mantispy brings Cell Painting and other image-based profiling data into the scverse ecosystem, from reading what a pipeline wrote through quality control, normalization and batch correction to evaluating what survived.
Everything operates on an `AnnData`, so scanpy's PCA, neighbors, UMAP and Leiden work on the same object, and {func}`~mantispy.io.read_plate` reads the images behind the profiles as `SpatialData`.

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} {octicon}`desktop-download;1.5em;sd-mr-1` Installation
:link: installation
:link-type: doc

New to *mantispy*? Check out the installation guide.
:::

:::{grid-item-card} {octicon}`rocket;1.5em;sd-mr-1` Quickstart
:link: tutorials/overview
:link-type: doc

The whole workflow on one page, from a public screen to called hits.
:::

:::{grid-item-card} {octicon}`play;1.5em;sd-mr-1` Tutorials
:link: tutorials/general
:link-type: doc

The tutorials walk you through real-world applications of mantispy.
:::

:::{grid-item-card} {octicon}`code-square;1.5em;sd-mr-1` API reference
:link: api
:link-type: doc

The API reference contains a detailed description of the mantispy API.
:::

:::{grid-item-card} {octicon}`comment-discussion;1.5em;sd-mr-1` Discussion
:link: https://discourse.scverse.org/

Need help? Reach out on our forum to get your questions answered.
:::

:::{grid-item-card} {octicon}`mark-github;1.5em;sd-mr-1` GitHub
:link: https://github.com/scverse/mantispy

Found a bug? Interested in improving mantispy? Check out our GitHub for the latest developments.
:::

::::

## Citation

```{eval-rst}
.. include:: about/cite.md
    :start-line: 2
    :parser: myst
```

```{toctree}
:caption: General
:hidden: true
:maxdepth: 2

installation
api
changelog
contributing
references
```

```{toctree}
:caption: Tutorials
:hidden: true
:maxdepth: 2

tutorials/general
tutorials/compound_screens
tutorials/genetic_screens
tutorials/single_cells
```

```{toctree}
:caption: Case studies
:hidden: true
:maxdepth: 1

case_studies/index
```

```{toctree}
:caption: About
:hidden: true
:maxdepth: 2

about/background
about/cite
GitHub <https://github.com/scverse/mantispy>
Discourse <https://discourse.scverse.org/>
```
