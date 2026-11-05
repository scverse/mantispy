# Tools

```{eval-rst}
.. module:: mantispy.tl
.. currentmodule:: mantispy
```

Aggregate profiles and ask what the perturbations did.

## Aggregation

```{eval-rst}
.. autosummary::
    :toctree: generated

    tl.aggregate
    tl.consensus
```

## Replicates and reproducibility

```{eval-rst}
.. autosummary::
    :toctree: generated

    tl.map
    tl.similarity
    tl.percent_replicating
    tl.grit
    tl.transport
    tl.replicate_saturation
```

## Hits and effect sizes

```{eval-rst}
.. autosummary::
    :toctree: generated

    tl.hit_calling
    tl.empirical_fdr
    tl.guide_activity
    tl.aggregate_guides
    tl.edistance
    tl.effect_size
    tl.wasserstein_features
    tl.differential_features
    tl.feature_signature
    tl.cytotoxicity
```

## Dose response

```{eval-rst}
.. autosummary::
    :toctree: generated

    tl.dose_response
    tl.dose_features
    tl.dose_direction
    tl.dose_trajectory
```

## Mechanism of action

```{eval-rst}
.. autosummary::
    :toctree: generated

    tl.nn_moa_classify
    tl.moa_enrichment
```

## Feature sets

```{eval-rst}
.. autosummary::
    :toctree: generated

    tl.feature_sets
    tl.enrich
    tl.rank_features
    tl.rank_sets
```

## Clusters and gene sets

```{eval-rst}
.. autosummary::
    :toctree: generated

    tl.cluster
    tl.gene_set_network
    tl.ora
    tl.enrich_hits
    tl.pathway_coherence
    tl.network_enrichment
```

## Single cells

```{eval-rst}
.. autosummary::
    :toctree: generated

    tl.cluster_composition
    tl.subpopulation_hits
    tl.cell_cycle_phase
    tl.neighbors_local_density
```

## Circular covariates

A cell-cycle clock is an angle, so its statistics are circular. Every function takes `period=`, the length of one turn in the units the angle is stored in.

```{eval-rst}
.. autosummary::
    :toctree: generated

    tl.circular_correlation
    tl.circular_neighbor_error
    tl.circular_shift
    tl.orient_cycle
    tl.feature_cycle_association
```
