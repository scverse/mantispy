# Tools

```{eval-rst}
.. module:: mantispy.tl
.. currentmodule:: mantispy
```

## Aggregation

```{eval-rst}
.. autosummary::
    :toctree: generated

    tl.aggregate
    tl.consensus
```

| Function | Stores |
| --- | --- |
| `tl.aggregate` | returns a new object: `obs` gains `Metadata_CellCount` (`count_key`), and `Metadata_SiteCount` (`site_key`) when the fields of view are known; `uns["mantispy"]` gains `aggregated_from` and a `resolution` of `"well"` or `"perturbation"` |
| `tl.consensus` | returns a new object at `"perturbation"` resolution; `obs["Metadata_ReplicateCount"]`, `uns["mantispy"]["consensus_weights"]` |

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

| Function | Stores |
| --- | --- |
| `tl.map` | `uns["mantispy"][key_added]`, `obs[key_added]`, `obs[key_added + "_qvalue"]` |
| `tl.similarity` | `obsp[key_added]` |
| `tl.percent_replicating` | `uns["mantispy"][key_added]` and `..._summary` |
| `tl.grit` | `obs[key_added]`, `uns["mantispy"][key_added]` |
| `tl.transport` | `uns["mantispy"][key_added]` and `..._units`, `obs[key_added + "_agreement"]` |
| `tl.replicate_saturation` | `uns["mantispy"][key_added]` |

`tl.map` has four modes, named after the questions the field asks.
`"activity"` matches the copairs reference implementation (0.9267 against 0.9267 over 301 JUMP compounds):

| mode | question | needs |
| --- | --- | --- |
| `"activity"` | is this perturbation distinguishable from the negative controls? | controls |
| `"consistency"` | do perturbations sharing an annotation look alike, against those that do not? | `annotation_key` |
| `"replicability"` | do a perturbation's replicates retrieve each other against the other perturbations on their plate? (mAP-nonrep) | `Metadata_Plate`, controls to leave out |
| `"cross_plate"` | do a perturbation's replicates on other plates retrieve each other against everything else? | `Metadata_Plate` |

## Hits and effect sizes

```{eval-rst}
.. autosummary::
    :toctree: generated

    tl.hit_calling
    tl.edistance
    tl.effect_size
    tl.wasserstein_features
    tl.differential_features
    tl.feature_signature
    tl.cytotoxicity
```

| Function | Stores |
| --- | --- |
| `tl.hit_calling` | `uns["mantispy"][key_added]`, `obs[key_added + "_distance"]`, `obs[key_added + "_row_distance"]`, `obs[key_added + "_qvalue"]`, `obs[key_added + "_reference_held_out"]` |
| `tl.edistance` | `uns["mantispy"][key_added]`, or `..._pairwise` when `reference=None` |
| `tl.effect_size` | `varm[key_added]`, `uns["mantispy"][key_added]` and `..._groups` |
| `tl.wasserstein_features` | `varm[key_added]`, `uns["mantispy"][key_added]` and `..._groups` |
| `tl.cytotoxicity` | `uns["mantispy"][key_added]`, `obs[key_added + "_suspect"]` |

## Dose response

```{eval-rst}
.. autosummary::
    :toctree: generated

    tl.dose_response
    tl.dose_features
    tl.dose_direction
    tl.dose_trajectory
```

| Function | Stores |
| --- | --- |
| `tl.dose_response` | `uns["mantispy"][key_added]` |
| `tl.dose_features` | `uns["mantispy"][key_added]`, one row per compound and feature |
| `tl.dose_direction` | `uns["mantispy"][key_added]`, one row per compound and concentration, `obs[key_added + "_phase"]` |
| `tl.dose_trajectory` | returns a new object: compounds by features-and-positions at `"perturbation"` resolution |

## Mechanism of action

```{eval-rst}
.. autosummary::
    :toctree: generated

    tl.nn_moa_classify
    tl.moa_enrichment
```

| Function | Stores |
| --- | --- |
| `tl.nn_moa_classify` | `obs[key_added + "_predicted"]`, `uns["mantispy"][key_added]` and `..._confusion` |
| `tl.moa_enrichment` | `uns["mantispy"][key_added]` |

## Feature sets

```{eval-rst}
.. autosummary::
    :toctree: generated

    tl.feature_sets
    tl.enrich
    tl.rank_features
    tl.rank_sets
```

| Function | Stores |
| --- | --- |
| `tl.feature_sets` | returns a decoupler network; stores nothing |
| `tl.enrich` | `obsm["score_<method>"]`, `obsm["padj_<method>"]` (written by decoupler) |
| `tl.rank_features` | `uns["mantispy"][key_added]` |
| `tl.rank_sets` | `uns["mantispy"][key_added]` |

## Clusters and gene sets

```{eval-rst}
.. autosummary::
    :toctree: generated

    tl.cluster
    tl.gene_sets
    tl.ora
    tl.enrich_hits
    tl.pathway_coherence
    tl.network_enrichment
```

| Function | Stores |
| --- | --- |
| `tl.cluster` | `obs[key_added]` (categorical labels); for `method="hierarchical"` also `uns["mantispy"][key_added + "_linkage"]` (the tree) and `uns["mantispy"][key_added]` (chosen cut and leaf labels) |
| `tl.gene_sets` | returns a gene-set network; stores nothing |
| `tl.ora` | `uns["mantispy"][key_added]`, one row per group and set with `n`, `odds_ratio`, `pvalue`, `qvalue` |
| `tl.enrich_hits` | `uns["mantispy"][key_added]` |
| `tl.pathway_coherence` | `uns["mantispy"][key_added]`, sorted by coherence |
| `tl.network_enrichment` | `uns["mantispy"][key_added]`: the 2x2 `table`, `odds_ratio`, `pvalue`, `threshold` and pair counts |

## Single cells

```{eval-rst}
.. autosummary::
    :toctree: generated

    tl.cluster_composition
    tl.subpopulation_hits
    tl.cell_cycle_phase
    tl.neighbors_local_density
```

| Function | Stores |
| --- | --- |
| `tl.cluster_composition` | returns a new object: wells by clusters; `uns["mantispy"]["composition_test"]` |
| `tl.subpopulation_hits` | `uns["mantispy"][key_added]` |
| `tl.cell_cycle_phase` | `obs[key_added]` |
| `tl.neighbors_local_density` | `obs[key_added]` |
