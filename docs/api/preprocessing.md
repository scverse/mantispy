# Preprocessing

```{eval-rst}
.. module:: mantispy.pp
.. currentmodule:: mantispy
```

## Annotation

```{eval-rst}
.. autosummary::
    :toctree: generated

    pp.annotate_controls
    pp.annotate_jump
    pp.find_perturbation_key
    pp.standardize_feature_names
```

| Function | Stores |
| --- | --- |
| `pp.annotate_controls` | `obs["Metadata_Control"]`, and `obs["Metadata_Control_Type"]` when `poscon` is given |
| `pp.annotate_jump` | `obs`: `Metadata_JCP2022`, `Metadata_Perturbation`, `Metadata_Control`; `Metadata_InChIKey` for compounds, `Metadata_Gene`, `Metadata_Control_Type` and `Metadata_ChromosomeArm` for CRISPR |
| `pp.standardize_feature_names` | `var_names`, `var["original_name"]` |

## Quality control

```{eval-rst}
.. autosummary::
    :toctree: generated

    pp.calculate_qc_metrics
    pp.filter_cells
    pp.filter_features
    pp.outliers
    pp.image_qc
    pp.filter_images
    pp.well_qc
    pp.downsample
```

| Function | Stores |
| --- | --- |
| `pp.calculate_qc_metrics` | `obs`: `qc_n_nan_features`, `qc_nan_fraction`, `qc_is_border`, `qc_area_outlier`, `qc_pass`; `var`: `qc_n_nan`, `qc_variance`, `qc_n_unique` |
| `pp.filter_cells` | subsets `obs` in place |
| `pp.filter_features` | subsets `var` in place |
| `pp.outliers` | `obs[key_added]`, `obs[key_added + "_score"]` |
| `pp.image_qc` | `uns["mantispy"]["image_qc"]`, `obs["qc_image_pass"]` |
| `pp.well_qc` | `uns["mantispy"]["well_qc"]`, `obs["qc_well_pass"]` |
| `pp.downsample` | returns a new object holding the sampled rows |

## Normalization

```{eval-rst}
.. autosummary::
    :toctree: generated

    pp.normalize
    pp.rank_int
```

| Function | Stores |
| --- | --- |
| `pp.normalize` | `X`, or `layers[key_added]`; `layers["raw"]` when `keep_raw=True` |
| `pp.rank_int` | `X`, or `layers[key_added]` |

`pp.normalize` also writes `var["degenerate_scale"]`, flagging features with no spread in some group.
Those are divided by `epsilon` rather than by zero and come back at ~1e17; drop them before computing anything from distances.
Writing to a layer suffixes the column, so `key_added="sphered"` flags `var["degenerate_scale_sphered"]` and one call cannot overwrite what another measured.

## Feature selection

```{eval-rst}
.. autosummary::
    :toctree: generated

    pp.feature_select
    pp.subset_features
    pp.feature_select_chatterjee
    pp.feature_reproducibility
    pp.feature_batch_sensitivity
```

| Function | Stores |
| --- | --- |
| `pp.feature_select` | `var[key_added]`, `uns["mantispy"]["feature_select"]` |
| `pp.feature_select_chatterjee` | `var[key_added]`, `var["chatterjee_xi"]` |
| `pp.feature_reproducibility` | `var[key_added]`, `var[key_added + "_selected"]` |
| `pp.feature_batch_sensitivity` | `var[key_added + "_pvalue"]`, `_qvalue`, `_sensitive` |

## Batch correction

```{eval-rst}
.. autosummary::
    :toctree: generated

    pp.sphere
    pp.tvn
    pp.correct_plate_position
    pp.regress_out
    pp.harmony
```

| Function | Stores |
| --- | --- |
| `pp.sphere` | `X`, or `layers[key_added]` |
| `pp.tvn` | `obsm[key_added]` |
| `pp.correct_plate_position` | `X` or `layers[key_added]`, `uns["mantispy"]["plate_position"]` |
| `pp.regress_out` | `X`, or `layers[key_added]` |
| `pp.harmony` | `obsm[key_added]`; needs `mantispy[harmony]` |
