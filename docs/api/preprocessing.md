# Preprocessing

```{eval-rst}
.. module:: mantispy.pp
.. currentmodule:: mantispy
```

Annotate, quality-control, normalize, select and batch-correct profiles.

## Annotation

```{eval-rst}
.. autosummary::
    :toctree: generated

    pp.annotate_controls
    pp.annotate_jump
    pp.find_perturbation_key
    pp.standardize_feature_names
```

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

## Normalization

```{eval-rst}
.. autosummary::
    :toctree: generated

    pp.normalize
    pp.rank_int
```

## Feature selection

```{eval-rst}
.. autosummary::
    :toctree: generated

    pp.feature_select
    pp.subset_features
    pp.decorr_threshold_sweep
    pp.feature_select_chatterjee
    pp.feature_reproducibility
    pp.feature_batch_sensitivity
```

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
