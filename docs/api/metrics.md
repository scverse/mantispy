# Metrics

```{eval-rst}
.. module:: mantispy.metrics
.. currentmodule:: mantispy
```

Metrics do not modify the object.
Each returns a tidy frame with `metric`, `representation`, `key` and `value`, so results from several calls stack.

```{eval-rst}
.. autosummary::
    :toctree: generated

    metrics.silhouette_label
    metrics.silhouette_batch
    metrics.lisi
    metrics.pc_regression
    metrics.batch_variance_explained
    metrics.known_relationships
    metrics.evaluate_correction
    metrics.diagnose_testing
```
