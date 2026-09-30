# Metrics

```{eval-rst}
.. module:: mantispy.metrics
.. currentmodule:: mantispy
```

Metrics do not modify the object.
Most return a tidy frame with `metric`, `representation`, `key` and `value`, so results from several calls stack; `evaluate_integration` instead returns scib-metrics' benchmark table (or a per-representation mAP frame), so its output switches on what is installed.

```{eval-rst}
.. autosummary::
    :toctree: generated

    metrics.known_relationships
    metrics.evaluate_integration
    metrics.diagnose_testing
    metrics.pc_regression
    metrics.batch_variance_explained
```
