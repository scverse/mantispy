"""Metrics for judging a correction.

:func:`evaluate_correction` runs the batch-integration panel over one or more representations and returns a tidy frame with ``metric``, ``representation``, ``key`` and ``value``, so results from different metrics and representations stack into one table.
:func:`diagnose_testing` returns its own table of checks on differential testing.

PC-regression (:func:`pc_regression`, :func:`batch_variance_explained`) is native and dependency-free; the batch-mixing metrics (iLISI, cLISI, the batch and label silhouettes) come from scib-metrics when it is installed, and :func:`known_relationships` follows EFAAR.
Batch metrics and biological-signal metrics trade off against each other, so read them together.
"""

from mantispy.metrics._diagnose import diagnose_testing
from mantispy.metrics._evaluate import evaluate_correction
from mantispy.metrics._relationships import known_relationships
from mantispy.metrics._variance import batch_variance_explained, pc_regression, variance_carried

__all__ = [
    "batch_variance_explained",
    "diagnose_testing",
    "evaluate_correction",
    "known_relationships",
    "pc_regression",
    "variance_carried",
]
