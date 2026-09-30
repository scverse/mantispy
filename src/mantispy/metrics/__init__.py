"""Metrics for judging a correction.

:func:`evaluate_correction` runs the batch-integration panel over one or more representations and returns a tidy frame with ``metric``, ``representation``, ``key`` and ``value``, so results from different metrics and representations stack into one table.
:func:`diagnose_testing` returns its own table of checks on differential testing.

The integration panel (iLISI, cLISI, the batch and label silhouettes and PC-regression) wraps scib-metrics, and :func:`known_relationships` follows EFAAR.
Batch metrics and biological-signal metrics trade off against each other, so read them together.
"""

from mantispy.metrics._diagnose import diagnose_testing
from mantispy.metrics._evaluate import evaluate_correction
from mantispy.metrics._relationships import known_relationships
from mantispy.metrics._variance import variance_carried

__all__ = [
    "diagnose_testing",
    "evaluate_correction",
    "known_relationships",
    "variance_carried",
]
