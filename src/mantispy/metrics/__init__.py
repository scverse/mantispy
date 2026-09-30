"""Metrics for judging a correction.

:func:`evaluate_integration` runs the integration benchmark scib-metrics owns over one or more representations and draws it as a mantispy heatmap, returning the numeric results frame; it raises :class:`ImportError` when neither scib-metrics nor copairs is installed.
:func:`diagnose_testing` returns its own table of checks on differential testing.

PC-regression (:func:`pc_regression`, :func:`batch_variance_explained`) is native and dependency-free and audits what a representation spends its variance on; the integration benchmark comes from scib-metrics when it is installed, and :func:`known_relationships` follows EFAAR.
Batch metrics and biological-signal metrics trade off against each other, so read them together.
"""

from mantispy.metrics._diagnose import diagnose_testing
from mantispy.metrics._evaluate import evaluate_integration
from mantispy.metrics._relationships import known_relationships
from mantispy.metrics._variance import batch_variance_explained, pc_regression, variance_carried

__all__ = [
    "batch_variance_explained",
    "diagnose_testing",
    "evaluate_integration",
    "known_relationships",
    "pc_regression",
    "variance_carried",
]
