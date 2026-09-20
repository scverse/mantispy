"""Metrics for judging a correction.

Each correction metric returns a tidy frame with ``metric``, ``representation``, ``key`` and ``value``, so results from different metrics and representations stack into one table, which :func:`evaluate_correction` builds. :func:`diagnose_testing` returns its own table of checks on differential testing.

The definitions follow scib and are reimplemented here, so scib is not a dependency. Batch metrics and biological-signal metrics trade off against each other, so read them together.
"""

from mantispy.metrics._diagnose import diagnose_testing
from mantispy.metrics._evaluate import evaluate_correction
from mantispy.metrics._lisi import lisi
from mantispy.metrics._relationships import known_relationships
from mantispy.metrics._silhouette import silhouette_batch, silhouette_label
from mantispy.metrics._variance import batch_variance_explained, pc_regression

__all__ = [
    "diagnose_testing",
    "batch_variance_explained",
    "evaluate_correction",
    "known_relationships",
    "lisi",
    "pc_regression",
    "silhouette_batch",
    "silhouette_label",
]
