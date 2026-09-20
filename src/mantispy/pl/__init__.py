"""Plotting.

Every plot returns Matplotlib axes and does not modify the object it draws.

Two plots have no function of their own. A plate map of a per-well flag is ``mt.pl.plate(adata, color="qc_well_pass")``, and embeddings side by side are a loop over ``sc.pl.embedding``.
"""

from mantispy.pl._diagnostics import control_drift, image_qc, outliers, plate_effects
from mantispy.pl._evaluation import batch_variance, map, metrics, replicate_correlation, similarity
from mantispy.pl._features import feature_correlation, feature_groups
from mantispy.pl._heterogeneity import cell_cycle, cluster_composition, density, subpopulation_hits
from mantispy.pl._hits import dose_direction, dose_response, effect_sizes, feature_volcano, hits
from mantispy.pl._moa import distance_heatmap, moa_confusion, moa_enrichment, pathway_coherence, sets_heatmap
from mantispy.pl._plate import plate
from mantispy.pl._qc import cell_counts, cytotoxicity, feature_distributions, nan_matrix, qc, replicate_saturation
from mantispy.pl._signature import feature_signature
from mantispy.pl._transport import setting_agreement, transport

__all__ = [
    "setting_agreement",
    "batch_variance",
    "cell_counts",
    "cell_cycle",
    "cluster_composition",
    "control_drift",
    "cytotoxicity",
    "density",
    "distance_heatmap",
    "dose_direction",
    "dose_response",
    "effect_sizes",
    "feature_correlation",
    "feature_distributions",
    "feature_groups",
    "feature_signature",
    "feature_volcano",
    "hits",
    "image_qc",
    "map",
    "metrics",
    "moa_confusion",
    "moa_enrichment",
    "nan_matrix",
    "outliers",
    "pathway_coherence",
    "plate",
    "plate_effects",
    "qc",
    "replicate_correlation",
    "replicate_saturation",
    "sets_heatmap",
    "similarity",
    "transport",
    "subpopulation_hits",
]
