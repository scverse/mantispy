"""Synthetic plates to try things on, and Cell Painting Gallery accessions to check them against."""

from mantispy.ds._blobs import blobs
from mantispy.ds._datasets import (
    JUMP_LITE_MODELS,
    bbbc021,
    chroma,
    corum,
    cp_posh,
    jump_cells,
    jump_crispr,
    jump_export,
    jump_lite,
    jump_lite_targets,
    jump_plate,
    jump_target2,
    neuropainting,
    oasis_pilot,
    pki,
    pooled_rare,
    rohban,
    scallops_arv471,
)
from mantispy.ds._resources import gene_sets, interactions
from mantispy.ds._synthetic import DEFAULT_CHANNELS, synthetic_plate

__all__ = [
    "DEFAULT_CHANNELS",
    "bbbc021",
    "blobs",
    "chroma",
    "corum",
    "cp_posh",
    "gene_sets",
    "interactions",
    "jump_cells",
    "JUMP_LITE_MODELS",
    "jump_crispr",
    "jump_lite",
    "jump_lite_targets",
    "jump_export",
    "jump_plate",
    "jump_target2",
    "neuropainting",
    "oasis_pilot",
    "pki",
    "pooled_rare",
    "rohban",
    "scallops_arv471",
    "synthetic_plate",
]
