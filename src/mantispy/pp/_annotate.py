"""Annotate perturbations and controls.

Several later steps (``pp.sphere``, ``tl.grit``) default to ``reference="negcon"``,
which reads ``Metadata_Control``; :func:`annotate_controls` writes that column.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from anndata import AnnData

from mantispy._core._utils import as_frame, categorize_metadata, get_logger, inplace_or_copy

#: Columns checked, in order, when the perturbation column is not named explicitly.
PERTURBATION_KEYS = (
    "Metadata_Perturbation",
    "Metadata_Compound",
    "Metadata_Treatment",
    "Metadata_BroadSample",
)


def find_perturbation_key(adata: AnnData, perturbation_key: str | None = None) -> str:
    """Resolve which ``obs`` column holds the perturbation identity."""
    if perturbation_key is not None:
        if perturbation_key not in adata.obs:
            raise KeyError(f"obs has no column {perturbation_key!r}")
        return perturbation_key
    for candidate in PERTURBATION_KEYS:
        if candidate in adata.obs:
            return candidate
    raise KeyError(
        f"none of {list(PERTURBATION_KEYS)} is in obs; pass perturbation_key= to say which "
        "column identifies the perturbation"
    )


@inplace_or_copy()
def annotate_controls(
    adata: AnnData,
    negcon: Sequence[str] = ("DMSO",),
    poscon: Sequence[str] | None = None,
    perturbation_key: str | None = None,
    copy: bool = False,
) -> AnnData | None:
    """Mark negative (and optionally positive) controls.

    Args:
        adata: Object to annotate.
        negcon: Perturbation values that are negative controls.
        poscon: Perturbation values that are positive controls, if any.
        perturbation_key: Column holding the perturbation. Auto-detected from
            ``PERTURBATION_KEYS`` when omitted.
        copy: Return a modified copy instead of mutating in place.

    Returns:
        ``None``, or the modified copy when ``copy=True``. Writes ``obs["Metadata_Control"]``
        and, when ``poscon`` is given, ``obs["Metadata_Control_Type"]``.
    """
    key = find_perturbation_key(adata, perturbation_key)
    values = adata.obs[key].astype(str)

    is_negcon = values.isin([str(v) for v in negcon]).to_numpy()
    if not is_negcon.any():
        get_logger().warning(
            "no negative controls found: none of %s appear in obs[%r]. "
            "Steps defaulting to reference='negcon' will fail until this is set.",
            list(negcon),
            key,
        )
    adata.obs["Metadata_Control"] = is_negcon

    if poscon is not None:
        control_type = np.where(is_negcon, "negcon", "")
        control_type[values.isin([str(v) for v in poscon]).to_numpy()] = "poscon"
        adata.obs["Metadata_Control_Type"] = control_type
    return None  # the decorator returns the copy when copy=True


@inplace_or_copy(expects="well")
def annotate_jump(adata: AnnData, kind: str = "compound", copy: bool = False) -> AnnData | None:
    """Join the JUMP annotation onto profiles read from the Cell Painting Gallery.

    A JUMP plate parquet records only the source, plate and well of a profile. What the
    well contained is published in a separate repository keyed by ``Metadata_JCP2022``.

    Args:
        adata: Well-level JUMP profiles carrying ``Metadata_Source``, ``Metadata_Plate`` and
            ``Metadata_Well``.
        kind: Which annotation to join. ``"compound"`` today.
        copy: Return an annotated copy instead of annotating in place.

    Returns:
        ``None``, or the annotated copy. Adds ``Metadata_JCP2022`` (the perturbation
        identifier), ``Metadata_Perturbation``, ``Metadata_InChIKey`` and
        ``Metadata_Control``, which marks JUMP's DMSO wells.

    Notes:
        Downloads about 14 MB of annotation once and caches it. Wells the annotation does not
        cover are kept and logged, since an unannotated well is still a measurement.
    """
    from mantispy.io._jump import join_jump_annotation

    obs = as_frame(adata.obs)
    joined = join_jump_annotation(obs.copy(), kind=kind)
    joined.index = obs.index
    adata.obs = categorize_metadata(joined)
    get_logger().info(
        "annotate_jump: %d perturbations over %d wells, %d of them controls",
        int(joined["Metadata_JCP2022"].nunique()),
        len(joined),
        int(joined["Metadata_Control"].sum()),
    )
    return None
