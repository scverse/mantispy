"""JUMP Cell Painting profiles and their perturbation annotation.

A JUMP plate parquet carries three metadata columns (source, plate, well) and 4762
features. The compound or gene each well received is recorded in a separate repository,
keyed by ``Metadata_JCP2022``. This module reads the profiles with
:func:`~mantispy.io.read_profiles` and joins the annotation onto them.

Sources, all public over HTTPS:

* profiles: Cell Painting Gallery, accession ``cpg0016-jump``
* annotation: ``jump-cellpainting/datasets`` on GitHub

References:
    Chandrasekaran et al. (2024) Nature Methods 21:1114, the JUMP Cell Painting datasets.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pandas as pd
from anndata import AnnData

from mantispy._core._utils import get_logger
from mantispy.io._profiles import read_profiles

#: JUMP's annotation tables, pinned by sha256 in the dataset registry because the upstream repository is mutable.
#: A changed table fails the checksum instead of changing the annotation.
TABLES = ("plate", "well", "compound")

#: JUMP's negative control: DMSO, under its JCP identifier.
NEGATIVE_CONTROL = "JCP2022_033924"

#: Kinds of perturbation the annotation covers, and the table each is described by.
KINDS = ("compound",)

_JOIN_ON = ["Metadata_Source", "Metadata_Plate", "Metadata_Well"]


def jump_metadata(name: str) -> pd.DataFrame:
    """Read one of JUMP's annotation tables, ``"plate"``, ``"well"`` or ``"compound"``."""
    if name not in TABLES:
        raise ValueError(f"name must be one of {TABLES}, got {name!r}")
    from mantispy.ds._datasets import _files

    (path,) = _files("_jump_annotation", select=lambda file_name: file_name == f"jump_{name}.csv.gz")
    return pd.read_csv(path)


def read_jump(paths: str | Path | Sequence[str | Path], annotate: bool = True, **kwargs) -> AnnData:
    """Read JUMP plate profiles, optionally joining the annotation.

    Args:
        paths: One or more ``{plate}.parquet`` files in the Cell Painting Gallery layout.
        annotate: Join the well and compound tables, which map the three metadata columns to a
            perturbation. Downloads about 14 MB once and caches it.
        kwargs: Passed to :func:`~mantispy.io.read_profiles`. ``on_column_mismatch="intersect"``
            is useful when plates come from different sources.

    Returns:
        An :class:`~anndata.AnnData` at well resolution. With ``annotate`` it carries
        ``Metadata_JCP2022``, ``Metadata_Perturbation``, ``Metadata_InChIKey`` and
        ``Metadata_Control``.

    Notes:
        JUMP plates from different sources share their feature names but not always the same
        set of features; pass ``on_column_mismatch="intersect"`` when mixing sources.
    """
    adata = read_profiles(paths, resolution="well", **kwargs)
    if annotate:
        from mantispy.pp._annotate import annotate_jump

        annotate_jump(adata)
    return adata


def join_jump_annotation(obs: pd.DataFrame, kind: str = "compound") -> pd.DataFrame:
    """Join the JUMP annotation onto an ``obs`` frame keyed by source, plate and well."""
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")

    missing = [column for column in _JOIN_ON if column not in obs]
    if missing:
        raise KeyError(
            f"obs is missing {missing}, the columns the JUMP annotation is keyed by. Read the plate "
            "parquet with mt.io.read_jump to get them."
        )

    wells = jump_metadata("well")
    for frame in (obs, wells):
        for column in _JOIN_ON:
            frame[column] = frame[column].astype(str)

    joined = obs.merge(wells, on=_JOIN_ON, how="left", validate="m:1")
    unannotated = int(joined["Metadata_JCP2022"].isna().sum())
    if unannotated:
        get_logger().warning(
            "%d of %d wells have no JUMP annotation; their Metadata_JCP2022 is missing", unannotated, len(joined)
        )

    compounds = jump_metadata("compound")[["Metadata_JCP2022", "Metadata_InChIKey"]]
    joined = joined.merge(compounds, on="Metadata_JCP2022", how="left", validate="m:1")
    joined["Metadata_Perturbation"] = joined["Metadata_JCP2022"].astype(str)
    joined["Metadata_Control"] = (joined["Metadata_JCP2022"] == NEGATIVE_CONTROL).to_numpy()
    return joined
