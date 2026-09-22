"""JUMP Cell Painting profiles and their perturbation annotation.

A JUMP plate parquet carries three metadata columns (source, plate, well) and 4762 features.
The compound or gene each well received is recorded in a separate repository, keyed by ``Metadata_JCP2022``.
This module reads the profiles with :func:`~mantispy.io.read_profiles` and joins the annotation onto them.

Sources, all public over HTTPS:

* profiles: Cell Painting Gallery, accession ``cpg0016-jump``
* annotation: ``jump-cellpainting/datasets`` on GitHub

References:
    :cite:t:`Chandrasekaran_2023`, the JUMP Cell Painting datasets.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd
from anndata import AnnData

from mantispy._core.logging import get_logger
from mantispy.io._profiles import _read_frame, read_profiles

#: JUMP's annotation tables, pinned by sha256 in the dataset registry because the upstream repository is mutable.
#: A changed table fails the checksum instead of changing the annotation.
TABLES = ("plate", "well", "compound", "crispr", "perturbation_control", "gene_chromosome_map")

#: JUMP's negative control: DMSO, under its JCP identifier.
NEGATIVE_CONTROL = "JCP2022_033924"

#: Kinds of perturbation the annotation covers, and the table each is described by.
KINDS = ("compound", "crispr")

_JOIN_ON = ["Metadata_Source", "Metadata_Plate", "Metadata_Well"]


def jump_metadata(name: str) -> pd.DataFrame:
    """Read one of JUMP's annotation tables, such as ``"well"``, ``"compound"`` or ``"crispr"``.

    Args:
        name: Which table to read, one of :data:`TABLES`.

    Returns:
        The table as JUMP publishes it, downloaded once into :attr:`mantispy.settings.cache_dir` and checked against the sha256 the dataset registry pins.

    Raises:
        ValueError: `name` is not one of :data:`TABLES`.
    """
    if name not in TABLES:
        raise ValueError(f"name must be one of {TABLES}, got {name!r}")
    from mantispy.ds._datasets import _files

    (path,) = _files("_jump_annotation", select=lambda file_name: file_name.split(".")[0] == f"jump_{name}")
    return _read_frame(path)


def read_jump(paths: str | Path | Sequence[str | Path], annotate: bool = True, **kwargs: Any) -> AnnData:
    """Read JUMP plate profiles, optionally joining the annotation.

    Args:
        paths: One or more ``{plate}.parquet`` files in the Cell Painting Gallery layout.
        annotate: Join the well and compound tables, which map the three metadata columns to a perturbation.
            Downloads about 14 MB once and caches it.
        kwargs: Passed to :func:`~mantispy.io.read_profiles`.
            ``on_column_mismatch="intersect"`` is useful when plates come from different sources.

    Returns:
        An :class:`~anndata.AnnData` at well resolution.
        With ``annotate`` it carries ``Metadata_JCP2022``, ``Metadata_Perturbation``, ``Metadata_InChIKey`` and ``Metadata_Control``.

    Notes:
        JUMP plates from different sources share their feature names but not always the same set of features; pass ``on_column_mismatch="intersect"`` when mixing sources.
    """
    adata = read_profiles(paths, resolution="well", **kwargs)
    if annotate:
        from mantispy.pp._annotate import annotate_jump

        annotate_jump(adata)
    return adata


def join_jump_annotation(obs: pd.DataFrame, kind: str = "compound") -> pd.DataFrame:
    """Join the JUMP annotation onto an ``obs`` frame keyed by source, plate and well.

    Args:
        obs: A frame carrying ``Metadata_Source``, ``Metadata_Plate`` and ``Metadata_Well``, whose three key columns are cast to strings in place before the join, or one that already carries ``Metadata_JCP2022``, as JUMP's assembled profiles do.
        kind: Which perturbation the annotation is read for, one of :data:`KINDS`.

    Returns:
        A new frame with ``Metadata_JCP2022``, ``Metadata_Perturbation`` and ``Metadata_Control`` joined onto `obs`, missing on the wells the annotation does not cover.
        For ``"compound"`` it adds ``Metadata_InChIKey``, and ``Metadata_Control`` marks :data:`NEGATIVE_CONTROL`.
        For ``"crispr"`` ``Metadata_Gene`` and ``Metadata_Perturbation`` are the gene symbol, ``Metadata_Control_Type`` is ``"negcon"``, ``"poscon"`` or ``"trt"``, ``Metadata_Control`` marks the no-guide and non-targeting wells, and ``Metadata_ChromosomeArm`` is the arm the gene sits on, such as ``"1p"``, missing for a gene without a mapped locus.

    Raises:
        ValueError: `kind` is not one of :data:`KINDS`.
        KeyError: `obs` has no ``Metadata_JCP2022`` and is missing one of the three columns the annotation is keyed by.
    """
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")

    joined = obs if "Metadata_JCP2022" in obs else _join_wells(obs)

    if kind == "compound":
        compounds = jump_metadata("compound")[["Metadata_JCP2022", "Metadata_InChIKey"]]
        joined = joined.merge(compounds, on="Metadata_JCP2022", how="left", validate="m:1")
        joined["Metadata_Perturbation"] = joined["Metadata_JCP2022"].astype(str)
        joined["Metadata_Control"] = (joined["Metadata_JCP2022"] == NEGATIVE_CONTROL).to_numpy()
        return joined

    genes = jump_metadata("crispr")[["Metadata_JCP2022", "Metadata_Symbol"]].rename(
        columns={"Metadata_Symbol": "Metadata_Gene"}
    )
    controls = jump_metadata("perturbation_control")
    controls = controls.loc[
        controls["Metadata_modality"] == "crispr", ["Metadata_JCP2022", "Metadata_pert_type"]
    ].rename(columns={"Metadata_pert_type": "Metadata_Control_Type"})
    for table in (genes, controls):
        joined = joined.merge(table, on="Metadata_JCP2022", how="left", validate="m:1")
    joined["Metadata_Control_Type"] = joined["Metadata_Control_Type"].fillna("trt")
    joined["Metadata_Perturbation"] = joined["Metadata_Gene"].fillna(joined["Metadata_JCP2022"]).astype(str)
    joined["Metadata_Control"] = (joined["Metadata_Control_Type"] == "negcon").to_numpy()
    joined["Metadata_ChromosomeArm"] = joined["Metadata_Gene"].map(_chromosome_arms())
    return joined


def _join_wells(obs: pd.DataFrame) -> pd.DataFrame:
    """Map source, plate and well to ``Metadata_JCP2022`` through JUMP's well table."""
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
    return joined


def _chromosome_arms() -> pd.Series:
    """The chromosome arm of every gene symbol, read off its cytogenetic locus as jump-profiling-recipe does."""
    loci = jump_metadata("gene_chromosome_map").drop_duplicates("Approved_symbol").set_index("Approved_symbol")["Locus"]
    return loci.astype(str).str.extract(r"^(\w+?[pq])", expand=False)
