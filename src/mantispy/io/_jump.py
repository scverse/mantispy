"""JUMP Cell Painting profiles and their perturbation annotation.

A JUMP plate parquet carries three metadata columns (source, plate, well) and 4762 features.
The compound or gene each well received is recorded in a separate repository, keyed by ``Metadata_JCP2022``.

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

import numpy as np
import pandas as pd
from anndata import AnnData

from mantispy._core.logging import get_logger
from mantispy.io._profiles import _read_frame, read_profiles

#: Pinned by sha256 in the dataset registry because the upstream repository is mutable.
TABLES = (
    "plate",
    "well",
    "compound",
    "crispr",
    "orf",
    "perturbation_control",
    "gene_chromosome_map",
    "gene_expression",
)

#: JUMP's negative control: DMSO, under its JCP identifier.
NEGATIVE_CONTROL = "JCP2022_033924"

KINDS = ("compound", "crispr", "orf")

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
        With ``annotate`` it carries ``Metadata_JCP2022``, ``Metadata_Perturbation``, ``Metadata_Perturbation_Type`` (``"compound"``), ``Metadata_InChIKey`` and ``Metadata_Control``.
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
        A new frame with ``Metadata_JCP2022``, ``Metadata_Perturbation``, ``Metadata_Perturbation_Type`` and ``Metadata_Control`` joined onto `obs`, missing on the wells the annotation does not cover.
        For ``"compound"`` the perturbation is the ``Metadata_JCP2022`` compound id, ``Metadata_Perturbation_Type`` is ``"compound"``, it adds ``Metadata_InChIKey``, and ``Metadata_Control`` marks :data:`NEGATIVE_CONTROL`.
        For ``"crispr"`` and ``"orf"`` ``Metadata_Gene`` and ``Metadata_Perturbation`` are the gene symbol (its reagents are the replicates), ``Metadata_Perturbation_Type`` is the kind, ``Metadata_Control_Type`` is ``"negcon"``, ``"poscon"`` or ``"trt"``, ``Metadata_Control`` marks the negative-control wells, and ``Metadata_ChromosomeArm`` is the arm the gene sits on, such as ``"1p"``, missing for a gene without a mapped locus.

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
        joined["Metadata_Perturbation_Type"] = "compound"
        joined["Metadata_Control"] = (joined["Metadata_JCP2022"] == NEGATIVE_CONTROL).to_numpy()
        return joined

    genes = jump_metadata(kind)[["Metadata_JCP2022", "Metadata_Symbol"]].rename(
        columns={"Metadata_Symbol": "Metadata_Gene"}
    )
    controls = jump_metadata("perturbation_control")
    controls = controls.loc[controls["Metadata_modality"] == kind, ["Metadata_JCP2022", "Metadata_pert_type"]].rename(
        columns={"Metadata_pert_type": "Metadata_Control_Type"}
    )
    for table in (genes, controls):
        joined = joined.merge(table, on="Metadata_JCP2022", how="left", validate="m:1")
    joined["Metadata_Control_Type"] = joined["Metadata_Control_Type"].fillna("trt")
    # The gene is the replication unit here: JUMP's mAP benchmark scores the genetic arms at the gene level,
    # treating the several reagents per gene as its replicates. The reagent stays in Metadata_JCP2022.
    joined["Metadata_Perturbation"] = joined["Metadata_Gene"].fillna(joined["Metadata_JCP2022"]).astype(str)
    joined["Metadata_Perturbation_Type"] = kind
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


def unexpressed_genes(
    expression: str | Path | pd.DataFrame | None = None,
    *,
    cell_line: str | None = None,
    models: str | Path | pd.DataFrame | None = None,
    tpm_cutoff: float = 0.0,
    zfpkm_cutoff: float = -3.0,
) -> set[str]:
    """Gene symbols that are not expressed in the screened cell line, as an empirical null for hit calling.

    With no `expression` this reads JUMP's Recursion U2OS reference and calls a gene unexpressed when its zFPKM is below `zfpkm_cutoff`, matching ``df[df.zfpkm < cutoff].gene.unique()`` in jump-profiling-recipe's chromosome-arm correction.

    With `expression` it reads a DepMap expression matrix you have downloaded, so the null can be built for any cell line DepMap covers.
    Nothing is downloaded or re-hosted; point it at the file from the DepMap data page, such as ``OmicsExpressionProteinCodingGenesTPMLogp1.csv`` (one row per model, gene columns named ``"SYMBOL (ENTREZ)"``, holding log2(TPM+1)).
    A gene is unexpressed when its value is at or below `tpm_cutoff`, which is zero for the zero-TPM genes PERISCOPE uses.

    Args:
        expression: A DepMap expression matrix, as a path or a loaded frame indexed by model, or ``None`` for the JUMP reference.
        cell_line: Which row of `expression` to read, a DepMap model id such as ``"ACH-000012"`` or, with `models`, a cell-line name such as ``"U2OS"``.
        models: DepMap's ``Model.csv``, as a path or frame, used to resolve a cell-line name in `cell_line` to its model id; unused when `cell_line` is already a model id.
        tpm_cutoff: The log2(TPM+1) value at or below which a gene is called unexpressed in the DepMap path.
        zfpkm_cutoff: The zFPKM below which a gene is called unexpressed in the JUMP path.

    Returns:
        The set of unexpressed gene symbols.

    Raises:
        ValueError: `expression` is given without `cell_line`, or the cell line is not found.
    """
    if expression is None:
        reference = jump_metadata("gene_expression")
        return set(reference.loc[reference["zfpkm"] < zfpkm_cutoff, "gene"].astype(str))
    if cell_line is None:
        raise ValueError("cell_line is required when expression is given, to pick which model's row to read")
    frame = expression if isinstance(expression, pd.DataFrame) else pd.read_csv(expression, index_col=0)
    model_id = _resolve_model(cell_line, frame, models)
    symbols = frame.columns.to_series().str.replace(r"\s*\(\d+\)\s*$", "", regex=True)
    row = pd.Series(np.asarray(frame.loc[model_id]).ravel(), index=frame.columns)
    values = pd.to_numeric(row, errors="coerce").to_numpy()
    return set(symbols[values <= tpm_cutoff].astype(str))


def _resolve_model(cell_line: str, expression: pd.DataFrame, models: str | Path | pd.DataFrame | None) -> str:
    """Turn a model id or cell-line name into the model id that indexes the expression matrix."""
    if cell_line in expression.index:
        return cell_line
    if models is None:
        raise ValueError(
            f"cell_line={cell_line!r} is not a row of the expression matrix; pass models=Model.csv to resolve a "
            "cell-line name to its DepMap model id."
        )
    table = models if isinstance(models, pd.DataFrame) else pd.read_csv(models)
    wanted = cell_line.strip().upper()
    name_columns = [column for column in ("StrippedCellLineName", "CellLineName", "cell_line_name") if column in table]
    hits = table.loc[
        table[name_columns].apply(lambda col: col.astype(str).str.upper().str.strip()).eq(wanted).any(axis=1)
    ]
    ids = hits["ModelID"].astype(str).unique() if "ModelID" in hits else hits.iloc[:, 0].astype(str).unique()
    ids = [model for model in ids if model in expression.index]
    if not ids:
        raise ValueError(f"cell_line={cell_line!r} matched no model present in the expression matrix")
    return ids[0]
