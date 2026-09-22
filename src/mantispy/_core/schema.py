"""The mantispy AnnData contract.

An object is described by three things: the ``Metadata_`` columns in ``obs`` that identify where a profile came from, the parsed annotation columns in ``var`` that say what each feature measures, and a small ``uns["mantispy"]`` dict holding the schema version, the resolution, and provenance.

Resolution is advisory.
:func:`validate` uses it to decide which identifier columns are required, but no function raises because it was given an object at another resolution.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from .features import COLUMNS as VAR_COLUMNS
from .features import empty_annotation
from .logging import get_logger
from .plate import normalize_well

if TYPE_CHECKING:
    from anndata import AnnData

SCHEMA_VERSION = "1.0"

#: Versions :func:`migrate` can bring to :data:`SCHEMA_VERSION`, oldest first.
SUPPORTED_VERSIONS: tuple[str, ...] = ("0.1", "1.0")

#: Resolutions a mantispy object can be at, coarsest last.
RESOLUTIONS = ("cell", "well", "perturbation")

#: Identifier columns required in ``obs``, per resolution.
REQUIRED_OBS: dict[str, tuple[str, ...]] = {
    "cell": ("Metadata_Plate", "Metadata_Well"),
    "well": ("Metadata_Plate", "Metadata_Well"),
    # A consensus profile no longer belongs to a plate or a well.
    "perturbation": (),
}

#: Columns mantispy understands but does not require. OPS entries are reserved for 0.8.
RESERVED_OBS: tuple[str, ...] = (
    "Metadata_Batch",
    "Metadata_Source",
    "Metadata_Site",
    "Metadata_ImageNumber",
    "Metadata_ObjectNumber",
    "Metadata_Perturbation",
    "Metadata_Compound",
    "Metadata_Concentration",
    # What the plate map wrote, where it records one dose to several precisions and Metadata_Concentration
    # holds the one the well was meant to get.
    "Metadata_ConcentrationRecorded",
    "Metadata_MOA",
    "Metadata_CellLine",
    "Metadata_Control",
    # Cells behind a row and the fields of view that contributed them, written by tl.aggregate
    "Metadata_CellCount",
    "Metadata_SiteCount",
    "Metadata_Center_X",
    "Metadata_Center_Y",
    "Metadata_Control_Type",
    "Metadata_CellCyclePhase",
    "Metadata_LocalDensity",
    "Metadata_ReplicateCount",
    # JUMP identifiers, written by pp.annotate_jump
    "Metadata_JCP2022",
    "Metadata_InChIKey",
    "Metadata_PlateType",
    # reserved for optical pooled screening, unused before 0.8, except Metadata_Gene, the gene a genetic
    # perturbation targets, which pp.annotate_jump(kind="crispr") writes
    "Metadata_Barcode",
    "Metadata_Gene",
    "Metadata_sgRNA",
    "Metadata_BarcodeQuality",
)

#: Annotation columns mantispy writes to ``var`` and reads back, none of them required.
#: Listed so that downstream tools can rely on the names without reading the docstrings.
OPTIONAL_VAR: tuple[str, ...] = (
    "selected",
    "selected_chatterjee",
    "chatterjee_xi",
    "degenerate_scale",
    "icc",
    "icc_selected",
    "batch_pvalue",
    "batch_qvalue",
    "batch_sensitive",
    "qc_n_nan",
    "qc_variance",
    "qc_n_unique",
    "original_name",
)

#: Annotation columns required in ``var``.
REQUIRED_VAR: tuple[str, ...] = tuple(VAR_COLUMNS)

#: Keys describing the object itself, as opposed to a result computed from it.
UNS_KEYS: tuple[str, ...] = (
    "schema_version",
    "resolution",
    "channels",
    "dataset",
    "image_table",
    "params",
    "aggregated_from",
    "consensus_from",
    "truth",
)

#: Result tables mantispy writes under ``uns["mantispy"]`` and reads back.
#: Each is read by a plot or another tool, which makes it part of the contract.
UNS_RESULTS: tuple[str, ...] = (
    "feature_select",
    "image_qc",
    "well_qc",
    "plate_position",
    "map",
    "percent_replicating",
    "grit",
    "effect",
    "wasserstein",
    "consensus_weights",
    "hits",
    "edistance",
    "edistance_pairwise",
    "dose_response",
    "moa",
    "moa_confusion",
    "moa_enrichment",
    "rank_features",
    "rank_sets",
    "composition_test",
    "subpopulation_hits",
    "replicate_saturation",
    "cytotoxicity",
    "pathway_coherence",
    "enrich_hits",
)


@dataclass
class ValidationReport:
    """Result of :func:`~mantispy.io.validate`. Truthy when there are no errors."""

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """Whether the object satisfies the schema."""
        return not self.errors

    def __bool__(self) -> bool:
        return self.ok

    def __repr__(self) -> str:
        return f"ValidationReport(ok={self.ok}, errors={len(self.errors)}, warnings={len(self.warnings)})"

    def __str__(self) -> str:
        lines = [f"ERROR: {error}" for error in self.errors]
        lines += [f"WARNING: {warning}" for warning in self.warnings]
        return "\n".join(lines) or "valid"


def stamp(adata: AnnData, resolution: str | None = None, *, fill_var: bool = True) -> None:
    """Write the schema version, and optionally the resolution, into ``uns``, and supply the annotation columns ``var`` is missing.

    Args:
        adata: The object to stamp, in place.
        resolution: What one row is; left as it is when ``None``.
        fill_var: Supply any of :data:`REQUIRED_VAR` that ``var`` does not carry, from :func:`~mantispy._core.features.empty_annotation`.
            Every tool that builds a new object stamps it, so filling here is what keeps a tool from returning something :func:`validate` rejects.
            :func:`~mantispy.io.write` passes ``False``, so that writing an object whose annotation a caller has damaged still reports it rather than quietly repairing it.
    """
    store = adata.uns.setdefault("mantispy", {})
    store["schema_version"] = SCHEMA_VERSION
    if resolution is not None:
        if resolution not in RESOLUTIONS:
            raise ValueError(f"resolution must be one of {RESOLUTIONS}, got {resolution!r}")
        store["resolution"] = resolution
    if fill_var and (absent := [column for column in REQUIRED_VAR if column not in adata.var]):
        empty = empty_annotation(adata.var.index)
        for column in absent:
            adata.var[column] = empty[column]


def get_resolution(adata: AnnData) -> str:
    """Recorded resolution of ``adata``, defaulting to ``"cell"``."""
    resolution = adata.uns.get("mantispy", {}).get("resolution")
    return resolution if resolution in RESOLUTIONS else "cell"


def resolution_for(columns: Iterable[str]) -> str:
    """Resolution an object grouped by ``columns`` is at.

    Args:
        columns: Columns the grouping supplies, which become the identifier columns of the result.

    Returns:
        The finest resolution whose :data:`REQUIRED_OBS` columns ``columns`` covers, so that :func:`validate` requires of the result only what it carries.

    Notes:
        ``"cell"`` is never returned, since a grouping replaces the cells and requires the same columns as ``"well"`` anyway.
        A grouping finer than the well, by site or by anything else, is still per-well or finer, and it carries the plate and well columns a well-resolution object needs.
    """
    supplied = set(columns)
    for resolution in RESOLUTIONS[1:]:
        if set(REQUIRED_OBS[resolution]) <= supplied:
            return resolution
    return RESOLUTIONS[-1]


def validate(adata: AnnData, *, raise_on_error: bool = False) -> ValidationReport:
    """Check ``adata`` against the mantispy schema.

    Args:
        adata: The object to check.
        raise_on_error: Raise :class:`ValueError` instead of returning a failing report.

    Returns:
        A :class:`~mantispy._core.schema.ValidationReport`; truthy when the object is valid.
    """
    report = ValidationReport()
    resolution = get_resolution(adata)

    if adata.n_vars == 0:
        report.errors.append(
            "the object has no features, so the other checks would pass on an empty matrix. "
            "If it came from mt.io.read_profiles, check the objects= argument."
        )
        if raise_on_error and not report.ok:
            raise ValueError("\n".join(report.errors))
        return report

    for column in REQUIRED_OBS[resolution]:
        if column not in adata.obs:
            report.errors.append(f"obs is missing required column {column!r} (resolution {resolution!r})")

    for column in REQUIRED_VAR:
        if column not in adata.var:
            report.errors.append(f"var is missing required column {column!r}")

    store = adata.uns.get("mantispy")
    if not isinstance(store, dict) or "schema_version" not in store:
        report.errors.append("uns['mantispy']['schema_version'] is missing")
    elif store["schema_version"] != SCHEMA_VERSION:
        seen = store["schema_version"]
        remedy = (
            "read the file with mt.io.read, which migrates it to "
            if seen in SUPPORTED_VERSIONS
            else f"this build knows {SUPPORTED_VERSIONS} and cannot migrate from it to "
        )
        report.errors.append(f"schema_version {seen!r}: {remedy}{SCHEMA_VERSION!r}")
    elif store.get("resolution") is not None and store["resolution"] not in RESOLUTIONS:
        report.warnings.append(f"unknown resolution {store['resolution']!r}")

    if adata.X is not None and getattr(adata.X, "dtype", None) != np.float32:
        report.errors.append(f"X must be float32, got {getattr(adata.X, 'dtype', None)}")

    if "Metadata_Well" in adata.obs:
        unparsable = []
        for well in adata.obs["Metadata_Well"].astype(str).unique():
            try:
                normalize_well(well)
            except ValueError:
                unparsable.append(well)
        if unparsable:
            report.errors.append(f"Metadata_Well has unparsable values: {sorted(unparsable)[:5]}")

    if adata.n_vars and "is_feature" in adata.var and not adata.var["is_feature"].any():
        report.warnings.append("no column in var is marked is_feature")

    if resolution == "well" and "Metadata_CellCount" not in adata.obs:
        report.warnings.append(
            "obs has no Metadata_CellCount, so mt.tl.cytotoxicity cannot separate a hit from cell loss; "
            "aggregate single cells with mt.tl.aggregate, which writes it, or add a per-well count"
        )

    if raise_on_error and not report.ok:
        raise ValueError("\n".join(report.errors))
    return report


def migrate(adata: AnnData, copy: bool = False) -> AnnData | None:
    """Bring an object written by an earlier mantispy up to :data:`SCHEMA_VERSION`.

    Args:
        adata: Object to migrate.
        copy: Return a migrated copy instead of migrating in place.

    Returns:
        ``None``, or the migrated copy.

    Raises:
        ValueError: If the object carries a version this build does not know.

    Notes:
        0.1 to 1.0 only updates the version stamp.
        The 1.0 freeze added names to the vocabulary (optional ``var`` columns, JUMP identifiers, the result tables) and removed none, so no data moves.
        Later versions that do move data add their migration here.
    """
    target = adata.copy() if copy else adata
    store = target.uns.setdefault("mantispy", {})
    seen = store.get("schema_version")

    if seen == SCHEMA_VERSION:
        return target if copy else None
    if seen not in SUPPORTED_VERSIONS:
        raise ValueError(
            f"cannot migrate an object stamped {seen!r}; this build knows {SUPPORTED_VERSIONS}. "
            "It was probably written by a newer mantispy; upgrade to read it."
        )

    store["schema_version"] = SCHEMA_VERSION
    get_logger().info("migrated an object from schema %s to %s", seen, SCHEMA_VERSION)
    return target if copy else None
