"""The mantispy AnnData contract.

An object is described by three things: the ``Metadata_`` columns in ``obs`` that identify where a profile came from, the parsed annotation columns in ``var`` that say what each feature measures, and a ``uns["mantispy"]`` dict holding the core descriptors (schema version and status, the resolution, the grouping that defines one observation, and the provenance history).

Under schema 2.0 the resolution is explicit, not advisory: a mantispy object MUST carry ``uns["mantispy"]["resolution"]`` and ``uns["mantispy"]["grouped_by"]``, and functions that need them fail clearly when they are absent rather than guessing.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from .features import COLUMNS as VAR_COLUMNS
from .features import measurement_kind
from .frames import as_frame
from .logging import get_logger
from .masks import feature_mask
from .plate import normalize_well

if TYPE_CHECKING:
    from anndata import AnnData

#: The current schema. 2.0 is a breaking, still-experimental redesign of the previous internal 1.0
#: schema (kept only for migration); see ``spec/morphology-schema-2.0.md``.
SCHEMA_VERSION = "2.0"

#: Status of :data:`SCHEMA_VERSION`. ``"experimental"`` until the migration is complete.
SCHEMA_STATUS = "experimental"

#: Statuses :func:`validate` accepts. An object keeps the status it was stamped with, so an
#: ``"experimental"`` object still validates once the maintainers flip :data:`SCHEMA_STATUS` to
#: ``"stable"``; only a status outside this set is rejected.
SCHEMA_STATUSES: tuple[str, ...] = ("experimental", "stable")

#: Versions :func:`migrate` can bring to :data:`SCHEMA_VERSION`, oldest first.
SUPPORTED_VERSIONS: tuple[str, ...] = ("0.1", "1.0", "2.0")

#: Resolutions a mantispy object can be at, coarsest last. These are structural tiers, not a biological
#: ontology: ``object`` is one primary segmented object, ``well`` one physical well profile, and
#: ``aggregate`` anything coarser, whose exact unit is carried by ``uns["mantispy"]["grouped_by"]``.
RESOLUTIONS = ("object", "well", "aggregate")

#: Old (schema 1.0) resolution names :func:`migrate` translates to the current ones.
RESOLUTION_RENAMES: dict[str, str] = {"cell": "object", "perturbation": "aggregate"}

#: Allowed values of ``var["feature_kind"]``: a directly measured morphology feature, a
#: learned-embedding dimension, or a feature a mantispy transformation produced (composition,
#: signature, trajectory). The name parser only produces ``measurement``.
FEATURE_KINDS = ("measurement", "embedding", "derived")

#: Identifier columns required in ``obs``, per resolution. An object-level row must be identifiable
#: without parsing ``obs_names``: ``Metadata_ImageID`` says which field of view it came from,
#: ``Metadata_ObjectType`` names the primary object set, and ``Metadata_ObjectNumber`` is the instance
#: number within the image; the pair ``(Metadata_ImageID, Metadata_ObjectNumber)`` is unique.
REQUIRED_OBS: dict[str, tuple[str, ...]] = {
    "object": ("Metadata_Plate", "Metadata_Well", "Metadata_ImageID", "Metadata_ObjectType", "Metadata_ObjectNumber"),
    "well": ("Metadata_Plate", "Metadata_Well"),
    # A consensus profile no longer belongs to a plate or a well.
    "aggregate": (),
}

#: Columns mantispy understands but does not require. OPS entries are reserved for 0.8.
RESERVED_OBS: tuple[str, ...] = (
    "Metadata_Batch",
    "Metadata_Source",
    "Metadata_Site",
    # Canonical opaque field-of-view id (Metadata_ImageID), the source CellProfiler image number it
    # may derive from, the primary-object set name, and the object instance number within an image.
    "Metadata_ImageID",
    "Metadata_ImageNumber",
    "Metadata_ObjectType",
    "Metadata_ObjectNumber",
    "Metadata_Perturbation",
    # What the perturbation is, one of "compound", "orf", "crispr" or "untreated". Control-ness stays in
    # Metadata_Control / Metadata_Control_Type, which are orthogonal to the kind of perturbation.
    "Metadata_Perturbation_Type",
    "Metadata_Compound",
    "Metadata_Concentration",
    # The dose as the plate map wrote it, where Metadata_Concentration holds the one the well was meant to get.
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
    # Genetic-perturbation annotation written by the ORF, CRISPR and variant loaders and, for Metadata_Gene, by
    # pp.annotate_jump(kind="crispr"); the barcode fields stay reserved for optical pooled screening in 0.8.
    "Metadata_Barcode",
    "Metadata_Gene",
    "Metadata_sgRNA",
    "Metadata_Construct",
    "Metadata_Allele",
    "Metadata_BarcodeQuality",
)

#: Annotation columns mantispy writes to ``var`` and reads back, none of them required.
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

#: The core descriptors every schema-2.0 object MUST carry under ``uns["mantispy"]``.
REQUIRED_UNS: tuple[str, ...] = (
    "schema_version",
    "schema_status",
    "resolution",
    "grouped_by",
    "history",
)

#: Keys describing the object itself: the required descriptors plus the optional ones.
UNS_KEYS: tuple[str, ...] = (
    *REQUIRED_UNS,
    "channels",
    "dataset",
    "image_table",
    "params",
    "aggregated_from",
    "consensus_from",
    "truth",
)

#: Result tables mantispy writes under ``uns["mantispy"]`` and reads back.
UNS_RESULTS: tuple[str, ...] = (
    "feature_select",
    "image_qc",
    "well_qc",
    "plate_position",
    "plate_position_detection",
    "chromosome_arm",
    "map",
    "percent_replicating",
    "grit",
    "effect",
    "wasserstein",
    "consensus_weights",
    "hits",
    "empirical_fdr",
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


def default_grouped_by(resolution: str) -> list[str]:
    """The grouping a ``resolution`` implies, where it is unambiguous.

    ``object`` groups by nothing and ``well`` by plate and well. ``aggregate`` has no safe default,
    since only the caller knows which columns define one aggregate, so it raises.
    """
    if resolution == "object":
        return []
    if resolution == "well":
        return list(REQUIRED_OBS["well"])
    raise ValueError(
        "resolution='aggregate' has no default grouped_by; pass grouped_by=[...] naming the "
        "columns that define one observation, e.g. ['Metadata_Gene']"
    )


def _as_columns(value: str | Iterable[Any] | None) -> list[str]:
    """Normalise a grouped_by-like value to a list of column names.

    Handles ``None``, a single column name, and a list/tuple/array of names (an h5ad round trip brings a
    stored list back as a numpy array, so this must not rely on truthiness, which is ambiguous on arrays).
    """
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(column) for column in value]


def _recover_grouped_by(store: dict, adata: AnnData, resolution: str) -> list[str]:
    """Best-effort grouping for a migrated object: the default, else the grouping its provenance names."""
    if resolution != "aggregate":
        return default_grouped_by(resolution)
    params = store.get("params")
    consensus = params.get("consensus") if isinstance(params, dict) else None
    for source in (store.get("aggregated_from"), consensus):
        if isinstance(source, dict):
            columns = _as_columns(source.get("by"))
            if columns:
                return columns
    return ["Metadata_Perturbation"] if "Metadata_Perturbation" in adata.obs else []


#: Columns that, where present, compose :func:`make_image_id`, most significant first.
_IMAGE_ID_COLUMNS = ("Metadata_Source", "Metadata_Plate", "Metadata_Well", "Metadata_Site", "Metadata_ImageNumber")


def make_image_id(obs: pd.DataFrame) -> np.ndarray:
    """A deterministic, opaque field-of-view id built from the stable identity columns ``obs`` carries.

    Joins whichever of :data:`_IMAGE_ID_COLUMNS` are present with ``"|"``, most significant first. The
    id is globally unique only insofar as those columns are: a plate/well/site or plate/image-number
    combination identifies one field, while plate and well alone (a dataset that lost its per-field
    identity) identify a whole well. Consumers MUST treat the value as opaque and never parse it.
    """
    columns = [column for column in _IMAGE_ID_COLUMNS if column in obs]
    if not columns:
        raise ValueError(
            f"cannot mint Metadata_ImageID: obs carries none of {_IMAGE_ID_COLUMNS}. Add a source "
            "identity column (e.g. Metadata_Plate) before stamping at object resolution."
        )
    parts = [obs[column].astype(str).to_numpy() for column in columns]
    return np.array(["|".join(values) for values in zip(*parts, strict=True)], dtype=object)


def object_number_within(image_id: np.ndarray | pd.Series) -> np.ndarray:
    """A 1-based object instance number running within each image, for data that lost the source one.

    Only for a source that gives no per-object number; the pair ``(image_id, result)`` is then unique.
    """
    series = pd.Series(np.asarray(image_id, dtype=object))
    return (series.groupby(series, sort=False).cumcount() + 1).to_numpy()


def ensure_object_identity(obs: pd.DataFrame, object_type: str = "Object") -> pd.DataFrame:
    """Fill the object-resolution identity columns ``obs`` lacks, returning ``obs``.

    Mints :func:`make_image_id` where ``Metadata_ImageID`` is absent, defaults ``Metadata_ObjectType``,
    and synthesises a within-image ``Metadata_ObjectNumber`` only where the source gave none. A column
    already present is left untouched, so a reader that knows the real values (a CellProfiler export)
    keeps them; the caller passes ``object_type`` when it knows the primary object set.
    """
    if "Metadata_ImageID" not in obs:
        obs["Metadata_ImageID"] = make_image_id(obs)
    if "Metadata_ObjectType" not in obs:
        obs["Metadata_ObjectType"] = object_type
    if "Metadata_ObjectNumber" not in obs:
        obs["Metadata_ObjectNumber"] = object_number_within(obs["Metadata_ImageID"].to_numpy())
    return obs


def stamp(
    adata: AnnData,
    resolution: str | None = None,
    *,
    grouped_by: Iterable[str] | None = None,
    history: list | None = None,
) -> None:
    """Stamp ``adata`` as a schema-2.0 object: the five :data:`REQUIRED_UNS` descriptors.

    ``resolution`` defaults to the one already recorded (so re-stamping keeps it) and raises if none is.
    ``grouped_by`` defaults per :func:`default_grouped_by` (``aggregate`` has none and must be
    given), but a stale grouping is dropped when the resolution changes. ``history`` keeps an existing
    list or starts empty. Nothing is written until every value resolves, so a rejected stamp leaves the
    object untouched.
    """
    store = adata.uns.get("mantispy", {})
    prior = store.get("resolution")
    resolution = resolution if resolution is not None else prior
    if resolution is None:
        raise ValueError(
            f"no resolution to stamp; pass resolution=... (one of {RESOLUTIONS}). A mantispy object "
            "must carry an explicit resolution; it is no longer assumed to be 'object'."
        )
    if resolution not in RESOLUTIONS:
        raise ValueError(f"resolution must be one of {RESOLUTIONS}, got {resolution!r}")
    if grouped_by is None:
        kept = store.get("grouped_by") if resolution == prior else None
        grouped_by = kept if kept is not None else default_grouped_by(resolution)

    # A fresh dict assigned back, not an in-place update: on a view uns is the dict's own, so mutating it
    # writes nowhere the caller sees. Assigning routes through AnnData to the parent's dict.
    adata.uns["mantispy"] = {
        **store,
        "schema_version": SCHEMA_VERSION,
        "schema_status": SCHEMA_STATUS,
        "resolution": resolution,
        "grouped_by": list(grouped_by),
        "history": list(history) if history is not None else list(store.get("history", [])),
    }


def get_resolution(adata: AnnData, default: str | None = None) -> str:
    """Recorded resolution of ``adata``.

    Args:
        adata: The object to read the resolution from.
        default: Returned when the object carries no resolution, instead of raising. Read-only and
            plotting consumers pass ``default="object"`` to tolerate an unstamped object; callers that
            genuinely need a recorded resolution leave it ``None`` and get a clear error.

    Raises:
        ValueError: No resolution is recorded and no ``default`` was given. Schema 2.0 no longer
            assumes ``"object"``; stamp the object with :func:`stamp` (``mt.io.stamp``) first.
    """
    resolution = adata.uns.get("mantispy", {}).get("resolution")
    if resolution is None:
        if default is not None:
            return default
        raise ValueError(
            "object has no uns['mantispy']['resolution']; stamp it with mt.io.stamp(adata, "
            "resolution=...) before calling a function that needs the resolution"
        )
    return resolution


def resolution_for(columns: Iterable[str]) -> str:
    """Resolution an object grouped by ``columns`` is at.

    Args:
        columns: Columns the grouping supplies, which become the identifier columns of the result.

    Returns:
        The finest resolution whose :data:`REQUIRED_OBS` columns ``columns`` covers, so that :func:`validate` requires of the result only what it carries.

    Notes:
        ``"object"`` is never returned, since a grouping replaces the cells and requires the same columns as ``"well"`` anyway.
        A grouping finer than the well, by site or by anything else, is still per-well or finer, and it carries the plate and well columns a well-resolution object needs.
    """
    supplied = set(columns)
    for resolution in RESOLUTIONS[1:]:
        if set(REQUIRED_OBS[resolution]) <= supplied:
            return resolution
    return RESOLUTIONS[-1]


def _check_grouped_by(adata: AnnData, store: dict, resolution: str | None, report: ValidationReport) -> None:
    """Check ``uns['mantispy']['grouped_by']`` is present, well-typed and consistent with resolution."""
    if "grouped_by" not in store:
        report.errors.append("uns['mantispy']['grouped_by'] is missing")
        return
    grouped_by = store["grouped_by"]
    if isinstance(grouped_by, np.ndarray):  # h5ad round-trips a uns list as an array
        grouped_by = grouped_by.tolist()
    if not isinstance(grouped_by, list | tuple) or not all(isinstance(column, str) for column in grouped_by):
        report.errors.append(f"grouped_by must be a list of column names, got {grouped_by!r}")
        return
    if resolution == "object" and grouped_by:
        report.errors.append(f"object resolution must have grouped_by=[], got {list(grouped_by)}")
    if resolution == "aggregate" and not grouped_by:
        report.errors.append("aggregate resolution requires a non-empty grouped_by naming the grouping columns")
    missing = [column for column in grouped_by if column not in adata.obs]
    if missing:
        report.errors.append(f"grouped_by references column(s) missing from obs: {missing}")


def _check_history(store: dict, report: ValidationReport) -> None:
    """Check ``uns['mantispy']['history']`` is present and a list."""
    if "history" not in store:
        report.errors.append("uns['mantispy']['history'] is missing")
    elif not isinstance(store["history"], list | np.ndarray):
        report.errors.append(f"history must be a list, got {type(store['history']).__name__}")


def _check_feature_kind(adata: AnnData, report: ValidationReport) -> None:
    """Check every feature row carries a ``feature_kind`` from :data:`FEATURE_KINDS`.

    The column is required (:data:`REQUIRED_VAR`) and, on a valid schema-2.0 object, non-null for every
    feature: a null is as much an error as a value outside the vocabulary. A producer that builds a
    non-measurement feature declares its kind; where mantispy cannot know it (an externally built
    object), the caller declares it through ``mt.io.stamp(..., feature_kind=...)``.
    """
    if "feature_kind" not in adata.var or "is_feature" not in adata.var:
        return
    try:
        mask = feature_mask(adata, "is_feature")
    except (ValueError, TypeError) as error:
        # validate reports, it never raises: a malformed is_feature is itself the finding here.
        report.errors.append(f"var['is_feature'] is not a usable boolean flag: {error}")
        return
    kinds = as_frame(adata.var)["feature_kind"]
    bad = kinds[mask & ~kinds.isin(FEATURE_KINDS)]
    if len(bad):
        found = sorted(bad.astype(object).fillna("null").astype(str).unique())
        report.errors.append(f"var['feature_kind'] must be one of {FEATURE_KINDS} for every feature; found {found}")


def _check_object_identity(adata: AnnData, store: dict, report: ValidationReport) -> None:
    """At object resolution, the (image, object) pair identifies a row and joins to the image table.

    Runs only once the required identity columns are present (a missing one is already reported).
    """
    obs = as_frame(adata.obs)
    if not {"Metadata_ImageID", "Metadata_ObjectNumber"} <= set(obs.columns):
        return
    pair = obs[["Metadata_ImageID", "Metadata_ObjectNumber"]]
    duplicated = int(pair.duplicated().sum())
    if duplicated:
        report.errors.append(
            f"(Metadata_ImageID, Metadata_ObjectNumber) is not unique: {duplicated} duplicate object(s); "
            "the pair must identify one primary object at object resolution"
        )
    table = store.get("image_table")
    if isinstance(table, pd.DataFrame):
        # The table is keyed by Metadata_ImageID (its index), but an h5ad round trip can demote the
        # index to a column, so accept either.
        known = set(table.index.astype(str))
        if "Metadata_ImageID" in table.columns:
            known |= set(table["Metadata_ImageID"].astype(str))
        unknown = set(obs["Metadata_ImageID"].astype(str)) - known
        if unknown:
            report.errors.append(
                f"{len(unknown)} Metadata_ImageID value(s) in obs have no row in "
                "uns['mantispy']['image_table']; every object must resolve to exactly one image"
            )


def validate(adata: AnnData, *, raise_on_error: bool = False) -> ValidationReport:
    """Check ``adata`` against the mantispy schema.

    Args:
        adata: The object to check.
        raise_on_error: Raise :class:`ValueError` instead of returning a failing report.

    Returns:
        A :class:`~mantispy._core.schema.ValidationReport`; truthy when the object is valid.
    """
    report = ValidationReport()

    if adata.n_vars == 0:
        report.errors.append(
            "the object has no features, so the other checks would pass on an empty matrix. "
            "If it came from mt.io.read_profiles, check the objects= argument."
        )
        if raise_on_error and not report.ok:
            raise ValueError("\n".join(report.errors))
        return report

    store = adata.uns.get("mantispy")
    if not isinstance(store, dict):
        store = {}

    if "schema_version" not in store:
        report.errors.append("uns['mantispy']['schema_version'] is missing")
    elif store["schema_version"] != SCHEMA_VERSION:
        seen = store["schema_version"]
        remedy = (
            "read the file with mt.io.read, which migrates it to "
            if seen in SUPPORTED_VERSIONS
            else f"this build knows {SUPPORTED_VERSIONS} and cannot migrate from it to "
        )
        report.errors.append(f"schema_version {seen!r}: {remedy}{SCHEMA_VERSION!r}")

    if "schema_status" not in store:
        report.errors.append("uns['mantispy']['schema_status'] is missing")
    elif store["schema_status"] not in SCHEMA_STATUSES:
        report.errors.append(f"unknown schema_status {store['schema_status']!r}; must be one of {SCHEMA_STATUSES}")

    resolution = store.get("resolution")
    if resolution is None:
        report.errors.append(
            "uns['mantispy']['resolution'] is missing; schema 2.0 requires an explicit resolution "
            f"(one of {RESOLUTIONS})"
        )
    elif resolution not in RESOLUTIONS:
        report.errors.append(f"unknown resolution {resolution!r}; must be one of {RESOLUTIONS}")

    _check_grouped_by(adata, store, resolution, report)
    _check_history(store, report)

    if resolution in REQUIRED_OBS:
        for column in REQUIRED_OBS[resolution]:
            if column not in adata.obs:
                report.errors.append(f"obs is missing required column {column!r} (resolution {resolution!r})")
    if resolution == "object":
        _check_object_identity(adata, store, report)

    for column in REQUIRED_VAR:
        if column not in adata.var:
            report.errors.append(f"var is missing required column {column!r}")
    _check_feature_kind(adata, report)

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
        0.1 to 1.0 only updated the version stamp. 1.0 to 2.0 is a compatibility seam that fills the new
        core descriptors: the old resolution names are translated (``cell`` to ``object``,
        ``perturbation`` to ``aggregate``; a missing one reads as ``object``), ``grouped_by`` is
        recovered from the resolution and recorded provenance, and ``feature_kind`` is inferred only
        where the CellProfiler annotation proves a feature is a measurement. A feature with no such
        annotation cannot be classified from a 1.0 object, so migration raises rather than mislabel it;
        re-stamp through ``mt.io.stamp(..., feature_kind=...)`` to declare those. Pre-PyPI, correctness
        is preferred over migrating an arbitrary internal 1.0 object seamlessly.
    """
    target = adata.copy() if copy else adata
    store = target.uns.get("mantispy", {})
    seen = store.get("schema_version") if isinstance(store, dict) else None

    if seen == SCHEMA_VERSION:
        return target if copy else None
    if seen not in SUPPORTED_VERSIONS:
        raise ValueError(
            f"cannot migrate an object stamped {seen!r}; this build knows {SUPPORTED_VERSIONS}. "
            "It was probably written by a newer mantispy; upgrade to read it."
        )

    # Resolve every new descriptor and run every check that can fail BEFORE writing anything, so a
    # rejected migration leaves the object untouched: migrate(copy=False) must not half-migrate it.
    resolution = RESOLUTION_RENAMES.get(store.get("resolution"), store.get("resolution"))
    if resolution not in RESOLUTIONS:
        resolution = "object"
    grouped_by = (
        _as_columns(store["grouped_by"]) if "grouped_by" in store else _recover_grouped_by(store, target, resolution)
    )
    if resolution == "aggregate" and not grouped_by:
        raise ValueError(
            "cannot migrate: an aggregate object with no recoverable grouping. Re-stamp with "
            "mt.io.stamp(adata, resolution='aggregate', grouped_by=[...]) naming the grouping columns."
        )

    feature_kind = None
    if "feature_kind" not in target.var:
        var = as_frame(target.var)
        is_feature = feature_mask(target, "is_feature")
        has_cp = (
            var["feature_group"].notna().to_numpy() if "feature_group" in var else np.zeros(target.n_vars, dtype=bool)
        )
        unclassifiable = int((is_feature & ~has_cp).sum())
        if unclassifiable:
            raise ValueError(
                f"cannot migrate: {unclassifiable} feature(s) carry no CellProfiler annotation, so their "
                "feature_kind cannot be inferred from a 1.0 object. Re-stamp with "
                "mt.io.stamp(adata, resolution=..., feature_kind=...) to declare it (e.g. 'embedding')."
            )
        feature_kind = measurement_kind(is_feature)

    # Object identity (M2): derive the fields a 1.0 object predates. make_image_id raises clearly when
    # obs carries no source identity at all, matching migrate's "correctness over a silent guess" stance.
    image_id = object_type = object_number = None
    if resolution == "object":
        obs = as_frame(target.obs)
        image_id = obs["Metadata_ImageID"].to_numpy() if "Metadata_ImageID" in obs else make_image_id(obs)
        object_number = (
            obs["Metadata_ObjectNumber"].to_numpy()
            if "Metadata_ObjectNumber" in obs
            else object_number_within(image_id)
        )
        # The primary-object set was never recorded on a 1.0 object; "Object" is an honest, generic name.
        object_type = obs["Metadata_ObjectType"].to_numpy() if "Metadata_ObjectType" in obs else "Object"

    store = target.uns.setdefault("mantispy", {})
    store["schema_version"] = SCHEMA_VERSION
    store["schema_status"] = SCHEMA_STATUS
    store.setdefault("history", [])
    store["resolution"] = resolution
    store["grouped_by"] = grouped_by
    if feature_kind is not None:
        target.var["feature_kind"] = feature_kind
    if resolution == "object":
        target.obs["Metadata_ImageID"] = image_id
        target.obs["Metadata_ObjectType"] = object_type
        target.obs["Metadata_ObjectNumber"] = object_number

    get_logger().info("migrated an object from schema %s to %s", seen, SCHEMA_VERSION)
    return target if copy else None
