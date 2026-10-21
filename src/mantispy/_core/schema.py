"""The mantispy AnnData contract.

An object is described by three things: the ``Metadata_`` columns in ``obs`` that identify where a profile came from, the parsed annotation columns in ``var`` that say what each feature measures, and a ``uns["mantispy"]`` dict holding the core descriptors (schema version and status, the resolution, the grouping that defines one observation, and the provenance history).

Under schema 2.0 the resolution is explicit, not advisory: a mantispy object MUST carry ``uns["mantispy"]["resolution"]`` and ``uns["mantispy"]["grouped_by"]``, and functions that need them fail clearly when they are absent rather than guessing.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from .features import COLUMNS as VAR_COLUMNS
from .features import measurement_kind
from .frames import as_frame
from .logging import get_logger
from .masks import _flag_mask, feature_mask
from .plate import normalize_well

if TYPE_CHECKING:
    from anndata import AnnData

#: The current schema. 2.0 is a breaking, still-experimental redesign of the previous internal 1.0
#: schema (kept only for migration); see ``spec/morphology-schema-2.0.md``.
SCHEMA_VERSION = "2.0"

#: Versions :func:`migrate` can bring to :data:`SCHEMA_VERSION`, oldest first.
SUPPORTED_VERSIONS: tuple[str, ...] = ("0.1", "1.0", "2.0")

#: Resolutions a mantispy object can be at, coarsest last. These are structural tiers, not a biological
#: ontology: ``object`` is one primary segmented object, ``well`` one physical well profile, and
#: ``aggregate`` anything coarser, whose exact unit is carried by ``uns["mantispy"]["grouped_by"]``.
RESOLUTIONS = ("object", "well", "aggregate")

#: Old (schema 1.0) resolution names :func:`migrate` translates to the current ones.
RESOLUTION_RENAMES: dict[str, str] = {"cell": "object", "perturbation": "aggregate"}

#: The obs columns that identify one primary object, filled by :func:`ensure_object_identity`.
OBJECT_IDENTITY = ("Metadata_ImageID", "Metadata_ObjectType", "Metadata_ObjectNumber")

#: Allowed values of ``var["feature_kind"]``: a directly measured morphology feature, a
#: learned-embedding dimension, or a feature a mantispy transformation produced (composition,
#: signature, trajectory). The name parser only produces ``measurement``.
FEATURE_KINDS = ("measurement", "embedding", "derived")

#: Allowed values of ``obs["Metadata_Control_Type"]`` (spec 10.2). ``negcon``, ``poscon`` and ``empty``
#: are controls; ``treatment`` is not. ``Metadata_Control`` is the boolean ``type != "treatment"``.
CONTROL_TYPES = ("negcon", "poscon", "treatment", "empty")

#: Concentration units mantispy recognises for dose-response work (spec 11.2). The list is a recommendation,
#: not a closed vocabulary: any single, explicit unit is accepted; what is forbidden is assuming one.
CONCENTRATION_UNITS = ("M", "mM", "uM", "nM", "pM")

#: Identifier columns required in ``obs``, per resolution. An object-level row must be identifiable
#: without parsing ``obs_names``; the pair ``(Metadata_ImageID, Metadata_ObjectNumber)`` is unique.
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
    # The source object number migrate keeps when it had to renumber colliding 1.0 objects.
    "Metadata_SourceObjectNumber",
    "Metadata_Perturbation",
    # What the perturbation is, one of "compound", "orf", "crispr" or "untreated". Control-ness stays in
    # Metadata_Control / Metadata_Control_Type, which are orthogonal to the kind of perturbation.
    "Metadata_Perturbation_Type",
    "Metadata_Compound",
    "Metadata_Concentration",
    # The explicit unit of Metadata_Concentration (M, mM, uM, nM, pM); required whenever a concentration is stored.
    "Metadata_ConcentrationUnit",
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
    "membership",
    "truth",
    "embedding",
    "perturbation",
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
    "composition_dispersion",
    "subpopulation_hits",
    "replicate_saturation",
    "cytotoxicity",
    "pathway_coherence",
    "enrich_hits",
    "differential",
    "transport",
    "ora",
    "gene_aggregation",
    "network_enrichment",
    "dose_features",
    "dose_direction",
    "cluster",
    "sphere_epsilon",
)

#: Suffixes of a companion table written beside a result (e.g. ``effect_groups`` beside ``effect``). A key
#: ending in one of these is registered when its base (the part before the suffix) is a registered result.
#: Companions that are registered in their own right (``edistance_pairwise``, ``moa_confusion``) are not
#: listed here, so each companion has a single registration.
_RESULT_COMPANION_SUFFIXES: tuple[str, ...] = ("_groups", "_summary", "_units", "_prior_df", "_linkage")


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
    block = obs[columns]
    if block.isna().to_numpy().any():
        # astype(str) would otherwise stringify a NaN into the id, so refuse clearly rather than build a
        # "nan"-bearing identifier that silently collides or breaks the join.
        incomplete = [column for column in columns if block[column].isna().any()]
        raise ValueError(
            f"cannot mint Metadata_ImageID: {incomplete} contain missing values; fill them before "
            "stamping at object resolution."
        )
    rows = block.astype(str).to_numpy()
    return np.array(["|".join(row) for row in rows], dtype=object)


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


#: Where perturbation identity lives unless an object records otherwise.
PERTURBATION_COLUMN = "Metadata_Perturbation"
CONTROL_COLUMN = "Metadata_Control_Type"


def _perturbation_record(adata: AnnData, store: dict, perturbation: Mapping[str, str] | None) -> dict[str, str] | None:
    """The ``perturbation`` record :func:`stamp` should write, or ``None`` to leave the stored one alone.

    An explicit record must name an existing ``obs`` column. Without one, an object that records none
    gets the default columns when ``obs`` has the default perturbation column.
    """
    if perturbation is not None:
        unknown = set(perturbation) - {"column", "control_column"}
        if unknown or not isinstance(perturbation.get("column"), str):
            raise ValueError("perturbation must be a mapping with a string 'column' and optional 'control_column'")
        record = {
            "column": perturbation["column"],
            "control_column": perturbation.get("control_column", CONTROL_COLUMN),
        }
        if record["column"] not in adata.obs:
            raise ValueError(f"perturbation column {record['column']!r} is not a column of obs")
        control = perturbation.get("control_column")
        if control is not None and control not in adata.obs:
            raise ValueError(f"perturbation control_column {control!r} is not a column of obs")
        return record
    if "perturbation" not in store and PERTURBATION_COLUMN in adata.obs:
        return {"column": PERTURBATION_COLUMN, "control_column": CONTROL_COLUMN}
    return None


def _recorded_column(adata: AnnData, field: str, default: str) -> str:
    store = adata.uns.get("mantispy")
    record = store.get("perturbation") if isinstance(store, dict) else None
    value = record.get(field) if isinstance(record, dict) else None
    return value if isinstance(value, str) and value in adata.obs else default


def resolve_perturbation_column(adata: AnnData, column: str | None = None) -> str:
    """The obs column holding perturbation identity: ``column`` if given, else the recorded one, else the default."""
    return column if column is not None else _recorded_column(adata, "column", PERTURBATION_COLUMN)


def record_perturbation(adata: AnnData) -> None:
    """Record the default perturbation columns on ``adata`` when ``obs`` carries ``Metadata_Perturbation``."""
    store = adata.uns.get("mantispy")
    store = store if isinstance(store, dict) else {}
    if PERTURBATION_COLUMN in adata.obs:
        adata.uns["mantispy"] = {
            **store,
            "perturbation": {"column": PERTURBATION_COLUMN, "control_column": CONTROL_COLUMN},
        }


def resolve_control_column(adata: AnnData) -> str:
    """The obs column holding the control type: the recorded one when it exists, else ``Metadata_Control_Type``."""
    return _recorded_column(adata, "control_column", CONTROL_COLUMN)


def stamp(
    adata: AnnData,
    resolution: str | None = None,
    *,
    grouped_by: Iterable[str] | None = None,
    history: list | None = None,
    perturbation: Mapping[str, str] | None = None,
) -> None:
    """Stamp ``adata`` as a schema-2.0 object: the five :data:`REQUIRED_UNS` descriptors.

    ``resolution`` defaults to the one already recorded (so re-stamping keeps it) and raises if none is.
    ``grouped_by`` defaults per :func:`default_grouped_by` (``aggregate`` has none and must be
    given), but a stale grouping is dropped when the resolution changes. ``history`` keeps an existing
    list or starts empty. ``perturbation`` names the ``obs`` columns holding perturbation identity
    (``column``) and control type (``control_column``); unset, an object that records none gets the
    defaults when ``obs`` has ``Metadata_Perturbation``. Nothing is written until every value resolves, so a rejected stamp leaves the
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

    entries = list(history) if history is not None else list(store.get("history", []))
    if not entries:
        # Seed once, on the first stamp; later re-stamps (tools re-stamp constantly) leave the log alone.
        from mantispy._core.provenance import history_record

        entries.append(history_record("stamp", params={"resolution": resolution, "grouped_by": list(grouped_by)}))

    recorded = _perturbation_record(adata, store, perturbation)

    # A fresh dict assigned back, not an in-place update: on a view uns is the dict's own, so mutating it
    # writes nowhere the caller sees. Assigning routes through AnnData to the parent's dict.
    adata.uns["mantispy"] = {
        **store,
        "schema_version": SCHEMA_VERSION,
        "resolution": resolution,
        "grouped_by": list(grouped_by),
        "history": entries,
        **({"perturbation": recorded} if recorded is not None else {}),
    }
    # Every stamped object carries its well geometry: fill Metadata_Row/Col from the well when it has one.
    from mantispy._core.plate import fill_well_grid

    fill_well_grid(as_frame(adata.obs))


def recorded_resolution(adata: AnnData) -> str | None:
    """The resolution ``adata`` records, in current terms, or ``None`` when it records none.

    An object written by an earlier mantispy and loaded without :func:`migrate` (``anndata.read_h5ad``)
    still carries the old names, so ``cell`` and ``perturbation`` are translated rather than read as some
    other resolution. A value that is no resolution at all raises.
    """
    store = adata.uns.get("mantispy")
    value = store.get("resolution") if isinstance(store, dict) else None
    if value is None:
        return None
    value = RESOLUTION_RENAMES.get(str(value), str(value))
    if value not in RESOLUTIONS:
        raise ValueError(
            f"unknown resolution {value!r} in uns['mantispy']; must be one of {RESOLUTIONS}. Read files with "
            "mt.io.read, which migrates an object from an earlier mantispy."
        )
    return value


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
    resolution = recorded_resolution(adata)
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


def _check_result_keys(store: dict, report: ValidationReport) -> None:
    """Warn on a persistent result key under ``uns['mantispy']`` that is not registered (spec §15.3).

    A key is fine when it is a core descriptor (:data:`UNS_KEYS`), a registered result
    (:data:`UNS_RESULTS`), or a companion of a registered result (a registered base plus a
    :data:`_RESULT_COMPANION_SUFFIXES` suffix, such as ``effect_groups``). Anything else is a result the
    package does not know about: usually a new table whose key was never added to :data:`UNS_RESULTS`, or a
    caller's custom ``key_added``. It is a warning, not an error, because a custom ``key_added`` is allowed;
    the package's own default keys must all be registered, which a test enforces.
    """
    results = set(UNS_RESULTS)
    known = set(UNS_KEYS) | results
    for key in store:
        if not isinstance(key, str) or key in known:
            continue
        if any(key.endswith(suffix) and key.removesuffix(suffix) in results for suffix in _RESULT_COMPANION_SUFFIXES):
            continue
        report.warnings.append(
            f"uns['mantispy'][{key!r}] is not a registered result key (not a recognized mantispy result); "
            "ignore this if it is your own custom key_added"
        )


def _check_embedding_provenance(store: dict, report: ValidationReport) -> None:
    """Warn when the optional ``uns['mantispy']['embedding']`` record has no string ``model``."""
    if "embedding" not in store:
        return
    record = store["embedding"]
    if not isinstance(record, dict) or not isinstance(record.get("model"), str):
        report.warnings.append("uns['mantispy']['embedding'] should be a mapping with a string 'model'")


def _check_perturbation_record(adata: AnnData, store: dict, report: ValidationReport) -> None:
    """Suggest recording the perturbation column, and warn when a recorded one is malformed or missing."""
    if "perturbation" not in store:
        if PERTURBATION_COLUMN in adata.obs:
            report.warnings.append(
                f"obs has {PERTURBATION_COLUMN!r} but the object does not record which column identifies "
                "perturbations; mt.io.stamp(adata, perturbation={'column': ...}) records it for the tools"
            )
        return
    record = store["perturbation"]
    if not isinstance(record, dict) or not isinstance(record.get("column"), str):
        report.warnings.append("the recorded perturbation should be a mapping with a string 'column'")
    elif record["column"] not in adata.obs:
        report.warnings.append(f"the recorded perturbation column {record['column']!r} is not a column of obs")
    elif isinstance(record.get("control_column"), str) and record["control_column"] not in adata.obs:
        report.warnings.append(
            f"the recorded perturbation control_column {record['control_column']!r} is not a column of obs"
        )


def _check_obs_writable(adata: AnnData, report: ValidationReport) -> None:
    """Error on an object-dtype obs column holding list, tuple, set, dict or array cells.

    Such a column cannot be written to h5ad, so catching it here fails early instead of at write time.
    """
    obs = as_frame(adata.obs)
    for column in obs.columns:
        series = obs[column]
        if series.dtype != object:  # ponytail: only object dtype can hold containers; every value is checked
            continue
        containers = series.dropna().map(lambda value: isinstance(value, list | tuple | set | dict | np.ndarray))
        if containers.any():
            kind = type(series.dropna()[containers].iloc[0]).__name__
            report.errors.append(
                f"obs column {column!r} holds {kind} cells, which cannot be written to h5ad; "
                "convert it to a scalar column"
            )


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


def _check_controls(adata: AnnData, report: ValidationReport) -> None:
    """Check the control fields against spec 10: co-presence, vocabulary, and the boolean relationship.

    The control fields are conditional, not part of the minimum schema, so a check runs only once one of
    them is present. ``Metadata_Control_Type`` MUST accompany ``Metadata_Control`` (and the reverse), its
    values MUST come from :data:`CONTROL_TYPES`, and ``Metadata_Control`` MUST equal
    ``Metadata_Control_Type != "treatment"`` row by row.
    """
    obs = as_frame(adata.obs)
    has_flag = "Metadata_Control" in obs
    has_type = "Metadata_Control_Type" in obs
    if not (has_flag or has_type):
        return
    if has_flag != has_type:
        missing = "Metadata_Control_Type" if has_flag else "Metadata_Control"
        present = "Metadata_Control" if has_flag else "Metadata_Control_Type"
        report.errors.append(
            f"obs has {present!r} but not {missing!r}; spec 10.2 requires both. "
            "Run mt.pp.annotate_controls, which writes the pair."
        )
        return

    # Missing values are counted on their own: under pandas 3 astype(str) keeps them as NaN, which cannot be
    # sorted alongside the labels.
    raw = obs["Metadata_Control_Type"]
    n_missing = int(raw.isna().sum())
    if n_missing:
        report.errors.append(f"Metadata_Control_Type has {n_missing} missing value(s); every row needs a control class")
    types = raw.astype(object).where(raw.notna(), "").astype(str)
    bad = sorted(set(types[raw.notna()].unique()) - set(CONTROL_TYPES))
    if bad:
        report.errors.append(f"Metadata_Control_Type must be one of {CONTROL_TYPES}; found {bad}")

    flag = obs["Metadata_Control"]
    try:
        is_control = _flag_mask(pd.Series(flag), "obs['Metadata_Control']", "a control row", "")
    except (ValueError, TypeError) as error:
        report.errors.append(f"obs['Metadata_Control'] is not a usable boolean flag: {error}")
        return
    expected = (types != "treatment").to_numpy()
    mismatch = int(((is_control != expected) & raw.notna().to_numpy()).sum())
    if mismatch:
        report.errors.append(
            f"Metadata_Control disagrees with Metadata_Control_Type on {mismatch} row(s); "
            'spec 10.2 fixes Metadata_Control == (Metadata_Control_Type != "treatment")'
        )


#: A delimiter-joined perturbation label: ``;`` or ``|`` between word characters, or a comma between two
#: letter runs (so ``1,2-dichloroethane`` does not match).
_COMPOSITE_LABEL = r"\w[;|]\w|[A-Za-z]{2},\s?[A-Za-z]{2}"


def _check_composite_perturbation(adata: AnnData, report: ValidationReport) -> None:
    """Warn when ``Metadata_Perturbation`` holds delimiter-joined labels such as ``"MKX;GENE2"``.

    This is a heuristic flag for a silent mislabel, not a representation: a joined label is treated as one
    perturbation by every grouping and test.
    """
    # Full composite-perturbation handling awaits the finding-1.1 (multi-membership) schema decision.
    obs = as_frame(adata.obs)
    if "Metadata_Perturbation" not in obs:
        return
    labels = obs["Metadata_Perturbation"].dropna().astype(str)
    if labels.empty:
        return
    composite = labels.str.contains(_COMPOSITE_LABEL, regex=True)
    if composite.mean() >= 0.05:
        report.warnings.append(
            f"Metadata_Perturbation has {int(composite.sum())} composite value(s) joined by a delimiter "
            f"(e.g. {labels[composite].iloc[0]!r}); each is treated as one perturbation, not several"
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
    _check_result_keys(store, report)
    _check_embedding_provenance(store, report)
    _check_perturbation_record(adata, store, report)
    _check_obs_writable(adata, report)

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
        wells = pd.Series(adata.obs["Metadata_Well"])
        if wells.isna().any():
            report.errors.append(f"Metadata_Well has {int(wells.isna().sum())} missing value(s)")
        unparsable = []
        for well in wells.dropna().astype(str).unique():
            try:
                normalize_well(well)
            except ValueError:
                unparsable.append(well)
        if unparsable:
            report.errors.append(f"Metadata_Well has unparsable values: {sorted(unparsable)[:5]}")

    _check_controls(adata, report)
    _check_composite_perturbation(adata, report)

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


#: How a 1.0 ``Metadata_Control_Type`` label reads under spec 10. 1.0 wrote ``""`` for a treatment
#: (``annotate_controls(poscon=...)``) and JUMP's ``"trt"``; any other label is a dataset's own and falls back
#: to the 1.0 boolean, which meant negative control.
_LEGACY_CONTROL_TYPES: dict[str, str] = {
    "negcon": "negcon",
    "poscon": "poscon",
    "empty": "empty",
    "treatment": "treatment",
    "trt": "treatment",
    "": "treatment",
}
_LEGACY_TRUE = frozenset({"true", "1", "1.0", "yes", "y", "t"})
_LEGACY_FALSE = frozenset({"false", "0", "0.0", "no", "n", "f", ""})


def _legacy_control_flag(values: pd.Series) -> np.ndarray:
    """Read a 1.0 ``Metadata_Control`` boolean, whatever spelling a platemap gave it.

    A missing value reads as ``False``, as a 1.0 platemap that left a well unset meant. A value that is
    neither a recognisable true nor false raises, since guessing would rewrite the controls irrecoverably.
    """
    if pd.api.types.is_bool_dtype(values):
        return values.to_numpy(dtype=bool)
    text = values.astype(object).where(values.notna(), "").astype(str).str.strip().str.lower()
    unknown = sorted(set(text) - _LEGACY_TRUE - _LEGACY_FALSE)
    if unknown:
        raise ValueError(
            f"cannot migrate: Metadata_Control holds {unknown[:5]}, which read as neither true nor false. "
            "Recode the column to booleans before reading the file."
        )
    return text.isin(_LEGACY_TRUE).to_numpy()


def _legacy_controls(obs: pd.DataFrame) -> tuple[np.ndarray, np.ndarray] | None:
    """Translate a 1.0 control annotation to the spec-10 pair ``(Metadata_Control_Type, Metadata_Control)``.

    Under 1.0 ``Metadata_Control`` meant negative control, and ``Metadata_Control_Type``, where a producer
    wrote one, held either the classes above or a dataset's own labels. Known labels map directly; any other
    label takes its class from the 1.0 flag. The flag is then rewritten as ``type != "treatment"``, so it
    marks every control rather than only the negative ones. Returns ``None`` when there is nothing to translate.
    """
    has_flag, has_type = "Metadata_Control" in obs, "Metadata_Control_Type" in obs
    if not (has_flag or has_type):
        return None
    flag = _legacy_control_flag(pd.Series(obs["Metadata_Control"])) if has_flag else None
    if has_type:
        labels = obs["Metadata_Control_Type"].astype(object).where(obs["Metadata_Control_Type"].notna(), "")
        labels = labels.astype(str).str.strip()
        mapped = labels.map(_LEGACY_CONTROL_TYPES)
        unknown = mapped.isna().to_numpy()
        if unknown.any():
            if flag is None:
                raise ValueError(
                    f"cannot migrate: Metadata_Control_Type holds {sorted(set(labels[unknown]))[:5]}, which are not "
                    f"control classes {CONTROL_TYPES}, and there is no Metadata_Control to read them from."
                )
            get_logger().info(
                "Metadata_Control_Type labels %s are not control classes; they take their class from the 1.0 "
                "Metadata_Control flag (negcon where it was set, treatment otherwise)",
                sorted(set(labels[unknown]))[:5],
            )
            mapped[unknown] = np.where(flag[unknown], "negcon", "treatment")
        types = mapped.to_numpy(dtype=object)
    else:
        assert flag is not None  # one of the two columns is present
        types = np.where(flag, "negcon", "treatment").astype(object)
    return types, types != "treatment"


def _resolve_colliding_objects(identity: pd.DataFrame) -> None:
    """Make ``(Metadata_ImageID, Metadata_ObjectNumber)`` unique on a migrating 1.0 object, in place.

    A 1.0 object can carry per-image object numbers while the image id minted from its columns is coarser
    than the field, so distinct objects collide. The source numbers are kept wherever they can be: where
    ``region`` (the SpatialData label element) separates the fields, it refines the id and the numbers stay
    the label instances; where the id already resolves single fields (site or image number), the collision
    is a genuine duplicate and is refused rather than hidden. Only an id with no field identity at all is
    renumbered within it, with the source numbers kept in ``Metadata_SourceObjectNumber``.
    """
    numbers = identity["Metadata_ObjectNumber"]
    if "region" in identity and not identity[["region", "Metadata_ObjectNumber"]].duplicated().any():
        identity["Metadata_ImageID"] = identity["Metadata_ImageID"].astype(str) + "|" + identity["region"].astype(str)
        if not identity[["Metadata_ImageID", "Metadata_ObjectNumber"]].duplicated().any():
            return
    if {"Metadata_Site", "Metadata_ImageNumber"} & set(identity.columns):
        duplicated = int(identity[["Metadata_ImageID", "Metadata_ObjectNumber"]].duplicated().sum())
        raise ValueError(
            f"cannot migrate: {duplicated} object(s) repeat a (Metadata_ImageID, Metadata_ObjectNumber) pair although "
            "the image id resolves single fields, so they are duplicate rows rather than distinct objects. "
            "Remove the duplicates before reading the file."
        )
    get_logger().warning(
        "Metadata_ObjectNumber repeats within the image ids this object's columns allow (no site or image number), "
        "so objects are renumbered within each image; the source numbers are kept in Metadata_SourceObjectNumber"
    )
    identity["Metadata_SourceObjectNumber"] = numbers.to_numpy()
    identity["Metadata_ObjectNumber"] = object_number_within(identity["Metadata_ImageID"].to_numpy())


def set_control_columns(obs: pd.DataFrame, is_control: Any) -> pd.DataFrame:
    """Write the control pair (spec §10) into ``obs`` from one boolean control mask.

    Sets ``Metadata_Control`` to the mask and ``Metadata_Control_Type`` to the ``negcon``/``treatment`` it
    implies, so the invariant ``Metadata_Control == (Metadata_Control_Type != "treatment")`` holds by
    construction. For a negcon/treatment layout (no positive controls); a dataset with poscons sets the
    type column itself. Returns ``obs`` for chaining.
    """
    mask = np.asarray(is_control, dtype=bool)
    obs["Metadata_Control"] = mask
    obs["Metadata_Control_Type"] = np.where(mask, "negcon", "treatment")
    return obs


def require_concentration_unit(
    obs: pd.DataFrame,
    dose_key: str = "Metadata_Concentration",
    unit_key: str = "Metadata_ConcentrationUnit",
) -> str | None:
    """Raise unless a stored concentration carries a single, explicit unit (spec §11.2).

    A dose-response function calls this before reading ``dose_key``: mantispy MUST NOT assume a unit, so a
    concentration with no unit, or with more than one unit across the rows, is an error the caller must
    fix. Returns the unit (for reporting) when one consistent unit is present, or ``None`` when there is no
    concentration column to check. The unit is not restricted to :data:`CONCENTRATION_UNITS`; any single
    explicit value is accepted.
    """
    if dose_key not in obs.columns:
        return None
    if unit_key not in obs.columns:
        raise ValueError(
            f"obs has {dose_key!r} but no {unit_key!r}; a dose-response function needs an explicit "
            f"concentration unit (e.g. one of {CONCENTRATION_UNITS}). mantispy does not assume a unit."
        )
    # A missing unit on a dosed row is an error, not something to drop: dropping it would silently assume a
    # unit for that row. Check for NaN explicitly before astype(str) — under pandas 3 a NaN survives astype
    # as a float, so a {"nan"} string filter would miss it and sorted() would raise TypeError on a mixed set.
    present = obs.loc[obs[dose_key].notna(), unit_key]
    text = present.astype(str)
    blank = present.isna() | text.isin(["", "nan"])
    if blank.all():
        raise ValueError(
            f"{unit_key!r} is empty for every row that has a {dose_key!r}; set an explicit concentration unit."
        )
    if blank.any():
        raise ValueError(
            f"{unit_key!r} is missing for some rows that have a {dose_key!r}; every dosed row needs an "
            "explicit unit, since mantispy does not assume one."
        )
    units = sorted(set(text))
    if len(units) > 1:
        raise ValueError(
            f"{unit_key!r} has more than one unit {units}; a dose-response function requires a single "
            "consistent unit. Convert the concentrations to one unit first."
        )
    return units[0]


def _rekey_legacy_image_table(store: dict, identity: pd.DataFrame) -> None:
    """Re-key a 1.0 ``image_table`` from the source image number to ``Metadata_ImageID`` (spec 8).

    A 1.0 export keyed the table by the integer CellProfiler image number; under 2.0 obs joins to it
    through ``Metadata_ImageID``, so the two would otherwise be disjoint. Re-keys when obs still carries
    ``Metadata_ImageNumber`` to map through. A field no object came from (segmentation found nothing) has
    no object to map through, so its id is minted from the table's own identity columns the way obs mints
    it, or else kept as its image number; such a row joins to no object either way.
    """
    table = store.get("image_table")
    if not isinstance(table, pd.DataFrame) or table.index.name == "Metadata_ImageID":
        return
    if "Metadata_ImageNumber" not in identity:
        return
    numbers = pd.Series(table.index.astype(str), index=table.index)
    mapping = (
        identity.assign(_key=identity["Metadata_ImageNumber"].astype(str)).groupby("_key")["Metadata_ImageID"].first()
    )
    ids = numbers.map(mapping)
    unmapped = ids.isna().to_numpy()
    if unmapped.any():
        frame = table.assign(Metadata_ImageNumber=table.index.to_numpy())
        columns = [column for column in _IMAGE_ID_COLUMNS if column in identity]
        usable = all(column in frame for column in columns) and not frame.loc[unmapped, columns].isna().to_numpy().any()
        ids[unmapped] = make_image_id(frame.loc[unmapped, columns]) if usable else numbers[unmapped].to_numpy()
    store["image_table"] = table.set_axis(pd.Index(ids.astype(str).to_numpy(), name="Metadata_ImageID"))


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

    # Object identity (M2): derive the columns a 1.0 object predates. ensure_object_identity mints the
    # opaque image id on a copy now (raising clearly when obs carries no source identity, matching
    # migrate's "correctness over a silent guess" stance) so a rejected migration writes nothing; the
    # primary-object set was never recorded on a 1.0 object, so it defaults to the generic "Object".
    identity = ensure_object_identity(as_frame(target.obs).copy()) if resolution == "object" else None
    controls = _legacy_controls(as_frame(target.obs))
    if identity is not None and identity[["Metadata_ImageID", "Metadata_ObjectNumber"]].duplicated().any():
        _resolve_colliding_objects(identity)

    store = target.uns.setdefault("mantispy", {})
    store["schema_version"] = SCHEMA_VERSION
    store.setdefault("history", [])
    store["resolution"] = resolution
    store["grouped_by"] = grouped_by
    if feature_kind is not None:
        target.var["feature_kind"] = feature_kind
    if identity is not None:
        for column in (*OBJECT_IDENTITY, "Metadata_SourceObjectNumber"):
            if column in identity:
                target.obs[column] = identity[column].to_numpy()
        _rekey_legacy_image_table(store, identity)
    if controls is not None:
        # Spec 10 widens the boolean to any control and moves the class into Metadata_Control_Type.
        target.obs["Metadata_Control_Type"] = controls[0]
        target.obs["Metadata_Control"] = controls[1]

    get_logger().info("migrated an object from schema %s to %s", seen, SCHEMA_VERSION)
    return target if copy else None
