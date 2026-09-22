"""Read and write profile tables.

The design of :func:`read_profiles` follows ``scverse/cell-painting-io`` (MIT), whose readers were developed against 44 Cell Painting Gallery accessions.
What differs between real datasets is a parameter here rather than an assumption: metadata prefixes, missing-value sentinels, columns that disagree between files, and metadata that exists only in the directory name.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from pathlib import Path
from typing import Literal

import anndata as ad
import numpy as np
import pandas as pd

from mantispy._core.features import _infer_channels, parse_feature_names
from mantispy._core.frames import as_frame, categorize_metadata
from mantispy._core.logging import get_logger, report_drop
from mantispy._core.plate import normalize_well
from mantispy._core.provenance import record_params
from mantispy._core.schema import (
    REQUIRED_OBS,
    RESOLUTIONS,
    SCHEMA_VERSION,
    SUPPORTED_VERSIONS,
    migrate,
    validate,
)
from mantispy._core.schema import stamp as _record
from mantispy.io._cellprofiler import export_prefix, read_export

#: Column-name prefixes that mark metadata. Real accessions use all four.
METADATA_PREFIXES: tuple[str, ...] = ("Image_Metadata_", "Metadata_", "metadata_", "meta_")

#: When fewer features than this survive the default object filter, the drop is logged as a warning instead of at info level.
THIN_FEATURE_SET = 10

#: CellProfiler objects treated as features by default: the three compartments, as in ``pycytominer.infer_cp_features``.
#: ``Image`` is excluded because its measurements are whole-field, not per-cell.
DEFAULT_OBJECTS: tuple[str, ...] = ("Cells", "Cytoplasm", "Nuclei")

#: pycytominer's per-well counts and the names mantispy reads them under, the first one present winning.
#: ``Metadata_Object_Count``, the cells it aggregated, equals ``Metadata_Count_Cells`` wherever both are published.
#: ``Metadata_Site_Count`` counts the fields of view that contributed cells, as :func:`mantispy.tl.aggregate` does, which need not be all those imaged.
_UPSTREAM_COUNTS = {
    "Metadata_Count_Cells": "Metadata_CellCount",
    "Metadata_Object_Count": "Metadata_CellCount",
    "Metadata_Site_Count": "Metadata_SiteCount",
}


def _read_frame(path: Path) -> pd.DataFrame:
    if path.suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if path.suffix in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t")
    return pd.read_csv(path)


def _adopt_counts(obs: pd.DataFrame) -> pd.DataFrame:
    """Copy pycytominer's per-well counts in `obs` to the names mantispy reads, keeping the originals, and return it."""
    for source, target in _UPSTREAM_COUNTS.items():
        if source in obs and target not in obs:
            obs[target] = obs[source].to_numpy(dtype=float)
    return obs


def _strip_prefix(name: str, prefixes: Sequence[str]) -> str:
    for prefix in sorted(prefixes, key=len, reverse=True):
        if name.startswith(prefix):
            return f"Metadata_{name[len(prefix) :]}"
    return name


def _stack(files: list[Path], on_column_mismatch: str) -> tuple[pd.DataFrame, list[int], list[Path]]:
    frames = [_read_frame(path) for path in files]
    # An empty file reads as all-object columns and would turn the other frames' columns to object in concat, so empty files are dropped first.
    kept = [index for index, frame in enumerate(frames) if len(frame)]
    if kept and len(kept) < len(frames):
        files = [files[index] for index in kept]
        frames = [frames[index] for index in kept]

    if len({tuple(frame.columns) for frame in frames}) > 1:
        if on_column_mismatch == "raise":
            raise ValueError(
                "profile files disagree on columns; pass on_column_mismatch='intersect' to keep "
                "the shared ones (batches with disjoint feature sets are common)"
            )
        shared = set.intersection(*(set(frame.columns) for frame in frames))
        if not shared:
            raise ValueError("profile files share no columns")
        order = [column for column in frames[0].columns if column in shared]
        frames = [frame[order] for frame in frames]

    frame = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
    return frame, [len(f) for f in frames], files


def from_dataframe(
    df: pd.DataFrame,
    metadata_prefixes: Sequence[str] = METADATA_PREFIXES,
    metadata_columns: Sequence[str] = (),
    channels: Sequence[str] | None = None,
    sentinels: float | Collection[float] | None = None,
    keep_non_features: bool = False,
    objects: Sequence[str] | None = DEFAULT_OBJECTS,
    resolution: str = "well",
) -> ad.AnnData:
    """Build an AnnData from a wide profile table.

    Numeric columns that parse as features become ``X``; text and ``Metadata_`` columns become ``obs``.
    Numeric columns that parse as non-features (object numbers, parent links, centroids) are dropped and logged by default, because published profile tables often carry dozens of them and they mean nothing at well level.

    Args:
        df: The table to convert.
        metadata_prefixes: Prefixes marking metadata columns.
            The matching prefix is normalized to ``Metadata_``.
        metadata_columns: Columns to treat as metadata even though they are numeric and unprefixed.
        channels: Channel vocabulary passed to the feature-name parser.
        sentinels: Values in the feature matrix that stand for missing, replaced with NaN.
        keep_non_features: Keep those dropped numeric non-feature columns in ``obs`` instead.
        objects: Which CellProfiler objects count as features.
            The default is the three compartments, matching ``pycytominer.infer_cp_features``.
            ``None`` keeps every object, including the whole-field ``Image`` measurements.
        resolution: Resolution to record.
            Profile tables are usually well-level.

    Returns:
        An :class:`~anndata.AnnData` of observations by features at the recorded resolution, with the parsed feature annotation in ``var`` and the schema stamp, the resolution and the channel vocabulary it parsed with in ``uns["mantispy"]``.
        pycytominer's per-well ``Metadata_Count_Cells``, or ``Metadata_Object_Count`` without it, and ``Metadata_Site_Count`` are copied to ``Metadata_CellCount`` and ``Metadata_SiteCount``.

    Raises:
        ValueError: `df` has no rows, no column parses as a feature on `objects`, or two metadata prefixes normalize onto the same column name.
        KeyError: A name in `metadata_columns` is not in `df`.
    """
    if len(df) == 0:
        raise ValueError(
            f"the table has {len(df.columns)} column(s) and no rows. A column holding no values has no "
            "dtype to be recognised as a feature by, so this would read as an empty object with every "
            "feature column filed into obs. Check that the file holds more than a header."
        )
    prefixes = tuple(metadata_prefixes)
    named = set(metadata_columns)
    if missing := named - set(df.columns):
        raise KeyError(f"metadata columns not in data: {sorted(missing)}")

    numeric = set(df.select_dtypes("number").columns)
    meta_columns = [c for c in df.columns if c.startswith(prefixes) or c in named or c not in numeric]
    candidates = [c for c in df.columns if c not in set(meta_columns)]

    # Resolve the vocabulary here so the object can record what it was parsed with.
    # Inference reads channels off the columns it is given, so a subset of a plate can infer a smaller vocabulary and parse a column differently: on a six-column sample `Cells_Correlation_Correlation_AGP_DNA` parses as channel 'DNA', feature 'Correlation_AGP', where the full plate gives channel 'AGP|DNA', feature 'Correlation'.
    vocabulary = list(channels) if channels is not None else sorted(_infer_channels(candidates))
    parsed = parse_feature_names(candidates, channels=vocabulary)
    if channels is None and vocabulary:
        get_logger().info(
            "inferred the channel vocabulary %s from the column names; pass channels= to fix it, "
            "since a different set of columns can infer a different vocabulary",
            vocabulary,
        )
    keep = parsed["is_feature"].to_numpy()
    if objects is not None:
        wanted = set(objects)
        excluded = ~parsed["object"].isin(wanted).to_numpy()
        if (keep & excluded).any():
            report_drop(
                f"feature(s) measured on {sorted(set(parsed['object'][keep & excluded].astype(str)))}",
                int((keep & excluded).sum()),
                int(keep.sum()),
                remedy="pass objects=None to keep every object",
                # Warn when too few features remain, not when a large fraction is dropped.
                # This filter is the default, and on a small export the whole-field Image_ columns often outnumber the per-cell ones.
                escalate=int((keep & ~excluded).sum()) < THIN_FEATURE_SET,
            )
        keep = keep & ~excluded
    feature_names = parsed.index[keep].tolist()

    if not feature_names and candidates:
        objects_seen = sorted(set(parsed["object"].dropna().astype(str)))
        raise ValueError(
            f"none of the {len(candidates)} numeric column(s) parsed as a feature on "
            f"{list(objects) if objects is not None else 'any object'}"
            + (f"; the columns name these objects instead: {objects_seen}" if objects_seen else "")
            + ". Pass objects=None to keep every object, objects=(...) to name yours, or check that the "
            "columns follow <Object>_<FeatureGroup>_<Feature>_<Channel>."
        )

    discarded = [c for c in candidates if c not in set(feature_names)]
    if keep_non_features:
        meta_columns += discarded
    elif discarded:
        get_logger().info(
            "dropped %d numeric non-feature column(s) such as %s; pass keep_non_features=True to keep them",
            len(discarded),
            ", ".join(discarded[:3]),
        )

    X = df[feature_names].to_numpy(dtype=np.float32)
    if sentinels is not None:
        # to_numpy can hand back a read-only view of a single-dtype block.
        values = [sentinels] if isinstance(sentinels, int | float) else list(sentinels)
        X = np.where(np.isin(X, np.asarray(values, dtype=X.dtype)), np.float32(np.nan), X)

    obs = df[meta_columns].rename(columns=lambda c: _strip_prefix(c, prefixes)).reset_index(drop=True)
    if obs.columns.has_duplicates:
        collided = sorted(set(obs.columns[obs.columns.duplicated()]))
        originals = {
            name: [column for column in meta_columns if _strip_prefix(column, prefixes) == name] for name in collided
        }
        raise ValueError(
            f"two metadata prefixes collapse onto the same column name: {originals}, which would leave obs "
            "with duplicate columns. Drop one of them, or pass metadata_prefixes= with only the prefix this "
            "file uses."
        )
    if "Metadata_Well" in obs:
        obs["Metadata_Well"] = [normalize_well(well) for well in obs["Metadata_Well"].astype(str)]
    obs = categorize_metadata(_adopt_counts(obs))
    obs.index = pd.Index([str(i) for i in range(len(obs))])

    adata = ad.AnnData(X=X, obs=obs, var=parsed.loc[feature_names])
    _record(adata, resolution=resolution)
    # Recorded whether given or inferred, so the parse can be reproduced from the object.
    if vocabulary:
        adata.uns["mantispy"]["channels"] = list(vocabulary)
    return adata


def _ancestor_name(path: Path, depth: int) -> str:
    """The name `depth` directories up: 1 is the directory holding a file, or a directory that was given itself."""
    base = path if path.is_dir() else path.parent
    return base.name if depth == 1 else base.parents[depth - 2].name


def _join_platemap(obs: pd.DataFrame, platemap: str | Path | pd.DataFrame) -> pd.DataFrame:
    """Left-join a platemap onto ``obs`` by well, and by plate when both carry one."""
    frame = platemap.copy() if isinstance(platemap, pd.DataFrame) else _read_frame(Path(platemap))
    if "Metadata_Well" not in frame:
        raise ValueError("platemap must contain a 'Metadata_Well' column")
    frame["Metadata_Well"] = [normalize_well(well) for well in frame["Metadata_Well"].astype(str)]
    on = ["Metadata_Well"] + (["Metadata_Plate"] if "Metadata_Plate" in frame and "Metadata_Plate" in obs else [])
    left = obs.astype(dict.fromkeys(on, str))
    frame = frame.astype(dict.fromkeys(on, str))
    # validate="m:1": a duplicated well in the platemap would otherwise multiply that well's rows.
    try:
        joined = left.merge(
            frame, on=on, how="left", suffixes=("", "_platemap"), validate="m:1", indicator="_platemap_match"
        )
    except pd.errors.MergeError as error:
        duplicated = frame[frame.duplicated(on, keep=False)][on].drop_duplicates()
        raise ValueError(
            f"the platemap has more than one row for {len(duplicated)} of its {on} combination(s), "
            f"such as {duplicated.head(3).to_dict('records')}, which would multiply the rows of those "
            "wells. De-duplicate the platemap first."
        ) from error
    # A platemap that names the wrong wells joins onto nothing and leaves every column it was read for missing, which otherwise looks exactly like a successful read.
    unmatched = int((joined.pop("_platemap_match") == "left_only").sum())
    if unmatched:
        get_logger().warning(
            "%d of %d rows have no platemap row for %s; the platemap's columns are missing on those rows",
            unmatched,
            len(joined),
            on,
        )
    joined.index = obs.index
    return categorize_metadata(joined)


def _image_table(image: pd.DataFrame, obs: pd.DataFrame) -> pd.DataFrame:
    """Per-image quality measurements, keyed by ImageNumber and carrying plate/well.

    The plate and well columns let ``pp.image_qc`` threshold per plate instead of pooling every plate together.
    """
    quality = [c for c in image.columns if "ImageQuality" in c]
    table = image[["ImageNumber", *quality]].set_index("ImageNumber")
    if present := [c for c in ("Metadata_Plate", "Metadata_Well", "Metadata_Site") if c in obs.columns]:
        table = table.join(obs.groupby("Metadata_ImageNumber", observed=True)[present].first(), how="left")
    return table


def read_profiles(
    paths: str | Path | Sequence[str | Path],
    metadata_prefixes: Sequence[str] = METADATA_PREFIXES,
    metadata_columns: Sequence[str] = (),
    index_columns: Sequence[str] | None = None,
    channels: Sequence[str] | None = None,
    sentinels: float | Collection[float] | None = None,
    keep_non_features: bool = False,
    objects: Sequence[str] | None = DEFAULT_OBJECTS,
    on_column_mismatch: Literal["raise", "intersect"] = "raise",
    path_columns: Mapping[str, int] | None = None,
    platemap: str | Path | pd.DataFrame | None = None,
    primary_object: str = "Cells",
    strict_one_to_one: bool = True,
    resolution: str | None = None,
) -> ad.AnnData:
    """Read profiles into an AnnData of observations by features.

    What `paths` points at decides how it is read:

    - one or more CSV, TSV or parquet files, stacked row-wise;
    - a directory an ``ExportToSpreadsheet`` run wrote, one row per `primary_object` with every other object joined onto it through the ``Parent_`` column that links the two, and the ``MeasureImageQuality`` columns of ``Image.csv`` kept in ``uns["mantispy"]["image_table"]``;
    - a directory of parquet parts, as CytoTable writes them.

    Numeric columns that parse as CellProfiler features become ``X``.
    Text and ``Metadata_`` columns become ``obs``, with every metadata prefix normalized to ``Metadata_``.
    Numeric columns that parse as non-features (object numbers, parent links, centroids) are dropped and logged, because published profile tables often carry dozens of them and they mean nothing at well level.

    Args:
        paths: A file or a directory, or several files to stack.
        metadata_prefixes: Prefixes marking metadata columns.
            The matching prefix is normalized to ``Metadata_``.
        metadata_columns: Columns to treat as metadata even though they are numeric and unprefixed.
        index_columns: ``obs`` columns, named as they are after the prefix is normalized, joined with ``:`` into the observation index.
            Without them the index is the row number.
        channels: Channel vocabulary passed to the feature-name parser.
            Read from the ``FileName_`` columns of an export directory, and inferred from the feature names otherwise.
        sentinels: Values in the feature matrix that stand for missing, replaced with NaN.
        keep_non_features: Keep the numeric columns that do not parse as features in ``obs`` instead.
        objects: Which CellProfiler objects count as features, and for an export directory which object tables are read.
            The default is the three compartments, matching ``pycytominer.infer_cp_features``.
            ``None`` keeps every object, including the whole-field ``Image`` measurements.
        on_column_mismatch: ``"raise"``, or ``"intersect"`` to keep the shared columns in the first file's order.
            Batches with disjoint feature sets do occur.
        path_columns: Metadata read from the path: maps a column name to how many directories up to take the name of, counting the directory holding a file, or a directory that was given, as 1.
        platemap: A table, or a path to one, with a ``Metadata_Well`` column and optionally ``Metadata_Plate``, left-joined onto ``obs``.
        primary_object: For an export directory, the object one row of the result is.
        strict_one_to_one: For an export directory, raise when another object does not match the primary object exactly once.
            ``False`` keeps the first match.
        resolution: Resolution to record, ``"cell"`` for a directory and ``"well"`` for files when omitted.

    Returns:
        An :class:`~anndata.AnnData` at the recorded resolution, with the parsed feature annotation in ``var``, the metadata in ``obs`` with pycytominer's per-well counts copied to ``Metadata_CellCount`` and ``Metadata_SiteCount``, and the schema stamp, the resolution, the channel vocabulary, this call's parameters and, from an export directory, the per-image quality table under ``uns["mantispy"]``.

    Raises:
        ValueError: No paths were given, `on_column_mismatch` is not one of the two accepted values, a directory was given together with other paths, the files disagree on columns while `on_column_mismatch` is ``"raise"``, a file holds a header and no rows, no column parses as a feature on `objects`, `index_columns` do not identify observations uniquely, the platemap repeats a well, or an export object cannot be linked one to one.
        FileNotFoundError: A directory holds neither an ``Image.csv`` nor parquet parts, or has no table for `primary_object`.
        KeyError: A name in `metadata_columns` or `index_columns` is not in the data.

    Examples:
        >>> import mantispy as mt
        >>> wells = mt.io.read_profiles("BR00116991_augmented.csv.gz", sentinels=-999)  # doctest: +SKIP
        >>> cells = mt.io.read_profiles("analysis/", platemap="platemap.csv")  # doctest: +SKIP
    """
    # Checked before anything is read: an unrecognized value used to fall through to the intersect branch, so a typo quietly dropped every column the files disagreed on.
    if on_column_mismatch not in {"raise", "intersect"}:
        raise ValueError(f"on_column_mismatch must be 'raise' or 'intersect', got {on_column_mismatch!r}")
    files = [Path(paths)] if isinstance(paths, str | Path) else [Path(p) for p in paths]
    if not files:
        raise ValueError("no profile files given")
    given = [str(path) for path in files]

    image = None
    vocabulary = list(channels) if channels is not None else None
    if any(path.is_dir() for path in files):
        if len(files) > 1:
            raise ValueError(
                f"got {len(files)} paths and at least one is a directory; read one directory at a time. "
                "ExportToSpreadsheet numbers images from 1 in every directory, so stacked directories would "
                "share ImageNumbers."
            )
        directory = files[0]
        resolution = resolution or "cell"
        if export_prefix(directory) is not None:
            frame, image, found = read_export(directory, primary_object, objects, strict_one_to_one)
            vocabulary = vocabulary if vocabulary is not None else (found or None)
            lengths = [len(frame)]
        else:
            parts = sorted(directory.glob("**/*.parquet"))
            if not parts:
                raise FileNotFoundError(
                    f"{directory} holds neither an ExportToSpreadsheet Image.csv nor .parquet parts"
                )
            get_logger().info("read_profiles: %d parquet part(s) under %s", len(parts), directory)
            frame, lengths, files = _stack(parts, on_column_mismatch)
    else:
        resolution = resolution or "well"
        frame, lengths, files = _stack(files, on_column_mismatch)

    if path_columns:
        derived = {
            name: np.repeat([_ancestor_name(path, depth) for path in files], lengths)
            for name, depth in path_columns.items()
        }
        frame = pd.concat([frame, pd.DataFrame(derived, index=frame.index)], axis=1)

    adata = from_dataframe(
        frame,
        metadata_prefixes=metadata_prefixes,
        metadata_columns=metadata_columns,
        channels=vocabulary,
        sentinels=sentinels,
        keep_non_features=keep_non_features,
        objects=objects,
        resolution=resolution,
    )
    obs = as_frame(adata.obs)
    if platemap is not None:
        obs = _join_platemap(obs, platemap)
    if index_columns:
        if missing := [column for column in index_columns if column not in obs.columns]:
            raise KeyError(f"index columns not in metadata: {missing}")
        index = obs[list(index_columns)].astype(str).agg(":".join, axis=1)
        if index.duplicated().any():
            raise ValueError(f"{list(index_columns)} do not identify observations uniquely")
        obs.index = pd.Index(index.to_numpy())
    adata.obs = obs
    if image is not None:
        adata.uns["mantispy"]["image_table"] = _image_table(image, obs)

    record_params(
        adata,
        "read_profiles",
        {
            "paths": given,
            "objects": objects,
            "channels": vocabulary,
            "primary_object": primary_object,
            "strict_one_to_one": strict_one_to_one,
            "resolution": resolution,
        },
    )
    get_logger().info("read %d observations x %d features from %s", adata.n_obs, adata.n_vars, given[0])
    return adata


def read(path: str | Path, backed: Literal["r", "r+"] | None = None, migrate_schema: bool = True) -> ad.AnnData:
    """Read a mantispy h5ad or zarr store, checking its schema version.

    Args:
        path: File to read.
            A ``.zarr`` suffix selects the zarr reader.
        backed: ``"r"`` leaves ``X`` on disk and reads it a group at a time.
            ``obs`` and ``var`` are always in memory, so QC, feature flags and metadata work unchanged, but a function that rewrites ``X`` needs ``copy=True`` or ``key_added=``.
            h5ad only; zarr stores are read whole.
        migrate_schema: Bring an older but known schema up to the current one on read.
            Set it to ``False`` to see the version the file carries.

    Returns:
        The object, backed or in memory.
        ``adata.to_memory()`` loads a backed one into memory.

    Raises:
        ValueError: `backed` was given for a zarr store, or the file carries a schema version this build cannot migrate from or was told not to migrate.
    """
    path = Path(path)
    if backed is not None and path.suffix == ".zarr":
        raise ValueError("backed reads are h5ad only; a zarr store is read whole")
    adata = ad.read_zarr(path) if path.suffix == ".zarr" else ad.read_h5ad(path, backed=backed)
    stored = adata.uns.get("mantispy", {}).get("schema_version")
    if stored == SCHEMA_VERSION:
        return adata
    if migrate_schema and stored in SUPPORTED_VERSIONS:
        migrate(adata)
        return adata
    raise ValueError(
        f"file was written with schema_version {stored!r}, this version supports {SCHEMA_VERSION!r}"
        + (". Read it with mt.io.read(path, migrate_schema=True)" if stored in SUPPORTED_VERSIONS else "")
    )


def write(adata: ad.AnnData, path: str | Path) -> None:
    """Validate and write ``adata`` as h5ad (default) or zarr (``.zarr`` suffix).

    Args:
        adata: Object to write, stamped with the current schema version before it is checked.
        path: Where to write it.
            A ``.zarr`` suffix selects the zarr writer.

    Raises:
        ValueError: The object does not satisfy the schema, with every failure :func:`~mantispy.io.validate` found in the message.
    """
    path = Path(path)
    # Stamp the version so an unstamped object can be written, but leave var as the caller built it:
    # a missing annotation column is something to report here, not to repair on the way out.
    _record(adata, fill_var=False)
    validate(adata, raise_on_error=True)
    if path.suffix == ".zarr":
        adata.write_zarr(path)
    else:
        adata.write_h5ad(path)


def stamp(adata: ad.AnnData, resolution: str | None = None, copy: bool = False) -> ad.AnnData | None:
    """Mark an :class:`~anndata.AnnData` built elsewhere as a mantispy object.

    Every reader here, and every tool that returns a new object, records this already. This is the entry point for an object that did not come from one of them: a published ``h5ad``, another pipeline's output, a subset assembled in a notebook, or a matrix of learned embeddings with its metadata alongside.

    Args:
        adata: The object to stamp.
        resolution: What one row is: ``"cell"``, ``"well"`` or ``"perturbation"``. ``obs`` has to carry the columns that resolution requires. The default keeps whatever resolution the object already records, and falls back to ``"well"`` for an object that records none, so re-stamping a subset does not quietly demote it.
        copy: Return a stamped copy instead of stamping in place.

    Returns:
        ``None``, or the stamped copy.
        Writes the schema version and the resolution to ``uns["mantispy"]``, and adds the missing feature-annotation columns to ``var``.

    Raises:
        ValueError: ``resolution`` is not one of the three, or ``obs`` lacks a column that resolution requires.

    Notes:
        Only the ``obs`` columns the resolution requires are checked, because that is what the rest of the package dispatches on. :func:`validate` gives the full report, including what it warns about rather than blocks. ``X`` is one of the things it rather than this checks: the package stores features as ``float32``, and a matrix that came out of scikit-learn or :func:`numpy.load` is ``float64``, so an embedding usually wants ``adata.X = adata.X.astype("float32")`` before it is written.

        Any of the feature-annotation columns the schema requires that ``var`` does not already have are added empty, and columns already present are left as they are. They are not filled by parsing the feature names: the parser finds structure in names that have none — it reads ``openphenom_nahualX_17`` as the ``nahualX`` group of an ``openphenom`` object — and an embedding would then carry feature families named after the model's own tensors. An object read by :func:`read_profiles` already has the parsed annotation and keeps it.

    Examples:
        Bringing in a matrix of learned embeddings, one row per well:

        >>> import anndata as ad
        >>> import mantispy as mt
        >>> adata = ad.AnnData(embeddings, obs=metadata)  # doctest: +SKIP
        >>> mt.io.stamp(adata, resolution="well")  # doctest: +SKIP
    """
    if resolution is None:
        resolution = adata.uns.get("mantispy", {}).get("resolution", "well")
    if resolution not in RESOLUTIONS:
        raise ValueError(f"resolution must be one of {RESOLUTIONS}, got {resolution!r}")

    # Checked before the copy, so a call that is going to be rejected does not duplicate X first.
    missing_obs = [column for column in REQUIRED_OBS[resolution] if column not in adata.obs]
    if missing_obs:
        raise ValueError(
            f"obs is missing {missing_obs}, which every {resolution}-resolution object needs. Add the "
            "column(s), or stamp at a resolution whose requirements obs meets."
        )

    target = adata.copy() if copy else adata
    # _record supplies the annotation columns var does not carry.
    _record(target, resolution=resolution)
    return target if copy else None
