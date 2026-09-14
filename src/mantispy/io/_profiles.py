from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from pathlib import Path
from typing import Literal

import anndata as ad
import numpy as np
import pandas as pd

METADATA_PREFIXES: Sequence[str] = ("Image_Metadata_", "Metadata_", "metadata_", "meta_")

CHANNEL_ALIASES: Mapping[str, str] = {
    "dna": "dna",
    "hoechst": "dna",
    "rna": "rna",
    "agp": "agp",
    "er": "er",
    "mito": "mito",
    "brightfield": "brightfield",
    "lowzbf": "brightfield_low",
    "bflow": "brightfield_low",
    "highzbf": "brightfield_high",
    "bfhigh": "brightfield_high",
}


def _read_parquet_or_csv(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _strip_longest_prefix(name: str, prefixes: Sequence[str]) -> str:
    for prefix in sorted(prefixes, key=len, reverse=True):
        if name.startswith(prefix):
            return name[len(prefix) :]
    return name


def _annotate_features(var: pd.DataFrame, *, aliases: Mapping[str, str] = CHANNEL_ALIASES) -> None:
    """Annotate a feature table in place with what each CellProfiler feature name encodes.

    Names follow ``<compartment>_<family>_<measurement>[_<channel>][_<parameters>]``.
    Compartment and family are read positionally, channels are the name tokens that match a known channel.
    ``Correlation`` features measure colocalization between a pair of channels, hence two channel columns; ``AreaShape`` and ``Neighbors`` are geometry and have none.
    The measurement itself and its parameters are not extracted.

    Args:
        var: The feature table to annotate, indexed by feature name.
            Gains the columns ``compartment``, ``family``, ``channel``, ``channel_2`` and ``n_channels``.
        aliases: Maps a lower-cased name token to the channel it stands for.
    """
    tokens = [name.lower().split("_") for name in var.index]
    matched = [[aliases[t] for t in token if t in aliases] for token in tokens]
    var["compartment"] = pd.Categorical([t[0] for t in tokens])
    var["family"] = pd.Categorical([t[1] if len(t) > 1 else "" for t in tokens])
    var["channel"] = pd.Categorical([c[0] if c else None for c in matched])
    var["channel_2"] = pd.Categorical([c[1] if len(c) > 1 else None for c in matched])
    var["n_channels"] = np.array([len(c) for c in matched], dtype=np.int8)


def _align_columns(
    frames: Sequence[pd.DataFrame], on_column_mismatch: Literal["raise", "intersect"]
) -> list[pd.DataFrame]:
    if len({tuple(frame.columns) for frame in frames}) == 1:
        return list(frames)
    if on_column_mismatch == "raise":
        msg = "files disagree on columns; pass on_column_mismatch='intersect' to keep the shared ones"
        raise ValueError(msg)
    shared = set.intersection(*(set(frame.columns) for frame in frames))
    if not shared:
        msg = "files share no columns"
        raise ValueError(msg)
    order = [column for column in frames[0].columns if column in shared]
    return [frame[order] for frame in frames]


def read_profiles(
    paths: Path | str | Sequence[Path | str],
    *,
    metadata_prefixes: Sequence[str] = METADATA_PREFIXES,
    metadata_columns: Sequence[str] = (),
    sentinels: float | Collection[float] | None = None,
    index_columns: Sequence[str] | None = None,
    on_column_mismatch: Literal["raise", "intersect"] = "raise",
    path_columns: Mapping[str, int] | None = None,
    annotate_features: bool = True,
) -> ad.AnnData:
    """Read CellProfiler profiles into an :class:`~anndata.AnnData` of observations × features.

    Numeric columns become the feature matrix, everything else :attr:`~anndata.AnnData.obs`.
    What has differed between Cell Painting datasets is a parameter, not an assumption: missing-value sentinels, metadata prefixes, and columns that disagree across plates.

    Args:
        paths: One profile file, or several to stack row-wise.
            ``.parquet`` is read as parquet, anything else as CSV.
        metadata_prefixes: Column-name prefixes marking metadata.
            The matching prefix is stripped from the ``obs`` column name.
        metadata_columns: Columns to treat as metadata even though they are numeric and unprefixed.
        sentinels: Values in the feature matrix standing for missing, replaced with ``NaN``.
        index_columns: Metadata columns, named as they are after prefix stripping, joined with ``:`` into the observation index.
        on_column_mismatch: What to do when the files disagree on columns: ``"raise"``, or ``"intersect"`` to keep the shared columns in the column order of the first file.
        path_columns: Metadata columns taken from the file path, mapping each column name to how many directories up from the file to read the name of.
        annotate_features: Annotate :attr:`~anndata.AnnData.var` with the compartment, family and channels each feature name encodes.

    Returns:
        An :class:`~anndata.AnnData` whose ``X`` is ``float32``, whose ``obs`` holds the metadata, and whose ``var`` is indexed by feature name.

    Raises:
        ValueError: No files were given, the files share no columns, they disagree on columns while `on_column_mismatch` is ``"raise"``, or `index_columns` do not identify observations uniquely.
        KeyError: A name in `metadata_columns` or `index_columns` is not in the data.

    Examples:
        Read one plate of well-level profiles, with ``-999`` standing for missing:

        >>> import mantispy as mt
        >>> adata = mt.io.read_profiles(  # doctest: +SKIP
        ...     "BR00116991_normalized.csv.gz",
        ...     sentinels=-999,
        ...     index_columns=("Plate", "Well"),
        ... )

        Stack several plates whose column sets drifted, tagging each row with the plate its file sat in:

        >>> adata = mt.io.read_profiles(  # doctest: +SKIP
        ...     sorted(Path("profiles").glob("*/*.parquet")),
        ...     on_column_mismatch="intersect",
        ...     path_columns={"Metadata_Plate": 1},
        ... )
    """
    files = [Path(paths)] if isinstance(paths, str | Path) else [Path(path) for path in paths]
    if not files:
        msg = "no profile files given"
        raise ValueError(msg)

    frames = [_read_parquet_or_csv(path) for path in files]
    # a file with no rows is read as all-object and would drag the dtypes of the others with it through concat
    kept = [index for index, frame in enumerate(frames) if len(frame)]
    if kept and len(kept) < len(frames):
        files = [files[index] for index in kept]
        frames = [frames[index] for index in kept]
    frames = _align_columns(frames, on_column_mismatch)
    df = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
    if path_columns:
        lengths = [len(frame) for frame in frames]
        derived = {
            name: np.repeat([path.parents[depth - 1].name for path in files], lengths)
            for name, depth in path_columns.items()
        }
        df = pd.concat([df, pd.DataFrame(derived, index=df.index)], axis=1)

    prefixes = tuple(metadata_prefixes)
    named = set(metadata_columns)
    if missing_named := named - set(df.columns):
        msg = f"metadata columns not in data: {sorted(missing_named)}"
        raise KeyError(msg)
    prefixed = {column for column in df.columns if column.startswith(prefixes)}
    numeric = set(df.select_dtypes("number").columns)
    meta_columns = [c for c in df.columns if c in prefixed or c in named or c not in numeric]
    feature_columns = [c for c in df.columns if c not in set(meta_columns)]

    x = df[feature_columns].to_numpy(np.float32)
    if sentinels is not None:
        values = [sentinels] if isinstance(sentinels, int | float) else list(sentinels)
        x[np.isin(x, np.asarray(values, dtype=x.dtype))] = np.nan

    obs = df[meta_columns].rename(columns=lambda c: _strip_longest_prefix(c, prefixes))
    for column in obs.select_dtypes(include=["object", "str"]).columns:
        obs[column] = obs[column].astype("string")
    obs.index = obs.index.astype(str)
    if index_columns:
        if missing := [column for column in index_columns if column not in obs.columns]:
            msg = f"index columns not in metadata: {missing}"
            raise KeyError(msg)
        index = obs[list(index_columns)].astype(str).agg(":".join, axis=1)
        if index.duplicated().any():
            msg = f"{list(index_columns)} do not identify observations uniquely"
            raise ValueError(msg)
        obs.index = pd.Index(index, name="observation")

    var = pd.DataFrame(index=pd.Index(feature_columns, name="feature"))
    if annotate_features:
        _annotate_features(var)
    return ad.AnnData(X=x, obs=obs, var=var)
