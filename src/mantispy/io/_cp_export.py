"""Read the tables of a CellProfiler export, as ExportToSpreadsheet or ExportToDatabase wrote them, into one form.

Both modules write an image table and one table per object, but they name, split and type them differently.
Each reader here returns a :class:`RawExport` that holds the tables as ExportToSpreadsheet names them: an object's columns without its name, ``ImageNumber`` and ``ObjectNumber`` as the keys.
"""

from __future__ import annotations

import csv
import io
import itertools
import re
import sqlite3
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from mantispy._core.logging import get_logger

#: The suffixes of an SQLite database ExportToDatabase writes.
SQLITE_SUFFIXES = (".db", ".sqlite", ".sqlite3")
#: The delimiter ExportToSpreadsheet writes behind each file suffix.
_DELIMITERS = {".csv": ",", ".txt": "\t"}
#: CellProfiler's integer columns, which a table split from a wider one holds as floats.
_INTEGER_PREFIXES = ("Number_Object_Number", "Parent_", "Children_")
#: ExportToDatabase's default column-name limit, used when the pipeline does not say.
_MAX_NAME = 64
#: Words of CellProfiler's measurement names, for restoring the names ExportToDatabase shortened.
_WORDS = frozenset(
    """Correlation Costes Manders Overlap RWC K Slope Intensity Integrated Mean Median Std Max Min Lower Upper
    Quartile Edge Mass Displacement MAD Location Center CenterMassIntensity MaxIntensity Granularity Texture
    AngularSecondMoment Contrast DifferenceEntropy DifferenceVariance Entropy InfoMeas1 InfoMeas2
    InverseDifferenceMoment SumAverage SumEntropy SumVariance Variance RadialDistribution FracAtD MeanFrac RadialCV
    ZernikeMagnitude ZernikePhase AreaShape Neighbors Children Parent Count Number Object Threshold Orig Image
    Width Height FirstClosestObjectNumber SecondClosestObjectNumber FirstClosestDistance SecondClosestDistance
    AngleBetweenNeighbors PercentTouching NumberOfNeighbors Expanded Adjacent IntegratedIntensityEdge
    MeanIntensityEdge StdIntensityEdge MinIntensityEdge MaxIntensityEdge MassDisplacement LowerQuartileIntensity
    UpperQuartileIntensity MedianIntensity MADIntensity MeanIntensity StdIntensity MinIntensity MaxIntensity
    IntegratedIntensity""".split()
)


@dataclass
class RawExport:
    """The tables of one export, named as ExportToSpreadsheet names them.

    Attributes:
        image: The image table: ``ImageNumber``, the ``Count_<Object>`` columns, image metadata and measurements.
        objects: One table per object that has rows, keyed by the object's name, with ``ImageNumber`` and ``ObjectNumber`` and the object's columns without its name.
        pipeline: The pipeline text the export stores, or ``None``.
        source: Where the export was read from.
        renamed_image_number: Whether an imported ``Metadata_ImageNumber`` was renamed.
    """

    image: pd.DataFrame
    objects: dict[str, pd.DataFrame]
    pipeline: str | None
    source: Path
    renamed_image_number: bool = field(default=False)


def export_prefix(path: Path) -> str | None:
    """The file-name prefix an ``ExportToSpreadsheet`` run used, or ``None`` if `path` is not such a directory.

    Args:
        path: The directory to look in, which such a run writes ``<prefix>Image.csv`` or, with tab delimiters, ``<prefix>Image.txt`` into.

    Returns:
        The shortest prefix a ``*Image.csv`` or ``*Image.txt`` in `path` carries, which is ``""`` for a run configured without one, and ``None`` when `path` is not a directory or holds no such file.
    """
    if not path.is_dir():
        return None
    files = [file for suffix in _DELIMITERS for file in path.glob(f"*Image{suffix}")]
    prefixes = sorted((file.name.removesuffix(f"Image{file.suffix}") for file in files), key=len)
    return prefixes[0] if prefixes else None


def _imported_image_number(frame: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    """Rename an imported ``Metadata_ImageNumber``, which would read as CellProfiler's own ``ImageNumber``."""
    if "ImageNumber" in frame and "Metadata_ImageNumber" in frame:
        return frame.rename(columns={"Metadata_ImageNumber": "Metadata_ImportedImageNumber"}), True
    return frame, False


def _read_table(path: Path) -> pd.DataFrame:
    """One table of an export, with a repeated header kept once.

    Asked to add image metadata, ExportToSpreadsheet copies it into each object table and writes a ``Metadata_`` column twice when the table already carried it.
    pandas would read the second copy as ``Metadata_Well.1``, so the copies are compared and one is kept.

    Raises:
        ValueError: Two columns of the same name hold different values.
    """
    delimiter = _DELIMITERS[path.suffix]
    with path.open(newline="") as handle:
        header = next(csv.reader(handle, delimiter=delimiter), [])
    frame = pd.read_csv(path, sep=delimiter)
    if len(header) == frame.shape[1]:
        frame.columns = pd.Index(header)
    for name in frame.columns[frame.columns.duplicated()].unique():
        copies = frame.loc[:, frame.columns == name]
        if not all(copies.iloc[:, 0].equals(copies.iloc[:, i]) for i in range(1, copies.shape[1])):
            raise ValueError(f"{path.name} has {copies.shape[1]} columns named {name!r} holding different values")
    return frame.loc[:, ~frame.columns.duplicated()]


def _object_names(image: pd.DataFrame) -> list[str]:
    """The objects the image table counts, which are the objects the pipeline identified."""
    return [column.removeprefix("Count_") for column in image.columns if column.startswith("Count_")]


def _split_combined(path: Path, delimiter: str) -> dict[str, pd.DataFrame]:
    """The object tables of a file ExportToSpreadsheet wrote with every object in one table.

    The first header row names the object of each column and the second the column; the rows line up the objects by their object number.
    """
    with path.open(newline="") as handle:
        reader = csv.reader(handle, delimiter=delimiter)
        groups, names = next(reader), next(reader)
    body = pd.read_csv(path, sep=delimiter, header=None, skiprows=2)
    image_number = body[groups.index("Image")]
    tables = {}
    for obj in dict.fromkeys(groups):
        if obj == "Image":
            continue
        positions = [i for i, group in enumerate(groups) if group == obj]
        frame = body[positions].set_axis([names[i] for i in positions], axis=1)
        frame = frame.loc[frame["Number_Object_Number"].notna()]
        frame = pd.concat([image_number.loc[frame.index].rename("ImageNumber"), frame], axis=1)
        if "ObjectNumber" not in frame:
            frame = _with_object_number(frame)
        tables[obj] = _integers(frame.reset_index(drop=True))
    return tables


def _with_object_number(frame: pd.DataFrame) -> pd.DataFrame:
    """The table with ``ObjectNumber`` after ``ImageNumber``, copied from ``Number_Object_Number``.

    Built in one concat, since inserting a column into a wide table trips pandas' fragmentation warning.
    """
    number = frame["Number_Object_Number"].rename("ObjectNumber")
    return pd.concat([frame[["ImageNumber"]], number, frame.drop(columns="ImageNumber")], axis=1)


def _integers(frame: pd.DataFrame) -> pd.DataFrame:
    """Cast CellProfiler's integer columns back from the floats a split table holds them as."""
    columns = [c for c in frame.columns if c in ("ImageNumber", "ObjectNumber") or c.startswith(_INTEGER_PREFIXES)]
    whole = [c for c in columns if frame[c].notna().all()]
    return frame.astype(dict.fromkeys(whole, "int64"))


def _pipeline_from_csv(directory: Path, prefix: str, suffix: str) -> str | None:
    path = directory / f"{prefix}Experiment{suffix}"
    if not path.is_file():
        return None
    frame = pd.read_csv(path, sep=_DELIMITERS[suffix])
    values = frame.set_index("Key")["Value"] if "Key" in frame else pd.Series(dtype=object)
    text = values.get("Pipeline_Pipeline")
    return str(text) if isinstance(text, str) else None


def extract_csv(directory: Path, files: Mapping[str, str | Path] | None = None) -> RawExport:
    """The tables of a directory an ExportToSpreadsheet run wrote.

    Args:
        directory: The directory.
        files: Object tables under custom file names, by object name.

    Raises:
        FileNotFoundError: `directory` holds no ``Image.csv`` or ``Image.txt``.
        ValueError: A table repeats a column with different values.
    """
    prefix = export_prefix(directory)
    if prefix is None:
        raise FileNotFoundError(f"{directory} holds no ExportToSpreadsheet Image.csv or Image.txt")
    suffix = ".csv" if (directory / f"{prefix}Image.csv").is_file() else ".txt"
    image, renamed = _imported_image_number(_read_table(directory / f"{prefix}Image{suffix}"))
    names = _object_names(image)
    paths = {name: directory / f"{prefix}{name}{suffix}" for name in names}
    paths |= {name: Path(path) for name, path in (files or {}).items()}

    objects: dict[str, pd.DataFrame] = {}
    for path in dict.fromkeys(path for path in paths.values() if path.is_file()):
        with path.open(newline="") as handle:
            first = next(csv.reader(handle, delimiter=_DELIMITERS[path.suffix]), [""])[0]
        if first == "Image":
            objects |= _split_combined(path, _DELIMITERS[path.suffix])
    for name, path in paths.items():
        if name not in objects and path.is_file():
            objects[name] = _read_table(path)
    for name in list(objects):
        objects[name], hit = _imported_image_number(objects[name])
        renamed |= hit
    return RawExport(image, objects, _pipeline_from_csv(directory, prefix, suffix), directory, renamed)


# --- ExportToDatabase -------------------------------------------------------


def _shorten(name: str, limit: int) -> str:
    """The name ExportToDatabase writes for a column, as CellProfiler's ``ColumnNameMapping`` shortens it.

    Lower-case vowels go first, then other lower-case letters, then upper-case letters, each from the end of the name, until it fits.
    """
    name = re.sub(r"[^0-9a-zA-Z_$]", "_", name)
    if len(name) <= limit:
        return name
    for start in [position for position in (name.find("_"), 0) if position != -1]:
        to_remove = len(name) - limit
        if to_remove <= 0:
            continue
        removed = 0
        for letters in ("aeiou", "bcdfghjklmnpqrstvwxyz", "ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
            for i in range(len(name) - 1, start - 1, -1):
                if name[i] in letters:
                    name = name[:i] + name[i + 1 :]
                    removed += 1
                    if removed == to_remove:
                        break
            if removed == to_remove:
                break
    return name


def _is_subsequence(short: str, word: str) -> bool:
    letters = iter(word)
    return all(letter in letters for letter in short)


def _restore(name: str, limit: int, vocabulary: frozenset[str]) -> str | None:
    """The full name a shortened column had, or ``None`` when no single candidate shortens to it."""
    options = []
    for token in name.split("_"):
        expansions = {w for w in vocabulary if len(w) > len(token) and w[0] == token[:1] and _is_subsequence(token, w)}
        options.append(sorted({token} | expansions))
    total = 1
    for option in options:
        total *= len(option)
    if total > 200_000:
        return None
    hits = {
        candidate
        for parts in itertools.product(*options)
        if _shorten(candidate := "_".join(parts), limit) == name
        and all(part in vocabulary or part.isdigit() for part in parts)
    }
    return hits.pop() if len(hits) == 1 else None


def _restore_names(frames: dict[str, pd.DataFrame], limit: int, extra: set[str]) -> None:
    """Restore, in place, the column names ExportToDatabase shortened to `limit` characters.

    A name exactly `limit` long may have been shortened; its tokens are expanded with words from every name that was not, and the expansion is kept only when CellProfiler's own shortening gives the name back.
    """
    vocabulary = set(_WORDS) | {token for name in extra for token in name.split("_")}
    for frame in frames.values():
        vocabulary |= {token for column in frame.columns if len(column) < limit for token in column.split("_")}
    words = frozenset(word for word in vocabulary if word and not word.isdigit())
    kept = []
    for key, frame in frames.items():
        mapping = {}
        for column in frame.columns:
            if len(column) == limit:
                restored = _restore(column, limit, words)
                if restored is None:
                    kept.append(column)
                else:
                    mapping[column] = restored
        frames[key] = frame.rename(columns=mapping)
    if kept:
        get_logger().warning(
            "kept %d column name(s) ExportToDatabase shortened, such as %s, as written: no single full name "
            "shortens to them",
            len(kept),
            kept[:3],
        )


def _typed(frame: pd.DataFrame, declared: dict[str, str]) -> pd.DataFrame:
    """The table with the dtypes its columns were declared with, as a CSV export would read them.

    A column holding only NULL reads as objects, so it gets its declared numeric type.
    Text metadata is parsed as numbers where every value is one, as pandas reads ``Metadata_Site`` from a CSV.
    """
    casts: dict[str, str] = {}
    for column, kind in declared.items():
        kind = kind.upper()
        if any(word in kind for word in ("FLOAT", "REAL", "DOUBLE", "INT")) and frame[column].dtype == object:
            casts[column] = "float64"
    frame = frame.astype(casts)
    for column in frame.columns:
        if (column.startswith("Metadata_") or "_Metadata_" in column) and not is_numeric_dtype(frame[column]):
            values = [_text(value) for value in frame[column]]
            parsed = pd.read_csv(io.StringIO("\n".join(["x", *values])), keep_default_na=True)["x"]
            if is_numeric_dtype(parsed):
                frame[column] = parsed.to_numpy()
    return frame


def _text(value: object) -> str:
    """A metadata value as the text a CSV export would hold.

    ExportToDatabase stores a metadata value extracted as an integer as its eight little-endian bytes in a text column.
    """
    if isinstance(value, bytes):
        return (
            str(int.from_bytes(value, "little", signed=True)) if len(value) == 8 else value.decode("utf-8", "replace")
        )
    return "" if value is None or (isinstance(value, float) and np.isnan(value)) else str(value)


def _read_sql(connection: sqlite3.Connection, table: str) -> pd.DataFrame:
    declared = {row[1]: str(row[2]) for row in connection.execute(f'PRAGMA table_info("{table}")')}
    return _typed(pd.read_sql_query(f'SELECT * FROM "{table}"', connection), declared)


def _strip(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    """The table with ``<name>_`` taken off its column names, which ExportToDatabase puts on every column but the keys."""
    return frame.rename(columns=lambda column: column.removeprefix(f"{name}_"))


def extract_sqlite(path: Path) -> RawExport:
    """The tables of an SQLite database an ExportToDatabase run wrote.

    The run may have written one table per object, one ``Per_Object`` table holding every object aligned by object number, or both with ``Per_Object`` as a view.
    The view inner-joins the objects on their number and loses the objects other objects have no partner for, so it is not read.

    Raises:
        FileNotFoundError: The database holds no ``Per_Image`` table.
    """
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        kinds = dict(connection.execute("SELECT name, type FROM sqlite_master WHERE type IN ('table', 'view')"))
        prefixes = sorted((name.removesuffix("Per_Image") for name in kinds if name.endswith("Per_Image")), key=len)
        if not prefixes:
            raise FileNotFoundError(f"{path} holds no ExportToDatabase Per_Image table; found {sorted(kinds)}")
        prefix = prefixes[0]
        pipeline = None
        for table in (f"{prefix}Per_Experiment", "Per_Experiment"):
            if table in kinds:
                row = connection.execute(f'SELECT Pipeline_Pipeline FROM "{table}"').fetchone()
                value = row[0] if row else None
                pipeline = value.decode("utf-8", "replace") if isinstance(value, bytes) else value
                break
        limit = _MAX_NAME
        if pipeline and (match := re.search(r"Maximum # of characters in a column name:(\d+)", pipeline)):
            limit = int(match.group(1))

        frames = {"Image": _read_sql(connection, f"{prefix}Per_Image")}
        names = _object_names(_strip(frames["Image"], "Image"))
        single = kinds.get(f"{prefix}Per_Object") == "table"
        if single:
            frames["Object"] = _read_sql(connection, f"{prefix}Per_Object")
        else:
            for name in names:
                if kinds.get(f"{prefix}Per_{name}") == "table":
                    frames[name] = _read_sql(connection, f"{prefix}Per_{name}")
    finally:
        connection.close()

    channels = {
        re.sub(r"^Image_(FileName|PathName|URL)_", "", column)
        for column in frames["Image"].columns
        if re.match(r"Image_(FileName|PathName|URL)_", column)
    }
    _restore_names(frames, limit, set(names) | channels)
    image, renamed = _imported_image_number(_strip(frames.pop("Image"), "Image"))
    if single:
        wide = frames.pop("Object")
        for name in names:
            own = [
                column
                for column in wide.columns
                if column.startswith(f"{name}_")
                and not any(len(other) > len(name) and column.startswith(f"{other}_") for other in names)
            ]
            if f"{name}_Number_Object_Number" in own:
                part = wide.loc[wide[f"{name}_Number_Object_Number"].notna(), ["ImageNumber", *own]]
                frames[name] = part.reset_index(drop=True)
    objects = {}
    for name, frame in frames.items():
        frame = _with_object_number(_strip(frame, name))
        objects[name], hit = _imported_image_number(_integers(frame))
        renamed |= hit
    return RawExport(image, objects, pipeline, path, renamed)
