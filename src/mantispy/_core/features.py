"""Parse CellProfiler and cp_measure column names into structured annotations.

This is the only module that parses feature names.
Its output populates ``adata.var``, and nothing downstream re-parses names.

Two grammars are handled. CellProfiler's::

    [<Object>_]<Group>_<feature words>[_<channel>...][_<numeric params>][_<NofM>]

and cp_measure's, which separates its tokens with slashes, names the channel by index and glues the group to the feature in camel case::

    <object>_<channel>/<aggregation>/<group><Feature words>[_<numeric params>][_<NofM>]

A name is read as cp_measure's only when it matches that shape in full, which a CellProfiler name cannot because CellProfiler never emits a ``/``.

Columns that are not measurements (object numbers, parent/child links, locations, file names, metadata) get ``is_feature = False`` so callers can route them somewhere other than ``X``.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Sequence
from importlib.resources import files

import numpy as np
import pandas as pd

#: Columns of the annotation table, in order.
COLUMNS = [
    "object",
    "feature_group",
    "feature",
    "channel",
    "scale",
    "angle",
    "gray_levels",
    "radial_bin",
    "params",
    "is_feature",
]

_TEXT_COLUMNS = ["object", "feature_group", "feature", "channel", "radial_bin", "params"]
_FLOAT_COLUMNS = ["scale", "angle", "gray_levels"]

#: Groups that never hold a usable profile feature.
NON_FEATURE_GROUPS = frozenset(
    {
        "Number",
        "ObjectNumber",
        "Location",
        "Parent",
        "Children",
        "Metadata",
        "Group",
        "Count",
        "FileName",
        "PathName",
        "URL",
        "Series",
        "Frame",
        "Scaling",
        "MD5Digest",
        "ExecutionTime",
        "ModuleError",
        "Width",
        "Height",
        "Threshold",
    }
)

#: Groups whose names carry one or more channel tokens.
CHANNEL_BEARING_GROUPS = frozenset(
    {
        "Intensity",
        "Texture",
        "Granularity",
        "RadialDistribution",
        "Correlation",
        "Colocalization",
        "ImageQuality",
        "IntensityDistribution",
    }
)

#: Groups that hold measurements but carry no channel.
PLAIN_FEATURE_GROUPS = frozenset({"AreaShape", "Neighbors", "AreaOccupied", "Zernike", "Skeleton"})

_KNOWN_GROUPS = NON_FEATURE_GROUPS | CHANNEL_BEARING_GROUPS | PLAIN_FEATURE_GROUPS

#: Maps a lower-cased channel token to the stain it stands for, so that datasets naming a channel differently stay comparable.
#: Adapted from scverse/cell-painting-io (MIT).
#: Nothing is renamed unless a caller asks for canonical channels.
CHANNEL_ALIASES: dict[str, str] = {
    "dna": "dna",
    "hoechst": "dna",
    "dapi": "dna",
    "nuclei": "dna",
    "rna": "rna",
    "agp": "agp",
    "actin": "agp",
    "er": "er",
    "mito": "mito",
    "mitochondria": "mito",
    "brightfield": "brightfield",
    "lowzbf": "brightfield_low",
    "bflow": "brightfield_low",
    "highzbf": "brightfield_high",
    "bfhigh": "brightfield_high",
}


def canonical_channel(channel: str | None, aliases: dict[str, str] | None = None) -> str | None:
    """Map a channel name onto its canonical stain, or return it unchanged.

    Multi-channel values such as ``"DNA|ER"`` are mapped component-wise.
    Unknown names pass through lower-cased, so an unfamiliar vocabulary still compares consistently with itself.
    """
    if channel is None or (isinstance(channel, float) and pd.isna(channel)):
        return None
    lookup = CHANNEL_ALIASES if aliases is None else aliases
    return "|".join(lookup.get(part.lower(), part.lower()) for part in str(channel).split("|"))


#: Bare column names CellProfiler emits that are never features.
_BARE_NON_FEATURES = frozenset({"ImageNumber", "ObjectNumber", "TableNumber"})

#: ``AreaShape`` measurements that say where an object is rather than what it looks like. CellProfiler 4 writes
#: the centroid here instead of under ``Location``.
_POSITIONAL_AREASHAPE = frozenset({"Center", "BoundingBoxMinimum", "BoundingBoxMaximum"})

_RADIAL_BIN_RE = re.compile(r"^\d+of\d+$")
_NUMERIC_RE = re.compile(r"^-?\d+(\.\d+)?$")

#: ``cell_0/max/textureContrast_3_03_256``: object, channel index, per-object aggregation, then the group glued to the
#: feature in camel case. CellProfiler separates every token with an underscore and never emits a ``/``, so a name has
#: to match this whole shape before it is read this way.
_CP_MEASURE_RE = re.compile(
    r"^(?P<object>[a-z]+)_(?P<channel>\d+)/(?P<agg>[a-z]+)/(?P<group>[a-z_]+)(?P<feature>[A-Z].*)$"
)


def _is_numeric(token: str) -> bool:
    return bool(_NUMERIC_RE.match(token))


def _infer_channels(names: Sequence[str]) -> list[str]:
    """Guess the channel vocabulary from a list of column names.

    A token counts as a channel when it trails a channel-bearing group, is neither numeric nor a radial bin, and shows up with at least two distinct feature names in that group.
    Used only when the caller does not pass ``channels``.
    """
    seen: dict[str, set[str]] = defaultdict(set)
    for name in names:
        tokens = str(name).split("_")
        for i, token in enumerate(tokens):
            if token in CHANNEL_BEARING_GROUPS and i + 2 < len(tokens):
                for candidate in tokens[i + 2 :]:
                    if not _is_numeric(candidate) and not _RADIAL_BIN_RE.match(candidate):
                        seen[candidate].add(tokens[i + 1])
    return sorted(token for token, features in seen.items() if len(features) >= 2)


def _split_object(tokens: list[str]) -> tuple[str | None, str, list[str]]:
    """Split a token list into ``(object, feature_group, rest)``."""
    if tokens[0] == "Image":
        return "Image", tokens[1], tokens[2:]
    if tokens[0] in _KNOWN_GROUPS:
        return None, tokens[0], tokens[1:]
    if tokens[1] in _KNOWN_GROUPS or len(tokens) >= 3:
        return tokens[0], tokens[1], tokens[2:]
    return None, tokens[0], tokens[1:]


def _read_suffixes(row: dict, rest: list[str], group: str) -> None:
    """Move the radial bin and the numeric suffix of ``rest`` into their own columns, leaving the feature name."""
    radial = [token for token in rest if _RADIAL_BIN_RE.match(token)]
    if radial:
        row["radial_bin"] = radial[0]
        rest = [token for token in rest if token not in radial]

    numeric = [token for token in rest if _is_numeric(token)]
    if numeric:
        # Keep the full numeric suffix: Zernike_2_0 and Zernike_2_2 must stay distinct.
        row["params"] = "_".join(numeric)
        rest = [token for token in rest if not _is_numeric(token)]
        if group.lower() == "texture":
            for key, value in zip(("scale", "angle", "gray_levels"), numeric, strict=False):
                row[key] = float(value)
        else:
            row["scale"] = float(numeric[0])

    row["feature"] = "_".join(rest) if rest else group
    row["is_feature"] = True


def _parse_cp_measure(name: str) -> dict:
    """Annotate one ``cp_measure`` name, whose channel is an index rather than the stain's name.

    cp_measure is handed one channel at a time and numbers them in the order it was given them, so the index is all
    the name carries. It is kept as the channel rather than resolved to a stain, because the mapping lives in the
    acquisition metadata and guessing it would put a wrong stain on every intensity feature in the screen.
    """
    row: dict = dict.fromkeys(COLUMNS)
    match = _CP_MEASURE_RE.match(name)
    if match is None:
        # The grammar was chosen from the file as a whole, so a name that does not fit it is a mixed or
        # hand-edited file rather than a name to guess at.
        row["is_feature"] = False
        return row

    # A group ending in "_" means the name separated it from the feature, as in "sizeshape_Solidity".
    group = match["group"].rstrip("_")
    row["object"] = match["object"]
    row["feature_group"] = group
    row["channel"] = match["channel"]
    _read_suffixes(row, match["feature"].split("_"), group)
    return row


def _parse_one(name: str, channels: frozenset[str]) -> dict:
    row: dict = dict.fromkeys(COLUMNS)
    row["is_feature"] = False

    tokens = str(name).split("_")
    if name in _BARE_NON_FEATURES or len(tokens) < 2:
        return row

    obj, group, rest = _split_object(tokens)
    row["object"] = obj
    row["feature_group"] = group

    if group in NON_FEATURE_GROUPS or (group == "AreaShape" and rest and rest[0] in _POSITIONAL_AREASHAPE):
        return row
    if not rest:
        row["feature"] = group
        row["is_feature"] = True
        return row

    found_channels = [token for token in rest if token in channels]
    if found_channels:
        row["channel"] = "|".join(found_channels)
        rest = [token for token in rest if token not in channels]

    _read_suffixes(row, rest, group)
    return row


def parse_feature_names(names: Sequence[str], channels: Sequence[str] | None = None) -> pd.DataFrame:
    """Parse ``names`` into a table of feature annotations indexed by name.

    Which grammar to read is decided once, from ``names`` as a whole, because it is a property of the file that wrote them rather than of any one column.

    Args:
        names: Column names from a CellProfiler or cp_measure table.
        channels: The channel vocabulary.
            When omitted it is inferred from ``names``, which is less reliable than passing the channels the reader found in ``Image.csv``.

    Returns:
        A frame indexed by ``names`` with the columns listed in :data:`COLUMNS`.
        Text columns are ``category`` dtype (so they survive an h5ad round trip even when they are entirely missing) and ``is_feature`` is ``bool``.
    """
    names = list(names)
    if any(_CP_MEASURE_RE.match(str(name)) for name in names):
        # cp_measure names their own channel by index, so there is no vocabulary to infer or to be given.
        rows = [_parse_cp_measure(str(name)) for name in names]
    else:
        channel_set = frozenset(channels if channels is not None else _infer_channels(names))
        rows = [_parse_one(str(name), channel_set) for name in names]
    parsed = pd.DataFrame(rows, index=pd.Index(names), columns=COLUMNS)
    for column in _TEXT_COLUMNS:
        parsed[column] = parsed[column].astype("category")
    for column in _FLOAT_COLUMNS:
        parsed[column] = parsed[column].astype("float64")
    parsed["is_feature"] = parsed["is_feature"].astype(bool)
    return parsed


def empty_annotation(names: Sequence[str] | pd.Index) -> pd.DataFrame:
    """The annotation table for features whose names carry no CellProfiler structure.

    Learned embeddings, cluster compositions and feature-family signatures all have columns that are features but are not measurements of a compartment in a channel. The schema still asks for the annotation columns, so they are supplied empty rather than guessed at: :func:`parse_feature_names` reads ``openphenom_nahualX_17`` as the ``nahualX`` group of the ``openphenom`` object, which would give such an object feature families named after the model's own tensors.

    Args:
        names: The feature names, which are used only as the index.

    Returns:
        A frame indexed by ``names`` with the columns of :data:`COLUMNS`, every annotation column null and ``is_feature`` true.
        Text columns are ``category`` dtype, as :func:`parse_feature_names` returns them, so an entirely missing column survives an h5ad round trip.
    """
    index = pd.Index(names)
    empty = pd.DataFrame(index=index, columns=COLUMNS, dtype=object)
    for column in _TEXT_COLUMNS:
        empty[column] = pd.Categorical([None] * len(index))
    for column in _FLOAT_COLUMNS:
        empty[column] = np.full(len(index), np.nan)
    empty["is_feature"] = True
    return empty


def load_blocklist(name: str = "default") -> list[str]:
    """Return the feature names blocked by default (the pycytominer blocklist)."""
    if name != "default":
        raise ValueError(f"unknown blocklist: {name!r}")
    text = (files("mantispy._core.data") / "blocklist_cellprofiler.txt").read_text()
    return [line.strip() for line in text.splitlines() if line.strip() and not line.startswith("#")]


def blocklist_hits(candidates: Sequence[np.ndarray], blocklist: str | Sequence[str] = "default") -> np.ndarray:
    """Boolean mask, ``True`` where any of ``candidates`` names a blocked feature.

    ``candidates`` is normally the current ``var_names`` together with ``var["original_name"]``, so a blocklist matches before and after :func:`~mantispy.pp.standardize_feature_names`.
    That function rewrites channel-bearing names into a grammar no blocklist entry matches, but keeps the incoming name in ``original_name``.
    Both call sites that apply a blocklist (:func:`~mantispy.pp.filter_features` and the ``blocklist`` operation of :func:`~mantispy.pp.feature_select`) use this function.
    """
    blocked = set(load_blocklist(blocklist) if isinstance(blocklist, str) else blocklist)
    return np.logical_or.reduce([np.isin(np.asarray(candidate), list(blocked)) for candidate in candidates])
