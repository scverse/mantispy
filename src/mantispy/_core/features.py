"""Parse CellProfiler and cp_measure column names into structured annotations.

This is the only module that parses feature names.
Its output populates ``adata.var``, and nothing downstream re-parses names.

The grammar handled here is::

    [<Object>_]<Group>_<feature words>[_<channel>...][_<numeric params>][_<NofM>]

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

_RADIAL_BIN_RE = re.compile(r"^\d+of\d+$")
_NUMERIC_RE = re.compile(r"^-?\d+(\.\d+)?$")


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


def _parse_one(name: str, channels: frozenset[str]) -> dict:
    row: dict = dict.fromkeys(COLUMNS)
    row["is_feature"] = False

    tokens = str(name).split("_")
    if name in _BARE_NON_FEATURES or len(tokens) < 2:
        return row

    obj, group, rest = _split_object(tokens)
    row["object"] = obj
    row["feature_group"] = group

    if group in NON_FEATURE_GROUPS:
        return row
    if not rest:
        row["feature"] = group
        row["is_feature"] = True
        return row

    found_channels = [token for token in rest if token in channels]
    if found_channels:
        row["channel"] = "|".join(found_channels)
        rest = [token for token in rest if token not in channels]

    radial = [token for token in rest if _RADIAL_BIN_RE.match(token)]
    if radial:
        row["radial_bin"] = radial[0]
        rest = [token for token in rest if token not in radial]

    numeric = [token for token in rest if _is_numeric(token)]
    if numeric:
        # Keep the full numeric suffix: Zernike_2_0 and Zernike_2_2 must stay distinct.
        row["params"] = "_".join(numeric)
        rest = [token for token in rest if not _is_numeric(token)]
        if group == "Texture":
            for key, value in zip(("scale", "angle", "gray_levels"), numeric, strict=False):
                row[key] = float(value)
        else:
            row["scale"] = float(numeric[0])

    row["feature"] = "_".join(rest) if rest else group
    row["is_feature"] = True
    return row


def parse_feature_names(names: Sequence[str], channels: Sequence[str] | None = None) -> pd.DataFrame:
    """Parse ``names`` into a table of feature annotations indexed by name.

    Args:
        names: Column names from a CellProfiler or cp_measure table.
        channels: The channel vocabulary.
            When omitted it is inferred from ``names``, which is less reliable than passing the channels the reader found in ``Image.csv``.

    Returns:
        A frame indexed by ``names`` with the columns listed in :data:`COLUMNS`.
        Text columns are ``category`` dtype (so they survive an h5ad round trip even when they are entirely missing) and ``is_feature`` is ``bool``.
    """
    names = list(names)
    channel_set = frozenset(channels if channels is not None else _infer_channels(names))
    parsed = pd.DataFrame(
        [_parse_one(name, channel_set) for name in names],
        index=pd.Index(names),
        columns=COLUMNS,
    )
    for column in _TEXT_COLUMNS:
        parsed[column] = parsed[column].astype("category")
    for column in _FLOAT_COLUMNS:
        parsed[column] = parsed[column].astype("float64")
    parsed["is_feature"] = parsed["is_feature"].astype(bool)
    return parsed


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
