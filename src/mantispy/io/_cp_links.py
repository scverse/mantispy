"""Classify the ``Parent_`` links of a CellProfiler export by the module that made them.

Every export carries the pipeline that wrote it (``Experiment.csv`` or ``Per_Experiment``), and the module that identified an object says what its link to the parent means.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import pandas as pd

from mantispy._core.logging import get_logger

#: What a link from a child table to a parent object means.
#: ``compartment``: one child per parent, a region of the same cell (IdentifySecondaryObjects, IdentifyTertiaryObjects, ExpandOrShrinkObjects).
#: ``subset``: the child is one of the parent's objects, kept by FilterObjects or MaskObjects.
#: ``aggregate``: many children per parent, related by RelateObjects.
#: ``carrier``: a copy of an object that holds only its link (ResizeObjects, RelateObjects' saved children, the measurement-free input of a filter).
LINK_KINDS = ("compartment", "subset", "aggregate", "carrier")

#: Module name to (setting naming the child, settings naming the parent, link kind).
_MODULES = {
    "IdentifySecondaryObjects": ("Name the objects to be identified", ("Select the input objects",), "compartment"),
    "IdentifyTertiaryObjects": (
        "Name the tertiary objects to be identified",
        ("Select the larger identified objects", "Select the smaller identified objects"),
        "compartment",
    ),
    "ExpandOrShrinkObjects": ("Name the output objects", ("Select the input objects",), "compartment"),
    "RelateObjects": ("Child objects", ("Parent objects",), "aggregate"),
    "FilterObjects": ("Name the output objects", ("Select the objects to filter",), "subset"),
    "MaskObjects": ("Name the masked objects", ("Select objects to be masked",), "subset"),
    "ResizeObjects": ("Name the output object", ("Select the input object",), "carrier"),
    "SplitOrMergeObjects": ("Name the new objects", ("Select the input objects",), "split"),
}

#: The compartments a pipeline without its text is assumed to relate one to one, when their counts agree.
_COMPARTMENTS = frozenset({"Cells", "Cytoplasm", "Nuclei"})

_MODULE_RE = re.compile(r"\n(?=[A-Za-z0-9_]+:\[module_num)")
_HEADER_RE = re.compile(r"([A-Za-z0-9_]+):\[module_num:(\d+)\|.*?enabled:(True|False)")


def pipeline_links(text: str) -> dict[tuple[str, str], str]:
    """The ``(child, parent)`` links the enabled modules of a pipeline make, and the kind of each.

    Args:
        text: The pipeline, as CellProfiler stores it in ``Pipeline_Pipeline``.

    Returns:
        A mapping from ``(child, parent)`` object names to one of :data:`LINK_KINDS`, or ``"split"`` for SplitOrMergeObjects.
    """
    links: dict[tuple[str, str], str] = {}
    for block in _MODULE_RE.split(text):
        header = _HEADER_RE.match(block)
        if header is None or header.group(3) != "True" or header.group(1) not in _MODULES:
            continue
        settings: dict[str, str] = {}
        for line in block.splitlines()[1:]:
            key, _, value = line.strip().partition(":")
            settings.setdefault(key, value)
        child_key, parent_keys, kind = _MODULES[header.group(1)]
        child = settings.get(child_key)
        for parent in (settings.get(key) for key in parent_keys):
            if child and parent and parent != "None":
                links[(child, parent)] = kind
        if (
            header.group(1) == "RelateObjects"
            and settings.get("Do you want to save the children with parents as a new object set?") == "Yes"
        ):
            links[(settings.get("Name the output object", ""), str(child))] = "carrier"
    return links


def classify(
    tables: Mapping[str, pd.DataFrame],
    measured: Mapping[str, bool],
    pipeline: str | None,
    counts: pd.DataFrame,
    relations: Mapping[str, Mapping[str, str]] | None,
) -> tuple[dict[str, dict[str, str]], dict[tuple[str, str], str]]:
    """The kind of every ``Parent_`` link of every table, keyed by child and then by parent.

    Args:
        tables: The object tables, with CellProfiler's unprefixed column names.
        measured: Whether each table holds a measurement of its own, beyond location and links.
        pipeline: The pipeline text, if the export holds it.
        counts: The image table, whose ``Count_<Object>`` columns the classification without a pipeline compares.
        relations: Kinds given by the user, which override any other source.

    Returns:
        For each table, its parent objects and the kind of the link to each; and the ``(child, parent)`` links neither the pipeline nor a name rule classifies, with the kind the tables suggest for each.

    Raises:
        ValueError: A link was made by SplitOrMergeObjects, or a kind in `relations` is unknown.
    """
    from_pipeline = pipeline_links(pipeline) if pipeline else {}
    given = {(child, parent): kind for child, parents in (relations or {}).items() for parent, kind in parents.items()}
    if unknown := {kind for kind in given.values() if kind not in LINK_KINDS}:
        raise ValueError(f"relations= kinds must be among {LINK_KINDS}, got {sorted(unknown)}")
    links: dict[str, dict[str, str]] = {}
    unexplained: dict[tuple[str, str], str] = {}
    for child, frame in tables.items():
        links[child] = {}
        for parent in (column.removeprefix("Parent_") for column in frame.columns if column.startswith("Parent_")):
            kind = given.get((child, parent)) or from_pipeline.get((child, parent))
            if kind is None:
                kind = _by_name(child, parent, tables, counts)
            if kind == "split":
                raise ValueError(
                    f"{child} was made from {parent} by SplitOrMergeObjects, whose objects need not match any one "
                    f"{parent}; pass relations={{{child!r}: {{{parent!r}: ...}}}} to say how to read the link"
                )
            if kind == "subset" and not measured.get(parent, True):
                kind = "carrier"
            if kind is None:
                unexplained[(child, parent)] = _likely(child, parent, tables, measured)
                continue
            links[child][parent] = kind
    return links, unexplained


def _likely(child: str, parent: str, tables: Mapping[str, pd.DataFrame], measured: Mapping[str, bool]) -> str:
    """The kind a link the pipeline does not explain most likely has, offered to the user to confirm.

    A child without measurements is a copy that carries the link, a parent with several children aggregates them, and otherwise the child is a compartment of its parent.
    """
    if not measured.get(child, True):
        return "carrier"
    frame = tables[child]
    linked = frame.loc[frame[f"Parent_{parent}"].fillna(0) > 0, ["ImageNumber", f"Parent_{parent}"]]
    return "aggregate" if linked.duplicated().any() else "compartment"


def unexplained_message(unexplained: Mapping[tuple[str, str], str], pipeline: bool) -> str:
    """What the links in `unexplained` lack, and the ``relations=`` that reads them as the tables suggest."""
    names = ", ".join(f"{child} -> {parent}" for child, parent in sorted(unexplained))
    suggested: dict[str, dict[str, str]] = {}
    for (child, parent), kind in sorted(unexplained.items()):
        suggested.setdefault(child, {})[parent] = kind
    source = "the pipeline does not make them" if pipeline else "the export holds no pipeline"
    return (
        f"cannot tell what the link(s) {names} mean, as {source}; pass relations={suggested} if the pipeline made "
        f"them so (the kinds are {', '.join(LINK_KINDS)})"
    )


def _by_name(child: str, parent: str, tables: Mapping[str, pd.DataFrame], counts: pd.DataFrame) -> str | None:
    """The kind of a link the pipeline does not explain, when the column names or the counts leave no doubt."""
    columns = tables[child].columns
    if any(c.startswith("Distance_") and c.endswith(f"_{parent}") for c in columns) or (
        parent in tables and any(c.startswith(f"Mean_{child}_") for c in tables[parent].columns)
    ):
        return "aggregate"
    pair = (f"Count_{child}", f"Count_{parent}")
    if {child, parent} <= _COMPARTMENTS and set(pair) <= set(counts.columns):
        if counts[pair[0]].equals(counts[pair[1]]):
            return "compartment"
    return None


def carriers(links: Mapping[str, Mapping[str, str]], measured: Mapping[str, bool]) -> list[str]:
    """The tables that only carry links: measurement-free, and every link they take part in a carrier link."""
    involved: dict[str, list[str]] = {}
    for child, parents in links.items():
        for parent, kind in parents.items():
            involved.setdefault(child, []).append(kind)
            involved.setdefault(parent, []).append(kind)
    return sorted(
        name for name, kinds in involved.items() if not measured.get(name, True) and set(kinds) == {"carrier"}
    )


def link_edges(links: Mapping[str, Mapping[str, str]]) -> Sequence[tuple[str, str, str]]:
    """The links as sorted ``(child, parent, kind)`` triples, carrier links left out."""
    return sorted(
        (child, parent, kind)
        for child, parents in links.items()
        for parent, kind in parents.items()
        if kind != "carrier"
    )


@dataclass
class Omissions:
    """What a read left out, gathered over its objects and logged once, one warning line per kind."""

    #: The root of the profile, or None for the per-object readers.
    root: str | None
    #: Links nothing classifies, with the kind the tables suggest for each.
    unexplained: dict[tuple[str, str], str] = field(default_factory=dict)
    #: Whether the export held its pipeline, which the message about `unexplained` says.
    pipeline: bool = True
    #: Objects of which one root has several, left out without ``aggregate=``.
    several: set[str] = field(default_factory=set)
    #: Objects with no link to the root.
    unlinked: set[str] = field(default_factory=set)
    #: Per object, how many of its rows link to no root, and how many it has.
    orphans: dict[str, list[int]] = field(default_factory=dict)
    #: Per compartment, how many roots have none.
    missing: dict[str, int] = field(default_factory=dict)

    def add_orphans(self, name: str, left_out: int, total: int) -> None:
        """Count `left_out` of `total` rows of `name` that link to no root."""
        counts = self.orphans.setdefault(name, [0, 0])
        counts[0] += left_out
        counts[1] += total

    def log(self) -> None:
        """Log what was left out, at warning level."""
        log = get_logger()
        if self.unexplained:
            log.warning(
                "left out %s: %s",
                sorted({child for child, _ in self.unexplained}),
                unexplained_message(self.unexplained, self.pipeline),
            )
        if self.several:
            log.warning(
                "left out %s, of which one %s can have several; pass aggregate= (for example aggregate='mean') to "
                "roll them into it",
                sorted(self.several),
                self.root,
            )
        if self.unlinked:
            log.warning(
                "left out %s, which are not linked to %s; their counts per image are in uns['mantispy']['image_table']",
                sorted(self.unlinked),
                self.root,
            )
        clauses = []
        if orphans := [f"{left} of {total} {name}" for name, (left, total) in sorted(self.orphans.items()) if left]:
            clauses.append(f"left out {', '.join(orphans)} object(s), which link to no {self.root}")
        clauses += [
            f"{count} {self.root} object(s) have no {name} and get NaN for its features"
            for name, count in sorted(self.missing.items())
            if count
        ]
        if clauses:
            log.warning("%s; the per-object readers keep every row", "; ".join(clauses))
