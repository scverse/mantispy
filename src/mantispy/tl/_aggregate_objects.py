"""Roll the objects of a CellProfiler export into one table, one row per root object."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
from anndata import AnnData

from mantispy._core._numba import MEAN, MEDIAN, STD, group_counts, grouped_stat
from mantispy._core._reduce import get_matrix
from mantispy._core.features import _TEXT_COLUMNS, annotation
from mantispy._core.frames import as_frame, categorize_metadata
from mantispy._core.provenance import record_params
from mantispy._core.schema import stamp
from mantispy.io._cp_links import Omissions, carriers, link_edges

#: The statistics a child's features are rolled into its root with; ``count`` is the number of children.
STATISTICS = ("count", "mean", "median", "std")
_KERNELS = {"mean": MEAN, "median": MEDIAN, "std": STD}


#: NamesAndTypes' record of where in a file each image sat, which the per-object tables keep and a profile does not.
_FILE_BOOKKEEPING = ("Metadata_ChannelNumber", "Metadata_FileLocation", "Metadata_Frame", "Metadata_Series")
#: CellProfiler's own counts, numbers and links of an object, which the per-object tables keep and a profile does not.
_OBJECT_BOOKKEEPING = re.compile(r"_(Children_.+_Count|Location_Center_Z|Number_Object_Number|Parent_.+)$")


def profile_obs(obs: pd.DataFrame, root: str) -> pd.DataFrame:
    """`obs` of a profile without CellProfiler's bookkeeping, and without the root's centroid twice.

    The identity the schema asks for (image and object numbers, plate, well, site, centroid, the numbers of joined objects) stays.
    """
    duplicates = {f"{root}_Location_Center_X", f"{root}_Location_Center_Y"} if "Metadata_Center_X" in obs else set()
    return obs.drop(
        columns=[
            column
            for column in obs.columns
            if column in _FILE_BOOKKEEPING
            or column in duplicates
            or (not column.startswith("Metadata_") and _OBJECT_BOOKKEEPING.search(column))
        ]
    )


def statistics(func: str | Sequence[str]) -> tuple[str, ...]:
    """`func` as a tuple of statistic names.

    Raises:
        ValueError: A name is not one of :data:`STATISTICS`, or none is given.
    """
    names = (func,) if isinstance(func, str) else tuple(func)
    if not names or not set(names) <= set(STATISTICS):
        raise ValueError(f"func must name one or more of {STATISTICS}, got {func!r}")
    return names


def aggregate_objects(
    objects: Mapping[str, AnnData] | Any,
    root: str = "Cells",
    func: str | Sequence[str] = STATISTICS,
    relations: Mapping[str, Mapping[str, str]] | None = None,
) -> AnnData:
    """Roll the per-object tables of an export into one table, one row per `root` object.

    The objects one to one with `root` (its compartments, such as the cytoplasm and nucleus of a cell) and its ancestors are joined onto it.
    The objects of which a root has several, such as the nucleoli of a nucleus or the foci of a cell, are summarised per root with each statistic of `func`.
    Which is which follows the link kinds each table records in ``uns["mantispy"]["links"]``, as :func:`~mantispy.io.read_cellprofiler_csv` writes them, never the counts observed.
    A child is reached through any chain of links, so the foci of a filtered subset of foci still roll into the cell that subset belongs to.

    Missing values of a child are skipped; a root with no child has a count of 0 and NaN for every other statistic.
    ``std`` is the sample standard deviation (``ddof=1``), NaN for a single child, and ``median`` averages the two middle values of an even count.
    Children that link to no root (a parent number of 0) are left out before the statistics, with one warning giving their number for every child.

    Args:
        objects: The per-object tables, keyed by object name, or a SpatialData holding them as its tables.
        root: The object one row of the result is.
        func: The statistics to take, from ``"count"``, ``"mean"``, ``"median"`` and ``"std"``.
        relations: Link kinds that override the recorded ones, by child and then by parent, for example ``{"Nucleoli": {"Nuclei": "aggregate"}}``.

    Returns:
        An object-resolution :class:`~anndata.AnnData` with the root's features, those of the joined objects, and per child the column ``<Child>_count`` and ``<Child>_<column>_<stat>`` for each of its features and statistics.
        ``var`` keeps the parsed annotation of the child's column and adds ``aggregation``, the statistic, and ``parent``, the root.
        CellProfiler's own ``<Parent>_Mean_<Child>_`` columns are left out for each child rolled up here.
        ``obs`` is the root's, with the number of each joined object as ``Metadata_<Object>ObjectNumber`` and without the bookkeeping :func:`~mantispy.io.read_cellprofiler` leaves out.

    Raises:
        ValueError: There is no table for `root`, `func` names an unknown statistic, a compartment is not one to one with its parent, a child links to an object number its parent table lacks, or two result columns share a name.

    Examples:
        >>> import mantispy as mt
        >>> objects = mt.io.read_cellprofiler_csv("analysis/")  # doctest: +SKIP
        >>> cells = mt.tl.aggregate_objects(objects, root="Cells", func=("count", "mean"))  # doctest: +SKIP
    """
    tables = dict(objects.tables) if hasattr(objects, "tables") else dict(objects)
    adata = combine(tables, root, func=statistics(func), relations=relations)
    record_params(adata, "aggregate_objects", {"root": root, "func": list(statistics(func)), "relations": relations})
    return adata


def candidate_roots(tables: Mapping[str, AnnData], links: Mapping[str, Mapping[str, str]]) -> list[str]:
    """The objects worth naming as a root: those with features or with a link to another table, else every object."""
    linked = {name for edge in link_edges(links) for name in edge[:2]}
    return sorted(name for name in tables if tables[name].n_vars or name in linked) or sorted(tables)


def _take(values: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """``values[rows]``, with -1 where a row is -1."""
    return np.where(rows >= 0, values[np.clip(rows, 0, None)] if len(values) else -1, -1)


def _unique_children(up: np.ndarray, n_parents: int, child: str, parent: str) -> np.ndarray:
    """For each parent row, the one child row that links to it, or -1.

    Raises:
        ValueError: A parent has more than one such child.
    """
    linked = up >= 0
    if (several := pd.Series(up[linked]).duplicated().sum()) > 0:
        raise ValueError(
            f"{child} is linked to {parent} one to one, but {several} {parent} object(s) have more than one {child}. "
            f"Pass relations={{{child!r}: {{{parent!r}: 'aggregate'}}}} to summarise them instead."
        )
    children = np.full(n_parents, -1)
    children[up[linked]] = np.flatnonzero(linked)
    return children


def _resolve(
    tables: Mapping[str, AnnData], root: str, edges: Sequence[tuple[str, str, str]]
) -> dict[str, tuple[str, np.ndarray, bool]]:
    """How each object reaches `root` along the links.

    ``("join", rows, one_to_one)`` gives each root row its row of the object (or -1); ``("aggregate", rows, False)`` gives each row of the object its root row (or -1).
    """
    keys = {
        name: pd.MultiIndex.from_arrays(
            [
                as_frame(t.obs)["Metadata_ImageID"].astype(str).to_numpy(),
                as_frame(t.obs)["Metadata_ObjectNumber"].to_numpy(),
            ]
        )
        for name, t in tables.items()
    }

    def parents(child: str, parent: str) -> np.ndarray:
        obs = as_frame(tables[child].obs)
        numbers = obs[f"Metadata_{parent}ObjectNumber"].fillna(0).to_numpy()
        rows = keys[parent].get_indexer(
            pd.MultiIndex.from_arrays([obs["Metadata_ImageID"].astype(str).to_numpy(), numbers])
        )
        if (missing := int(((rows < 0) & (numbers > 0)).sum())) > 0:
            raise ValueError(f"{missing} {child} object(s) link to a {parent} object number that is not in its table")
        return rows

    state: dict[str, tuple[str, np.ndarray, bool]] = {root: ("join", np.arange(tables[root].n_obs), True)}
    queue = [root]
    while queue:
        current = queue.pop(0)
        how, rows, one_to_one = state[current]
        n_current = tables[current].n_obs
        for child, parent, kind in edges:
            if parent == current and child not in state:
                up = parents(child, current)
                if how == "aggregate":
                    state[child] = ("aggregate", _take(rows, up), False)
                elif kind != "aggregate":
                    state[child] = ("join", _take(_unique_children(up, n_current, child, current), rows), one_to_one)
                elif one_to_one:
                    inverse = np.full(n_current, -1)
                    inverse[rows[rows >= 0]] = np.flatnonzero(rows >= 0)
                    state[child] = ("aggregate", _take(inverse, up), False)
                else:
                    continue
                queue.append(child)
            elif child == current and parent not in state:
                up = parents(current, parent)
                if how == "join":
                    state[parent] = ("join", _take(up, rows), one_to_one and kind != "aggregate")
                elif kind != "aggregate":
                    unique = _unique_children(up, tables[parent].n_obs, current, parent)
                    state[parent] = ("aggregate", _take(rows, unique), False)
                else:
                    continue
                queue.append(parent)
    return state


def _matrix(adata: AnnData) -> np.ndarray:
    return get_matrix(adata)


def _gather(values: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """Rows of `values` in the order of `rows`, NaN where a row is -1."""
    taken = values[np.clip(rows, 0, None)] if len(values) else np.empty((len(rows), *values.shape[1:]))
    if (rows < 0).any():
        taken = taken.astype(np.result_type(taken.dtype, np.float64) if taken.dtype.kind != "f" else taken.dtype)
        taken[rows < 0] = np.nan
    return taken


def _gather_column(column: pd.Series, rows: np.ndarray) -> np.ndarray:
    values = column.to_numpy()
    if not (rows < 0).any():
        return values[rows]
    if column.dtype.kind in "iufb":
        return _gather(values.astype(np.float64), rows)
    taken = values.astype(object)[np.clip(rows, 0, None)]
    taken[rows < 0] = None
    return taken


def combine(
    tables: Mapping[str, AnnData],
    root: str,
    *,
    func: Sequence[str] | None,
    relations: Mapping[str, Mapping[str, str]] | None = None,
    strict: bool = False,
    omissions: Omissions | None = None,
) -> AnnData:
    """One table per `root` object: compartments and ancestors joined, and children rolled up with `func` when it is given.

    What is left out is added to `omissions`, which the caller logs once; without it, it is logged here.

    Raises:
        ValueError: `strict` is set and a compartment row has no root or a root has no compartment, or as :func:`aggregate_objects`.
    """
    report = Omissions(root) if omissions is None else omissions
    stats = tuple(func or ())
    links = {name: dict(t.uns.get("mantispy", {}).get("links", {})) for name, t in tables.items()}
    for child, parents in (relations or {}).items():
        links.setdefault(child, {}).update(parents)
    skipped = set(carriers(links, {name: t.n_vars > 0 for name, t in tables.items()}))
    if root not in tables:
        raise ValueError(f"there is no {root!r} table; pass root= one of {candidate_roots(tables, links)}")
    edges = [
        (child, parent, kind)
        for child, parent, kind in link_edges(links)
        if child in tables and parent in tables and not {child, parent} & skipped
    ]
    state = _resolve(tables, root, edges)
    joined = sorted(name for name, (how, _, _) in state.items() if how == "join" and name != root)
    children = sorted(name for name, (how, _, _) in state.items() if how == "aggregate")
    report.unlinked |= set(tables) - set(state) - skipped
    if children and func is None:
        report.several |= set(children)
        children = []

    base = tables[root]
    n_root = base.n_obs
    obs = as_frame(base.obs).copy()
    blocks: list[np.ndarray] = [_matrix(base)]
    variables: list[pd.DataFrame] = [as_frame(base.var)]
    for name in joined:
        _, rows, one_to_one = state[name]
        table = tables[name]
        if one_to_one:
            orphans = table.n_obs - len(np.unique(rows[rows >= 0]))
            missing = int((rows < 0).sum())
            if strict and (orphans or missing):
                raise ValueError(
                    f"{orphans} {name} object(s) have no {root}, and {missing} {root} object(s) have no {name}; "
                    "pass strict=False to leave the first out and give the second NaN for its features"
                )
            report.add_orphans(name, orphans, table.n_obs)
            report.missing[name] = report.missing.get(name, 0) + missing
        blocks.append(_gather(_matrix(table), rows))
        variables.append(as_frame(table.var))
        table_obs = as_frame(table.obs)
        obs[f"Metadata_{name}ObjectNumber"] = _gather_column(table_obs["Metadata_ObjectNumber"], rows)
        for column in (c for c in table_obs.columns if c.startswith(f"{name}_")):
            obs[column] = _gather_column(table_obs[column], rows)

    rolled = [*joined, root]
    for name in children:
        _, rows, _ = state[name]
        table = tables[name]
        linked = rows >= 0
        report.add_orphans(name, int((~linked).sum()), table.n_obs)
        codes = rows[linked]
        if "count" in stats:
            blocks.append(group_counts(codes, n_root).astype(np.float64)[:, None])
            variables.append(
                annotation(
                    [f"{name}_count"],
                    object=name,
                    feature_group="Count",
                    feature="count",
                    feature_kind="derived",
                    aggregation="count",
                    parent=root,
                )
            )
        X = _matrix(table)[linked]
        for stat in (s for s in stats if s != "count" and table.n_vars):
            blocks.append(grouped_stat(X, codes, n_root, _KERNELS[stat], ddof=1))
            var = as_frame(table.var).copy()
            var.index = pd.Index([f"{feature}_{stat}" for feature in var.index])
            var["aggregation"] = stat
            var["parent"] = root
            variables.append(var)
        # CellProfiler's own per-parent means of this child duplicate the ones rolled up here.
        drop = tuple(f"{owner}_Mean_{name}_" for owner in rolled)
        for position in range(len(rolled)):
            keep = ~variables[position].index.str.startswith(drop)
            if not keep.all():
                variables[position] = variables[position].loc[keep]
                blocks[position] = blocks[position][:, keep]

    var = pd.concat(variables)
    for column in (c for c in [*_TEXT_COLUMNS, "aggregation", "parent"] if c in var):
        # Concatenating categoricals with different categories degrades them to object; keep every table's categories.
        known = [
            v[column].cat.categories
            for v in variables
            if column in v and isinstance(v[column].dtype, pd.CategoricalDtype)
        ]
        categories = sorted(set().union(*known, var[column].dropna().unique()))
        var[column] = pd.Categorical(var[column], categories=categories)
    if var.index.has_duplicates:
        raise ValueError(f"the rolled-up columns repeat names: {sorted(set(var.index[var.index.duplicated()]))[:5]}")
    X = np.hstack(blocks) if blocks else np.empty((n_root, 0))
    result = ad.AnnData(X=X, obs=categorize_metadata(profile_obs(obs, root)), var=var)
    stamp(result, resolution="object")
    store = base.uns.get("mantispy", {})
    if "image_table" in store:
        result.uns["mantispy"]["image_table"] = store["image_table"]
    if omissions is None:
        report.log()
    return result
