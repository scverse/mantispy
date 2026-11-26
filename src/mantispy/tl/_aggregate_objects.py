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
from mantispy.io._cellprofiler import IDENTITY_GROUPS
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
    score: str | None = None,
) -> AnnData:
    """Roll the per-object tables of an export into one table, one row per `root` object.

    The objects one to one with `root` (its compartments, such as the cytoplasm and nucleus of a cell) and its ancestors are joined onto it.
    The objects of which a root has several, such as the nucleoli of a nucleus or the foci of a cell, are summarised per root with each statistic of `func`.
    Which is which follows the link kinds each table records in ``uns["mantispy"]["links"]``, as :func:`~mantispy.io.read_cellprofiler_csv` writes them, never the counts observed.
    A child is reached through any chain of links, so the foci of a filtered subset of foci still roll into the cell that subset belongs to.

    Missing values of a child are skipped; a root with no child has a count of 0 and NaN for every other statistic.
    ``std`` is the sample standard deviation (``ddof=1``), NaN for a single child, and ``median`` averages the two middle values of an even count.
    Children that link to no root (a parent number of 0) are left out before the statistics, with one warning giving their number for every child.

    A child's columns that name a value instead of measuring one, its text columns and the barcode calls (``Barcode_*``) of a pooled screen, are summarised in ``obs`` by their most common value per root.
    Missing values are not counted, and a tie goes to the smallest value.

    Args:
        objects: The per-object tables, keyed by object name, or a SpatialData holding them as its tables.
        root: The object one row of the result is.
        func: The statistics to take, from ``"count"``, ``"mean"``, ``"median"`` and ``"std"``.
        relations: Link kinds that override the recorded ones, by child and then by parent, for example ``{"Nucleoli": {"Nuclei": "aggregate"}}``.
        score: A numeric ``obs`` column of a child, such as ``"Foci_Barcode_MatchedTo_Score"``, averaged over the children that hold the most common value of each of that child's summarised columns.

    Returns:
        An object-resolution :class:`~anndata.AnnData` with the root's features, those of the joined objects, and per child the column ``<Child>_count`` and ``<Child>_<column>_<stat>`` for each of its features and statistics.
        ``var`` keeps the parsed annotation of the child's column and adds ``aggregation``, the statistic, and ``parent``, the root.
        CellProfiler's own ``<Parent>_Mean_<Child>_`` columns are left out for each child rolled up here.
        ``obs`` is the root's, with the number of each joined object as ``Metadata_<Object>ObjectNumber`` and without the bookkeeping :func:`~mantispy.io.read_cellprofiler` leaves out, and per summarised column ``<column>_majority`` (the most common value, as text), ``<column>_fraction`` (the share of the root's children that hold it), ``<column>_n_distinct`` (the number of distinct values, 0 without a child) and ``<column>_tie`` (whether another value is as common, missing without a child).
        With `score`, ``<column>_majority_score`` holds its mean over the children that hold the most common value.

    Raises:
        KeyError: `score` is not a column of a child rolled into `root`.
        ValueError: There is no table for `root`, `func` names an unknown statistic, a compartment is not one to one with its parent, a child links to an object number its parent table lacks, or two result columns share a name.

    Examples:
        >>> import mantispy as mt
        >>> objects = mt.io.read_cellprofiler_csv("analysis/")  # doctest: +SKIP
        >>> cells = mt.tl.aggregate_objects(objects, root="Cells", func=("count", "mean"))  # doctest: +SKIP
    """
    tables = dict(objects.tables) if hasattr(objects, "tables") else dict(objects)
    adata = combine(tables, root, func=statistics(func), relations=relations, score=score)
    record_params(
        adata,
        "aggregate_objects",
        {"root": root, "func": list(statistics(func)), "relations": relations, "score": score},
    )
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


def categorical_annotation(var: pd.DataFrame, variables: Sequence[pd.DataFrame]) -> pd.DataFrame:
    """`var`, stacked from `variables`, with each text annotation one categorical holding the categories of all of them.

    Concatenating categoricals with different categories degrades them to object.
    """
    for column in (c for c in [*_TEXT_COLUMNS, "aggregation", "parent"] if c in var):
        known = [
            v[column].cat.categories
            for v in variables
            if column in v and isinstance(v[column].dtype, pd.CategoricalDtype)
        ]
        categories = sorted(set().union(*known, var[column].dropna().unique()))
        var[column] = pd.Categorical(var[column], categories=categories)
    return var


def _identity_columns(obs: pd.DataFrame, name: str) -> list[str]:
    """The columns of a child's ``obs`` that name a value rather than measure one: text, and integer identity groups."""
    identity = tuple(f"{name}_{group}_" for group in IDENTITY_GROUPS)
    return [
        column
        for column in obs.columns
        if column.startswith(f"{name}_")
        and (
            not pd.api.types.is_numeric_dtype(obs[column])
            or (pd.api.types.is_integer_dtype(obs[column]) and column.startswith(identity))
        )
    ]


def _majority(
    values: pd.Series, codes: np.ndarray, n_root: int, score: np.ndarray | None = None
) -> dict[str, np.ndarray | pd.api.extensions.ExtensionArray]:
    """Per root, the most common value of the children, its fraction, the number of distinct values and whether it ties.

    Missing values are not counted; a tie goes to the smallest value.
    With `score`, also the mean score of the children that hold the winning value.
    """
    present = values.notna().to_numpy()
    codes = codes[present]
    value_codes, uniques = pd.factorize(values[present], sort=True)
    n_values = max(len(uniques), 1)
    pairs, counts = np.unique(codes.astype(np.int64) * n_values + value_codes, return_counts=True)
    roots, kinds = pairs // n_values, pairs % n_values
    n_distinct = np.bincount(roots, minlength=n_root).astype(np.int64)
    total = np.bincount(roots, weights=counts, minlength=n_root)
    # Most children first and, among equal counts, the smallest value, so the first row of each root is its winner.
    order = np.lexsort((kinds, -counts, roots))
    first = order[np.r_[True, roots[order][1:] != roots[order][:-1]]] if len(order) else order
    winner = np.full(n_root, -1)
    winner[roots[first]] = kinds[first]
    top = np.zeros(n_root)
    top[roots[first]] = counts[first]
    has = n_distinct > 0
    tie = np.bincount(roots[counts == top[roots]], minlength=n_root) > 1
    labels = np.asarray(uniques.astype(str), dtype=object)
    majority = np.full(n_root, None, dtype=object)
    majority[has] = labels[winner[has]]
    ties = pd.array(tie, dtype="boolean")
    ties[~has] = pd.NA
    result: dict[str, np.ndarray | pd.api.extensions.ExtensionArray] = {
        "majority": majority,
        "fraction": np.divide(top, total, out=np.full(n_root, np.nan), where=has),
        "n_distinct": n_distinct,
        "tie": ties,
    }
    if score is not None:
        held = (value_codes == winner[codes]) & ~np.isnan(score[present])
        sums = np.bincount(codes[held], weights=score[present][held], minlength=n_root)
        held_counts = np.bincount(codes[held], minlength=n_root)
        result["majority_score"] = np.divide(sums, held_counts, out=np.full(n_root, np.nan), where=held_counts > 0)
    return result


def combine(
    tables: Mapping[str, AnnData],
    root: str,
    *,
    func: Sequence[str] | None,
    relations: Mapping[str, Mapping[str, str]] | None = None,
    strict: bool = False,
    omissions: Omissions | None = None,
    score: str | None = None,
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

    if score is not None and not any(score in tables[name].obs for name in children):
        raise KeyError(f"score {score!r} is not a column of the objects rolled into {root}: {children}")
    rolled = [*joined, root]
    named: dict[str, Any] = {}
    for name in children:
        _, rows, _ = state[name]
        table = tables[name]
        linked = rows >= 0
        report.add_orphans(name, int((~linked).sum()), table.n_obs)
        codes = rows[linked]
        child_obs = as_frame(table.obs).loc[linked]
        scores = child_obs[score].to_numpy(dtype=np.float64) if score in child_obs else None
        for column in _identity_columns(child_obs, name):
            for key, values in _majority(child_obs[column], codes, n_root, scores).items():
                named[f"{column}_{key}"] = values
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

    var = categorical_annotation(pd.concat(variables), variables)
    if var.index.has_duplicates:
        raise ValueError(f"the rolled-up columns repeat names: {sorted(set(var.index[var.index.duplicated()]))[:5]}")
    if named:
        obs = pd.concat([obs, pd.DataFrame(named, index=obs.index)], axis=1)
    X = np.hstack(blocks) if blocks else np.empty((n_root, 0))
    result = ad.AnnData(X=X, obs=categorize_metadata(profile_obs(obs, root)), var=var)
    stamp(result, resolution="object")
    store = base.uns.get("mantispy", {})
    if "image_table" in store:
        result.uns["mantispy"]["image_table"] = store["image_table"]
    if omissions is None:
        report.log()
    return result
