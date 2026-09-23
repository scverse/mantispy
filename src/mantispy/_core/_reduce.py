"""Matrix access and grouping for every grouped operation.

Nothing outside ``_core`` reads ``adata.X`` or ``adata.layers`` directly, as ``tests/test_api_guards.py`` checks.
Every read goes through :func:`get_matrix` and every grouping through :func:`group_codes`, so a backed, chunked implementation would go into those two functions.

Built on them:

* :func:`reduce_grouped`: a per-group statistic through the numba kernels (the read path).
* :func:`transform_grouped`: rewrite the matrix group by group (the write path), over :func:`iter_groups`.
* :func:`iter_groups`: the ``(key, rows, block)`` iteration, which yields a group's rows in increasing order.

:func:`reduce_grouped` orders its own groups rather than going through :func:`iter_groups`, because it applies ``mask`` to each group's rows before reading them.
Both orderings have to stay increasing: :func:`get_matrix` hands an increasing index straight to h5py and sorts any other one, at the cost of a full-size copy.

A dozen call sites across ``pp`` and ``tl`` loop over the groups themselves, because what they compute per group (a whitening, a permutation null, a chi-square) is not a statistic the kernels can express.
Most take their rows from :func:`~mantispy._core._numba.group_offsets`, re-exported here, rather than scanning ``codes == group`` once per group: that scan is O(n_obs) per group, so a loop over g groups costs O(g * n_obs) where one stable ordering serves every group.
A handful still scan, among them ``tl/_heterogeneity.py``, ``pp/_batch.py`` and ``pp/_sphere.py``; see #113 for the measured cost.
Each loop would need rewriting for a streaming backend.
They still take the matrix and the grouping from this module, so the code to change is easy to find.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from ._numba import MAD, MEAN, MEDIAN, QUANTILE, STD, group_counts, group_offsets, grouped_stat
from .frames import as_frame
from .logging import get_logger

if TYPE_CHECKING:
    from anndata import AnnData

__all__ = [
    "MAD",
    "MEAN",
    "MEDIAN",
    "QUANTILE",
    "STD",
    "get_matrix",
    "group_codes",
    "group_offsets",
    "group_rows",
    "iter_groups",
    "reduce_grouped",
    "representation",
    "transform_grouped",
]


def group_rows(codes: np.ndarray, n_groups: int) -> list[np.ndarray]:
    """Row indices of each group, taken from one stable ordering rather than by scanning the codes per group."""
    order, offsets = group_offsets(codes, n_groups)
    return [order[offsets[group] : offsets[group + 1]] for group in range(n_groups)]


def get_matrix(adata: AnnData, layer: str | None = None, rows: np.ndarray | None = None) -> np.ndarray:
    """Return the requested matrix as a dense ``float32`` array.

    ``rows`` reads only those rows, so a backed object holds one group in memory instead of the whole screen.
    h5py accepts a fancy index only in increasing order, so rows asked for in any other order are sorted for the read and the requested order restored afterwards.
    Every grouped path here asks for a group's rows in increasing order already, and that restoring gather is a full-size copy of what was just read, so it is done only when the order actually differs.
    Rows that cover the whole matrix in order are read as one slice rather than selected point by point, and an in-order read hands back the block the backend produced rather than a copy of it, so treat the result as read-only.
    """
    matrix: Any = adata.X if layer is None else adata.layers[layer]
    if matrix is None:
        raise ValueError("adata has no matrix to read" if layer is None else f"no layer {layer!r}")

    if rows is not None:
        wanted = np.asarray(rows)
        if _reads_from_disk(matrix):
            # Only an integer index takes a shortcut, and its order is settled by comparison rather than by
            # np.diff: subtraction wraps on an unsigned dtype, which would read a descending index as an
            # increasing one, and on a boolean array it is a not-equal, which says nothing about order at all.
            # Anything else goes the way every read went before, and is refused by the backend if it is invalid.
            increasing = wanted.dtype.kind in "iu" and bool(np.all(wanted[1:] > wanted[:-1]))
            if not increasing:
                # The order h5py refuses: read the rows sorted, then put them back as they were asked for.
                order = np.argsort(wanted, kind="stable")
                block = matrix[wanted[order]]
                inverse = np.empty_like(order)
                inverse[order] = np.arange(order.size)
                matrix = block[inverse]
            elif (
                wanted.size
                and 0 <= wanted[0]
                and wanted[-1] < matrix.shape[0]
                and wanted[-1] - wanted[0] + 1 == wanted.size
            ):
                # Increasing with no gaps is a range, and asking for it as one lets the backend read a
                # contiguous block instead of selecting the rows point by point. Every group of a file
                # stored in the grouping's own order looks like this, and so does ``by=None``, which asks
                # for all of them. The ends are checked against the dataset first, because a slice would
                # silently clamp to what is there where a fancy index is refused.
                matrix = matrix[int(wanted[0]) : int(wanted[-1]) + 1]
            else:
                matrix = matrix[wanted]
        else:
            matrix = matrix[wanted]
    elif _reads_from_disk(matrix) and not hasattr(matrix, "__array__"):
        # anndata's CSRDataset and CSCDataset have no ``__array__``, so the read below sees a sequence of
        # sparse rows and raises rather than returning the matrix. Asking for every row gives the scipy
        # matrix that ``toarray`` then flattens out. An h5py dataset does have one and is left alone,
        # because reading through it converts to float32 as it goes instead of afterwards.
        matrix = matrix[:]

    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    return np.asarray(matrix, dtype=np.float32)


def _warn_if_not_streamable(matrix: Any) -> None:
    """Say so when a per-group loop over this on-disk matrix will read the whole of it every time.

    anndata indexes a CSR dataset by row without leaving the file, which is what makes streaming work.
    On a CSC dataset the same index falls back to ``to_memory()``, so a loop over g groups reads and
    densifies the entire matrix g times rather than once, and the groups are where that is least visible.
    """
    if getattr(matrix, "format", None) == "csc":
        get_logger().warning(
            "the matrix on disk is stored column-major (CSC), which cannot be read row by row: every "
            "group's read loads the whole matrix. Store it row-major before writing, with "
            "adata.X = adata.X.tocsr(), or read the object into memory with mt.io.read(path)."
        )


def _reads_from_disk(matrix: Any) -> bool:
    """Whether this matrix is an on-disk dataset rather than an array in memory.

    ``shape`` and ``dtype`` are not enough, because a scipy sparse matrix has both and is in memory.
    ``toarray`` separates them: every in-memory sparse container has it, and an h5py or zarr dataset does not.
    """
    if isinstance(matrix, np.ndarray) or hasattr(matrix, "toarray"):
        return False
    return hasattr(matrix, "shape") and hasattr(matrix, "dtype")


def representation(adata: AnnData, use_rep: str | None) -> np.ndarray:
    """``obsm[use_rep]`` if given, otherwise ``X``, as float64.

    Every tool that can score an embedding instead of the features uses this, so the input is chosen in one place and the missing-key error says how to compute an embedding.
    """
    if use_rep is None:
        return get_matrix(adata).astype(np.float64)
    if use_rep not in adata.obsm:
        raise KeyError(f"obsm has no representation {use_rep!r}; run sc.pp.pca first, or pass use_rep=None")
    return np.asarray(adata.obsm[use_rep], dtype=np.float64)


def group_codes(adata: AnnData, by: str | Sequence[str] | None) -> tuple[np.ndarray, pd.Index]:
    """Per-row integer group codes plus the ordered group keys.

    Values are used without conversion to strings, so numeric metadata keeps its dtype and ``0.4`` and ``0.40`` are one group.
    Missing values raise instead of forming a group.
    """
    if by is None:
        return np.zeros(adata.n_obs, dtype=np.int32), pd.Index(["all"])

    columns = [by] if isinstance(by, str) else list(by)
    missing = [column for column in columns if column not in adata.obs]
    if missing:
        raise KeyError(f"obs is missing grouping column(s): {missing}")

    frame = as_frame(adata.obs)[columns]
    if frame.isna().to_numpy().any():
        empty = [column for column in columns if frame[column].isna().any()]
        raise ValueError(f"grouping column(s) contain missing values: {empty}. Filter or fill them first.")

    values: pd.Index = pd.Index(frame[columns[0]]) if len(columns) == 1 else pd.MultiIndex.from_frame(frame)
    codes, uniques = values.factorize(sort=True)
    # MultiIndex is already an Index; wrapping it again would flatten it to tuples.
    keys = uniques if isinstance(uniques, pd.Index) else pd.Index(uniques)
    return codes.astype(np.int32), keys


def iter_groups(
    adata: AnnData, by: str | Sequence[str] | None, layer: str | None = None
) -> Iterator[tuple[Any, np.ndarray, np.ndarray]]:
    """Yield ``(key, row_index, block)`` per group, in group-key order."""
    codes, keys = group_codes(adata, by)
    _warn_if_not_streamable(adata.X if layer is None else adata.layers[layer])
    order = np.argsort(codes, kind="stable")
    bounds = np.searchsorted(codes[order], np.arange(len(keys) + 1))
    for index, key in enumerate(keys):
        rows = order[bounds[index] : bounds[index + 1]]
        yield key, rows, get_matrix(adata, layer, rows=rows)


def reduce_grouped(
    adata: AnnData,
    by: str | Sequence[str] | None,
    stat: int,
    layer: str | None = None,
    mask: np.ndarray | None = None,
    q: float = 0.5,
    ddof: int = 1,
) -> tuple[np.ndarray, pd.Index, np.ndarray]:
    """Per-group statistic over the feature matrix.

    Args:
        adata: Object to reduce.
        by: Grouping column(s), or ``None`` for a single group.
        stat: One of :data:`MEAN`, :data:`MEDIAN`, :data:`MAD`, :data:`STD`, :data:`QUANTILE`.
        layer: Layer to read instead of ``X``.
        mask: Boolean row mask restricting which rows contribute, e.g. controls only.
            Groups are still keyed by the full set of groups present in ``adata``.
        q: Quantile to compute for :data:`QUANTILE`.
        ddof: Delta degrees of freedom for :data:`STD`.

    Returns:
        ``(values, keys, counts)`` where ``values`` is ``(n_groups, n_vars)`` float64, ``keys`` indexes the groups and ``counts`` holds the contributing row count.

    Raises:
        ValueError: ``mask`` does not hold one entry per row of ``adata``.
    """
    codes, keys = group_codes(adata, by)
    source: Any = adata.X if layer is None else adata.layers[layer]
    selected = None if mask is None else np.asarray(mask, dtype=bool)
    # Checked here rather than left to whichever branch runs. Selecting a group's rows with the mask reads
    # only the entries that group owns, so a mask longer than the object would go unnoticed on a backed
    # object and raise in memory, which is the divergence the backed tests exist to catch.
    if selected is not None and selected.shape != (adata.n_obs,):
        raise ValueError(f"mask must be one boolean per row: got shape {selected.shape} for {adata.n_obs} rows")

    if _reads_from_disk(source):
        _warn_if_not_streamable(source)
        # One group at a time, so a screen that does not fit in memory still reduces.
        # Each group's statistic depends only on its own rows, so the result matches the single kernel call (tests/test_backed.py).
        # A group with no contributing rows is NaN, as in the in-memory kernel.
        # Zero would read as a measurement and center a plate with no controls left on 0.0.
        values = np.full((len(keys), adata.n_vars), np.nan)
        counts = np.zeros(len(keys), dtype=np.int64)
        # One stable ordering serves every group. Scanning ``codes == index`` per group instead is two
        # full-length passes each, so the index arithmetic grows with the group count and at well level
        # outweighs the reads it is preparing. Without a mask the ordering is already the answer, and
        # group_rows' slices are views, so the common case copies nothing.
        for index, members in enumerate(group_rows(codes, len(keys))):
            rows = members if selected is None else members[selected[members]]
            if not rows.size:
                continue
            block = get_matrix(adata, layer, rows=rows)
            values[index] = grouped_stat(block, np.zeros(rows.size, dtype=codes.dtype), 1, stat, q=q, ddof=ddof)[0]
            counts[index] = rows.size
        return values, keys, counts

    matrix = get_matrix(adata, layer)
    if selected is not None:
        codes, matrix = codes[selected], matrix[selected]
    values = grouped_stat(matrix, codes, len(keys), stat, q=q, ddof=ddof)
    return values, keys, group_counts(codes, len(keys))


def transform_grouped(
    adata: AnnData,
    by: str | Sequence[str] | None,
    func: Callable[[Any, np.ndarray], np.ndarray],
    layer: str | None = None,
) -> np.ndarray:
    """Rewrite the matrix one group at a time.

    ``func(key, block)`` receives a group's rows as ``float32`` and returns the replacement block.
    The output is written into one preallocated ``float32`` array, so the peak cost is the input plus the output, with no full-size ``float64`` temporaries.
    """
    # Shaped from the object, not read off the matrix: get_matrix with no rows reads all of it.
    out = np.empty(adata.shape, dtype=np.float32)
    for key, rows, block in iter_groups(adata, by, layer=layer):
        if rows.size:
            out[rows] = func(key, block).astype(np.float32, copy=False)
    return out
