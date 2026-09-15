"""Matrix access and grouping for every grouped operation.

Nothing outside ``_core`` reads ``adata.X`` or ``adata.layers`` directly, as
``tests/test_api_guards.py`` checks. Every read goes through :func:`get_matrix` and every
grouping through :func:`group_codes`, so a backed, chunked implementation would go into
those two functions.

Built on them:

* :func:`reduce_grouped`: a per-group statistic through the numba kernels (the read path).
* :func:`transform_grouped`: rewrite the matrix group by group (the write path).
* :func:`iter_groups`: the ``(key, rows, block)`` iteration both are built on.

Sixteen call sites across ``pp`` and ``tl`` run their own ``np.flatnonzero(codes == group)``
loop over the matrix from :func:`get_matrix`, because what they compute per group (a
whitening, a permutation null, a chi-square) is not a statistic the kernels can express.
Each loop would need rewriting for a streaming backend. They still take the matrix and
the grouping from this module, so the code to change is easy to find.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from ._numba import MAD, MEAN, MEDIAN, QUANTILE, STD, group_counts, grouped_stat
from ._utils import as_frame

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
    "iter_groups",
    "reduce_grouped",
    "representation",
    "transform_grouped",
]


def get_matrix(adata: AnnData, layer: str | None = None, rows: np.ndarray | None = None) -> np.ndarray:
    """Return the requested matrix as a dense ``float32`` array.

    ``rows`` reads only those rows, so a backed object holds one group in memory instead of
    the whole screen. h5py accepts a fancy index only in increasing order, so the indices
    are sorted for the read and the requested order is restored afterwards.
    """
    matrix: Any = adata.X if layer is None else adata.layers[layer]
    if matrix is None:
        raise ValueError("adata has no matrix to read" if layer is None else f"no layer {layer!r}")

    if rows is not None:
        wanted = np.asarray(rows)
        if _reads_from_disk(matrix):
            order = np.argsort(wanted, kind="stable")
            block = matrix[wanted[order]]
            inverse = np.empty_like(order)
            inverse[order] = np.arange(order.size)
            matrix = block[inverse]
        else:
            matrix = matrix[wanted]

    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    return np.asarray(matrix, dtype=np.float32)


def _reads_from_disk(matrix: Any) -> bool:
    """Whether this matrix is an on-disk dataset rather than an array in memory.

    ``shape`` and ``dtype`` are not enough, because a scipy sparse matrix has both and is in
    memory. ``toarray`` separates them: every in-memory sparse container has it, and an h5py
    or zarr dataset does not.
    """
    if isinstance(matrix, np.ndarray) or hasattr(matrix, "toarray"):
        return False
    return hasattr(matrix, "shape") and hasattr(matrix, "dtype")


def representation(adata: AnnData, use_rep: str | None) -> np.ndarray:
    """``obsm[use_rep]`` if given, otherwise ``X``, as float64.

    Every tool that can score an embedding instead of the features uses this, so the input
    is chosen in one place and the missing-key error says how to compute an embedding.
    """
    if use_rep is None:
        return get_matrix(adata).astype(np.float64)
    if use_rep not in adata.obsm:
        raise KeyError(f"obsm has no representation {use_rep!r}; run sc.pp.pca first, or pass use_rep=None")
    return np.asarray(adata.obsm[use_rep], dtype=np.float64)


def group_codes(adata: AnnData, by: str | Sequence[str] | None) -> tuple[np.ndarray, pd.Index]:
    """Per-row integer group codes plus the ordered group keys.

    Values are used without conversion to strings, so numeric metadata keeps its dtype and
    ``0.4`` and ``0.40`` are one group. Missing values raise instead of forming a group.
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
        mask: Boolean row mask restricting which rows contribute, e.g. controls only. Groups
            are still keyed by the full set of groups present in ``adata``.
        q: Quantile to compute for :data:`QUANTILE`.
        ddof: Delta degrees of freedom for :data:`STD`.

    Returns:
        ``(values, keys, counts)`` where ``values`` is ``(n_groups, n_vars)`` float64,
        ``keys`` indexes the groups and ``counts`` holds the contributing row count.
    """
    codes, keys = group_codes(adata, by)
    source: Any = adata.X if layer is None else adata.layers[layer]
    selected = np.ones(adata.n_obs, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)

    if _reads_from_disk(source):
        # One group at a time, so a screen that does not fit in memory still reduces. Each
        # group's statistic depends only on its own rows, so the result matches the single
        # kernel call (tests/test_backed.py).
        # A group with no contributing rows is NaN, as in the in-memory kernel. Zero would
        # read as a measurement and center a plate with no controls left on 0.0.
        values = np.full((len(keys), adata.n_vars), np.nan)
        counts = np.zeros(len(keys), dtype=np.int64)
        for index in range(len(keys)):
            rows = np.flatnonzero((codes == index) & selected)
            if not rows.size:
                continue
            block = get_matrix(adata, layer, rows=rows)
            values[index] = grouped_stat(block, np.zeros(rows.size, dtype=codes.dtype), 1, stat, q=q, ddof=ddof)[0]
            counts[index] = rows.size
        return values, keys, counts

    matrix = get_matrix(adata, layer)
    if mask is not None:
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

    ``func(key, block)`` receives a group's rows as ``float32`` and returns the
    replacement block. The output is written into one preallocated ``float32`` array, so
    the peak cost is the input plus the output, with no full-size ``float64`` temporaries.
    """
    out = np.empty_like(get_matrix(adata, layer))
    for key, rows, block in iter_groups(adata, by, layer=layer):
        if rows.size:
            out[rows] = func(key, block).astype(np.float32, copy=False)
    return out
