"""Matrix access and grouping for every grouped operation.

Nothing outside ``_core`` reads ``adata.X`` or ``adata.layers`` directly, as ``tests/test_api_guards.py`` checks.
Every read goes through :func:`get_matrix` and every grouping through :func:`group_codes`.
:func:`reduce_grouped` and :func:`iter_groups` both have to yield a group's rows in increasing order: :func:`get_matrix` hands an increasing index straight to h5py and sorts any other one, at the cost of a full-size copy.
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Iterator, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from mantispy._settings import settings

from ._numba import MAD, MEAN, MEDIAN, QUANTILE, STD, group_counts, group_offsets, grouped_stat
from .features import annotation
from .frames import as_frame
from .logging import get_logger

if TYPE_CHECKING:
    from types import FrameType

    from anndata import AnnData

__all__ = [
    "MAD",
    "MEAN",
    "MEDIAN",
    "QUANTILE",
    "STD",
    "float_dtype",
    "get_matrix",
    "group_codes",
    "group_offsets",
    "group_rows",
    "iter_groups",
    "is_lazy",
    "iter_waves",
    "materialize",
    "matrix_dtype",
    "out_of_core",
    "reduce_grouped",
    "reduced_var",
    "representation",
    "transform_grouped",
]


def group_rows(codes: np.ndarray, n_groups: int) -> list[np.ndarray]:
    """Row indices of each group, from one stable ordering of the codes."""
    order, offsets = group_offsets(codes, n_groups)
    return [order[offsets[group] : offsets[group + 1]] for group in range(n_groups)]


def _obsm_source(adata: AnnData, use_rep: str, layer: str | None) -> Any:
    """The 2-D ``obsm[use_rep]`` array, refusing a ``layer`` alongside it or a missing or non-2-D key."""
    if layer is not None:
        raise ValueError(f"use_rep={use_rep!r} and layer={layer!r} are mutually exclusive; pass only one")
    if use_rep not in adata.obsm:
        raise ValueError(f"no obsm {use_rep!r}; have {sorted(adata.obsm)}")
    matrix = adata.obsm[use_rep]
    # A lazy representation stays lazy, so a grouped read takes only the rows it needs.
    if not _reads_from_disk(matrix):
        matrix = np.asarray(matrix)
    if matrix.ndim != 2:
        raise ValueError(f"obsm {use_rep!r} must be 2-D, got {matrix.ndim}-D")
    return matrix


def reduced_var(adata: AnnData, use_rep: str | None, n_cols: int) -> pd.DataFrame:
    """The ``var`` for an object reduced by :func:`reduce_grouped`.

    ``use_rep`` reduces an embedding whose axes are not named features, so its ``var`` is a plain range index over ``n_cols``; otherwise the source ``var`` is carried over unchanged.
    """
    if use_rep is not None:
        # The axes of a learned representation are embedding dimensions, not measurements.
        return annotation(pd.Index([str(index) for index in range(n_cols)]), feature_kind="embedding")
    return as_frame(adata.var).copy()


def float_dtype(dtype: Any) -> np.dtype:
    """The dtype a matrix of ``dtype`` is read and written back as.

    A floating dtype is kept, so ``float32`` stays ``float32`` and ``float64`` stays ``float64``.
    Any other numeric dtype becomes ``float64``, which holds every integer up to 2**53 exactly and can represent the ``NaN`` a computation may produce.
    """
    dtype = np.dtype(dtype)
    return dtype if dtype.kind == "f" else np.dtype(np.float64)


def _source(adata: AnnData, layer: str | None, use_rep: str | None) -> Any:
    """The matrix :func:`get_matrix` reads: ``obsm[use_rep]``, a layer, or ``X``."""
    if use_rep is not None:
        return _obsm_source(adata, use_rep, layer)
    matrix = adata.X if layer is None else adata.layers[layer]
    if matrix is None:
        raise ValueError("adata has no matrix to read" if layer is None else f"no layer {layer!r}")
    return matrix


def out_of_core(adata: AnnData, layer: str | None = None, *, use_rep: str | None = None) -> bool:
    """Whether the matrix :func:`get_matrix` reads is on disk or lazy, so that a grouped operation has to stream it."""
    return _reads_from_disk(_source(adata, layer, use_rep))


def matrix_dtype(adata: AnnData, layer: str | None = None, *, use_rep: str | None = None) -> np.dtype:
    """The dtype :func:`get_matrix` returns for this matrix, read off the matrix without loading it."""
    return float_dtype(_source(adata, layer, use_rep).dtype)


def get_matrix(
    adata: AnnData,
    layer: str | None = None,
    rows: np.ndarray | None = None,
    *,
    use_rep: str | None = None,
    columns: slice | None = None,
) -> np.ndarray:
    """Return the requested matrix as a dense floating array, in the dtype :func:`float_dtype` gives it.

    A ``float32`` or ``float64`` matrix comes back in its own dtype, so a tool that writes its result in that dtype leaves the precision the caller chose untouched.

    ``rows`` reads only those rows, so a backed object holds one group in memory instead of the whole screen.
    h5py accepts a fancy index only in increasing order, so rows asked for in any other order are sorted for the read and the requested order restored afterwards.
    Rows that cover the whole matrix in order are read as one slice rather than selected point by point, and an in-order read hands back the block the backend produced rather than a copy of it, so treat the result as read-only.

    ``use_rep`` reads ``adata.obsm[use_rep]`` instead of ``X`` or a layer; it must be a 2-D representation and cannot be combined with ``layer``.
    ``columns`` reads only that block of columns of the rows asked for, in the same read.

    Reading the whole of an on-disk or lazy matrix larger than :attr:`~mantispy.settings.stream_bytes` warns, naming the function that asked for it, since that function does not stream.
    """
    matrix = _source(adata, layer, use_rep)
    if rows is None:
        _warn_if_whole_read(matrix)
    source = matrix

    def take(index: Any) -> Any:
        """``source[index]``, narrowed to ``columns`` in the same read."""
        return source[index] if columns is None else source[index, columns]

    if rows is not None:
        wanted = np.asarray(rows)
        if _reads_from_disk(matrix):
            # Compared rather than np.diff: subtraction wraps on unsigned dtypes and is a not-equal on booleans.
            increasing = wanted.dtype.kind in "iu" and bool(np.all(wanted[1:] > wanted[:-1]))
            if not increasing:
                order = np.argsort(wanted, kind="stable")
                block = take(wanted[order])
                inverse = np.empty_like(order)
                inverse[order] = np.arange(order.size)
                matrix = block[inverse]
            elif (
                wanted.size
                and 0 <= wanted[0]
                and wanted[-1] < matrix.shape[0]
                and wanted[-1] - wanted[0] + 1 == wanted.size
            ):
                # The ends are bounds-checked because a slice silently clamps where a fancy index is refused.
                matrix = take(slice(int(wanted[0]), int(wanted[-1]) + 1))
            else:
                matrix = take(wanted)
        else:
            matrix = take(wanted)
    elif columns is not None:
        matrix = take(slice(None))
    elif _reads_from_disk(matrix) and not hasattr(matrix, "__array__"):
        # anndata's CSRDataset and CSCDataset have no ``__array__``, so np.asarray below would raise on them.
        matrix = matrix[:]

    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    return np.asarray(matrix, dtype=float_dtype(matrix.dtype))


def _warn_if_whole_read(matrix: Any) -> None:
    """Say so when the whole of a large on-disk or lazy matrix is about to be read into memory."""
    if not _reads_from_disk(matrix):
        return
    n_bytes = int(np.prod(matrix.shape)) * float_dtype(matrix.dtype).itemsize
    if n_bytes > settings.stream_bytes:
        get_logger().warning(
            "%s reads the whole %d x %d matrix into memory, since it does not stream",
            _caller(),
            *matrix.shape,
            deep="subset the object first, or load it with adata.to_memory() if it fits",
        )


def _caller() -> str:
    """The first function outside ``mantispy._core`` on the stack, which is the one that asked for the read."""
    frame: FrameType | None = sys._getframe(1)
    while frame is not None and str(frame.f_globals.get("__name__", "")).startswith("mantispy._core"):
        frame = frame.f_back
    return "an unknown caller" if frame is None else frame.f_code.co_name


def is_lazy(matrix: Any) -> bool:
    """Whether this matrix is a dask array, which is lazy rather than read from a file."""
    return type(matrix).__module__.split(".")[0] == "dask"


def materialize(matrix: Any) -> np.ndarray:
    """Compute a lazy matrix into one array, in the dtype :func:`float_dtype` gives it.

    Each chunk is stored into the result as it is computed, where computing the whole would also hold the chunks to concatenate.
    """
    import dask.array as da

    out = np.empty(matrix.shape, dtype=float_dtype(matrix.dtype))
    da.store(matrix, out, lock=False)
    return out


def _warn_if_not_streamable(matrix: Any) -> None:
    """Say so when a per-group loop over this on-disk matrix will read the whole of it every time.

    anndata indexes a CSR dataset by row without leaving the file.
    On a CSC dataset the same index falls back to ``to_memory()``, so a loop over g groups reads and densifies the entire matrix g times rather than once.
    An in-memory scipy CSC matrix also reports ``format == "csc"`` but never leaves memory, so it must not warn.
    """
    if not _reads_from_disk(matrix):
        return
    if getattr(matrix, "format", None) == "csc":
        get_logger().warning(
            "the matrix on disk is stored by column (CSC), so each group's read loads all of it",
            deep="write it row-major with adata.X = adata.X.tocsr(), or read it into memory with mt.io.read(path)",
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
    """``obsm[use_rep]`` if given, otherwise ``X``, as float64."""
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
        raise KeyError(f"obs is missing grouping column(s): {missing}. obs has: {list(adata.obs.columns)}")

    frame = as_frame(adata.obs)[columns]
    if frame.isna().to_numpy().any():
        empty = [column for position, column in enumerate(columns) if frame.iloc[:, position].isna().any()]
        raise ValueError(f"grouping column(s) contain missing values: {empty}. Filter or fill them first.")

    if len(columns) == 1:
        codes, uniques = pd.Index(frame[columns[0]]).factorize(sort=True)
        keys = uniques if isinstance(uniques, pd.Index) else pd.Index(uniques)
        return codes.astype(np.int32), keys
    return _multi_group_codes(frame, columns)


def _multi_group_codes(frame: pd.DataFrame, columns: list[str]) -> tuple[np.ndarray, pd.Index]:
    """``group_codes`` for several columns, without the object-tuple array ``MultiIndex.factorize`` builds.

    Each column is factorized on its own (a hash over the values, dtype kept), and the per-column codes are
    folded into one mixed-radix integer. Because every column's codes are sorted, that integer is monotone in
    lexicographic order, so factorizing it reproduces ``MultiIndex.factorize(sort=True)``'s codes and key order.
    Materializing the tuples instead costs an object array per row, which dominates grouping at screen scale.
    """
    # By position, so a column named twice in ``columns`` is read as a Series rather than a two-column frame.
    per_column = [pd.factorize(frame.iloc[:, position], sort=True) for position in range(len(columns))]
    sizes = [len(uniques) for _, uniques in per_column]
    product = 1
    for size in sizes:
        product *= max(size, 1)
        if product > (1 << 62):
            # The mixed-radix fold would overflow int64; the object-tuple path is correct whatever the sizes.
            codes, uniques = pd.MultiIndex.from_frame(frame).factorize(sort=True)
            return codes.astype(np.int32), uniques if isinstance(uniques, pd.Index) else pd.Index(uniques)
    composite = per_column[0][0].astype(np.int64)
    for (column_codes, _), size in zip(per_column[1:], sizes[1:], strict=True):
        composite = composite * size + column_codes
    codes, present = pd.factorize(composite, sort=True)
    # Decode each present combination back to its per-column position, then name the groups with the
    # original values, so keys is the same MultiIndex as the object-tuple path would have produced.
    remainder = np.asarray(present, dtype=np.int64)
    decoded = []
    for size in reversed(sizes[1:]):
        decoded.append(remainder % size)
        remainder = remainder // size
    decoded.append(remainder)
    decoded.reverse()
    levels = [uniques.take(position) for (_, uniques), position in zip(per_column, decoded, strict=True)]
    keys = pd.MultiIndex.from_arrays(levels, names=columns)
    return codes.astype(np.int32), keys


def iter_waves(
    adata: AnnData,
    grouping: tuple[np.ndarray, pd.Index],
    layer: str | None = None,
    *,
    use_rep: str | None = None,
    mask: np.ndarray | None = None,
    by_columns: bool = False,
) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray, slice, np.ndarray]]:
    """Yield ``(groups, rows, codes, columns, block)``, reading several groups' rows of an on-disk or lazy matrix at once.

    Groups are taken in key order and packed together until their rows would exceed :attr:`~mantispy.settings.stream_bytes`; a group larger than that is read on its own.
    ``by_columns=True`` reads such a group in blocks of columns that each fit the budget, for a caller whose statistic is computed column by column.
    ``groups`` holds the positions of a wave's groups in the keys, ``rows`` the increasing row indices read into ``block``, ``codes`` the position of each row's group within ``groups``, and ``columns`` the columns ``block`` holds.
    A lazy matrix is read through those rows alone, so each wave touches only the chunks that hold them, and each of those once.
    ``mask`` restricts which rows are read, and a group it leaves empty is skipped.
    """
    codes, keys = grouping
    source = _source(adata, layer, use_rep)
    _warn_if_not_streamable(source)
    order, offsets = group_offsets(codes, len(keys))
    if mask is not None:
        # Selecting from the stable ordering keeps every group's rows in increasing order.
        order = order[mask[order]]
        offsets = np.concatenate([[0], np.cumsum(np.bincount(codes[mask], minlength=len(keys)))])
    sizes = np.diff(offsets)
    row_bytes = max(source.shape[1], 1) * float_dtype(source.dtype).itemsize
    limit = max(settings.stream_bytes // row_bytes, 1)

    def read(groups: list[int]) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray, slice, np.ndarray]]:
        # The groups of a wave are consecutive in the ordering, so their rows are one slice of it.
        members = order[offsets[groups[0]] : offsets[groups[-1] + 1]]
        local = np.repeat(np.arange(len(groups)), sizes[groups])
        increasing = np.argsort(members, kind="stable")
        rows = members[increasing]
        n_cols = source.shape[1]
        # Sized as column_block in _corr sizes a slice of the data, against the read budget.
        width = max(settings.stream_bytes // (rows.size * float_dtype(source.dtype).itemsize), 1)
        if not by_columns or rows.size <= limit or width >= n_cols:
            width = max(n_cols, 1)
        for start in range(0, max(n_cols, 1), width):
            columns = slice(start, min(start + width, n_cols))
            block = get_matrix(adata, layer, rows=rows, use_rep=use_rep, columns=None if width >= n_cols else columns)
            yield np.asarray(groups), rows, local[increasing], columns, block

    wave: list[int] = []
    total = 0
    for group in np.flatnonzero(sizes):
        if wave and total + sizes[group] > limit:
            yield from read(wave)
            wave, total = [], 0
        wave.append(int(group))
        total += int(sizes[group])
    if wave:
        yield from read(wave)


def iter_groups(
    adata: AnnData,
    by: str | Sequence[str] | None,
    layer: str | None = None,
    *,
    use_rep: str | None = None,
    grouping: tuple[np.ndarray, pd.Index] | None = None,
) -> Iterator[tuple[Any, np.ndarray, np.ndarray]]:
    """Yield ``(key, row_index, block)`` per group, in group-key order.

    An on-disk or lazy matrix is read in the waves :func:`iter_waves` packs, and a group with no rows is skipped there.
    ``grouping`` reuses a ``(codes, keys)`` pair a caller already computed with :func:`group_codes` for ``by``, so a function that groups the same object more than once factorizes it only once.
    """
    codes, keys = group_codes(adata, by) if grouping is None else grouping
    if out_of_core(adata, layer, use_rep=use_rep):
        for groups, rows, wave_codes, _, block in iter_waves(adata, (codes, keys), layer, use_rep=use_rep):
            wave_order, wave_offsets = group_offsets(wave_codes, groups.size)
            for local, group in enumerate(groups):
                picked = wave_order[wave_offsets[local] : wave_offsets[local + 1]]
                yield keys[group], rows[picked], block[picked]
        return
    order = np.argsort(codes, kind="stable")
    bounds = np.searchsorted(codes[order], np.arange(len(keys) + 1))
    for index, key in enumerate(keys):
        rows = order[bounds[index] : bounds[index + 1]]
        yield key, rows, get_matrix(adata, layer, rows=rows, use_rep=use_rep)


def reduce_grouped(
    adata: AnnData,
    by: str | Sequence[str] | None,
    stat: int,
    layer: str | None = None,
    mask: np.ndarray | None = None,
    q: float = 0.5,
    ddof: int = 1,
    *,
    use_rep: str | None = None,
    grouping: tuple[np.ndarray, pd.Index] | None = None,
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
        use_rep: Reduce ``adata.obsm[use_rep]`` instead of ``X``; ``values`` then has one column per axis of that representation.
            A lazy representation is computed whole, once, rather than streamed group by group like ``X``.
            Mutually exclusive with ``layer``.
        grouping: Reuse a ``(codes, keys)`` pair already computed with :func:`group_codes` for ``by``, so a caller that reduces the same object twice factorizes it once.

    Returns:
        ``(values, keys, counts)`` where ``values`` is ``(n_groups, n_cols)`` float64 (``n_cols`` is ``n_vars`` for ``X``/``layer`` and the representation's width for ``use_rep``), ``keys`` indexes the groups and ``counts`` holds the contributing row count.

    Raises:
        ValueError: ``mask`` does not hold one entry per row of ``adata``, or ``use_rep`` and ``layer`` are both given, or ``use_rep`` is not a 2-D ``obsm``.
    """
    codes, keys = group_codes(adata, by) if grouping is None else grouping
    source: Any
    if use_rep is not None:
        source = _obsm_source(adata, use_rep, layer)
    else:
        source = adata.X if layer is None else adata.layers[layer]
    n_cols = source.shape[1] if use_rep is not None else adata.n_vars
    selected = None if mask is None else np.asarray(mask, dtype=bool)
    # Checked up front because the backed path indexes the mask per group and would not notice a mask that is too long.
    if selected is not None and selected.shape != (adata.n_obs,):
        raise ValueError(f"mask must be one boolean per row: got shape {selected.shape} for {adata.n_obs} rows")

    # A lazy representation is computed once: it is as narrow as an in-memory one, and each wave over it would run its whole graph again.
    lazy_rep = use_rep is not None and is_lazy(source)
    if _reads_from_disk(source) and not lazy_rep:
        # A group with no contributing rows is NaN, as in the in-memory kernel.
        values = np.full((len(keys), n_cols), np.nan)
        counts = np.zeros(len(keys), dtype=np.int64)
        waves = iter_waves(adata, (codes, keys), layer, use_rep=use_rep, mask=selected, by_columns=True)
        for groups, _, wave_codes, columns, block in waves:
            values[groups, columns] = grouped_stat(block, wave_codes, groups.size, stat, q=q, ddof=ddof)
            counts[groups] = np.bincount(wave_codes, minlength=groups.size)
        return values, keys, counts

    if lazy_rep:
        matrix = materialize(source)
    else:
        matrix = get_matrix(adata, layer, use_rep=use_rep)
    # The kernel skips the rows the mask leaves out, so they are never copied out of the matrix.
    values = grouped_stat(matrix, codes, len(keys), stat, q=q, ddof=ddof, mask=selected)
    return values, keys, group_counts(codes if selected is None else codes[selected], len(keys))


def transform_grouped(
    adata: AnnData,
    by: str | Sequence[str] | None,
    func: Callable[[Any, np.ndarray], np.ndarray],
    layer: str | None = None,
    *,
    use_rep: str | None = None,
    n_cols: int | None = None,
    rowwise: bool = False,
    grouping: tuple[np.ndarray, pd.Index] | None = None,
) -> Any:
    """Rewrite the matrix one group at a time.

    ``func(key, block)`` receives a group's rows as :func:`get_matrix` reads them and returns the replacement block.
    The output is written into one preallocated array of that same dtype, so the peak cost is the input plus the output, with no full-size temporaries in a wider dtype.
    ``rowwise=True`` declares that ``func`` transforms every row on its own, as a fitted centre and scale do, so a lazy (dask) matrix can hand it a group's rows one chunk at a time; the result is then lazy too, in the same row chunks, and computed when it is read or written.
    In memory it also hands ``func`` a group in slices that fit :attr:`~mantispy.settings.stream_bytes`, which bounds the temporaries it makes.
    Otherwise every group reaches ``func`` whole, as a ranking needs.
    ``use_rep`` rewrites ``adata.obsm[use_rep]`` instead of ``X`` or a layer, and ``n_cols`` sets the width of the result when ``func`` changes it.
    ``grouping`` reuses a ``(codes, keys)`` pair already computed with :func:`group_codes` for ``by``.
    """
    codes, keys = group_codes(adata, by) if grouping is None else grouping
    source = _source(adata, layer, use_rep)
    dtype = float_dtype(source.dtype)
    width = source.shape[1] if n_cols is None else n_cols
    if rowwise and is_lazy(source):
        return _transform_lazy(source, codes, keys, func, dtype, width)
    # Shaped from the source, not read off it: get_matrix with no rows reads all of it.
    out = np.empty((source.shape[0], width), dtype=dtype)
    if rowwise and not _reads_from_disk(source):
        # A row-wise func is fed a group a slice at a time, read straight from the matrix, so neither the group's copy nor the temporaries func makes outgrow the read budget.
        step = max(settings.stream_bytes // (max(width, source.shape[1]) * 8), 1)
        order, offsets = group_offsets(codes, len(keys))
        for group, key in enumerate(keys):
            rows = order[offsets[group] : offsets[group + 1]]
            for start in range(0, rows.size, step):
                picked = rows[start : start + step]
                block = get_matrix(adata, layer, rows=picked, use_rep=use_rep)
                out[picked] = func(key, block).astype(dtype, copy=False)
        return out
    for key, rows, block in iter_groups(adata, by, layer=layer, use_rep=use_rep, grouping=(codes, keys)):
        if rows.size:
            out[rows] = func(key, block).astype(dtype, copy=False)
    return out


def _transform_lazy(
    source: Any,
    codes: np.ndarray,
    keys: pd.Index,
    func: Callable[[Any, np.ndarray], np.ndarray],
    dtype: np.dtype,
    width: int,
) -> Any:
    """:func:`transform_grouped` on a dask array: each row chunk is rewritten group by group, alongside its rows' codes."""
    import dask.array as da

    matrix = source.rechunk({1: -1})
    row_codes = da.from_array(codes, chunks=(matrix.chunks[0],))

    def rewrite(block: np.ndarray, block_codes: np.ndarray) -> np.ndarray:
        block = np.asarray(block, dtype=dtype)
        out = np.empty((block.shape[0], width), dtype=dtype)
        order, offsets = group_offsets(block_codes, len(keys))
        for group in np.flatnonzero(np.diff(offsets)):
            rows = order[offsets[group] : offsets[group + 1]]
            out[rows] = func(keys[int(group)], block[rows]).astype(dtype, copy=False)
        return out

    return da.blockwise(
        rewrite, "ij", matrix, "ij", row_codes, "i", dtype=dtype, adjust_chunks={"j": width}, concatenate=True
    )
