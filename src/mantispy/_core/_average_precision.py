"""Average precision of every query profile against its implied positive and negative pairs.

copairs builds the positive and negative pairs as explicit index tables and ranks each query's pairs with one global
sort. Here the pairs are implied by group codes instead: a positive of a query shares its positive label (and every
other ``pos_sameby`` column) and differs in every ``pos_diffby`` column, a negative shares the query's ``neg_sameby``
bucket, differs in every ``neg_diffby`` column and, for a multilabel column, shares none of the query's labels.
Nothing quadratic is stored. Each query's positives are scored first and sorted; its negatives are then streamed past
them and only counted, as the number ranked above each positive is all average precision needs.

The ranking follows copairs exactly: a pair's key is ``float32(1) - similarity`` and a tie between a positive and a
negative ranks the positive first. Every similarity is the float32 value copairs computes,
reproducing numpy's pairwise-summation tree, so the average precisions agree with copairs to the last bits of their
float64 sum.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from llvmlite import ir
from numba import njit, types
from numba.core import cgutils
from numba.extending import intrinsic
from scipy.spatial import distance as scipy_distance
from scipy.spatial.distance import cdist

from mantispy._core._numba import N_CHUNKS, group_offsets

if TYPE_CHECKING:
    from builtins import range as prange
else:
    from numba import prange

#: The distances with a pairwise kernel of their own, as copairs defines them; ``abs_cosine`` and ``correlation``
#: are cosines of transformed rows.
COSINE, ABS_COSINE, CORRELATION, EUCLIDEAN, MANHATTAN, CHEBYSHEV = range(6)
BUILTIN = {
    "cosine": COSINE,
    "abs_cosine": ABS_COSINE,
    "correlation": CORRELATION,
    "euclidean": EUCLIDEAN,
    "manhattan": MANHATTAN,
    "chebyshev": CHEBYSHEV,
}
#: The metric names scipy's cdist accepts, which copairs accepts too.
SCIPY_METRICS = frozenset(vars(scipy_distance)["_METRICS_NAMES"])
#: scipy metrics that estimate a scale from each batch of pairs, so a pair's similarity has no single value.
BATCH_DEPENDENT = ("seuclidean", "mahalanobis")
#: scipy metrics bounded in ``[0, 1]``, which copairs turns into a similarity as ``1 - d`` rather than ``1 / (1 + d)``.
_BOUNDED = ("jaccard", "hamming")

#: Column of :func:`ap_frame` naming the profile an AP row scores; not metadata, so not ``Metadata_``-prefixed.
PROFILE = "__profile__"
#: numpy sums at most this many float32 terms with eight running accumulators before it splits the range in two.
_PW_BLOCK = 128
#: Rows numpy sums with eight accumulators; shorter sums run one accumulator.
_PW_UNROLL = 8
#: Bytes of one similarity block (queries by bucket rows) for a scipy metric.
_BLOCK_BYTES = 2**26


def check_distance(distance: str) -> None:
    """Raise unless ``distance`` is supported."""
    if not isinstance(distance, str) or not (distance in BUILTIN or distance in SCIPY_METRICS):
        raise ValueError(
            f"distance must be one of {sorted(BUILTIN)} or a scipy.spatial.distance metric name, got {distance!r}"
        )
    if distance in BATCH_DEPENDENT:
        raise ValueError(
            f"distance={distance!r} estimates its scale from each batch of pairs, so a pair has no single "
            "similarity to rank by. Standardize the features first and use 'euclidean'."
        )


# ---------------------------------------------------------------------------------------------------------------------
# Exact float32 similarities: numpy's pairwise summation, without recursion so the kernels cache.


def summation_plan(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Numpy's pairwise-summation tree over ``n`` terms as leaves and a postfix program.

    numpy sums up to 128 terms with eight running accumulators and otherwise splits the range at half its length,
    rounded down to a multiple of eight, and adds the two halves. ``ops[k] >= 0`` pushes the sum of leaf ``ops[k]``,
    ``-1`` adds the two topmost sums, so the tree is evaluated in the order numpy's recursion evaluates it.
    """
    leaf_lo: list[int] = []
    leaf_n: list[int] = []
    ops: list[int] = []

    def visit(lo: int, size: int) -> None:
        if size <= _PW_BLOCK:
            ops.append(len(leaf_lo))
            leaf_lo.append(lo)
            leaf_n.append(size)
            return
        half = size // 2
        half -= half % _PW_UNROLL
        visit(lo, half)
        visit(lo + half, size - half)
        ops.append(-1)

    visit(0, n)
    return np.array(leaf_lo, np.int64), np.array(leaf_n, np.int64), np.array(ops, np.int64)


@njit(nogil=True, cache=True, inline="always")
def _term(a, b, kind):
    if kind == 0:
        return a * b
    if kind == 1:
        d = a - b
        return d * d
    return abs(a - b)


@njit(nogil=True, cache=True)
def _leaf_sum(x, y, lo, n, kind):
    """Numpy's float32 sum of ``term(x[i], y[i])`` over ``[lo, lo + n)``, ``n <= 128``.

    The indices are unsigned so numba drops its negative-index checks, which would otherwise block vectorization.
    """
    lo = np.uint64(lo)
    n = np.uint64(n)
    eight = np.uint64(8)
    if n < eight:
        res = np.float32(0.0)
        for i in range(n):
            res += _term(x[lo + i], y[lo + i], kind)
        return res
    r0 = _term(x[lo], y[lo], kind)
    r1 = _term(x[lo + np.uint64(1)], y[lo + np.uint64(1)], kind)
    r2 = _term(x[lo + np.uint64(2)], y[lo + np.uint64(2)], kind)
    r3 = _term(x[lo + np.uint64(3)], y[lo + np.uint64(3)], kind)
    r4 = _term(x[lo + np.uint64(4)], y[lo + np.uint64(4)], kind)
    r5 = _term(x[lo + np.uint64(5)], y[lo + np.uint64(5)], kind)
    r6 = _term(x[lo + np.uint64(6)], y[lo + np.uint64(6)], kind)
    r7 = _term(x[lo + np.uint64(7)], y[lo + np.uint64(7)], kind)
    stop = n - (n % eight)
    for i in range(eight, stop, eight):
        b = lo + i
        r0 += _term(x[b], y[b], kind)
        r1 += _term(x[b + np.uint64(1)], y[b + np.uint64(1)], kind)
        r2 += _term(x[b + np.uint64(2)], y[b + np.uint64(2)], kind)
        r3 += _term(x[b + np.uint64(3)], y[b + np.uint64(3)], kind)
        r4 += _term(x[b + np.uint64(4)], y[b + np.uint64(4)], kind)
        r5 += _term(x[b + np.uint64(5)], y[b + np.uint64(5)], kind)
        r6 += _term(x[b + np.uint64(6)], y[b + np.uint64(6)], kind)
        r7 += _term(x[b + np.uint64(7)], y[b + np.uint64(7)], kind)
    res = ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7))
    for i in range(stop, n):
        res += _term(x[lo + i], y[lo + i], kind)
    return res


@njit(nogil=True, cache=True)
def _pairwise_sum(x, y, kind, leaf_lo, leaf_n, ops, stack):
    """Numpy's float32 ``np.sum(term(x, y))`` along a row, evaluated from :func:`summation_plan`."""
    if ops.shape[0] == 1:
        return _leaf_sum(x, y, leaf_lo[0], leaf_n[0], kind)
    top = 0
    for k in range(ops.shape[0]):
        op = ops[k]
        if op >= 0:
            stack[top] = _leaf_sum(x, y, leaf_lo[op], leaf_n[op], kind)
            top += 1
        else:
            top -= 1
            stack[top - 1] = stack[top - 1] + stack[top]
    return stack[0]


@njit(nogil=True, cache=True)
def _similarity(rows, scale, i, j, metric, leaf_lo, leaf_n, ops, stack):
    """Copairs' float32 similarity of rows ``i`` and ``j`` (prepared by :func:`prepare_rows`), bit for bit."""
    x = rows[i]
    y = rows[j]
    one = np.float32(1.0)
    if metric == COSINE:
        return _pairwise_sum(x, y, 0, leaf_lo, leaf_n, ops, stack)
    if metric == ABS_COSINE:
        return abs(_pairwise_sum(x, y, 0, leaf_lo, leaf_n, ops, stack))
    if metric == CORRELATION:
        return _pairwise_sum(x, y, 0, leaf_lo, leaf_n, ops, stack) / np.sqrt(scale[i] * scale[j])
    if metric == EUCLIDEAN:
        return one / (one + np.sqrt(_pairwise_sum(x, y, 1, leaf_lo, leaf_n, ops, stack)))
    if metric == MANHATTAN:
        return one / (one + _pairwise_sum(x, y, 2, leaf_lo, leaf_n, ops, stack))
    m = np.float32(0.0)
    for f in range(x.shape[0]):
        d = abs(x[f] - y[f])
        if d > m:
            m = d
    return one / (one + m)


def prepare_rows(features: np.ndarray, distance: str) -> tuple[np.ndarray, np.ndarray]:
    """Rows and per-row scales the similarity kernel reads, transformed per row as copairs transforms each batch.

    Cosines read rows divided by their norm, the correlation reads centred rows and their squared norms; every
    transformation acts on one row at a time, so it gives the same float32 values whether applied to a batch of
    pairs or to the whole matrix once.
    """
    x = np.ascontiguousarray(features, dtype=np.float32)
    scale = np.ones(len(x), dtype=np.float32)
    if distance in ("cosine", "abs_cosine"):
        x = np.ascontiguousarray(x / np.linalg.norm(x, axis=1)[:, np.newaxis])
    elif distance == "correlation":
        x = np.ascontiguousarray(x - x.mean(axis=1, keepdims=True))
        scale = (x**2).sum(axis=1)
    return x, scale


# ---------------------------------------------------------------------------------------------------------------------
# Panels: the same float32 values as :func:`_similarity`, for a block of queries against a block of candidates.
#
# The candidates are copied into a panel with one row per feature, so ``panel[f, w]`` is feature ``f`` of candidate
# ``w``. Each (query, candidate) pair keeps its own eight accumulators, filled in numpy's order, and the arithmetic of
# each pair is the one :func:`_leaf_sum` does; only the loop over candidates is innermost, so the compiler runs it
# over vector lanes, and each panel value read serves :data:`QUERIES` queries.

#: Candidates in one panel.
PANEL = 64
#: Queries scored against a panel at once.
QUERIES = 8


@njit(nogil=True, cache=True)
def fill_panel(rows, cand, n_cand, panel):
    """Copy rows ``cand[:n_cand]`` into the columns of ``panel``, one panel row per feature."""
    for w in range(n_cand):
        r = rows[cand[w]]
        for f in range(r.shape[0]):
            panel[f, w] = r[f]


@njit(nogil=True, cache=True, inline="always")
def _accumulate(dst, a, p, width, kind, first):
    """``dst[w] = term(a, p[w])`` when ``first``, else ``dst[w] += term(a, p[w])``, over ``width`` lanes.

    The branches sit outside the loops, so each loop is a plain vector loop.
    """
    if kind == 0:
        if first:
            for w in range(width):
                dst[w] = a * p[w]
        else:
            for w in range(width):
                dst[w] += a * p[w]
    elif kind == 1:
        if first:
            for w in range(width):
                d = a - p[w]
                dst[w] = d * d
        else:
            for w in range(width):
                d = a - p[w]
                dst[w] += d * d
    elif first:
        for w in range(width):
            dst[w] = abs(a - p[w])
    else:
        for w in range(width):
            dst[w] += abs(a - p[w])


@njit(nogil=True, cache=True)
def _panel_leaf(rows, qs, nq, panel, width, lo, n, kind, acc, res):
    """:func:`_leaf_sum` of queries ``rows[qs[:nq]]`` against the first ``width`` panel columns, into ``res``."""
    if n < 8:
        for q in range(nq):
            x = rows[qs[q]]
            out = res[q]
            for w in range(width):
                out[w] = np.float32(0.0)
            for f in range(lo, lo + n):
                _accumulate(out, x[f], panel[f], width, kind, False)
        return
    stop = n - (n % 8)
    for k in range(8):  # accumulator k sums terms lo + k, lo + k + 8, ... in turn, as numpy's does
        p = panel[lo + k]
        for q in range(nq):
            _accumulate(acc[q, k], rows[qs[q], lo + k], p, width, kind, True)
        for i in range(8, stop, 8):
            f = lo + i + k
            p = panel[f]
            for q in range(nq):
                _accumulate(acc[q, k], rows[qs[q], f], p, width, kind, False)
    for q in range(nq):
        out = res[q]
        a = acc[q]
        for w in range(width):
            out[w] = ((a[0, w] + a[1, w]) + (a[2, w] + a[3, w])) + ((a[4, w] + a[5, w]) + (a[6, w] + a[7, w]))
        x = rows[qs[q]]
        for f in range(lo + stop, lo + n):
            _accumulate(out, x[f], panel[f], width, kind, False)


@njit(nogil=True, cache=True)
def panel_similarity(rows, scale, metric, leaf_lo, leaf_n, ops, qs, nq, cand, width, panel, acc, stack, out):
    """``out[q, w]``: :func:`_similarity` of ``rows[qs[q]]`` and ``rows[cand[w]]``, the panel filled from ``cand``.

    ``acc`` holds ``(QUERIES, 8, PANEL)`` and ``stack`` ``(depth, QUERIES, PANEL)`` scratch values.
    """
    one = np.float32(1.0)
    if metric == CHEBYSHEV:
        for q in range(nq):
            x = rows[qs[q]]
            for w in range(width):
                out[q, w] = np.float32(0.0)
            for f in range(x.shape[0]):
                a = x[f]
                p = panel[f]
                for w in range(width):
                    d = abs(a - p[w])
                    if d > out[q, w]:
                        out[q, w] = d
            for w in range(width):
                out[q, w] = one / (one + out[q, w])
        return
    kind = 0 if metric <= CORRELATION else (1 if metric == EUCLIDEAN else 2)
    top = 0
    for k in range(ops.shape[0]):  # the summation tree of _pairwise_sum, a panel of sums at a time
        op = ops[k]
        if op >= 0:
            _panel_leaf(rows, qs, nq, panel, width, leaf_lo[op], leaf_n[op], kind, acc, stack[top])
            top += 1
        else:
            top -= 1
            for q in range(nq):
                for w in range(width):
                    stack[top - 1, q, w] = stack[top - 1, q, w] + stack[top, q, w]
    for q in range(nq):
        for w in range(width):
            v = stack[0, q, w]
            if metric == ABS_COSINE:
                v = abs(v)
            elif metric == CORRELATION:
                v = v / np.sqrt(scale[qs[q]] * scale[cand[w]])
            elif metric == EUCLIDEAN:
                v = one / (one + np.sqrt(v))
            elif metric == MANHATTAN:
                v = one / (one + v)
            out[q, w] = v


def stack_depth(ops: np.ndarray) -> int:
    """The most partial sums :func:`summation_plan`'s program holds at once."""
    return int(np.cumsum(np.where(ops >= 0, 1, -1)).max(initial=1))


@intrinsic
def atomic_increment(typingctx, array, index):
    """``array[index] += 1`` in one atomic step, for a count that several threads raise."""
    if not (isinstance(array, types.Array) and array.ndim == 1 and isinstance(array.dtype, types.Integer)):
        return None

    def codegen(context, builder, signature, args):
        array_type, index_type = signature.args
        view = context.make_array(array_type)(context, builder, args[0])
        position = context.cast(builder, args[1], index_type, types.intp)
        pointer = cgutils.get_item_pointer(context, builder, array_type, view, [position], wraparound=False)
        builder.atomic_rmw("add", pointer, ir.Constant(context.get_value_type(array_type.dtype), 1), "monotonic")
        return context.get_dummy_value()

    return types.void(array, index), codegen


# ---------------------------------------------------------------------------------------------------------------------
# Pair layout from group codes


@dataclass
class Layout:
    """The implied pairs of one query, as group codes.

    ``lab_ptr``/``lab`` list each profile's positive labels (one for a single-label column, any number for a
    multilabel one), ``mem_ptr``/``mem`` each label's profiles. ``pos_other`` must match and ``pos_diff`` must
    differ for a positive; ``neg_bucket`` must match, ``neg_diff`` must differ and, when ``disjoint``, the label
    sets must not intersect for a negative.
    """

    n: int
    lab_ptr: np.ndarray
    lab: np.ndarray
    mem_ptr: np.ndarray
    mem: np.ndarray
    label_values: np.ndarray
    pos_other: np.ndarray
    pos_diff: np.ndarray
    neg_bucket: np.ndarray
    n_buckets: int
    neg_diff: np.ndarray
    disjoint: bool


def _combined_codes(meta: pd.DataFrame, columns: list[str]) -> tuple[np.ndarray, int]:
    if not columns:
        return np.zeros(len(meta), np.int64), 1
    codes = meta.groupby(columns, observed=True, sort=True, dropna=False).ngroup().to_numpy(np.int64)
    return codes, int(codes.max()) + 1 if len(codes) else 0


def _column_codes(meta: pd.DataFrame, columns: list[str]) -> np.ndarray:
    out = np.empty((len(meta), len(columns)), np.int64)
    for c, column in enumerate(columns):
        out[:, c] = pd.factorize(meta[column], use_na_sentinel=False)[0]
    return out


def _csr(owner: np.ndarray, values: np.ndarray, n_owners: int) -> tuple[np.ndarray, np.ndarray]:
    order = np.lexsort((values, owner))
    ptr = np.zeros(n_owners + 1, np.int64)
    np.add.at(ptr, owner + 1, 1)
    return np.cumsum(ptr), values[order].astype(np.int64)


def layout(
    meta: pd.DataFrame,
    pos_sameby: list[str],
    pos_diffby: list[str],
    neg_sameby: list[str],
    neg_diffby: list[str],
    multilabel_col: str | None = None,
) -> Layout:
    """Group codes of the pair definitions, as copairs would match them.

    A single-label query's label is the combination of its ``pos_sameby`` columns. A multilabel column holds a list
    of labels per row; a positive shares at least one of them, and the column must be a ``neg_diffby`` column, so a
    negative shares none.

    Raises:
        ValueError: A pair column holds a missing value, sameby and diffby overlap, a definition is empty, or the
            multilabel column is not a ``neg_diffby`` column.
    """
    for same, diff, kind in ((pos_sameby, pos_diffby, "positive"), (neg_sameby, neg_diffby, "negative")):
        if set(same) & set(diff):
            raise ValueError(f"the {kind} sameby and diffby columns must be disjoint, got {same} and {diff}")
        if not (same or diff):
            raise ValueError(f"the {kind} pairs need at least one sameby or diffby column")
    scalar = [c for c in dict.fromkeys([*pos_sameby, *pos_diffby, *neg_sameby, *neg_diffby]) if c != multilabel_col]
    if meta[scalar].isna().any(axis=None):
        raise ValueError("the metadata columns of the pair definitions must not hold missing values")
    n = len(meta)
    if multilabel_col is None:
        codes, n_labels = _combined_codes(meta, list(pos_sameby))
        label_values = np.arange(n_labels)
        lab_ptr, lab = np.arange(n + 1, dtype=np.int64), codes
        pos_other = np.zeros(n, np.int64)
        disjoint = False
        neg_diff_columns = list(neg_diffby)
    else:
        if multilabel_col not in pos_sameby:
            raise ValueError(f"the multilabel column {multilabel_col!r} must be a pos_sameby column")
        if multilabel_col not in neg_diffby:
            raise ValueError(
                f"the multilabel column {multilabel_col!r} must be a neg_diffby column, so a negative shares no label"
            )
        lists = meta[multilabel_col].to_list()
        owner = np.repeat(np.arange(n), [len(v) for v in lists])
        flat = pd.Series([item for v in lists for item in v], dtype=object)
        item_codes, uniques = pd.factorize(flat, sort=True)
        label_values = np.asarray(uniques, dtype=object)
        # A label listed twice in one row counts once.
        pairs = (
            np.unique(np.stack([owner, item_codes.astype(np.int64)], axis=1), axis=0)
            if len(owner)
            else np.zeros((0, 2), np.int64)
        )
        lab_ptr, lab = _csr(pairs[:, 0], pairs[:, 1], n)
        n_labels = len(label_values)
        pos_other, _ = _combined_codes(meta, [c for c in pos_sameby if c != multilabel_col])
        disjoint = True
        neg_diff_columns = [c for c in neg_diffby if c != multilabel_col]
    owners = np.repeat(np.arange(n, dtype=np.int64), np.diff(lab_ptr))
    mem_ptr, mem = _csr(lab, owners, n_labels)
    neg_bucket, n_buckets = _combined_codes(meta, list(neg_sameby))
    return Layout(
        n=n, lab_ptr=lab_ptr, lab=lab, mem_ptr=mem_ptr, mem=mem, label_values=label_values, pos_other=pos_other,
        pos_diff=_column_codes(meta, list(pos_diffby)), neg_bucket=neg_bucket, n_buckets=n_buckets,
        neg_diff=_column_codes(meta, neg_diff_columns), disjoint=disjoint,
    )  # fmt: skip


@njit(nogil=True, cache=True, inline="always")
def _differs(diff, i, j):
    for c in range(diff.shape[1]):
        if diff[i, c] == diff[j, c]:
            return False
    return True


@njit(nogil=True, cache=True, inline="always")
def _labels_disjoint(lab_ptr, lab, i, j):
    a, a_end = lab_ptr[i], lab_ptr[i + 1]
    b, b_end = lab_ptr[j], lab_ptr[j + 1]
    while a < a_end and b < b_end:
        if lab[a] == lab[b]:
            return False
        if lab[a] < lab[b]:
            a += 1
        else:
            b += 1
    return True


@njit(nogil=True, cache=True, inline="always")
def is_negative(i, j, neg_bucket, neg_diff, disjoint, lab_ptr, lab):
    """Whether profile ``j`` is a negative of query ``i``."""
    return (
        j != i
        and neg_bucket[i] == neg_bucket[j]
        and _differs(neg_diff, i, j)
        and (not disjoint or _labels_disjoint(lab_ptr, lab, i, j))
    )


@njit(nogil=True, cache=True)
def _count_positives(lab_ptr, lab, mem_ptr, mem, pos_other, pos_diff):
    """Positives of every (profile, label) entry of ``lab``."""
    counts = np.zeros(lab.shape[0], np.int64)
    for i in range(lab_ptr.shape[0] - 1):
        for e in range(lab_ptr[i], lab_ptr[i + 1]):
            label = lab[e]
            c = 0
            for t in range(mem_ptr[label], mem_ptr[label + 1]):
                j = mem[t]
                if j != i and pos_other[j] == pos_other[i] and _differs(pos_diff, i, j):
                    c += 1
            counts[e] = c
    return counts


@njit(parallel=True, nogil=True, cache=True)
def _positive_keys(
    rows, scale, metric, leaf_lo, leaf_n, ops, row_entry, row_profile, pos_ptr, lab, mem_ptr, mem, pos_other,
    pos_diff, keys, partner, n_chunks,
):  # fmt: skip
    """Fill each AP row's positives (``partner``) and their sorted rank keys, ``float32(1) - similarity``.

    ``metric < 0`` fills the partners only; the caller scores them and sorts.
    """
    n_rows = row_entry.shape[0]
    one = np.float32(1.0)
    for chunk in prange(n_chunks):
        stack = np.empty(64, np.float32)
        for r in range(chunk, n_rows, n_chunks):
            i = row_profile[r]
            label = lab[row_entry[r]]
            p = pos_ptr[r]
            for t in range(mem_ptr[label], mem_ptr[label + 1]):
                j = mem[t]
                if j != i and pos_other[j] == pos_other[i] and _differs(pos_diff, i, j):
                    partner[p] = j
                    if metric >= 0:
                        keys[p] = one - _similarity(rows, scale, i, j, metric, leaf_lo, leaf_n, ops, stack)
                    p += 1
            if metric >= 0:
                keys[pos_ptr[r] : pos_ptr[r + 1]].sort()


@njit(nogil=True, cache=True, inline="always")
def _bin_negative(key, r_lo, r_hi, keys, pos_ptr, bins, bin_ptr):
    """Count one negative of key ``key`` against each of the query's AP rows ``[r_lo, r_hi)``.

    The count is atomic, since a pair's two profiles are counted by one thread and other pairs of either profile by
    others.
    """
    for r in range(r_lo, r_hi):
        lo = pos_ptr[r]
        hi = pos_ptr[r + 1]
        # Ties rank the positive first, so the negative sits after every positive with a key <= its own.
        a, b = lo, hi
        while a < b:
            mid = (a + b) >> 1
            if keys[mid] <= key:
                a = mid + 1
            else:
                b = mid
        atomic_increment(bins, bin_ptr[r] + a - lo)


@njit(parallel=True, nogil=True, cache=True)
def _count_negatives(order, off, neg_bucket, neg_diff, disjoint, lab_ptr, lab, n_chunks):
    """The negatives of every profile, which its bucket ``order[off[b]:off[b + 1]]`` holds."""
    n = neg_bucket.shape[0]
    n_neg = np.zeros(n, np.int64)
    for chunk in prange(n_chunks):
        for i in range(chunk, n, n_chunks):
            b = neg_bucket[i]
            c = 0
            for t in range(off[b], off[b + 1]):
                if is_negative(i, order[t], neg_bucket, neg_diff, disjoint, lab_ptr, lab):
                    c += 1
            n_neg[i] = c
    return n_neg


@njit(parallel=True, nogil=True, cache=True)
def _negatives_exact(
    rows, scale, metric, leaf_lo, leaf_n, ops, depth, order, items, prof_row_ptr, neg_bucket, neg_diff, disjoint,
    lab_ptr, lab, keys, pos_ptr, bins, bin_ptr, n_chunks,
):  # fmt: skip
    """Count every negative pair of a query past the query's sorted positives, each pair's similarity scored once.

    ``items[k] = (r_lo, r_hi, c_lo, c_hi, q_end)`` pairs the rows ``order[r_lo:r_hi]`` with the candidates
    ``order[c_lo:c_hi]`` of one bucket, whose queries come first and end at ``q_end``. A pair at positions ``r < c``
    is counted for the row and, when the candidate is a query too, for the candidate: being a negative is symmetric,
    and so is the similarity, to the bit.
    """
    one = np.float32(1.0)
    n_items = items.shape[0]
    for chunk in prange(n_chunks):
        panel = np.zeros((rows.shape[1], PANEL), np.float32)
        acc = np.zeros((QUERIES, 8, PANEL), np.float32)
        stack = np.zeros((depth, QUERIES, PANEL), np.float32)
        sims = np.zeros((QUERIES, PANEL), np.float32)
        cand = np.empty(PANEL, np.int64)
        cpos = np.empty(PANEL, np.int64)
        qs = np.empty(QUERIES, np.int64)
        for k in range(chunk, n_items, n_chunks):
            r_lo, r_hi, c_lo, c_hi, q_end = items[k, 0], items[k, 1], items[k, 2], items[k, 3], items[k, 4]
            n_cand = 0  # the candidates that are a negative of a row before them
            for c in range(c_lo, c_hi):
                j = order[c]
                for r in range(r_lo, min(r_hi, c)):
                    if is_negative(order[r], j, neg_bucket, neg_diff, disjoint, lab_ptr, lab):
                        cand[n_cand] = j
                        cpos[n_cand] = c
                        n_cand += 1
                        break
            if n_cand == 0:
                continue
            fill_panel(rows, cand, n_cand, panel)
            for q0 in range(r_lo, r_hi, QUERIES):
                nq = min(QUERIES, r_hi - q0)
                for q in range(nq):
                    qs[q] = order[q0 + q]
                panel_similarity(
                    rows, scale, metric, leaf_lo, leaf_n, ops, qs, nq, cand, n_cand, panel, acc, stack, sims
                )
                for q in range(nq):
                    i = qs[q]
                    for w in range(n_cand):
                        j = cand[w]
                        if cpos[w] > q0 + q and is_negative(i, j, neg_bucket, neg_diff, disjoint, lab_ptr, lab):
                            key = one - sims[q, w]
                            _bin_negative(key, prof_row_ptr[i], prof_row_ptr[i + 1], keys, pos_ptr, bins, bin_ptr)
                            if cpos[w] < q_end:
                                _bin_negative(key, prof_row_ptr[j], prof_row_ptr[j + 1], keys, pos_ptr, bins, bin_ptr)


def _pair_items(off: np.ndarray, n_query: np.ndarray) -> np.ndarray:
    """:func:`_negatives_exact`'s blocks: each bucket's query rows against the candidates from the same row on."""
    items = [
        (r_lo, min(r_lo + PANEL, start + nq), c_lo, min(c_lo + PANEL, end), start + nq)
        for start, end, nq in zip(off[:-1].tolist(), off[1:].tolist(), n_query.tolist(), strict=True)
        for r_lo in range(start, start + nq, PANEL)
        for c_lo in range(r_lo, end, PANEL)
    ]
    return np.array(items, dtype=np.int64).reshape(-1, 5)


@njit(parallel=True, nogil=True, cache=True)
def _negatives_block(
    sims, queries, cands, prof_row_ptr, neg_bucket, neg_diff, disjoint, lab_ptr, lab, keys, pos_ptr, bins, bin_ptr,
    n_chunks,
):  # fmt: skip
    """Count the negatives of ``queries`` among ``cands`` past their sorted positives, from a block of similarities."""
    one = np.float32(1.0)
    nq = queries.shape[0]
    for chunk in prange(n_chunks):
        for b in range(chunk, nq, n_chunks):
            i = queries[b]
            r_lo = prof_row_ptr[i]
            r_hi = prof_row_ptr[i + 1]
            if r_hi == r_lo:
                continue
            for t in range(cands.shape[0]):
                j = cands[t]
                if is_negative(i, j, neg_bucket, neg_diff, disjoint, lab_ptr, lab):
                    _bin_negative(one - sims[b, t], r_lo, r_hi, keys, pos_ptr, bins, bin_ptr)


@njit(parallel=True, nogil=True, cache=True)
def _ap_from_bins(pos_ptr, bins, bin_ptr, n_chunks):
    """Average precision of each AP row from the negatives counted before each of its sorted positives."""
    n_rows = pos_ptr.shape[0] - 1
    ap = np.full(n_rows, np.nan)
    for chunk in prange(n_chunks):
        for r in range(chunk, n_rows, n_chunks):
            p = pos_ptr[r + 1] - pos_ptr[r]
            if p == 0:
                continue
            above = 0
            s = 0.0
            for d in range(p):
                above += bins[bin_ptr[r] + d]
                s += (d + 1) / (d + 1 + above)
            ap[r] = s / p
    return ap


def _block_similarity(rows: np.ndarray, a: np.ndarray, b: np.ndarray, distance: str) -> np.ndarray:
    """float32 similarities of rows ``a`` to rows ``b`` under a scipy metric, by scipy's cdist.

    scipy computes each pair's distance with one per-pair routine whatever the block, so a block gives the values
    copairs takes from the diagonal of its own blocks.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        d = cdist(rows[a], rows[b], distance)  # type: ignore[call-overload]
    return ((1 - d) if distance in _BOUNDED else 1 / (1 + d)).astype(np.float32)


def _pair_similarity(rows: np.ndarray, a: np.ndarray, b: np.ndarray, distance: str) -> np.ndarray:
    """float32 similarities of the pairs ``(a[k], b[k])`` for a scipy metric, from the diagonals of 64-row blocks."""
    out = np.empty(len(a), np.float32)
    for lo in range(0, len(a), 64):
        sl = slice(lo, lo + 64)
        with np.errstate(divide="ignore", invalid="ignore"):
            d = np.diag(cdist(rows[a[sl]], rows[b[sl]], distance))  # type: ignore[call-overload]
        out[sl] = (1 - d) if distance in _BOUNDED else 1 / (1 + d)
    return out


@dataclass
class APResult:
    """Average precision of every AP row and the positive pairs it ranked."""

    row_profile: np.ndarray  # the profile each AP row scores
    row_label: np.ndarray  # the label code each AP row scores toward
    average_precision: np.ndarray
    n_pos_pairs: np.ndarray
    n_total_pairs: np.ndarray
    pos_ptr: np.ndarray  # positives of AP row r: partner[pos_ptr[r]:pos_ptr[r + 1]]
    partner: np.ndarray
    n_neg: np.ndarray  # the negatives of every profile


def average_precision(
    lay: Layout, features: np.ndarray, distance: str = "cosine", n_chunks: int = N_CHUNKS
) -> APResult:
    """Average precision of each AP row.

    An AP row is a profile for a single-label column, and a (profile, label) with a positive for a multilabel one.
    """
    check_distance(distance)
    rows, scale = prepare_rows(features, distance)
    counts = _count_positives(lay.lab_ptr, lay.lab, lay.mem_ptr, lay.mem, lay.pos_other, lay.pos_diff)
    entry_profile = np.repeat(np.arange(lay.n, dtype=np.int64), np.diff(lay.lab_ptr))
    if lay.disjoint:
        row_entry = np.flatnonzero(counts > 0)
    else:
        row_entry = np.arange(len(lay.lab), dtype=np.int64)
    row_profile = entry_profile[row_entry]
    row_npos = counts[row_entry]
    pos_ptr = np.concatenate([[0], np.cumsum(row_npos)]).astype(np.int64)
    keys = np.empty(pos_ptr[-1], np.float32)
    partner = np.empty(pos_ptr[-1], np.int64)
    leaf_lo, leaf_n, ops = summation_plan(rows.shape[1])
    exact = distance in BUILTIN
    _positive_keys(
        rows, scale, BUILTIN[distance] if exact else -1, leaf_lo, leaf_n, ops, row_entry, row_profile, pos_ptr,
        lay.lab, lay.mem_ptr, lay.mem, lay.pos_other, lay.pos_diff, keys, partner, n_chunks,
    )  # fmt: skip
    if not exact:
        keys[:] = np.float32(1) - _pair_similarity(rows, np.repeat(row_profile, row_npos), partner, distance)
        for r in range(len(row_entry)):
            keys[pos_ptr[r] : pos_ptr[r + 1]].sort()
    # AP rows of each profile are contiguous; prof_row_ptr points into them.
    prof_row_ptr = np.searchsorted(row_profile, np.arange(lay.n + 1)).astype(np.int64)
    bin_ptr = (pos_ptr[:-1] + np.arange(len(row_entry))).astype(np.int64)
    bins = np.zeros(pos_ptr[-1] + len(row_entry), np.int64)
    # Only a profile with a positive needs its negatives ranked; it is a query, and each bucket lists its queries first.
    query = np.bincount(row_profile[row_npos > 0], minlength=lay.n) > 0
    order, off2 = group_offsets(2 * lay.neg_bucket + ~query, 2 * lay.n_buckets)
    off = off2[::2]
    n_neg = _count_negatives(order, off, lay.neg_bucket, lay.neg_diff, lay.disjoint, lay.lab_ptr, lay.lab, n_chunks)
    if exact:
        _negatives_exact(
            rows, scale, BUILTIN[distance], leaf_lo, leaf_n, ops, stack_depth(ops), order,
            _pair_items(off, off2[1::2] - off[:-1]), prof_row_ptr, lay.neg_bucket, lay.neg_diff, lay.disjoint,
            lay.lab_ptr, lay.lab, keys, pos_ptr, bins, bin_ptr, n_chunks,
        )  # fmt: skip
    else:
        for g in range(lay.n_buckets):
            cands = order[off[g] : off[g + 1]]
            if len(cands) == 0:
                continue
            step = max(1, _BLOCK_BYTES // (4 * len(cands)))
            for lo in range(0, len(cands), step):
                queries = cands[lo : lo + step]
                _negatives_block(
                    _block_similarity(rows, queries, cands, distance), queries, cands, prof_row_ptr, lay.neg_bucket,
                    lay.neg_diff, lay.disjoint, lay.lab_ptr, lay.lab, keys, pos_ptr, bins, bin_ptr, n_chunks,
                )  # fmt: skip
    ap = _ap_from_bins(pos_ptr, bins, bin_ptr, n_chunks)
    n_pos = row_npos
    n_total = n_pos + n_neg[row_profile]
    return APResult(
        row_profile=row_profile, row_label=lay.lab[row_entry], average_precision=ap, n_pos_pairs=n_pos,
        n_total_pairs=n_total, pos_ptr=pos_ptr, partner=partner, n_neg=n_neg,
    )  # fmt: skip


def normalized_average_precision(ap: np.ndarray, n_pos: np.ndarray, n_total: np.ndarray) -> np.ndarray:
    """``(AP - E[AP]) / (1 - E[AP])`` under random ranking, clipped to ``[-1, 1]``, as copairs normalizes it."""
    m = np.asarray(n_pos, dtype=np.int64)
    total = np.asarray(n_total, dtype=np.int64)
    harmonic = np.concatenate([[0.0], np.cumsum(1.0 / np.arange(1, int(total.max(initial=0)) + 1))])
    h = harmonic[total]
    with np.errstate(divide="ignore", invalid="ignore"):
        mu0 = (1.0 / total) * (((m - 1.0) / (total - 1.0)) * (total - h) + h)
    mu0 = np.where(total == 1, (m == 1).astype(float), mu0)
    mu0 = np.where(m == 0, 0.0, mu0)
    mu0 = np.where((m == total) & (total > 1), 1.0, mu0)
    out = np.clip((ap - mu0) / np.maximum(1 - mu0, 1e-10), -1.0, 1.0)
    return out


def ap_frame(meta: pd.DataFrame, lay: Layout, result: APResult, multilabel_col: str | None = None) -> pd.DataFrame:
    """The AP rows as copairs' ``average_precision`` frame, one row per AP row.

    It holds the metadata, ``average_precision``, ``normalized_average_precision``, ``n_pos_pairs`` and
    ``n_total_pairs``. A multilabel column holds the one label each row scores toward. ``PROFILE`` names the row of ``meta`` it scores.
    """
    frame = meta.iloc[result.row_profile].reset_index(drop=True)
    if multilabel_col is not None:
        frame[multilabel_col] = lay.label_values[result.row_label]
    frame[PROFILE] = result.row_profile
    frame["average_precision"] = result.average_precision
    frame["normalized_average_precision"] = normalized_average_precision(
        result.average_precision, result.n_pos_pairs, result.n_total_pairs
    )
    frame["n_pos_pairs"] = result.n_pos_pairs
    frame["n_total_pairs"] = result.n_total_pairs
    return frame
