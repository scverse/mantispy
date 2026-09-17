"""Feature-by-feature correlation, in batches.

Rows: ``np.corrcoef`` needs the whole matrix transposed into float64, 32 GB at
1M x 4000. Accumulating the Gram matrix in row chunks costs O(p^2) memory whatever the
row count, which is also what a streaming backend needs.

Missing values: pairwise-complete deletion gives every pair its own row set.
``pandas.DataFrame.corr`` computes it with one Cython pass per pair, which takes over
fourteen minutes on 50 640 JUMP wells by 3634 features. With ``M`` the finite mask and
``Z`` the values with missing entries zeroed, every pairwise moment is a matrix product:
``n = MᵀM``, ``Σxy = ZᵀZ``, ``Σx = ZᵀM``, ``Σx² = (Z∘Z)ᵀM``. These six products run in
BLAS, can be chunked by row, and give the same result as pandas.

Features: the full matrix is O(p^2), 3.2 GB at 20 000 features. :func:`correlated_pairs`
needs only the pairs above a threshold, so it correlates one column block at a time. The
peak is one block-by-p strip, capped by :data:`BLOCK_BYTES`, with the same arithmetic.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterator

import numpy as np
import pandas as pd

#: Bytes a single float64 row chunk may occupy. The row count follows from the feature
#: count, which sets the memory: 100 000 rows is 24 MB at 30 features and 3.2 GB at 4000.
CHUNK_BYTES = 256_000_000

#: Bytes one strip of the correlation matrix may occupy. The strip is
#: ``block x n_vars``, so the block width follows from this and the feature count.
BLOCK_BYTES = 256_000_000


def chunk_rows(n_vars: int) -> int:
    """Rows per chunk so that one float64 block stays inside :data:`CHUNK_BYTES`."""
    return max(int(CHUNK_BYTES / max(n_vars, 1) / 8), 1024)


def block_columns(n_vars: int) -> int:
    """Columns per block so that one ``block x n_vars`` strip stays inside :data:`BLOCK_BYTES`."""
    return min(n_vars, max(int(BLOCK_BYTES / 8 / max(n_vars, 1)), 256))


#: Mask-pair budget above which pairwise-complete Spearman warns that it will be slow.
_PATTERN_PAIR_BUDGET = 4096


def _rank_columns(X: np.ndarray) -> np.ndarray:
    return pd.DataFrame(X).rank(axis=0, method="average", na_option="keep").to_numpy()


def _prepare(X: np.ndarray, method: str) -> tuple[np.ndarray, np.ndarray]:
    """Validate, rank if asked, and report which columns hold non-finite values."""
    if method not in {"pearson", "spearman"}:
        raise ValueError(f"method must be 'pearson' or 'spearman', got {method!r}")
    X = np.asarray(X)
    if method == "spearman":
        X = _rank_columns(X)
    return X, np.flatnonzero(~np.isfinite(X).all(axis=0))


def corr_matrix(X: np.ndarray, method: str = "pearson", chunk_size: int | None = None) -> np.ndarray:
    """Correlation between the columns of ``X``.

    Args:
        X: Observations by features.
        method: ``"pearson"`` or ``"spearman"``. Spearman ranks the columns first and then runs
            the same code path.
        chunk_size: Rows per chunk in the moment accumulation. ``None`` picks a row count from the
            number of features so that one chunk stays around 256 MB.

    Returns:
        A ``(n_vars, n_vars)`` float64 matrix. Constant features correlate with nothing and
        come back as ``NaN``.

    Notes:
        The result is ``8 * n_vars ** 2`` bytes, 3.2 GB at 20 000 features. To find the pairs
        correlated above a threshold, :func:`correlated_pairs` does not hold the full matrix.
    """
    raw = np.asarray(X)
    X, dirty = _prepare(X, method)
    n_vars = X.shape[1]
    if dirty.size == 0:
        return _gram_corr(X, chunk_size)

    clean = np.setdiff1d(np.arange(n_vars), dirty, assume_unique=True)
    out = np.full((n_vars, n_vars), np.nan)
    if clean.size:
        out[np.ix_(clean, clean)] = _gram_corr(X, chunk_size, columns=clean)
    # Every pair touching a missing value, in one batch of six matrix products.
    against = np.arange(n_vars)
    values = _gappy_corr(raw, X, dirty, against, chunk_size, method)
    out[np.ix_(dirty, against)] = values
    out[np.ix_(against, dirty)] = values.T
    return out


def _pattern_groups(finite: np.ndarray, columns: np.ndarray) -> dict[bytes, list[int]]:
    """Positions within ``columns``, grouped by the column's missingness pattern."""
    groups: dict[bytes, list[int]] = {}
    for slot, column in enumerate(columns):
        groups.setdefault(finite[:, column].tobytes(), []).append(slot)
    return groups


def _gappy_corr(
    raw: np.ndarray, ranked: np.ndarray, left: np.ndarray, right: np.ndarray, chunk_size: int | None, method: str
) -> np.ndarray:
    """Pairwise-complete correlation of ``left`` against ``right``, re-ranking for Spearman.

    Pearson needs no correction, because dropping a pair's incomplete rows is the whole
    deletion rule and :func:`_pairwise_corr` does that.

    Spearman is Pearson on ranks, and a rank depends on which rows are present, so ranking
    each column once over all rows is correct only where both columns of a pair are
    complete. ``pandas.DataFrame.corr`` and ``scipy.stats.spearmanr(nan_policy="omit")``
    both re-rank each pair over the rows it shares. Ranking globally disagrees with both on
    the sign of the correlation in 8 of 300 trials of two columns at 25% missing, by up to
    0.10.

    Columns are grouped by missingness pattern, so the cost is one ranking per distinct
    pair of patterns rather than per pair of columns, over the columns of those two groups
    only. Real screens have few patterns because whole feature families go undefined
    together: rohban has one across 3634 columns, and JUMP TARGET-2 and pki have none at
    well level. Where every column is complete, the single group spans everything and the
    ranks are the global ones.
    """
    if method != "spearman":
        return _pairwise_corr(ranked, left, right, chunk_size)

    finite = np.isfinite(raw)
    out = np.full((left.size, right.size), np.nan)
    left_groups, right_groups = _pattern_groups(finite, left), _pattern_groups(finite, right)
    # Gaps that follow feature families share one pattern. Random gaps give each column its
    # own pattern and quadratic work: 69 s on 400 rows x 600 columns at 2% random missing,
    # against 0.04 s when the same columns share one pattern.
    if len(left_groups) * len(right_groups) > _PATTERN_PAIR_BUDGET:
        warnings.warn(
            f"spearman with pairwise-complete deletion is re-ranking {len(left_groups)} x {len(right_groups)} "
            "distinct missingness patterns, which will be slow. Use method='pearson', or drop the incomplete "
            "features first with mt.pp.feature_select (drop_na_columns) so the rest share one pattern.",
            UserWarning,
            stacklevel=3,
        )
    for left_mask, left_slots in left_groups.items():
        left_rows = np.frombuffer(left_mask, dtype=bool)
        for right_mask, right_slots in right_groups.items():
            rows = left_rows & np.frombuffer(right_mask, dtype=bool)
            if int(rows.sum()) < 2:
                continue
            left_columns, right_columns = left[left_slots], right[right_slots]
            columns = np.union1d(left_columns, right_columns)
            position = {column: slot for slot, column in enumerate(columns)}
            # On the rows both patterns share, every column of both groups is complete, so a
            # dense Pearson on the ranks is enough. _pairwise_corr would build six moment
            # matrices for nothing (500 s against 5 s when all patterns are distinct).
            block = _standardize(_rank_columns(raw[rows][:, columns]))
            out[np.ix_(left_slots, right_slots)] = (
                block[:, [position[column] for column in left_columns]].T
                @ block[:, [position[column] for column in right_columns]]
            )
    return out


def _standardize(block: np.ndarray) -> np.ndarray:
    """Center the columns and scale them to unit norm, so a dot product is a correlation."""
    centred = block - block.mean(axis=0)
    norm = np.sqrt(np.einsum("ij,ij->j", centred, centred))
    return centred / np.where(norm > 0, norm, np.nan)


def correlated_pairs(
    X: np.ndarray,
    threshold: float,
    method: str = "pearson",
    chunk_size: int | None = None,
    block_size: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Column pairs correlated above ``threshold``, and each column's total ``|r|``.

    The full matrix is never held. Clean column pairs are correlated a block at a time, so
    peak memory is set by :data:`BLOCK_BYTES` rather than by ``n_vars ** 2``.

    Args:
        X: Observations by features.
        threshold: Pairs whose signed correlation exceeds this are returned. The comparison is
            signed, as in pycytominer's ``correlation_threshold``.
        method: As :func:`corr_matrix`.
        chunk_size: As :func:`corr_matrix`.
        block_size: Columns per block. ``None`` picks one from :data:`BLOCK_BYTES`.

    Returns:
        ``(pairs, total)``. ``pairs`` is a ``(k, 2)`` array of column indices, each unordered pair
        once. ``total`` is ``sum(|r|)`` over each column of the full matrix, counting the diagonal
        and reading ``NaN`` as zero, which is the ranking pycytominer drops pairs by.
    """
    raw = np.asarray(X)
    X, dirty = _prepare(X, method)
    n_vars = X.shape[1]
    clean = np.setdiff1d(np.arange(n_vars), dirty, assume_unique=True)
    block_size = block_columns(n_vars) if block_size is None else block_size

    total = np.zeros(n_vars)
    pairs: list[np.ndarray] = []

    for start, left, earlier, strip in _iter_clean_blocks(X, clean, block_size, chunk_size):
        # Threshold the strip before overwriting it. A comparison against NaN is False, and
        # the strict lower triangle of the diagonal block holds each within-block pair once.
        # A mask avoids np.tril_indices, which would materialize every within-block pair.
        above = strip > threshold
        above[:, start:] &= np.tril(np.ones((left.size, left.size), dtype=bool), k=-1)
        rows, columns = np.nonzero(above)
        if rows.size:
            pairs.append(np.column_stack([left[rows], clean[columns]]))
        # |r| in place, so the strip is not copied again before summing.
        np.abs(strip, out=strip)
        np.nan_to_num(strip, copy=False, nan=0.0)
        # Everything left of the diagonal block counts in both directions of the sum;
        # the diagonal block is symmetric, so its column sums are the within-block total.
        total[left] += strip.sum(axis=1)
        total[earlier] += strip[:, :start].sum(axis=0)

    is_dirty = np.zeros(n_vars, dtype=bool)
    is_dirty[dirty] = True
    for start in range(0, n_vars if dirty.size else 0, block_size):
        against = np.arange(start, min(start + block_size, n_vars))
        values = _gappy_corr(raw, X, dirty, against, chunk_size, method)
        magnitude = np.nan_to_num(np.abs(values), nan=0.0)
        # A dirty column's own row of the matrix, and its contribution to every clean
        # column's, counted once each.
        total[dirty] += magnitude.sum(axis=1)
        total[against[~is_dirty[against]]] += magnitude[:, ~is_dirty[against]].sum(axis=0)
        rows, columns = np.nonzero(np.nan_to_num(values, nan=0.0) > threshold)
        columns = against[columns]
        # A dirty-dirty pair appears in both orders; keep the one that is not the diagonal.
        keep = ~is_dirty[columns] | (columns > dirty[rows])
        if keep.any():
            pairs.append(np.column_stack([dirty[rows[keep]], columns[keep]]))

    found = np.concatenate(pairs) if pairs else np.empty((0, 2), dtype=np.int64)
    return found.astype(np.int64), total


def _iter_clean_blocks(
    X: np.ndarray, clean: np.ndarray, block_size: int, chunk_size: int | None
) -> Iterator[tuple[int, np.ndarray, np.ndarray, np.ndarray]]:
    """Row strips of the clean part of the correlation matrix, one at a time.

    Yields ``(offset, block, earlier, strip)``. ``strip`` is ``block`` correlated against
    every clean column up to and including itself, so its last ``block.size`` columns are
    the symmetric diagonal block and the first ``offset`` are everything before it.

    Iterating over block pairs reads each column block once per partner, ``B**2`` casts of
    the matrix into float64 for ``B`` blocks, and at 16 000 features those casts dominate
    the run time. A strip reads the matrix ``B`` times and peaks at ``block_size * n_vars``
    instead of ``n_vars ** 2``, so the block size trades memory directly against reads.
    """
    for start in range(0, clean.size, block_size):
        left = clean[start : start + block_size]
        earlier = clean[:start]
        yield start, left, earlier, _gram_block(X, left, clean[: start + left.size], chunk_size)


def _gram_corr(X: np.ndarray, chunk_size: int | None, columns: np.ndarray | None = None) -> np.ndarray:
    """Pearson correlation from chunk-accumulated sums and cross-products.

    ``columns`` restricts the computation to a subset without copying it out of ``X``.
    Each chunk is sliced as it is read, so the peak is one chunk rather than a second full
    matrix. Valid only where no value is missing.

    Columns are centered on their finite mean first, as in :func:`_pairwise_corr`.
    Otherwise the moment form subtracts two large numbers: on a saturated 16-bit channel
    (mean 65535, spread 1e-3) independent columns come out at r = 1.0, and at 1e8 the
    clipped negative variance marks them constant.
    """
    n_obs = X.shape[0]
    n_vars = X.shape[1] if columns is None else columns.size
    chunk_size = chunk_rows(n_vars) if chunk_size is None else chunk_size
    centre = _finite_mean(X, chunk_size, columns)
    total = np.zeros(n_vars, dtype=np.float64)
    gram = np.zeros((n_vars, n_vars), dtype=np.float64)
    for start in range(0, n_obs, chunk_size):
        rows = slice(start, start + chunk_size)
        block = (X[rows] if columns is None else X[rows][:, columns]).astype(np.float64) - centre
        total += block.sum(axis=0)
        gram += block.T @ block

    mean = total / n_obs
    covariance = gram / n_obs - np.outer(mean, mean)
    deviation = np.sqrt(np.clip(np.diag(covariance), 0.0, None))

    with np.errstate(invalid="ignore", divide="ignore"):
        out = covariance / np.outer(deviation, deviation)
    constant = deviation == 0
    out[constant, :] = np.nan
    out[:, constant] = np.nan
    np.fill_diagonal(out, np.where(constant, np.nan, 1.0))
    return np.clip(out, -1.0, 1.0)


def _gram_block(X: np.ndarray, left: np.ndarray, right: np.ndarray, chunk_size: int | None) -> np.ndarray:
    """Correlation between two disjoint blocks of finite columns."""
    n_obs = X.shape[0]
    # Two float64 blocks of the chunk are alive at once, one per side.
    chunk_size = chunk_rows(2 * (left.size + right.size)) if chunk_size is None else chunk_size
    centre = _finite_mean(X, chunk_size)
    sums = [np.zeros(side.size) for side in (left, right)]
    squares = [np.zeros(side.size) for side in (left, right)]
    cross = np.zeros((left.size, right.size))
    for start in range(0, n_obs, chunk_size):
        rows = slice(start, start + chunk_size)
        blocks = [X[rows][:, side].astype(np.float64) - centre[side] for side in (left, right)]
        for index, block in enumerate(blocks):
            sums[index] += block.sum(axis=0)
            squares[index] += np.einsum("ij,ij->j", block, block)
        cross += blocks[0].T @ blocks[1]
    return _correlate(
        cross / n_obs - np.outer(sums[0] / n_obs, sums[1] / n_obs),
        (squares[0] / n_obs - (sums[0] / n_obs) ** 2)[:, None],
        (squares[1] / n_obs - (sums[1] / n_obs) ** 2)[None, :],
    )


def _pairwise_corr(X: np.ndarray, left: np.ndarray, right: np.ndarray, chunk_size: int | None) -> np.ndarray:
    """Pairwise-complete correlation of ``left`` against ``right``, from six moments.

    Each pair uses the rows where both of its columns are finite, the deletion rule of
    ``pandas.DataFrame.corr``. All pairs are computed at once, because each moment
    restricted to a pair's rows is a matrix product against the finite mask. Columns are
    centered on their finite mean first, which keeps the sums small enough that the
    moment form does not lose precision.
    """
    n_obs = X.shape[0]
    width = left.size + right.size
    # Six float64 blocks of the chunk are alive at once, so the budget is divided by six.
    chunk_size = max(int(CHUNK_BYTES / 8 / max(width, 1) / 6), 512) if chunk_size is None else chunk_size
    centre = _finite_mean(X, chunk_size)

    shape = (left.size, right.size)
    counts = np.zeros(shape)
    cross = np.zeros(shape)
    sum_left, sum_right = np.zeros(shape), np.zeros(shape)
    square_left, square_right = np.zeros(shape), np.zeros(shape)

    for start in range(0, n_obs, chunk_size):
        rows = slice(start, start + chunk_size)
        values = [X[rows][:, side].astype(np.float64) - centre[side] for side in (left, right)]
        masks = [np.isfinite(block).astype(np.float64) for block in values]
        zeroed = [np.where(mask.astype(bool), block, 0.0) for block, mask in zip(values, masks, strict=True)]
        del values
        counts += masks[0].T @ masks[1]
        cross += zeroed[0].T @ zeroed[1]
        sum_left += zeroed[0].T @ masks[1]
        sum_right += masks[0].T @ zeroed[1]
        square_left += np.square(zeroed[0]).T @ masks[1]
        square_right += masks[0].T @ np.square(zeroed[1])

    with np.errstate(invalid="ignore", divide="ignore"):
        scale = np.where(counts > 0, counts, np.nan)
        mean_left, mean_right = sum_left / scale, sum_right / scale
        out = _correlate(
            cross / scale - mean_left * mean_right,
            square_left / scale - mean_left**2,
            square_right / scale - mean_right**2,
        )
    return np.where(counts >= 2, out, np.nan)


def _finite_mean(X: np.ndarray, chunk_size: int, columns: np.ndarray | None = None) -> np.ndarray:
    """Per-column mean of the finite values, without a second copy of ``X``.

    ``columns`` restricts the mean to a subset in the same way as :func:`_gram_corr`: each
    row chunk is column-sliced as it is read, so ``X[:, columns]`` is never materialized.
    """
    n_vars = X.shape[1] if columns is None else columns.size
    total = np.zeros(n_vars)
    counts = np.zeros(n_vars)
    for start in range(0, X.shape[0], chunk_size):
        block = X[start : start + chunk_size] if columns is None else X[start : start + chunk_size][:, columns]
        finite = np.isfinite(block)
        total += np.where(finite, block, 0.0).sum(axis=0)
        counts += finite.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(counts > 0, total / np.where(counts > 0, counts, 1.0), 0.0)


def _correlate(covariance: np.ndarray, variance_left: np.ndarray, variance_right: np.ndarray) -> np.ndarray:
    """Covariance divided by the two deviations, with a constant column giving ``NaN``."""
    with np.errstate(invalid="ignore", divide="ignore"):
        scale = np.sqrt(np.clip(variance_left, 0.0, None) * np.clip(variance_right, 0.0, None))
        out = np.where(scale > 0, covariance / np.where(scale > 0, scale, 1.0), np.nan)
    return np.clip(out, -1.0, 1.0)
