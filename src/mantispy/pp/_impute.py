"""Fill missing feature values."""

from __future__ import annotations

import numpy as np
from anndata import AnnData

from mantispy._core._corr import chunk_rows
from mantispy._core._numba import MEDIAN, grouped_stat, knn_distances
from mantispy._core._reduce import get_matrix, group_rows
from mantispy._core.logging import get_logger
from mantispy._core.masks import reference_groups
from mantispy._core.mutation import inplace_or_copy

METHODS = ("knn", "reference_median", "zero")
#: Rows imputed per KNN call, which bounds the distance block sklearn holds at once.
KNN_BATCH = 1_000


def _any_missing(X: np.ndarray) -> bool:
    """Whether any value is ``NaN``.

    Scanned in row chunks so a matrix with no missing value is checked without materialising a full-size
    boolean mask, and a matrix that has one returns as soon as the first chunk with a ``NaN`` is reached.
    """
    step = chunk_rows(X.shape[1])
    for start in range(0, X.shape[0], step):
        if np.isnan(X[start : start + step]).any():
            return True
    return False


def _knn(block: np.ndarray, donors: np.ndarray, n_neighbors: int) -> np.ndarray:
    """Fill ``block`` from its nearest ``donors``, leaving the features no donor measures missing.

    Reproduces sklearn's ``KNNImputer`` value for value: distances are the parallel numba
    :func:`~mantispy._core._numba.knn_distances`, and the neighbour pick per feature is numpy's own
    ``argpartition`` on those distances, so the tie-break matches ``KNNImputer`` on any machine.
    """
    # sklearn drops a feature no donor measured, which would shift the columns, so those are left out here.
    measured = np.flatnonzero(~np.isnan(donors).all(axis=0))
    out = block.copy()
    if not measured.size:
        return out
    # Donors are the only float64 copy held for the whole call; receiver rows are upcast per batch.
    fit = np.ascontiguousarray(donors[:, measured], dtype=np.float64)
    missing_fit = np.isnan(fit)
    measured_fit = ~missing_fit
    missing_recv = np.isnan(block)[:, measured]

    # KNNImputer imputes an all-nan-distance receiver with the masked column mean of the donors.
    counts = measured_fit.sum(axis=0)
    sums = np.where(missing_fit, 0.0, fit).sum(axis=0)
    col_mean = np.divide(sums, counts, out=np.full(measured.size, np.nan), where=counts > 0)

    targets = np.flatnonzero(missing_recv.any(axis=1))
    for start in range(0, targets.size, KNN_BATCH):
        rows = targets[start : start + KNN_BATCH]
        recv = np.ascontiguousarray(block[rows][:, measured], dtype=np.float64)
        dist = knn_distances(recv, fit)  # (len(rows), n_donors)
        batch_missing = missing_recv[rows]
        filled = out[np.ix_(rows, measured)]
        for col in np.flatnonzero(batch_missing.any(axis=0)):
            local = np.flatnonzero(batch_missing[:, col])
            pot = np.flatnonzero(measured_fit[:, col])  # donors that measured this feature
            sub = dist[local][:, pot]
            all_nan = np.isnan(sub).all(axis=1)
            if all_nan.any():  # no shared coordinate with any donor of this feature
                filled[local[all_nan], col] = col_mean[col]
                if all_nan.all():
                    continue
                local, sub = local[~all_nan], sub[~all_nan]
            k = min(n_neighbors, pot.size)
            nearest = np.argpartition(sub, k - 1, axis=1)[:, :k]
            near_dist = np.take_along_axis(sub, nearest, axis=1)
            weight = ~np.isnan(near_dist)  # uniform weights, dropping donors too far to have a distance
            values = fit[pot, col][nearest]
            filled[local, col] = np.where(weight, values, 0.0).sum(axis=1) / weight.sum(axis=1)
        out[np.ix_(rows, measured)] = filled
    return out


@inplace_or_copy()
def impute(
    adata: AnnData,
    method: str = "knn",
    reference: str | None = None,
    by: str | list[str] | None = None,
    n_neighbors: int = 5,
    n_donors: int | None = 10_000,
    seed: int = 0,
    layer: str | None = None,
    key_added: str | None = None,
    copy: bool = False,
) -> AnnData | None:
    """Fill missing feature values, so that tools which refuse them, such as :func:`~mantispy.tl.map`, can run.

    Features that are missing in many rows carry little information and are better dropped first, with :func:`~mantispy.pp.feature_select` and its ``na_cutoff``.
    Cells missing many features can be dropped with ``max_nan_fraction`` in :func:`~mantispy.pp.calculate_qc_metrics` and :func:`~mantispy.pp.filter_cells`.
    Brieflow does both before imputing the rest with ``method="knn"``.
    Only ``NaN`` is filled; infinite values are left as they are.

    Args:
        adata: Object to impute.
        method: ``"knn"`` fills each missing value with the mean of that feature over the row's ``n_neighbors`` nearest reference rows that measured it, as sklearn's ``KNNImputer`` does, with distances over the features both rows measured.
            Distances mix every feature, so run it after :func:`~mantispy.pp.normalize`, which puts the features on one scale.
            ``"reference_median"`` fills with the feature's median over the reference rows of the row's group.
            ``"zero"`` fills with zero, which after a normalization centred on the controls is their median.
        reference: Rows that donate values: ``None`` for all rows, ``"negcon"`` for the negative controls, or the name of a boolean ``obs`` column, as in :func:`~mantispy.pp.normalize`.
            Unused by ``"zero"``.
        by: Column(s) of ``obs`` whose groups are imputed separately, so a row takes values only from reference rows of its own group, such as its plate.
            Unused by ``"zero"``.
        n_neighbors: Reference rows averaged per missing value by ``"knn"``.
        n_donors: ``"knn"`` searches a random sample of at most this many reference rows per group, rather than all of them, so the cost per imputed row is bounded.
            ``None`` searches every reference row.
        seed: Seed for the ``n_donors`` sample.
        layer: Read this layer instead of ``X``.
        key_added: Write to ``layers[key_added]`` instead of overwriting ``X``.
        copy: Return an imputed copy instead of imputing in place.

    Returns:
        ``None``, or the imputed copy.
        Writes ``X`` or ``layers[key_added]``, and ``var["n_imputed"]``, or ``var["n_imputed_<key_added>"]`` when writing to a layer, with the number of values filled in each feature.
        A value is left missing, with a warning, when no reference row of its group measured that feature.

    Raises:
        ValueError: If ``method`` is unknown, ``n_neighbors`` or ``n_donors`` is below 1, or ``reference`` selects no rows at all or none in some group.
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    if n_neighbors < 1 or (n_donors is not None and n_donors < 1):
        raise ValueError(f"n_neighbors and n_donors must be at least 1, got {n_neighbors} and {n_donors}")

    source = get_matrix(adata, layer)
    column = "n_imputed" if key_added is None else f"n_imputed_{key_added}"

    if method == "zero":
        out = np.array(source, copy=True)
        missing = np.isnan(out)
        out[missing] = 0.0
        filled = missing  # a filled NaN is no longer NaN, so every missing value is counted as imputed
    elif _any_missing(source):
        out = np.array(source, copy=True)
        missing = np.isnan(out)
        mask, codes, keys = reference_groups(adata, reference, by)
        if method == "reference_median":
            median = grouped_stat(out[mask], codes[mask], len(keys), MEDIAN)
            rows, columns = np.nonzero(missing)
            out[rows, columns] = median[codes[rows], columns]
        else:
            # An inf coordinate would poison the distances, and the contract leaves inf in place, so the
            # neighbour search runs on a copy that reads inf as NaN and only the originally-NaN cells take
            # the result back.
            finite = np.where(np.isinf(out), np.nan, out) if np.isinf(out).any() else out
            rng = np.random.default_rng(seed)
            # Only rows with a missing value are gathered, so no group-sized copy of the matrix is made.
            incomplete = missing.any(axis=1)
            for rows in group_rows(codes, len(keys)):
                targets = rows[incomplete[rows]]
                if not targets.size:
                    continue
                donors = rows[mask[rows]]
                if n_donors is not None and donors.size > n_donors:
                    donors = np.sort(rng.choice(donors, size=n_donors, replace=False))
                block = _knn(finite[targets], finite[donors], n_neighbors)
                out[targets] = np.where(missing[targets], block, out[targets])
        filled = missing & ~np.isnan(out)
    else:
        # Nothing is missing: no value is filled, so skip the matrix copy, the grouping and the neighbour
        # search entirely. X already holds the final values and is left untouched.
        out = None
        missing = None
        filled = None

    if filled is None:
        adata.var[column] = np.zeros(adata.n_vars, dtype=np.int64)
    else:
        adata.var[column] = filled.sum(axis=0).astype(np.int64)

    if out is not None:
        if key_added is None:
            adata.X = out
        else:
            adata.layers[key_added] = out
    elif key_added is not None:
        # No value changed, but the documented layer must still exist.
        adata.layers[key_added] = np.array(source, copy=True)

    if filled is not None:
        left = missing & ~filled
        if left.any():
            unfilled = left.any(axis=0)
            get_logger().warning(
                "impute: %d values in %d features stay missing, because no reference row of their group measured them",
                int(left.sum()),
                int(unfilled.sum()),
                deep=f"drop those features, e.g. {', '.join(map(str, adata.var_names[unfilled][:5]))}",
            )
    get_logger().info(
        "impute(%s) filled %d values in %d of %d features",
        method,
        0 if filled is None else int(filled.sum()),
        0 if filled is None else int(filled.any(axis=0).sum()),
        adata.n_vars,
    )
    return None
