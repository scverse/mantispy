"""Whitening (sphering) fitted on control profiles.

This is typical variation normalization.
Whitening by the covariance of the negative controls removes the variation they share, leaving the effects of the perturbations.
"""

from __future__ import annotations

import warnings

import numpy as np
from anndata import AnnData

from mantispy._core._reduce import get_matrix, group_codes
from mantispy._core.masks import reference_mask
from mantispy._core.mutation import inplace_or_copy

METHODS = ("ZCA", "ZCA-cor", "PCA", "PCA-cor")


def _fit(reference: np.ndarray, method: str, epsilon: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(center, scale, W)`` for the requested whitening.

    Follows pycytominer's ``Spherize``: epsilon is added to the singular values, and when there are no more rows than features the null directions are padded with the smallest non-zero singular value.
    Adding epsilon to clipped eigenvalues instead diverges from pycytominer by orders of magnitude where the matrix is near-singular.
    """
    # A missing value would otherwise surface as "LinAlgError: SVD did not converge".
    if not np.isfinite(reference).all():
        rows = int((~np.isfinite(reference).all(axis=1)).sum())
        columns = int((~np.isfinite(reference).all(axis=0)).sum())
        raise ValueError(
            f"sphering needs a complete reference, but {rows} of {reference.shape[0]} reference row(s) "
            f"have a missing or infinite value, across {columns} feature(s). Drop those features "
            "(mt.pp.feature_select with drop_na_columns, then mt.pp.subset_features) or those reference wells."
        )

    centre = reference.mean(axis=0)
    centered = reference - centre

    if method.endswith("-cor"):
        # StandardScaler: population standard deviation.
        scale = centered.std(axis=0, ddof=0)
        if np.any(scale == 0):
            raise ValueError(
                "sphering cannot standardize a feature with zero variance; drop constant "
                "features first, e.g. mt.pp.filter_features(adata, min_variance=1e-8)"
            )
        centered = centered / scale
    else:
        scale = np.ones(reference.shape[1])

    n_obs, n_vars = centered.shape
    if n_obs <= n_vars:
        warnings.warn(
            f"sphering is fitted on {n_obs} reference rows for {n_vars} features. With fewer rows "
            "than features the covariance is singular and the transform amplifies noise (on BBBC021 "
            "it lowered not-same-compound MOA retrieval from 78% to 22%). Select fewer features first, "
            "or use more controls.",
            UserWarning,
            stacklevel=4,
        )
    # Centering costs one degree of freedom, so a full-rank reference has rank
    # min(n_vars, n_obs - 1).
    rank = np.linalg.matrix_rank(centered)
    if rank != min(n_vars, n_obs - 1):
        raise ValueError(
            f"the reference matrix is not full rank: {n_obs} rows, {n_vars} features, rank {rank}. "
            "Sphering needs enough independent control profiles to estimate a covariance. "
            "Use more control wells, or reduce the feature set with mt.pp.feature_select."
        )

    # Only an underdetermined reference needs the null directions. full_matrices otherwise
    # allocates an (n_obs, n_obs) left factor, about a gigabyte at 7680 reference rows.
    _, singular, right = np.linalg.svd(centered, full_matrices=n_obs <= n_vars)
    if n_obs <= n_vars:
        singular = np.concatenate((singular[:rank], np.repeat(singular[rank - 1], n_vars - rank)))
    singular = singular + epsilon

    W = (right / singular[:, np.newaxis]).transpose() * np.sqrt(n_obs - 1)
    if method.startswith("ZCA"):
        W = W @ right
    return centre, scale, W


@inplace_or_copy()
def sphere(
    adata: AnnData,
    method: str = "ZCA-cor",
    reference: str | None = "negcon",
    epsilon: float = 1e-6,
    by: str | None = None,
    key_added: str | None = None,
    copy: bool = False,
) -> AnnData | None:
    """Whiten profiles with a transform fitted on the reference rows.

    Args:
        adata: Object to sphere. Usually well-level profiles.
        method: ``"ZCA"`` and ``"ZCA-cor"`` rotate back into the original feature basis, so the output columns still correspond to features and ``var`` still describes them. ``"PCA"`` and ``"PCA-cor"`` return principal components, which ``var`` no longer describes, and warn about it unless ``key_added`` is set. The ``-cor`` variants whiten the correlation instead of the covariance, so high-variance features do not dominate.
        reference: Rows to fit on: ``"negcon"`` for the controls, ``None`` for everything, or the name of a boolean ``obs`` column.
        epsilon: Regularization added to the singular values.
        by: Fit and apply separately within each group of this column, e.g. per batch.
        key_added: Write to ``layers[key_added]`` instead of overwriting ``X``.
        copy: Return a modified copy instead of mutating in place.

    Returns:
        ``None``, or the modified copy. Writes ``X`` or ``layers[key_added]``.

    Raises:
        ValueError: If ``method`` is unknown, ``reference`` selects no rows, a group has fewer than two reference rows, the reference holds missing or infinite values, a ``-cor`` method meets a zero-variance feature, or the reference matrix is not full rank.
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    if method.startswith("PCA") and key_added is None:
        warnings.warn(
            f"{method} rotates out of the feature basis, so the columns of X are principal "
            "components while var still describes the original features. Pass key_added= to "
            "keep X, or use ZCA or ZCA-cor, which rotate back into the feature basis.",
            UserWarning,
            stacklevel=3,
        )

    X = get_matrix(adata).astype(np.float64)
    mask = reference_mask(adata, reference)
    if not mask.any():
        raise ValueError(f"no reference rows selected by reference={reference!r}")

    codes, keys = group_codes(adata, by)
    out = np.empty_like(X)
    for group, key in enumerate(keys):
        rows = np.flatnonzero(codes == group)
        fit_rows = rows[mask[rows]]
        if fit_rows.size < 2:
            where = f" in group {key!r}" if by is not None else ""
            raise ValueError(
                f"sphering needs at least 2 reference rows to estimate a covariance, but "
                f"reference={reference!r} selects {fit_rows.size}{where}. Leave by=None to fit across "
                "groups, or check that the platemap labels the controls."
            )
        centre, scale, W = _fit(X[fit_rows], method, epsilon)
        out[rows] = ((X[rows] - centre) / scale) @ W

    result = out.astype(np.float32)
    if key_added is None:
        adata.X = result
    else:
        adata.layers[key_added] = result
    return None
