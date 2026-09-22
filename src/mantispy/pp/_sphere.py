"""Whitening fitted on control profiles, in two forms.

:func:`sphere` whitens by the covariance of the negative controls, which removes the variation they share and leaves the effects of the perturbations.
:func:`tvn` is typical variation normalization as :cite:t:`Celik_2024` defines it: the same idea followed by a per-batch alignment, so batches that disagree about what typical variation looks like are brought onto one another.
"""

from __future__ import annotations

import warnings

import numpy as np
from anndata import AnnData

from mantispy._core._numba import group_offsets
from mantispy._core._reduce import get_matrix, group_codes, representation
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
            "than features the covariance is singular and the transform amplifies noise. Select fewer "
            "features first, or use more controls.",
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


def _centre_scale(values: np.ndarray, reference: np.ndarray, where: str = "") -> np.ndarray:
    """Centre and scale every row by the mean and spread of the reference rows.

    The spread is the population standard deviation, as sklearn's ``StandardScaler`` computes it. A dimension with no spread among the reference rows is left on its own scale rather than divided by zero, and warns, as :func:`~mantispy.pp.normalize` does for the same condition — it cannot flag ``var``, because the columns being scaled are an embedding that ``var`` does not describe.

    "No spread" is read against the resolution of the values rather than as an exact zero, since a dimension an embedding is constant in still carries the rounding of the rotation that built it.

    Args:
        values: Rows to transform.
        reference: Boolean mask over ``values``, selecting the rows the mean and spread are taken from.
        where: Names the batch in the warning, when the reference rows are one batch's.

    Returns:
        The transformed rows.
    """
    block = values[reference]
    centre = block.mean(axis=0)
    scale = block.std(axis=0, ddof=0)
    # An exact zero is the wrong test for an embedding. A dimension the reference rows are constant in is constant
    # only up to the rounding the rotation that built it left behind, so its spread arrives as 1e-16 rather than 0,
    # passes this check, and is divided by anyway, putting that dimension 1e16 ahead of every other one. Below the
    # resolution of the values it is measured in, and the error in a standard deviation grows with the rows it is
    # taken over, a spread is not a spread.
    degenerate = scale <= block.shape[0] * np.finfo(scale.dtype).eps * np.abs(block).max(axis=0)
    if (no_spread := int(degenerate.sum())) > 0:
        warnings.warn(
            f"{no_spread} of {scale.size} dimension(s) have no spread among the {int(reference.sum())} "
            f"reference row(s){where}. Their scale is clamped to 1, so they pass through centred but "
            "unscaled and are not comparable with the rest. Reduce to fewer components, or use more "
            "controls.",
            UserWarning,
            stacklevel=4,
        )
    return (values - centre) / np.where(degenerate, 1.0, scale)


def _regularized_covariance(values: np.ndarray, epsilon: float) -> np.ndarray:
    """The covariance of ``values``, with ``epsilon`` added to its diagonal so it can be inverted."""
    return np.cov(values, rowvar=False, ddof=1) + epsilon * np.eye(values.shape[1])


def _symmetric_power(matrix: np.ndarray, power: float) -> np.ndarray:
    """Raise a symmetric positive definite matrix to a real power through its eigendecomposition.

    ``scipy.linalg.fractional_matrix_power`` handles any matrix, by a Schur decomposition that returns a complex result whose imaginary part is numerical noise. A regularized covariance is symmetric, so this is exact, real and cheaper, and it agrees with the general routine to floating-point noise.

    Args:
        matrix: A symmetric, positive definite matrix.
        power: The power to raise it to.

    Returns:
        The matrix raised to that power.

    Raises:
        ValueError: The matrix is singular or indefinite, which leaves the power undefined.
    """
    eigenvalues, vectors = np.linalg.eigh(matrix)
    if eigenvalues.min() <= 0:
        raise ValueError(
            f"a control covariance has a non-positive eigenvalue ({eigenvalues.min():.3g}), so it cannot be "
            "inverted. Raise epsilon, which is added to every covariance diagonal."
        )
    return (vectors * eigenvalues**power) @ vectors.T


@inplace_or_copy(expects=("well", "perturbation"))
def tvn(
    adata: AnnData,
    batch_key: str = "Metadata_Batch",
    reference: str | None = "negcon",
    use_rep: str | None = "X_pca",
    key_added: str = "X_tvn",
    epsilon: float = 0.5,
    copy: bool = False,
) -> AnnData | None:
    """Typical variation normalization, then align each batch's controls onto the pooled controls :cite:p:`Celik_2024`.

    The controls define what an untreated well looks like, so they are what the transform is fitted on: the profiles are centred and scaled on them, rotated onto the principal components of the controls alone, and centred and scaled on them again within each batch.
    The last step is CORAL — each batch is whitened by the covariance of its own controls and recoloured with the covariance of all of them, so a batch whose typical variation points in an unusual direction is brought onto the others rather than merely recentred.

    Args:
        adata: Object holding the profiles, usually one row per well.
        batch_key: ``obs`` column naming the batches to align. Each needs at least two reference rows.
        reference: Rows the transform is fitted on: ``"negcon"`` for the controls, ``None`` for everything, or the name of a boolean ``obs`` column.
        use_rep: Embedding to align, as :func:`~mantispy.pp.harmony` takes one, or ``None`` to align ``X`` itself. Fitting the rotation on the controls of a wide feature matrix is expensive, so the default expects a reduction first, normally ``sc.pp.pca``.
        key_added: ``obsm`` key for the result.
        epsilon: Added to the diagonal of every covariance before it is inverted. The profiles are on the controls' own scale by then, so their variances are near one and the reference value of 0.5 is a substantial shrink toward isotropy.
        copy: Return a modified copy instead of writing in place.

    Returns:
        ``None``, or the modified copy. Writes ``obsm[key_added]``.

    Raises:
        KeyError: ``obs`` has no column ``batch_key`` or no column named by ``reference``, or ``obsm`` holds nothing under ``use_rep``.
        ValueError: ``reference`` selects no rows, or fewer than two in some batch, which leaves that batch's covariance undefined.
        ValueError: A control covariance is singular even after ``epsilon``, so it cannot be inverted. Reachable by passing ``epsilon=0``.

    Notes:
        The rotation is fitted on the controls, so it keeps ``min(n_controls, n_features)`` components. With fewer controls than features the result is narrower than the input, which is why this writes ``obsm`` and never ``X``: ``var`` would no longer describe the columns.

        Batch correction methods disagree with each other often enough that one metric is not evidence. Compare this with :func:`~mantispy.pp.harmony` on the same object using :func:`~mantispy.metrics.evaluate_correction`, and on a screen with annotated perturbations also :func:`~mantispy.metrics.known_relationships`, which is the measure :cite:t:`Celik_2024` selects it by.

        Measured that way, it tends to trade replicate consistency for relationship recall, where :func:`~mantispy.pp.harmony` trades the other way. Neither buys the other's gain, so which of the two readouts the screen is for is the question to answer before running either.

        What it needs is controls, per batch and not in total, because the covariance it whitens each batch by is estimated from that batch's controls alone. A batch with fewer controls than the rotation has components cannot span the space, and the warning that says so is the sign to reduce to fewer components or to pool smaller batches together.
    """
    from sklearn.decomposition import PCA

    # Checked before the rotation, which is the expensive part, so a mistyped column costs nothing.
    if batch_key not in adata.obs:
        raise KeyError(f"obs has no column {batch_key!r} naming the batches to align")

    values = representation(adata, use_rep)
    controls = reference_mask(adata, reference)
    if not controls.any():
        raise ValueError(f"no reference rows selected by reference={reference!r}")

    values = _centre_scale(values, controls)
    # Fitted on the controls, so the components describe typical variation rather than the
    # perturbations, and the transform recentres everything on the control mean again.
    values = PCA().fit(values[controls]).transform(values)

    codes, keys = group_codes(adata, batch_key)
    # One stable ordering serves both passes, where `codes == group` would scan every row once
    # per batch. The two passes cannot be merged: the target below is taken from every control
    # row, after the first pass has rewritten them all.
    order, offsets = group_offsets(codes, len(keys))
    batches = [order[offsets[group] : offsets[group + 1]] for group in range(len(keys))]

    thin = []
    for key, rows in zip(keys, batches, strict=True):
        reference_rows = rows[controls[rows]]
        if reference_rows.size < 2:
            raise ValueError(
                f"batch {key!r} has {reference_rows.size} row(s) selected by reference={reference!r}, "
                "and aligning a batch needs at least 2 to estimate its covariance. Drop that batch, or "
                "check that the platemap labels its controls."
            )
        if reference_rows.size <= values.shape[1]:
            thin.append(f"{key!r} ({reference_rows.size})")
        values[rows] = _centre_scale(values[rows], controls[rows], where=f" of batch {key!r}")

    # _centre_scale clamps the directions such a batch leaves empty, and says so per batch. This names the cause
    # rather than the effect, and is the actionable form: the remedy is fewer components or larger batches.
    if thin:
        warnings.warn(
            f"{len(thin)} batch(es) have no more reference rows than the {values.shape[1]} component(s) "
            f"they are scaled in, such as {', '.join(thin[:3])}. Their controls cannot span the space, so "
            "the directions they leave empty are divided by near-zero spread and come out amplified. "
            "Reduce to fewer components, or pool smaller batches together.",
            UserWarning,
            stacklevel=3,
        )

    target = _symmetric_power(_regularized_covariance(values[controls], epsilon), 0.5)
    for rows in batches:
        source = _regularized_covariance(values[rows[controls[rows]]], epsilon)
        values[rows] = values[rows] @ _symmetric_power(source, -0.5) @ target

    adata.obsm[key_added] = values.astype(np.float32)
    return None
