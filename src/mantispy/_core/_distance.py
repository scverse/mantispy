"""Pairwise distance kernels shared by hit calling, e-distance and local density.

Distances are computed in row chunks. A full pairwise matrix between two blocks of n rows
is n^2 floats, so without chunking a per-perturbation loop over a single-cell object would
allocate a matrix larger than the data.
"""

from __future__ import annotations

import numpy as np

CHUNK_SIZE = 2048


def pairwise_sqeuclidean(A: np.ndarray, B: np.ndarray, chunk_size: int = CHUNK_SIZE) -> np.ndarray:
    """Squared Euclidean distances between the rows of ``A`` and the rows of ``B``."""
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    b_norm = np.einsum("ij,ij->i", B, B)
    out = np.empty((A.shape[0], B.shape[0]), dtype=np.float64)
    for start in range(0, A.shape[0], chunk_size):
        block = A[start : start + chunk_size]
        a_norm = np.einsum("ij,ij->i", block, block)
        out[start : start + chunk_size] = a_norm[:, None] + b_norm[None, :] - 2.0 * (block @ B.T)
    # The expansion can go slightly negative for near-identical rows.
    return np.maximum(out, 0.0)


def _mean_distance(A: np.ndarray, B: np.ndarray, exclude_diagonal: bool = False) -> float:
    distances = np.sqrt(pairwise_sqeuclidean(A, B))
    if exclude_diagonal:
        n = distances.shape[0]
        if n < 2:
            return 0.0
        return float((distances.sum() - np.trace(distances)) / (n * (n - 1)))
    return float(distances.mean())


def energy_distance(A: np.ndarray, B: np.ndarray) -> float:
    """Energy distance between two samples.

    Computed as ``2 E|a - b| - E|a - a'| - E|b - b'|``. It is zero if and only if the two
    distributions match, grows as they move apart, and assumes no shape for either.
    """
    return 2.0 * _mean_distance(A, B) - _mean_distance(A, A, True) - _mean_distance(B, B, True)


def mahalanobis_transform(reference: np.ndarray, regularization: float = 1e-6):
    """Center and whitening matrix that give the reference identity covariance.

    Distances measured after this transform are Mahalanobis distances under the
    reference's covariance, so "far from the controls" means the same in every direction.
    """
    reference = np.asarray(reference, dtype=np.float64)
    # np.cov spreads a single NaN over the whole matrix and eigh then fails to converge, so
    # the covariance uses complete rows only. Dropping rows keeps the estimate positive
    # semi-definite, which pairwise-complete covariance does not. The center is nan-aware
    # and uses every measured value.
    complete = reference[~np.isnan(reference).any(axis=1)]
    if complete.shape[0] < 2:
        raise ValueError(
            f"only {complete.shape[0]} of {reference.shape[0]} reference rows have no missing "
            "values, which is too few to estimate a covariance. Drop the incomplete features "
            "first, e.g. mt.pp.feature_select(adata, na_cutoff=0.0)."
        )
    centre = np.nanmedian(reference, axis=0)
    covariance = np.atleast_2d(np.cov(complete - centre, rowvar=False))
    covariance.flat[:: covariance.shape[0] + 1] += regularization

    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    eigenvalues = np.clip(eigenvalues, regularization, None)
    whitening = eigenvectors @ np.diag(1.0 / np.sqrt(eigenvalues)) @ eigenvectors.T
    return centre, whitening
