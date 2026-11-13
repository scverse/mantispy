"""Pick a PCA solver that stays bounded in memory on a tall, narrow table.

mantispy never reimplements a PCA; it hands the choice of solver to whichever library a site already
uses (scanpy, scikit-learn, or :func:`numpy.linalg.svd`). This helper only names the solver to pass:
when a table has many more rows than columns and few enough columns, eigendecomposing the small
feature covariance returns the same components as the full decomposition at a fraction of its memory
and time, so ``None`` picks it there and keeps each site's previous solver everywhere else.
"""

from __future__ import annotations

# Below this many columns the feature covariance is small enough to eigendecompose directly, and a
# table at least this many times taller than it is wide has enough rows that the covariance is well
# estimated and squaring its condition number costs no usable precision.
_TALL_RATIO = 10
_MAX_FEATURES = 2000


def pca_solver(n_obs: int, n_vars: int, svd_solver: str | None = None, *, fallback: str) -> str:
    """Name the PCA solver for a table of ``n_obs`` rows and ``n_vars`` columns.

    Args:
        n_obs: Rows the decomposition sees.
        n_vars: Columns the decomposition sees.
        svd_solver: ``None`` to choose, or any solver name the calling site understands to force it.
        fallback: The solver to keep when ``None`` does not switch, i.e. the site's previous one.

    Returns:
        ``"covariance_eigh"`` when ``svd_solver`` is ``None`` and the table is tall and narrow,
        ``fallback`` when it is ``None`` but not, and ``svd_solver`` unchanged otherwise.
    """
    if svd_solver is not None:
        return svd_solver
    if n_vars <= _MAX_FEATURES and n_obs >= _TALL_RATIO * n_vars:
        return "covariance_eigh"
    return fallback
