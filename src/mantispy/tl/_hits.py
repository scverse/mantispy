"""Hit calling: which perturbations moved away from the controls, and by how much."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from anndata import AnnData

from mantispy._core._distance import mahalanobis_transform
from mantispy._core._reduce import group_codes, group_offsets, representation
from mantispy._core._stats import benjamini_hochberg, permutation_pvalue, split_reference
from mantispy._core.frames import as_frame
from mantispy._core.logging import get_logger
from mantispy._core.masks import reference_mask
from mantispy._core.mutation import inplace_or_copy

METHODS = ("mahalanobis", "ks")


def ks_statistic(samples: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Two-sample KS statistic of each row of ``samples`` against ``reference``.

    Vectorized over rows, where ``scipy.stats.ks_2samp`` takes one pair at a time.
    The supremum of ``|F_s - F_r|`` is attained at a jump of either step function, and both one-sided differences evaluated at the points of ``samples`` cover every such jump.
    With the reference sorted once, all rows take two ``searchsorted`` calls.

    Args:
        samples: One sample per row; a one-dimensional array is treated as a single row.
        reference: The sample to compare against, sorted here rather than by the caller.

    Returns:
        One statistic per row of ``samples``, or ``NaN`` for every row when either side is empty.
    """
    samples = np.atleast_2d(samples)
    reference = np.sort(reference)
    n, m = samples.shape[1], reference.size
    if n == 0 or m == 0:
        return np.full(samples.shape[0], np.nan)

    ordered = np.sort(samples, axis=1)
    below = np.searchsorted(reference, ordered, side="left") / m
    at_or_below = np.searchsorted(reference, ordered, side="right") / m
    steps = np.arange(1, n + 1) / n
    upper = np.max(steps[None, :] - at_or_below, axis=1)
    lower = np.max(below - (steps - 1.0 / n)[None, :], axis=1)
    return np.maximum(np.maximum(upper, lower), 0.0)


def _statistic(distances: np.ndarray, control_distances: np.ndarray, method: str) -> np.ndarray:
    """One number per row of ``distances``: a median, or a KS statistic."""
    if method == "mahalanobis":
        return np.median(np.atleast_2d(distances), axis=1)
    return ks_statistic(distances, control_distances)


@inplace_or_copy()
def hit_calling(
    adata: AnnData,
    groupby: str = "Metadata_Perturbation",
    reference: str | None = "negcon",
    method: str = "mahalanobis",
    use_rep: str | None = None,
    n_permutations: int = 1000,
    threshold: float = 0.05,
    seed: int = 0,
    key_added: str = "hits",
    copy: bool = False,
) -> AnnData | None:
    """Call hits by testing each group's distance from the controls.

    Args:
        adata: Object to score, at cell or well resolution.
        groupby: Column defining the groups to test.
        reference: Which rows are the controls. They are split in half, one half to fit the covariance and the other to form the null (see Notes).
        method: ``"mahalanobis"`` scores the median distance of the group's rows from the control centroid, measured in the controls' covariance so that directions the controls already vary in count for less. ``"ks"`` scores the Kolmogorov-Smirnov statistic between the group's and the controls' distance distributions, which detects a shifted subpopulation that leaves the median unchanged. Use it at cell resolution. Its p-value comes from ``scipy.stats.ks_2samp``, so ``n_permutations`` does not apply.
        use_rep: Score ``obsm[use_rep]`` instead of ``X``. When the covariance-fitting half of the controls has no more rows than there are features, the covariance is singular and a warning suggests a PCA representation.
        n_permutations: Size of the permutation null. Applies to ``method="mahalanobis"`` only.
        threshold: q-value below which a group is called a hit in ``is_hit``.
        seed: Seed for the control split and the permutation null.
        key_added: Name for the outputs.
        copy: Return a modified copy instead of mutating in place.

    Returns:
        ``None``, or the modified copy.
        Writes ``uns["mantispy"][key_added]`` with ``group``, ``n_obs``, ``distance``, ``pvalue``, ``qvalue`` and ``is_hit``, and joins ``obs[key_added + "_distance"]`` and ``obs[key_added + "_qvalue"]`` back onto the rows.

    Raises:
        ValueError: ``method`` is not one of ``METHODS``, or ``reference`` selects fewer than four rows.

    Notes:
        The controls are split in half.
        One half estimates the centroid and the covariance, and the other supplies the null.
        A null drawn from the rows that defined the centroid would be in-sample while every tested group is out-of-sample, so it would come out too small and call pure noise as hits.
        Because only half the controls fit the covariance, the singular-covariance warning fires when there are fewer than about twice as many controls as features.

        The null is drawn from the controls only, and asks whether a group is further out than the same number of control rows would be.
        Drawing from every row would put real hits into the null, and a screen with many hits would look like one with none.

        Calibration degrades with few controls.
        On pure-noise screens of 12 groups of 12 rows with 10 features (the ``pure_noise_screen`` test fixture), the false positive rate at a nominal 0.05 was 0.10 with 48 controls, 0.077 with 192 and 0.052 with 384.
        Across four configurations from 20 to 80 features and 48 to 200 controls, ten seeds each, it was 2.1% overall.
        Neither figure is a bound for another screen, and both were measured with ``method="mahalanobis"``.
        Under ``method="ks"`` there is no permutation null; each group's distances are compared with those of the null half by ``scipy.stats.ks_2samp``.

        To check the rate on your own screen, :func:`~mantispy.metrics.diagnose_testing` relabels control wells as pseudo-treatments of your group sizes and reports the fraction called.
    """
    from scipy.stats import ks_2samp

    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")

    values = representation(adata, use_rep)
    is_control = reference_mask(adata, reference)
    n_control = int(is_control.sum())
    if n_control < 4:
        raise ValueError(
            f"hit calling needs at least four reference rows, and reference={reference!r} selects {n_control}; "
            "half fit the covariance and half form the null. On a consensus object the controls were probably "
            "dropped before mt.tl.consensus; keep them through consensus and drop the control row afterwards."
        )

    generator = np.random.default_rng(seed)
    fit_rows, null_rows = split_reference(np.flatnonzero(is_control), generator)
    if fit_rows.size <= values.shape[1]:
        warnings.warn(
            f"the covariance is estimated from {fit_rows.size} of {n_control} reference rows (half the "
            f"controls) for {values.shape[1]} features, so it is singular and the Mahalanobis distances are "
            "dominated by the regularization. Reduce the features with mt.pp.feature_select, or score a PCA "
            "representation with use_rep='X_pca'.",
            UserWarning,
            stacklevel=3,
        )

    centre, whitening = mahalanobis_transform(values[fit_rows])
    # After whitening, the Mahalanobis distance to the control centroid is the norm of the row.
    # Fill gaps before whitening, which mixes columns; one NaN would otherwise zero the row's distance.
    to_control = np.linalg.norm(np.nan_to_num(values - centre) @ whitening, axis=1)
    control_distances = to_control[null_rows]

    codes, keys = group_codes(adata, groupby)

    observed = np.empty(len(keys))
    sizes = np.empty(len(keys), dtype=int)
    pvalues = np.empty(len(keys))
    null = np.empty((len(keys), n_permutations))
    order, offsets = group_offsets(codes, len(keys))
    for index in range(len(keys)):
        rows = order[offsets[index] : offsets[index + 1]]
        sizes[index] = rows.size
        observed[index] = _statistic(to_control[rows], control_distances, method)[0]
        if method == "ks":
            # The KS statistic has a known null distribution.
            # A permutation null drawn from the reference rows is too small, since each draw is a subset of its own reference, and a further split cannot supply pseudo-groups as large as the real ones.
            pvalues[index] = float(ks_2samp(to_control[rows], control_distances).pvalue)
            continue
        draws = generator.choice(null_rows, size=(n_permutations, max(rows.size, 1)), replace=True)
        null[index] = _statistic(to_control[draws], control_distances, method)

    if method != "ks":
        pvalues = permutation_pvalue(observed, null)
    qvalues = benjamini_hochberg(pvalues)
    table = pd.DataFrame(
        {
            "group": [str(key) for key in keys],
            "n_obs": sizes,
            "distance": observed,
            "pvalue": pvalues,
            "qvalue": qvalues,
            "is_hit": qvalues < threshold,
        }
    )

    adata.uns.setdefault("mantispy", {})[key_added] = table
    lookup = table.set_index("group")
    labels = as_frame(adata.obs)[groupby].astype(str)
    adata.obs[f"{key_added}_distance"] = lookup["distance"].reindex(labels).to_numpy()
    adata.obs[f"{key_added}_qvalue"] = lookup["qvalue"].reindex(labels).to_numpy()
    get_logger().info("hit_calling(%s) called %d of %d groups", method, int(table["is_hit"].sum()), len(table))
    return None
