"""Which features can be trusted: reproducibility, and dependence on the batch."""

from __future__ import annotations

import numpy as np
from anndata import AnnData

from mantispy._core._reduce import get_matrix, group_codes
from mantispy._core._stats import benjamini_hochberg
from mantispy._core.logging import get_logger
from mantispy._core.mutation import inplace_or_copy


def intraclass_correlation(X: np.ndarray, codes: np.ndarray, n_groups: int) -> np.ndarray:
    """One-way ICC per column, the share of variance that lies between groups.

    ``ICC(1) = (MSB - MSW) / (MSB + (k0 - 1) * MSW)``, with ``k0`` the effective group size for an unbalanced design.
    Missing values are dropped per column, so a feature measured in fewer wells is scored on the wells it has rather than on zeros.

    Args:
        X: Values to score, one feature per column.
        codes: Group code per row, as ``group_codes`` returns.
        n_groups: Number of groups those codes index.

    Returns:
        One ICC per column, clipped to ``[-1, 1]``, and 0.0 for a column left with fewer than two observations or fewer than two groups.
    """
    observed = np.isfinite(X)
    values = np.where(observed, X, 0.0)

    counts = np.zeros((n_groups, X.shape[1]))
    sums = np.zeros((n_groups, X.shape[1]))
    np.add.at(counts, codes, observed)
    np.add.at(sums, codes, values)

    total = counts.sum(axis=0)
    present = counts > 0
    effective_groups = present.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        means = np.where(present, sums / np.where(counts == 0, 1.0, counts), 0.0)
        grand = sums.sum(axis=0) / np.where(total == 0, 1.0, total)

        between = (counts * (means - grand) ** 2).sum(axis=0) / np.maximum(effective_groups - 1, 1)
        residual = np.zeros_like(X)
        residual[observed] = (X - means[codes])[observed]
        within = (residual**2).sum(axis=0) / np.maximum(total - effective_groups, 1)

        # Effective group size: n_bar corrected for imbalance, as in Shrout & Fleiss.
        size = (total - (counts**2).sum(axis=0) / np.where(total == 0, 1.0, total)) / np.maximum(
            effective_groups - 1, 1
        )
        icc = (between - within) / (between + (size - 1.0) * within)

    icc = np.where((total < 2) | (effective_groups < 2) | ~np.isfinite(icc), 0.0, icc)
    return np.clip(icc, -1.0, 1.0)


@inplace_or_copy(expects=("well", "perturbation"))
def feature_reproducibility(
    adata: AnnData,
    groupby: str = "Metadata_Perturbation",
    min_icc: float = 0.2,
    key_added: str = "icc",
    copy: bool = False,
) -> AnnData | None:
    """Score each feature by how consistently replicates of a perturbation agree on it.

    A feature that varies only within replicate groups is noise, however large its variance.
    The intraclass correlation is the share of variance that lies between groups, which a variance filter does not measure.

    Args:
        adata: Profiles with several replicates per group, at well or perturbation resolution.
        groupby: ``obs`` column whose groups are the replicate sets.
        min_icc: Threshold for the boolean column. See the Notes on choosing it.
        key_added: ``var[key_added]`` holds the ICC; ``var[key_added + "_selected"]`` the flag.
        copy: Return a modified copy instead of mutating in place.

    Returns:
        ``None``, or the modified copy. Writes the ICC to ``var[key_added]`` and the flag to ``var[key_added + "_selected"]``.

    Raises:
        ValueError: If ``groupby`` has a single group, so that no variance can lie between groups.

    Notes:
        Filtering on ICC raised replicate-retrieval mAP on all three packaged screens: bbbc021 from 0.121 to 0.141 (ICC > 0.2) and 0.162 (> 0.4), rohban2017 from 0.097 to 0.143 and 0.136, and pki from 0.178 to 0.192 and 0.200.
        The best cutoff differs by dataset, so 0.2 is a conservative default; choose one from the distribution in ``var[key_added]``.

        ICC filtering can hurt other tasks.
        On BBBC021, not-same-compound MOA retrieval fell from 0.777 over all features to 0.767 at ICC > 0.2 and 0.757 at 0.4.
        ICC measures reproducibility within a treatment, which is a different property from agreement between compounds that share a mechanism.
    """
    codes, keys = group_codes(adata, groupby)
    if len(keys) < 2:
        raise ValueError(f"{groupby!r} has one group, so no variance can lie between groups")

    icc = intraclass_correlation(get_matrix(adata).astype(np.float64), codes, len(keys))
    adata.var[key_added] = icc
    adata.var[f"{key_added}_selected"] = icc > min_icc
    get_logger().info(
        "feature_reproducibility: %d of %d features above ICC %.2f (median %.3f)",
        int((icc > min_icc).sum()),
        adata.n_vars,
        min_icc,
        float(np.median(icc)),
    )
    return None


@inplace_or_copy()
def feature_batch_sensitivity(
    adata: AnnData,
    batch_key: str = "Metadata_Batch",
    threshold: float = 0.05,
    key_added: str = "batch",
    copy: bool = False,
) -> AnnData | None:
    """Test each feature for dependence on the batch, after whatever correction was applied.

    Args:
        adata: Profiles to test.
        batch_key: ``obs`` column holding the batch, imaging week or plate.
        threshold: q-value below which a feature is called batch sensitive.
        key_added: Prefix for ``var[key_added + "_pvalue"]``, ``_qvalue`` and ``_sensitive``.
        copy: Return a modified copy instead of mutating in place.

    Returns:
        ``None``, or the modified copy. Writes ``var[key_added + "_pvalue"]``, ``var[key_added + "_qvalue"]`` and the boolean ``var[key_added + "_sensitive"]``.

    Raises:
        ValueError: If ``batch_key`` holds fewer than two batches.

    Notes:
        Uses Kruskal-Wallis instead of ANOVA, because morphology features are not normally distributed and a few extreme wells should not decide the result.
        Expect a large fraction to come back sensitive, since per-plate centering does not remove plate structure: over BBBC021's treated wells, 338 of 344 features still depend on the plate at q < 0.05 after per-plate normalization, and rohban2017 and pki are similar.

        Compare the sensitive fraction before and after a correction.
    """
    from scipy.stats import kruskal

    codes, keys = group_codes(adata, batch_key)
    if len(keys) < 2:
        raise ValueError(f"batch sensitivity needs at least two batches, but {batch_key!r} has {len(keys)}")

    X = get_matrix(adata).astype(np.float64)
    blocks = [np.flatnonzero(codes == index) for index in range(len(keys))]
    pvalues = np.ones(adata.n_vars)
    for feature in range(adata.n_vars):
        samples = [column[np.isfinite(column)] for column in (X[rows, feature] for rows in blocks)]
        samples = [sample for sample in samples if sample.size > 1]
        if len(samples) > 1:
            try:
                pvalues[feature] = kruskal(*samples).pvalue
            except ValueError:
                pvalues[feature] = 1.0  # every value identical: no evidence of a difference

    qvalues = benjamini_hochberg(pvalues)
    adata.var[f"{key_added}_pvalue"] = pvalues
    adata.var[f"{key_added}_qvalue"] = qvalues
    adata.var[f"{key_added}_sensitive"] = qvalues < threshold
    get_logger().info(
        "feature_batch_sensitivity: %d of %d features depend on %s at q < %.2g",
        int((qvalues < threshold).sum()),
        adata.n_vars,
        batch_key,
        threshold,
    )
    return None
