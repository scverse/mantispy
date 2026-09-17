"""Consensus signatures, one profile per perturbation.

``tl.aggregate(by=("Metadata_Perturbation",))`` gives a median consensus.
modz is the weighted version used in the field.
Replicates that agree with the others count for more, so one bad well moves the signature less.

The weighting follows ``pycytominer.cyto_utils.modz.modz_base``, which comes from cmapPy:

1. correlate every replicate with every other over the features (Spearman by default);
2. set the diagonal to NaN and clip negative correlations to zero, so a replicate that anticorrelates with the rest counts as uninformative;
3. take a replicate's raw weight as its mean correlation with the others, floored at ``min_weight``;
4. normalize the weights to sum to one (equal weights if they are all zero) and round them to ``precision`` decimals;
5. take the weighted sum as the signature.
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
from anndata import AnnData
from scipy.stats import rankdata

from mantispy._core._numba import MEDIAN
from mantispy._core._reduce import get_matrix, group_codes, reduce_grouped
from mantispy._core.frames import as_frame
from mantispy._core.logging import report_drop
from mantispy._core.provenance import record_params
from mantispy._core.schema import stamp
from mantispy.tl._aggregate import _group_obs
from mantispy.tl._similarity import similarity_matrix

METHODS = ("modz", "median")
CORRELATIONS = ("spearman", "pearson")


def modz_weights(
    block: np.ndarray, correlation: str = "spearman", min_weight: float = 0.01, precision: int = 4
) -> np.ndarray:
    """Weight each replicate by how well it agrees with the others.

    A perturbation whose replicates all sit at ``min_weight`` has no reproducible signature, whatever its consensus profile looks like.

    Args:
        block: One perturbation's replicates, as rows, by features.
        correlation: ``"spearman"`` ranks the features first, ``"pearson"`` correlates the values.
        min_weight: Floor on a replicate's weight.
        precision: Decimals the weights are rounded to, as in pycytominer.

    Returns:
        One weight per row of ``block``, summing to one.
    """
    if block.shape[0] == 1:
        return np.ones(1)

    values = np.asarray(block, dtype=np.float64)
    if correlation == "spearman":
        # nan_policy="omit" ranks the present values and leaves NaN in place.
        # The default, "propagate", turns a replicate with one missing feature into an all-NaN row, which drops its weight to min_weight.
        values = rankdata(values, axis=1, nan_policy="omit")

    # similarity_matrix fills a gap with zero, below every rank, so replicates sharing a gap would correlate there and take the largest weights.
    # A replicate's own mean is the neutral fill: pearson centers the rows, so a filled feature then contributes nothing.
    gaps = np.isnan(values)
    if gaps.any():
        present = np.maximum((~gaps).sum(axis=1, keepdims=True), 1)
        centre = np.where(gaps, 0.0, values).sum(axis=1, keepdims=True) / present
        values = np.where(gaps, centre, values)

    matrix = similarity_matrix(values, metric="pearson").astype(np.float64)
    np.fill_diagonal(matrix, np.nan)

    weights = np.nanmean(np.clip(matrix, 0.0, None), axis=1)
    weights = np.clip(weights, min_weight, None)
    total = weights.sum()
    weights = np.full(weights.size, 1.0 / weights.size) if total == 0 else weights / total
    return np.round(weights, precision)


def consensus(
    adata: AnnData,
    by: str = "Metadata_Perturbation",
    method: str = "modz",
    correlation: str = "spearman",
    min_replicates: int = 2,
    min_weight: float = 0.01,
    precision: int = 4,
) -> AnnData:
    """One profile per group, weighting replicates by how well they agree.

    Args:
        adata: Profiles to summarize, normally well level.
        by: Column defining a perturbation.
        method: ``"modz"`` weights replicates by their agreement, so a single bad replicate moves the signature far less than it would a plain mean. ``"median"`` is the unweighted alternative, identical to ``tl.aggregate`` by the same column.
        correlation: How replicate agreement is measured: ``"spearman"`` (pycytominer's default, and insensitive to a few extreme features) or ``"pearson"``.
        min_replicates: Groups with fewer replicates are dropped.
        min_weight: Floor on a replicate's weight. A group whose replicates all land on the floor becomes an unweighted mean.
        precision: Decimals the weights are rounded to, as in pycytominer.

    Returns:
        A new object at ``"perturbation"`` resolution, one row per group, with ``Metadata_ReplicateCount`` and the metadata that is constant within a group.
        ``uns["mantispy"]["consensus_weights"]`` keeps the weight given to every input row, including the rows of groups dropped for having too few replicates, so a signature can be traced back to its replicates.
        Under ``method="median"`` no weights are computed and every row is recorded as 1.0, since a median is not a weighted sum.

    Raises:
        ValueError: ``method`` is not one of ``METHODS``, or ``correlation`` is not one of ``CORRELATIONS``.

    Notes:
        A missing value is filled with its own replicate's mean before the replicates are correlated.
        Zero would be an extreme value among ranks, and two replicates sharing a gap would look alike.
        The signature itself is a weighted sum, so a NaN feature stays NaN.

        modz is a weighted mean.
        With one outlying replicate it drifts about forty times less than the unweighted mean, but it does not beat a median.
        On BBBC021, not-same-compound MOA retrieval was 0.777 with ``method="median"`` and 0.660 with modz.
        It is the default because it matches pycytominer and is the usual definition of a consensus signature.
        Compare both methods on your own data.

        Normalize before taking a consensus, and first drop the features ``pp.normalize`` flags in ``var["degenerate_scale"]``.
        A feature that is constant among the controls is divided by epsilon, and a weighted mean carries the resulting values of order 1e17 into the signature, where a median would discard them.
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    if correlation not in CORRELATIONS:
        raise ValueError(f"correlation must be one of {CORRELATIONS}, got {correlation!r}")

    codes, keys = group_codes(adata, [by])
    counts = np.bincount(codes, minlength=len(keys))
    weights = np.ones(adata.n_obs)

    if method == "median":
        values, _, _ = reduce_grouped(adata, [by], MEDIAN)
    else:
        X = get_matrix(adata)
        values = np.zeros((len(keys), adata.n_vars), dtype=np.float64)
        for index in range(len(keys)):
            rows = np.flatnonzero(codes == index)
            block = modz_weights(X[rows], correlation, min_weight, precision)
            weights[rows] = block
            values[index] = block @ X[rows]

    obs = _group_obs(adata, [by], keys, codes, counts)
    # _group_obs writes the group size as Metadata_CellCount; here a group is replicates.
    obs = obs.rename(columns={"Metadata_CellCount": "Metadata_ReplicateCount"})

    keep = counts >= min_replicates
    report_drop("group(s)", int((~keep).sum()), int(keep.size), remedy=f"lower min_replicates below {min_replicates}")

    result = ad.AnnData(
        X=values[keep].astype(np.float32),
        obs=obs.loc[keep].reset_index(drop=True).set_axis(pd.Index([str(i) for i in range(int(keep.sum()))])),
        var=as_frame(adata.var).copy(),
    )
    stamp(result, resolution="perturbation")
    result.uns["mantispy"]["consensus_weights"] = pd.DataFrame(
        {"group": [str(keys[code]) for code in codes], "weight": weights}
    )
    record_params(
        result, "consensus", {"by": by, "method": method, "correlation": correlation, "min_replicates": min_replicates}
    )
    return result
