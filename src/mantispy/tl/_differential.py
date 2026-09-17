"""Differential features with the well as the experimental unit.

Cells in a well share its confluency, focus, plate position and treatment, so a per-cell
test treats dependent cells as independent replicates and finds a difference between almost
any two sets of cells. Splitting control wells into two arbitrary halves and testing per
cell calls 60% of features significant when the well-to-well spread is a quarter of the
cell-level spread. Aggregating to wells first calls none.

Wells are in turn nested in plates. A perturbation whose wells all sit on plates without
control wells cannot be separated from its plate, and an unblocked test is most confident in
that layout, so :func:`differential_features` checks the layout.

The test is the moderated t of Smyth (2004), the statistic behind ``limma``. Each feature's
residual variance is shrunk towards a prior estimated from all features. This suits
morphology profiles, with thousands of features and three or four replicates, and keeps a
feature that happens to look quiet in three wells from producing a large t. The gain is
about sevenfold in true positives at three replicates and none by six.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy import stats
from scipy.special import digamma, polygamma

from mantispy._core._reduce import get_matrix, group_codes
from mantispy._core._stats import benjamini_hochberg
from mantispy._core._utils import as_frame, get_logger, inplace_or_copy, reference_mask
from mantispy._core.schema import get_resolution


def _trigamma_inverse(value: float) -> float:
    """Solve ``trigamma(y) = value`` by the Newton iteration limma uses."""
    if value > 1e7:
        return float(1.0 / np.sqrt(value))
    if value < 1e-6:
        return 1.0 / value
    y = 0.5 + 1.0 / value
    for _ in range(50):
        tri = float(polygamma(1, y))
        step = float(tri * (1 - tri / value) / float(polygamma(2, y)))
        y += step
        if abs(step / max(y, 1e-300)) < 1e-8:
            break
    return y


def squeeze_variances(variances: np.ndarray, df: int) -> tuple[np.ndarray, float]:
    """Empirical Bayes posterior variances, and the prior degrees of freedom.

    Matches a scaled inverse chi-square to the observed log variances by the method of
    moments. A ``prior_df`` of ``inf`` means the variances were homogeneous enough that
    every feature is shrunk to the common prior.
    """
    usable = np.isfinite(variances) & (variances > 0)
    if usable.sum() < 2 or df < 1:
        return variances, 0.0

    scores = np.log(variances[usable]) - digamma(df / 2) + np.log(df / 2)
    spread = float(np.var(scores, ddof=1) - polygamma(1, df / 2))
    if spread <= 0:
        return np.where(usable, float(np.exp(np.mean(scores))), variances), np.inf

    prior_df = 2 * _trigamma_inverse(spread)
    prior_var = float(np.exp(np.mean(scores) + digamma(prior_df / 2) - np.log(prior_df / 2)))
    return (prior_df * prior_var + df * variances) / (prior_df + df), prior_df


def _design(is_treated: np.ndarray, blocks: np.ndarray | None) -> np.ndarray:
    """Intercept, the contrast of interest, and one column per extra block level."""
    columns = [np.ones(is_treated.size), is_treated.astype(float)]
    if blocks is not None:
        levels = pd.unique(blocks)
        columns.extend((blocks == level).astype(float) for level in levels[1:])
    return np.column_stack(columns)


def _fit(values: np.ndarray, design: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Least squares for every feature at once. Returns (coefficient, se_unit, sigma2, df)."""
    rank = np.linalg.matrix_rank(design)
    df = design.shape[0] - rank
    coefficients, *_ = np.linalg.lstsq(design, values, rcond=None)
    residual = values - design @ coefficients
    sigma2 = np.einsum("ij,ij->j", residual, residual) / max(df, 1)
    unit = float(np.sqrt(np.diag(np.linalg.pinv(design.T @ design))[1]))
    return coefficients[1], np.full(values.shape[1], unit), sigma2, df


@inplace_or_copy(expects=("well", "perturbation"))
def differential_features(
    adata: AnnData,
    groupby: str = "Metadata_Perturbation",
    reference: str | None = "negcon",
    contrast: str = "reference",
    block: str | None = "Metadata_Plate",
    min_replicates: int = 2,
    key_added: str = "differential",
    copy: bool = False,
) -> AnnData | None:
    """Moderated t-test per feature, per group, with wells as the replicates.

    Args:
        adata: Well-level profiles. Cell-level objects are refused; aggregate them first with
            :func:`~mantispy.tl.aggregate`, since the well is the unit that was randomized.
        groupby: Column naming the perturbation to test.
        reference: Rows to test against: ``"negcon"``, or the name of a boolean ``obs`` column.
        contrast: ``"reference"`` tests each group against the reference rows, which gives what
            the perturbation changed. ``"rest"`` tests it against every other perturbation and
            leaves the reference out, which gives what distinguishes it from the others. This is
            the marker-gene contrast, and it removes the component all active perturbations share.

            On BBBC021, mechanism retrieval from the resulting signatures is 0.631 for ``"rest"``
            and 0.505 for ``"reference"``. ``"rest"`` reproduces less well across plates because
            its comparison set depends on the rest of the screen. On the pki dose series, where
            the rest for a compound includes its own other doses, split-half agreement falls from
            0.566 to 0.448. Use ``"rest"`` to tell perturbations apart and ``"reference"`` for
            results to compare between screens.
        block: Column whose levels enter the model as fixed effects, normally the plate. Without
            it, plate variance stays in the residual and costs power; on a four-plate layout,
            blocking raised power from 0.68 to 0.94. ``None`` fits the contrast alone.
        min_replicates: Groups with fewer wells than this are left unscored.
        key_added: Name for the output table.
        copy: Return a modified copy instead of mutating in place.

    Returns:
        ``None``, or the modified copy. Writes ``uns["mantispy"][key_added]``, a tidy frame of
        ``group``, ``feature``, ``difference`` (the fitted contrast, in the units of ``X``),
        ``t``, ``pvalue`` and ``qvalue`` (Benjamini-Hochberg over the whole table), and
        ``uns["mantispy"][key_added + "_prior_df"]``, the prior degrees of freedom of the
        empirical Bayes step per group. Large values mean the variances were homogeneous and
        strongly shrunk.

    Notes:
        With ``block`` set, a group is skipped with a warning when none of its plates also holds
        rows it is compared against, since its treatment and its plate are then confounded. On
        a confounded null, the unblocked test reports 70% power with a 23% false positive rate
        because it detects the plate. None of the four screens packaged with mantispy has such
        a group.

        Features with a non-finite value in any well are left out of the fit and returned as
        ``NaN``, because one infinity makes the batched least squares return ``NaN`` for every
        feature.

        This test is meant for low-replicate screens. Across five configurations, the
        Mann-Whitney test in :func:`~mantispy.tl.effect_size` could not call a feature at three
        or fewer wells per treatment and was adequate from ten wells up, where this function is
        a rescaling of Cohen's d (Spearman 1.00). BBBC021 has three wells per treatment and the
        full JUMP TARGET-2 has 132.

        Calibration depends on the replicate count and the feature distribution, which vary by
        an order of magnitude between screens. Check the p-values on your own screen with
        :func:`~mantispy.metrics.diagnose_testing`, which relabels control wells as
        pseudo-treatments of the same size and reports the resulting false positive rate.
    """
    if get_resolution(adata) == "cell":
        raise ValueError(
            "differential_features needs well-level profiles; testing per cell treats cells as "
            "independent replicates and inflates the false discovery rate. Aggregate first with "
            "adata = mt.tl.aggregate(adata)."
        )

    if contrast not in {"reference", "rest"}:
        raise ValueError(f"contrast must be 'reference' or 'rest', got {contrast!r}")

    values = get_matrix(adata).astype(np.float64)
    is_control = reference_mask(adata, reference)
    if is_control.sum() < min_replicates:
        raise ValueError(
            f"reference={reference!r} selects {int(is_control.sum())} wells, fewer than min_replicates={min_replicates}"
        )

    obs = as_frame(adata.obs)
    if block is not None and block not in obs.columns:
        raise KeyError(f"obs has no column {block!r} to block on")
    blocks = obs[block].to_numpy() if block is not None else None

    usable = np.isfinite(values).all(axis=0)
    codes, keys = group_codes(adata, groupby)
    features = adata.var_names.to_numpy()

    frames, priors, skipped, confounded = [], {}, [], []
    for index, key in enumerate(keys):
        treated = (codes == index) & ~is_control
        # What this group is measured against: the reference rows, or every other
        # perturbation with the reference left out.
        against = is_control if contrast == "reference" else (~is_control & ~treated)
        if treated.sum() < min_replicates or against.sum() < min_replicates:
            skipped.append(str(key))
            continue

        rows = treated | against
        group_blocks = None
        if blocks is not None:
            group_blocks = blocks[rows]
            shared = set(blocks[treated]) & set(blocks[against])
            if not shared:
                confounded.append(str(key))
                continue
            # Only blocks holding both sides carry information about the contrast.
            keep = np.isin(blocks, sorted(shared)) & rows
            rows, group_blocks = keep, blocks[keep]

        design = _design(treated[rows], group_blocks)
        if np.linalg.matrix_rank(design) < design.shape[1] or design.shape[0] <= design.shape[1]:
            confounded.append(str(key))
            continue

        difference = np.full(adata.n_vars, np.nan)
        statistic = np.full(adata.n_vars, np.nan)
        pvalue = np.full(adata.n_vars, np.nan)

        coefficient, unit, sigma2, df = _fit(values[np.ix_(rows, np.flatnonzero(usable))], design)
        posterior, prior_df = squeeze_variances(sigma2, df)
        with np.errstate(invalid="ignore", divide="ignore"):
            scale = unit * np.sqrt(posterior)
            t = np.where(scale > 0, coefficient / np.where(scale > 0, scale, 1.0), np.nan)
        total_df = df + (prior_df if np.isfinite(prior_df) else 1e6)

        difference[usable] = coefficient
        statistic[usable] = t
        pvalue[usable] = 2 * stats.t.sf(np.abs(t), df=total_df)
        priors[str(key)] = float(prior_df)
        frames.append(
            pd.DataFrame(
                {
                    "group": str(key),
                    "feature": features,
                    "difference": difference,
                    "t": statistic,
                    "pvalue": pvalue,
                }
            )
        )

    if not frames:
        raise ValueError("no group had enough replicates in blocks shared with the reference")

    table = pd.concat(frames, ignore_index=True)
    table["qvalue"] = benjamini_hochberg(table["pvalue"].to_numpy())

    logger = get_logger()
    if skipped:
        logger.info(
            "differential_features left %d group(s) unscored for having < %d wells", len(skipped), min_replicates
        )
    if confounded:
        logger.warning(
            "differential_features skipped %d group(s) whose wells share no %s with the reference, "
            "so treatment and %s cannot be told apart: %s",
            len(confounded),
            block,
            block,
            ", ".join(confounded[:5]) + ("..." if len(confounded) > 5 else ""),
        )
    if not usable.all():
        logger.info(
            "differential_features returned %d feature(s) as NaN for holding a non-finite value", int((~usable).sum())
        )

    store = adata.uns.setdefault("mantispy", {})
    store[key_added] = table
    store[f"{key_added}_prior_df"] = priors
    return None
