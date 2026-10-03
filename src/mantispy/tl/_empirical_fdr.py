"""Calibrate a per-gene hit score against control genes that cannot have a phenotype."""

from __future__ import annotations

import warnings
from collections.abc import Collection

import numpy as np
import pandas as pd
from anndata import AnnData

from mantispy._core.frames import as_frame
from mantispy._core.logging import get_logger
from mantispy._core.mutation import inplace_or_copy

CRITERIA = ("p", "q")

_FEW_CONTROLS = 50


@inplace_or_copy()
def empirical_fdr(
    adata: AnnData,
    *,
    control_genes: Collection[str],
    score: str,
    group: str | None = None,
    alpha: float = 0.01,
    criterion: str = "p",
    key_added: str = "empirical_fdr",
    copy: bool = False,
) -> AnnData | None:
    """Score each gene against a control set that should produce no phenotype.

    Genes that are not expressed in the screened cell line cannot show a knockout phenotype, so the spread of their scores is an empirical null: a knockout that scores past what those genes reach is unlikely to be noise.
    This is the calibration PERISCOPE applies with non-expressed genes, generalized to any per-gene score and any control set (:func:`mantispy.io.unexpressed_genes` builds one).

    The score is yours to choose and must rise with the strength of the phenotype.
    A count of the features that separate a gene from the controls works well, from a per-feature test such as the Mann-Whitney p-values of :func:`~mantispy.tl.effect_size` or the moderated t of :func:`~mantispy.tl.differential_features`.
    An unbounded count separates the controls from the rest; a bounded score such as the activity mean average precision of :func:`~mantispy.tl.map` saturates, so a handful of off-target controls at its ceiling block the low tail.
    This calibrates against control genes that cannot respond, where :func:`~mantispy.tl.hit_calling` calibrates each group against control wells; use this when a set of non-responding genes is the better null.

    Args:
        adata: One row per gene, with the score in ``obs`` and a column (or index) naming the gene.
        control_genes: The genes that form the null, such as the unexpressed set.
            A bare string is rejected, since it would be read as a set of single characters.
        score: ``obs`` column holding the per-gene score, higher for a stronger phenotype.
        group: ``obs`` column naming the gene, matched against `control_genes`; the index is used when ``None``.
        alpha: The cutoff a gene must clear in ``criterion`` to be called a hit.
        criterion: Which quantity ``is_hit`` reads, ``"p"`` or ``"q"`` (see Notes).
        key_added: Prefix for the outputs.
        copy: Return a modified copy instead of mutating in place.

    Returns:
        ``None``, or the modified copy.
        Writes ``uns["mantispy"][key_added]`` with ``group``, ``score``, ``is_control``, ``pvalue``, ``qvalue`` and ``is_hit``, and joins ``obs[key_added + "_pvalue"]``, ``obs[key_added + "_qvalue"]``, ``obs[key_added + "_control"]`` and ``obs[key_added]`` back onto the rows.
        Control genes and genes with a missing score get a missing ``pvalue`` and ``qvalue`` and are never hits.

    Raises:
        TypeError: `control_genes` is a string rather than a collection of gene names.
        ValueError: `criterion` is not one of :data:`CRITERIA`, or no control gene is present in the object, or every control gene has a missing score.
        KeyError: `score`, or `group`, is not an ``obs`` column.

    Notes:
        Two quantities come out, answering different questions.
        The p-value is the share of control genes that reach a gene's score or beyond, with one added to the count and to the control total so a gene past every control gets a small positive value rather than zero, so a cutoff on it fixes the rate at which a control gene is called, the way PERISCOPE sets its threshold.
        It does not account for the number of genes tested, so the share of real false positives in a p-value hit list is higher than the cutoff whenever many genes carry no phenotype.
        The q-value is a target-decoy false discovery rate: at a gene's score it compares the fraction of controls reaching it with the fraction of all tested genes reaching it, and is made monotone from the permissive end.
        It estimates the share of the hit list that is false, so it is the honest rate to quote, and it can sit well above the p-value cutoff when the controls and the tested genes overlap.
        ``criterion`` picks which one ``is_hit`` reads; the other is written too.

        The controls must behave like the tested genes under the null for either number to mean what it says.
        A few controls with a strong, reproducible score (off-target reagents, or genes the expression reference mislabels) sit in the extreme tail and raise the floor the q-value can reach; tightening the expression cutoff rarely removes them, since an off-target effect is not an expression artifact.
        With few controls the null is coarse and a warning fires.
    """
    if isinstance(control_genes, str):
        raise TypeError("control_genes must be a collection of gene names, not a single string")
    if criterion not in CRITERIA:
        raise ValueError(f"criterion must be one of {CRITERIA}, got {criterion!r}")

    obs = as_frame(adata.obs)
    if score not in obs:
        raise KeyError(f"score={score!r} is not an obs column")
    if group is not None and group not in obs:
        raise KeyError(f"group={group!r} is not an obs column")

    names = (obs[group] if group is not None else pd.Series(adata.obs_names, index=obs.index)).astype(str).to_numpy()
    is_control = np.isin(names, np.asarray(list(control_genes), dtype=str))
    values = pd.to_numeric(obs[score], errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(values)

    control = is_control & finite
    target = ~is_control & finite
    n_control = int(control.sum())
    if not is_control.any():
        raise ValueError(
            "no control gene is present in the object; control_genes shares no name with "
            f"{'obs[' + group + ']' if group else 'the index'}."
        )
    if n_control == 0:
        raise ValueError(f"every control gene has a missing {score!r}, so there is no null to calibrate against")
    if n_control < _FEW_CONTROLS:
        warnings.warn(
            f"only {n_control} control genes are present, so the null is coarse and the smallest resolvable "
            "rate is one over that count. Widen the control set or loosen alpha.",
            UserWarning,
            stacklevel=3,
        )

    control_sorted = np.sort(values[control])
    target_sorted = np.sort(values[target])
    n_target = target_sorted.size

    def at_or_above(sorted_values: np.ndarray, query: np.ndarray) -> np.ndarray:
        """Count of ``sorted_values`` greater than or equal to each query value."""
        return sorted_values.size - np.searchsorted(sorted_values, query, side="left")

    pvalue = np.full(adata.n_obs, np.nan)
    qvalue = np.full(adata.n_obs, np.nan)
    if n_target:
        scores = values[target]
        decoy_count = at_or_above(control_sorted, scores)
        # Add one to the count and to the control total, so a gene past every control gets 1 / (n_control + 1)
        # rather than a zero tail probability that would overstate the strongest genes.
        p = (decoy_count + 1) / (n_control + 1)
        pvalue[target] = p
        # Every query is itself a target, so at_or_above(target_sorted, .) is at least one and target_rate is positive.
        target_rate = at_or_above(target_sorted, scores) / n_target
        fdr = (decoy_count / n_control) / target_rate
        order = np.argsort(scores)[::-1]
        monotone = np.minimum.accumulate(fdr[order][::-1])[::-1]
        q = np.empty(n_target)
        q[order] = monotone
        # A false discovery rate cannot be more confident than the gene's own tail probability.
        qvalue[target] = np.clip(np.maximum(q, p), 0.0, 1.0)

    chosen = pvalue if criterion == "p" else qvalue
    is_hit = np.zeros(adata.n_obs, dtype=bool)
    is_hit[target] = chosen[target] <= alpha

    table = pd.DataFrame(
        {
            "group": names,
            "score": values,
            "is_control": is_control,
            "pvalue": pvalue,
            "qvalue": qvalue,
            "is_hit": is_hit,
        }
    )
    adata.uns.setdefault("mantispy", {})[key_added] = table
    adata.obs[f"{key_added}_control"] = is_control
    adata.obs[f"{key_added}_pvalue"] = pvalue
    adata.obs[f"{key_added}_qvalue"] = qvalue
    adata.obs[key_added] = is_hit
    get_logger().info(
        "empirical_fdr(%s<=%.3g) called %d of %d genes against %d controls",
        criterion,
        alpha,
        int(is_hit.sum()),
        n_target,
        n_control,
    )
    return None
