"""Combine per-guide evidence into a gene-level hit call against a control-matched null."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy import stats

from mantispy._core.frames import as_frame
from mantispy._core.logging import get_logger
from mantispy._core.mutation import inplace_or_copy

METHODS = ("stouffer", "fisher")

_FEW_CONTROLS = 50
#: p-values are clipped into this range so a guide at 0 or 1 does not send its z to infinity.
_P_CLIP = 1e-12


def _combine(z: np.ndarray, p: np.ndarray, weight: np.ndarray, method: str) -> float:
    """Combine one group of guides into a single statistic, higher for a stronger phenotype."""
    if method == "stouffer":
        return float(np.sum(weight * z) / np.sqrt(np.sum(weight**2)))
    # Fisher: -2 sum ln(p) over the one-sided p-values, larger when any guide is strongly significant.
    return float(-2.0 * np.sum(np.log(p)))


@inplace_or_copy()
def aggregate_guides(
    adata: AnnData,
    *,
    score: str,
    guide: str,
    gene: str,
    control: str,
    method: str = "stouffer",
    weight: str | None = None,
    direction: str | None = None,
    n_null: int = 10000,
    alpha: float = 0.05,
    key_added: str = "gene_aggregation",
    seed: int = 0,
    copy: bool = False,
) -> AnnData | None:
    """Combine a gene's guides into one hit call, calibrated against non-targeting controls.

    A gene in a pooled CRISPR screen is targeted by many guides, and a call must pool them rather than treat each guide as its own result.
    This takes a per-guide score that rises with the strength of the phenotype, combines the guides of each gene, and calibrates the combined statistic against random same-size groups of non-targeting-control (NTC) guides.
    Matching the group size matters because the combined statistic's null depends on how many guides a gene has.

    The score is a one-sided per-guide p-value, small when the guide shows a phenotype, as produced by the activity mean average precision of :func:`~mantispy.tl.map`, or by :func:`~mantispy.tl.effect_size` or :func:`~mantispy.tl.hit_calling` against the negative controls.
    ``"stouffer"`` turns each p into a z and sums the z's, so it rewards a consistent effect across a gene's guides; ``"fisher"`` combines the p's and fires when any one guide is strongly significant, which is more powerful for a gene with a single potent guide but is carried by a lone off-target reagent.
    This combines per-guide evidence within a gene, where :func:`~mantispy.tl.empirical_fdr` calibrates an already-per-gene score against a set of non-responding genes.

    Args:
        adata: One row per guide, with the score, the guide id and the gene in ``obs``.
        score: ``obs`` column holding the one-sided per-guide p-value, small for a stronger phenotype.
        guide: ``obs`` column naming the guide; one scored row per guide is expected.
        gene: ``obs`` column naming the gene the guide targets.
        control: The value of `gene` that marks the non-targeting guides forming the null, such as ``"nontargeting"``.
        method: ``"stouffer"`` (default) or ``"fisher"`` (see above and Notes).
        weight: ``obs`` column of per-guide weights for a weighted Stouffer combination, such as a replicate or cell count; equal weights when ``None``. Only ``"stouffer"`` accepts weights.
        direction: ``obs`` column whose sign says which way a guide moved, so guides that disagree cancel instead of adding. Only ``"stouffer"`` uses it; a one-sided p already fixes the direction for ``"fisher"``.
        n_null: Random NTC groups drawn per distinct guide count to build the null.
        alpha: The q-value cutoff a gene must clear to be called a hit.
        key_added: Prefix for the outputs.
        seed: Seed for the null sampling.
        copy: Return a modified copy instead of mutating in place.

    Returns:
        ``None``, or the modified copy.
        Writes ``uns["mantispy"][key_added]`` with one row per gene: ``gene``, ``n_guides``, ``statistic``, ``pvalue``, ``qvalue`` and ``is_hit``, sorted by ``pvalue``.
        Joins ``obs[key_added + "_pvalue"]``, ``obs[key_added + "_qvalue"]`` and ``obs[key_added]`` onto the guide rows, broadcast from each guide's gene.
        Non-targeting guides, guides with a missing score, and genes left with no scored guide get a missing ``pvalue`` and ``qvalue`` and are never hits.

    Raises:
        ValueError: `method` is not one of ``METHODS``, `weight` or `direction` is given with ``"fisher"``, no control guide is present, or no gene has a scored guide.
        KeyError: `score`, `guide`, `gene`, `weight` or `direction` is not an ``obs`` column.

    Notes:
        The p-value is the share of same-size NTC groups whose combined statistic reaches the gene's or beyond, with one added to the count and the total so a gene past every NTC group gets a small positive value rather than zero.
        Each NTC group is drawn without replacement, a genuine same-size sample of the controls, so the pool of non-targeting guides must be well above the largest gene's guide count; a warning fires when a gene uses more than half the pool, since a same-size group is then nearly the whole pool, its null has almost no spread, and its p-value cannot be trusted (a gene with more guides than the whole pool falls back to sampling with replacement).
        The q-value is the Benjamini-Hochberg false discovery rate over the tested genes, made monotone from the permissive end and never below the gene's own p-value.

        The non-targeting guides must behave like a true-null gene for the calibration to hold.
        With few NTC guides the null is coarse and a warning fires.
        A gene whose guides disagree in direction scores low under ``"stouffer"`` unless `direction` lets them cancel; under ``"fisher"`` a one-sided p already points one way, so opposing guides cannot cancel.
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    if method == "fisher" and weight is not None:
        raise ValueError("weight is only supported for method='stouffer'; Fisher combines p-values directly")
    if method == "fisher" and direction is not None:
        raise ValueError("direction is only supported for method='stouffer'; a one-sided p already fixes the direction")

    obs = as_frame(adata.obs)
    for name, value in (("score", score), ("guide", guide), ("gene", gene)):
        if value not in obs:
            raise KeyError(f"{name}={value!r} is not an obs column")
    for name, optional in (("weight", weight), ("direction", direction)):
        if optional is not None and optional not in obs:
            raise KeyError(f"{name}={optional!r} is not an obs column")

    genes = obs[gene].astype(str).to_numpy()
    p = np.clip(pd.to_numeric(obs[score], errors="coerce").to_numpy(dtype=float), _P_CLIP, 1.0)  # clip keeps NaN
    z = stats.norm.isf(p)  # one-sided: small p -> large positive z
    if direction is not None:
        sign = np.sign(pd.to_numeric(obs[direction], errors="coerce").to_numpy(dtype=float))
        sign[sign == 0] = 1.0
        z = z * sign
    w = (
        pd.to_numeric(obs[weight], errors="coerce").to_numpy(dtype=float)
        if weight is not None
        else np.ones(adata.n_obs)
    )

    is_control = genes == str(control)
    scored = np.isfinite(p) & np.isfinite(z) & np.isfinite(w)
    control_scored = is_control & scored
    if not is_control.any():
        raise ValueError(f"no control guide is present; obs[{gene!r}] has no value equal to {control!r}")
    n_control = int(control_scored.sum())
    if n_control == 0:
        raise ValueError(f"every control guide has a missing {score!r}, so there is no null to calibrate against")
    if n_control < _FEW_CONTROLS:
        warnings.warn(
            f"only {n_control} non-targeting guides are scored, so the null is coarse. Widen the control set "
            "or loosen alpha.",
            UserWarning,
            stacklevel=3,
        )

    ctrl_z, ctrl_p, ctrl_w = z[control_scored], p[control_scored], w[control_scored]
    rng = np.random.default_rng(seed)

    # One combined statistic per gene, over its scored, non-control guides.
    tested = scored & ~is_control
    frame = pd.DataFrame({"gene": genes[tested], "z": z[tested], "p": p[tested], "w": w[tested]})
    gene_names: list[str] = []
    stat_list: list[float] = []
    k_list: list[int] = []
    for gene_name, g in frame.groupby("gene", sort=False):
        gene_names.append(str(gene_name))
        stat_list.append(_combine(g["z"].to_numpy(), g["p"].to_numpy(), g["w"].to_numpy(), method))
        k_list.append(len(g))
    if not gene_names:
        raise ValueError("no gene has a scored, non-control guide to aggregate")

    # NTC-size-matched null: for each distinct guide count, draw n_null groups of that many NTC guides,
    # without replacement so each group is a genuine same-size sample of the controls.
    if 2 * max(k_list) > n_control:
        warnings.warn(
            f"a gene has {max(k_list)} guides against only {n_control} non-targeting guides, more than half the "
            "pool, so a same-size NTC group is nearly the whole pool and its null has almost no spread, leaving "
            "the p-value unreliable. Use more controls, or drop such genes.",
            UserWarning,
            stacklevel=3,
        )
    null_by_k: dict[int, np.ndarray] = {}
    for k in sorted(set(k_list)):
        if k <= n_control:
            idx = np.argsort(rng.random((n_null, n_control)), axis=1)[:, :k]
        else:
            idx = rng.integers(0, n_control, size=(n_null, k))
        draws = np.array([_combine(ctrl_z[row], ctrl_p[row], ctrl_w[row], method) for row in idx])
        null_by_k[k] = np.sort(draws)

    names = np.array(gene_names)
    stat = np.array(stat_list)
    k_arr = np.array(k_list)
    pvalue = np.empty(names.size)
    for i, (s, k) in enumerate(zip(stat, k_arr)):
        nd = null_by_k[k]
        at_or_above = nd.size - np.searchsorted(nd, s, side="left")
        pvalue[i] = (at_or_above + 1) / (nd.size + 1)

    order = np.argsort(pvalue)
    ranked = pvalue[order] * pvalue.size / (np.arange(pvalue.size) + 1)
    monotone = np.minimum.accumulate(ranked[::-1])[::-1]
    qvalue = np.empty_like(pvalue)
    qvalue[order] = np.clip(np.maximum(monotone, pvalue[order]), 0.0, 1.0)
    is_hit = qvalue <= alpha

    table = (
        pd.DataFrame(
            {
                "gene": names,
                "n_guides": k_arr,
                "statistic": stat,
                "pvalue": pvalue,
                "qvalue": qvalue,
                "is_hit": is_hit,
            }
        )
        .sort_values("pvalue", kind="stable")
        .reset_index(drop=True)
    )
    adata.uns.setdefault("mantispy", {})[key_added] = table

    by_gene = table.set_index("gene")
    adata.obs[f"{key_added}_pvalue"] = by_gene["pvalue"].reindex(genes).to_numpy()
    adata.obs[f"{key_added}_qvalue"] = by_gene["qvalue"].reindex(genes).to_numpy()
    hit = by_gene["is_hit"].reindex(genes).to_numpy()
    adata.obs[key_added] = np.where(np.isnan(adata.obs[f"{key_added}_qvalue"].to_numpy()), False, hit).astype(bool)
    get_logger().info(
        "aggregate_guides(%s, q<=%.3g) called %d of %d genes against %d non-targeting guides",
        method,
        alpha,
        int(table["is_hit"].sum()),
        names.size,
        n_control,
    )
    return None
