"""Guide-level quality control: flag inactive, outlying and thin guides, and summarise how a gene's guides agree."""

from __future__ import annotations

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy import stats

from mantispy._core._numba import MEDIAN, _kernel_input, grouped_stat
from mantispy._core._reduce import get_matrix, group_codes
from mantispy._core.frames import as_frame
from mantispy._core.logging import get_logger
from mantispy._core.masks import reference_mask
from mantispy._core.mutation import inplace_or_copy

#: A guide's one-sided p-value is clipped to this before its z-score is taken, as the gene-level tests do.
_P_CLIP = 1e-12
#: The class label the control guides take when their activity sets the scale.
_REFERENCE = "__reference__"


def _majority_level(k: np.ndarray, alpha: float) -> np.ndarray:
    """The one-sided p-value each guide must fall below for a strict majority of ``k`` to reach a gene's ``alpha``.

    The ``k // 2 + 1``-th smallest of ``k`` uniform p-values is Beta-distributed, so its ``alpha`` quantile is the
    level at which the order rule's one-sided p-value equals ``alpha``.
    """
    r = k // 2 + 1
    return np.asarray(stats.beta.ppf(alpha, r, k - r + 1), dtype=float)


def _summary(guides: pd.DataFrame, by: list[str], alpha: float) -> pd.DataFrame:
    """Per ``by`` group: guides tested and usable, agreeing with the gene's direction, significant, Q, I^2, majority.

    ``guides`` carries each guide's signed z-score ``z``, weight ``n`` and one-sided p-value ``p_one`` in the direction
    it moved; ``alpha`` is the gene's one-sided level. A guide's effect is ``z / sqrt(n)`` with sampling variance
    ``1 / n``, so Q is ``sum(z^2) - sum(sqrt(n) z)^2 / sum(n)``.
    """
    agree = (guides["guide_sign"] == guides["gene_sign"]) & (guides["gene_sign"] != 0)
    k = guides.groupby(by, sort=False)["p_one"].transform("size").to_numpy()
    tallies = guides.assign(
        usable=~guides["qc_guide_fail"],
        agree=agree,
        significant=agree & (guides["p_one"].to_numpy() < _majority_level(k, alpha)),
        zz=guides["z"] ** 2,
        rz=np.sqrt(guides["n"]) * guides["z"],
    ).groupby(by, sort=False)
    table = pd.DataFrame(
        {
            "n_guides": tallies.size(),
            "n_usable": tallies["usable"].sum(),
            "n_agree": tallies["agree"].sum(),
            "n_significant": tallies["significant"].sum(),
            "cochran_q": tallies["zz"].sum() - tallies["rz"].sum() ** 2 / tallies["n"].sum(),
        }
    ).reset_index()
    df = table["n_guides"].to_numpy() - 1
    q = np.maximum(table["cochran_q"].to_numpy(), 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        table["cochran_q"] = np.where(df > 0, q, np.nan)
        table["cochran_pvalue"] = np.where(df > 0, stats.chi2.sf(q, np.maximum(df, 1)), np.nan)
        table["i2"] = np.where(df > 0, np.clip((q - df) / q, 0.0, 1.0), np.nan)
    table["majority"] = table["n_significant"] >= table["n_guides"] // 2 + 1
    return table


def _outlier_pvalues(guides: pd.DataFrame, by: list[str]) -> np.ndarray:
    """The two-sided p-value of the guide that differs most from its siblings, per group: its effect less theirs.

    The leave-one-out residual ``(z_i / sqrt(n_i) - mean_-i) / sqrt(1 / n_i + 1 / sum_-i n)`` is standard normal when
    the gene's guides share one effect. One far guide pulls its siblings' residuals too, so only the largest of a group
    is kept, and the rest are missing; so are a gene's of fewer than three guides, where the residuals of two guides
    mirror each other and cannot say which one is off.
    """
    g = guides.groupby(by, sort=False)
    n = guides["n"].to_numpy(dtype=float)
    rz = np.sqrt(n) * guides["z"].to_numpy(dtype=float)
    n_rest = g["n"].transform("sum").to_numpy(dtype=float) - n
    sum_rz = guides.assign(rz=rz).groupby(by, sort=False)["rz"].transform("sum").to_numpy(dtype=float)
    mean_rest = (sum_rz - rz) / n_rest
    with np.errstate(invalid="ignore", divide="ignore"):
        resid = (guides["z"].to_numpy(dtype=float) / np.sqrt(n) - mean_rest) / np.sqrt(1.0 / n + 1.0 / n_rest)
    k = g["n"].transform("size").to_numpy()
    size = np.nan_to_num(np.abs(resid), nan=-1.0)
    largest = size == guides.assign(size=size).groupby(by, sort=False)["size"].transform("max").to_numpy()
    return np.where((k >= 3) & largest, 2.0 * stats.norm.sf(np.abs(resid)), np.nan)


def _bootstrap_guides(adata: AnnData, params: dict) -> tuple[pd.DataFrame, np.ndarray]:
    """Every guide of a :func:`~mantispy.tl.bootstrap_features` call, from its recorded parameters, and each cell's guide.

    A guide is its gene label, as the result writes it, its ``replicate`` label and, with a ``condition``, its
    condition, as the result's guide table keys it; it carries whether it is a control guide and its activity p-value,
    from its median profile.
    """
    by = [params["groupby"]] if isinstance(params["groupby"], str) else list(params["groupby"])
    condition = params.get("condition")
    unit_codes, unit_keys = group_codes(adata, by)
    sub_by = [*by, params["replicate"], *([condition] if condition is not None and condition not in by else [])]
    guide_codes, guide_keys = group_codes(adata, sub_by)
    n_guides = len(guide_keys)
    first = np.full(n_guides, -1, dtype=np.int64)
    first[guide_codes[::-1]] = np.arange(adata.n_obs)[::-1]
    is_reference = reference_mask(adata, params["reference"])[first]
    labels = np.array([str(unit_keys[u]) for u in unit_codes[first]], dtype=object)
    obs = as_frame(adata.obs)
    replicate = obs[params["replicate"]].astype(str).to_numpy()[first]
    values = _kernel_input(get_matrix(adata, params.get("layer")))
    medians = grouped_stat(values, guide_codes, n_guides, MEDIAN)
    activity = _activity(medians, np.where(is_reference, _REFERENCE, labels))
    every = pd.DataFrame({"group": labels, "replicate": replicate})
    if condition is not None:
        every[condition] = obs[condition].astype(str).to_numpy()[first]
    every = every.assign(is_reference=is_reference, activity_pvalue=activity)
    return every, guide_codes


def _activity(profiles: np.ndarray, classes: np.ndarray) -> np.ndarray:
    """:func:`~mantispy.tl.guide_activity`'s one-sided p-value of each guide's profile against the control guides'."""
    from mantispy.tl import guide_activity

    obs = pd.DataFrame({"class": classes.astype(str)}, index=[str(i) for i in range(len(classes))])
    guides = AnnData(np.asarray(profiles, dtype=np.float64), obs=obs)
    guide_activity(guides, reference=_REFERENCE, group="class", key_added="activity")
    return np.asarray(as_frame(guides.obs)["activity"], dtype=float)


@inplace_or_copy()
def guide_qc(
    adata: AnnData,
    key: str = "bootstrap",
    *,
    alpha: float = 0.05,
    min_cells: int = 10,
    copy: bool = False,
) -> AnnData | None:
    """Flag guides that are inactive, outlying or thin, and summarise how each gene's guides back its call.

    A gene of a pooled screen is called from several guides, and a guide that does not cut, or cuts off target, can carry or sink its gene.
    This reads a gene-level result and its guides, flags each guide, and counts per gene the guides that are usable, that move the same way as the gene, and that are significant on their own.
    It reads :func:`~mantispy.tl.bootstrap_features` from ``uns["mantispy"][key]`` and its guide table ``key + "_guides"``, per gene and feature, and :func:`~mantispy.tl.aggregate_guides` from its gene table and the guide rows of ``obs`` it scored, per gene.

    A guide fails as

    - inactive when it is no different from the control guides while at least half of its siblings are: :func:`~mantispy.tl.guide_activity`'s p-value of its median profile, or the ``score`` :func:`~mantispy.tl.aggregate_guides` read, is at least ``alpha`` and theirs are below it;
    - outlying when it differs from its siblings, in size or direction, by more than its standard error: the leave-one-out residual of its effect against the rest of its gene, its smallest p-value over the features times their number, is below ``alpha``;
    - thin when it has fewer than ``min_cells`` cells.

    Flags select guides to look at; testing genes again without them, with ``exclude="qc_guide_fail"``, makes their p-values optimistic when the guides were flagged on the same cells and features, because a guide that disagreed with its siblings by chance is dropped and the rest agree more than they would.
    Exclude guides on evidence independent of the test: inactivity from a separate readout or half of the cells, or a thin guide.

    A guide is significant as the strict-majority rule, ``method="order"``, the default of :func:`~mantispy.tl.bootstrap_features`, counts it.
    That rule calls a gene of ``k`` guides when ``k // 2 + 1`` of them move its way, each at a one-sided p-value below a level set by ``k`` and ``alpha``, the level at which that many guides give the gene a p-value of ``alpha``.
    ``n_significant`` counts the guides below that level, and ``majority`` says whether they make a strict majority, which is whether the gene's p-value under the rule is below ``alpha``.
    So under the default a called gene always has a majority and is never a single-guide call; under ``"stouffer"`` or ``"fisher"`` one strong guide can carry a gene, and ``single_guide`` marks it.
    A guide that does not cut votes against its gene, so the rest of its guides must carry the majority alone: a gene with an inactive guide and no majority may be missed because of a bad reagent rather than its biology.
    Excluding a guide lowers ``k``, and with it the number of guides the gene needs and the level each must reach: a gene of four guides needs three, but of three, two.

    Args:
        adata: The object either function wrote its results to.
        key: The ``key_added`` of that call.
        alpha: A gene is called at a q-value below ``alpha`` (or, from :func:`~mantispy.tl.aggregate_guides`, when it is a hit), a guide is active or outlying at a p-value below ``alpha``, and significant at the level a strict majority needs for a gene's p-value below ``alpha``.
        min_cells: A guide with fewer cells fails as thin; from :func:`~mantispy.tl.aggregate_guides`, which sees no cells, it is not checked.
        copy: Return a modified copy instead of mutating in place.

    Returns:
        ``None``, or the modified copy.
        Writes ``uns["mantispy"]["guide_qc"]``, one row per tested guide: ``group``, ``replicate``, the ``condition`` column of a call that set one, ``n_cells``, ``activity_pvalue``, ``outlier_pvalue``, ``qc_guide_inactive_fail``, ``qc_guide_outlier_fail``, ``qc_guide_cells_fail`` and ``qc_guide_fail``, their union.
        Writes ``uns["mantispy"]["guide_qc_summary"]``, one row per gene (and feature, from :func:`~mantispy.tl.bootstrap_features`): ``n_guides`` (with a p-value), ``n_usable`` (of those, not failed), ``n_agree`` (moving the gene's way), ``n_significant`` (of those, below the strict majority's level), ``cochran_q``, ``cochran_pvalue`` and ``i2`` (see Notes), ``majority`` (``n_significant`` is at least ``n_guides // 2 + 1``), and ``single_guide``: the gene is called, has two or more guides, and exactly one of them is significant.
        Broadcasts ``obs["qc_guide_fail"]`` onto the rows of each guide, ``False`` for control guides.

    Raises:
        KeyError: ``uns["mantispy"]`` holds no result under ``key``.
        ValueError: ``alpha`` is not in ``(0, 1)``, ``min_cells`` is negative, or the result under ``key`` is not the latest of either function.

    Notes:
        A guide's signed z-score is read from its p-value and direction: from :func:`~mantispy.tl.bootstrap_features`, half its two-sided p-value in the direction of its ``effect``, against a null that already carries the spread between control guides; from :func:`~mantispy.tl.aggregate_guides`, its ``score`` in the direction of its ``direction`` column, or the same for every guide without one.
        Its effect is that z-score over the root of its weight, its cells (from :func:`~mantispy.tl.aggregate_guides`, its ``weight`` squared, or one).
        Cochran's Q sums the squared spread of a gene's guide effects around their weighted mean; its p-value is from the chi-squared distribution on one less than the guides, and ``i2`` is the share of the spread beyond sampling, ``max(0, (Q - df) / Q)``.
        A gene's direction is the sign of its ``effect``, or the direction most of its guides take, and a guide that does not move, or a gene whose effect is zero, agrees with nothing.
        A guide's level for ``k`` guides is the ``alpha`` quantile of the Beta distribution of the ``k // 2 + 1``-th smallest of ``k`` uniform p-values, at half ``alpha`` from :func:`~mantispy.tl.bootstrap_features`, whose p-values are two-sided; one guide's level is its own ``alpha``.
        A one-guide call is a lead, not a verdict: a true hit whose other guides cut weakly reaches it too.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if min_cells < 0:
        raise ValueError(f"min_cells must be at least 0, got {min_cells}")
    store = adata.uns.get("mantispy", {})
    if key not in store:
        raise KeyError(f"uns['mantispy'] holds no result under {key!r}")
    genes = store[key]
    keys = ["group", "replicate"]
    if f"{key}_guides" in store:
        params = store.get("params", {}).get("bootstrap_features", {})
        if params.get("key_added") != key:
            raise ValueError(f"uns['mantispy'][{key!r}] is not the latest result of mt.tl.bootstrap_features")
        if params.get("condition") is not None:
            keys.append(params["condition"])
        every, guide_codes = _bootstrap_guides(adata, params)
        per_guide = every.loc[~every["is_reference"], [*keys, "activity_pvalue"]]
        table = store[f"{key}_guides"].astype(dict.fromkeys(keys, str))
        guides = table.loc[
            np.isfinite(table["pvalue"].to_numpy(dtype=float)), [*keys, "feature", "effect", "pvalue", "n_cells"]
        ]
        guides = guides.merge(genes[["group", "feature", "effect"]], on=["group", "feature"], suffixes=("", "_gene"))
        sign = np.sign(guides["effect"].to_numpy(dtype=float))
        # the two-sided p-value halved is the one-sided one in the direction the guide moved
        p_one = guides["pvalue"].to_numpy(dtype=float) / 2
        guides = guides.assign(
            guide_sign=sign,
            gene_sign=np.sign(guides["effect_gene"].to_numpy(dtype=float)),
            p_one=p_one,
            z=sign * stats.norm.isf(np.clip(p_one, _P_CLIP, 0.5)),
            n=guides["n_cells"].to_numpy(dtype=float),
        )
        by = ["group", "feature"]
        # a gene's p-value is two-sided, twice the smaller of its one-sided ones
        level = alpha / 2
        per_guide = per_guide.merge(table.drop_duplicates(keys)[[*keys, "n_cells"]], on=keys, how="inner")
        called = genes[by].assign(called=(genes["qvalue"] < alpha).to_numpy())
        check_cells = True
    elif "is_hit" in genes:
        params = store.get("params", {}).get("aggregate_guides", {})
        if params.get("key_added") != key:
            raise ValueError(f"uns['mantispy'][{key!r}] is not the latest result of mt.tl.aggregate_guides")
        obs = as_frame(adata.obs)
        rows = np.flatnonzero(obs[params["gene"]].astype(str).isin(genes["group"]).to_numpy())
        sub = obs.iloc[rows]
        sign = np.sign(sub[params["direction"]].to_numpy(dtype=float)) if params["direction"] else np.ones(rows.size)
        score = sub[params["score"]].to_numpy(dtype=float)
        guides = pd.DataFrame(
            {
                "group": sub[params["gene"]].astype(str).to_numpy(),
                "replicate": sub[params["guide"]].astype(str).to_numpy(),
                "pvalue": score,
                "p_one": score,
                "guide_sign": sign,
                "z": sign * stats.norm.isf(np.clip(score, _P_CLIP, 1.0 - _P_CLIP)),
                "n": sub[params["weight"]].to_numpy(dtype=float) ** 2 if params["weight"] else np.ones(rows.size),
            }
        )
        guides = guides[np.isfinite(guides["pvalue"])]
        majority = np.sign(guides.groupby("group")["guide_sign"].sum())
        guides = guides.assign(gene_sign=guides["group"].map(majority).to_numpy())
        by = ["group"]
        level = alpha
        per_guide = pd.DataFrame(
            {"group": guides["group"], "replicate": guides["replicate"], "activity_pvalue": guides["pvalue"]}
        ).assign(n_cells=np.nan)
        called = genes[by].assign(called=genes["is_hit"].to_numpy(dtype=bool))
        check_cells = False
    else:
        raise ValueError(
            f"uns['mantispy'][{key!r}] is from neither mt.tl.bootstrap_features nor mt.tl.aggregate_guides"
        )

    # Flag each guide, then carry its verdict onto its rows of the gene tables.
    active = per_guide["activity_pvalue"].to_numpy() < alpha
    by_gene = per_guide.assign(active=active).groupby("group")["active"]
    sibling_active = by_gene.transform("sum").to_numpy() - active
    n_siblings = by_gene.transform("size").to_numpy() - 1
    per_guide["outlier_pvalue"] = (
        guides.assign(p_out=_outlier_pvalues(guides, by))
        .groupby(keys)["p_out"]
        .agg(lambda p: min(1.0, np.nanmin(p) * np.isfinite(p).sum()) if np.isfinite(p).any() else np.nan)
        .reindex(pd.MultiIndex.from_frame(per_guide[keys]))
        .to_numpy()
    )
    # most siblings active, so that one sibling active by chance does not flag the rest of a gene that does nothing
    per_guide["qc_guide_inactive_fail"] = (
        ~active & (sibling_active > 0) & (2 * sibling_active >= n_siblings) & np.isfinite(per_guide["activity_pvalue"])
    )
    per_guide["qc_guide_outlier_fail"] = per_guide["outlier_pvalue"].to_numpy() < alpha
    per_guide["qc_guide_cells_fail"] = (per_guide["n_cells"].to_numpy() < min_cells) if check_cells else False
    per_guide["qc_guide_fail"] = (
        per_guide["qc_guide_inactive_fail"] | per_guide["qc_guide_outlier_fail"] | per_guide["qc_guide_cells_fail"]
    )
    fail = per_guide.set_index(keys)["qc_guide_fail"]
    guides = guides.assign(qc_guide_fail=fail.reindex(pd.MultiIndex.from_frame(guides[keys])).fillna(False).to_numpy())
    summary = called[by].merge(_summary(guides, by, level), on=by, how="left")
    summary["majority"] = summary["majority"].fillna(False).astype(bool)
    for column in ("n_guides", "n_usable", "n_agree", "n_significant"):
        summary[column] = summary[column].fillna(0).astype(np.int64)
    summary["single_guide"] = (
        called["called"].to_numpy() & (summary["n_guides"] >= 2).to_numpy() & (summary["n_significant"] == 1).to_numpy()
    )

    # Broadcast the verdict onto the object's rows: cells of a bootstrap guide, or aggregate_guides' guide rows.
    if check_cells:
        by_code = fail.reindex(pd.MultiIndex.from_frame(every[keys])).fillna(False).to_numpy(bool)
        row_fail = (by_code & ~every["is_reference"].to_numpy())[guide_codes]
    else:
        obs = as_frame(adata.obs)
        keyed = pd.MultiIndex.from_arrays([obs[params["gene"]].astype(str), obs[params["guide"]].astype(str)])
        row_fail = fail.reindex(keyed).fillna(False).to_numpy(dtype=bool)

    store = adata.uns.setdefault("mantispy", {})
    store["guide_qc"] = per_guide.reset_index(drop=True)
    store["guide_qc_summary"] = summary
    adata.obs["qc_guide_fail"] = row_fail
    get_logger().info(
        "guide_qc failed %d of %d guides: %d inactive, %d outlying, %d thin",
        int(per_guide["qc_guide_fail"].sum()),
        len(per_guide),
        int(per_guide["qc_guide_inactive_fail"].sum()),
        int(per_guide["qc_guide_outlier_fail"].sum()),
        int(np.sum(per_guide["qc_guide_cells_fail"])),
    )
    return None
