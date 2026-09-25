"""Feature-set enrichment over the parsed feature annotation.

``var`` records the object, measurement family and channel of every feature, which defines a set-membership table.
Scoring those sets reports which kinds of measurement changed, such as the mitochondrial texture features, instead of a list of individual columns.

The sets are passed to decoupler, so the same call works for sets built from the feature names and for sets from prior knowledge.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
from anndata import AnnData

from mantispy._core._reduce import get_matrix
from mantispy._core._stats import benjamini_hochberg
from mantispy._core.frames import as_frame
from mantispy._core.logging import get_logger
from mantispy._core.mutation import inplace_or_copy

#: The single decoupler scorers, one call each.
SINGLE_METHODS = ("ulm", "mlm", "ora", "aucell", "gsea", "gsva", "zscore", "waggr", "viper")
#: Every accepted ``method``: the single scorers plus the ``"consensus"`` meta-method.
METHODS = (*SINGLE_METHODS, "consensus")

#: Default panel for ``method="consensus"``. A mix of model families, so no single method's bias
#: decides the call: a linear model, a z-score, a rank-based enrichment and an over-representation test.
CONSENSUS_PANEL = ("ulm", "zscore", "aucell", "ora")

#: Named shorthands for composite groupings.
COMPOSITES = {"group_by_channel": ["feature_group", "channel"]}


def _ora_n_up(adata: AnnData, top_fraction: float) -> int:
    """The ``n_up`` that makes ORA test the top ``top_fraction`` of features against the rest.

    decoupler keeps features ranked above ``n_up``, so the top k of n needs ``n_up = n - k``.
    Its own default would instead select the bottom 95%.
    """
    if not 0.0 < top_fraction < 1.0:
        raise ValueError(f"top_fraction must be in (0, 1), got {top_fraction!r}")
    return int(round(adata.n_vars * (1.0 - top_fraction)))


def feature_sets(adata: AnnData, by: str | Sequence[str] = "feature_group") -> pd.DataFrame:
    """Build a decoupler network from the parsed feature annotation.

    Args:
        adata: Object whose ``var`` carries the parser's columns.
        by: A ``var`` column, several of them (joined with ``|``), or ``"group_by_channel"`` for the ``feature_group``-and-``channel`` combination.

    Returns:
        A frame with ``source``, ``target`` and ``weight``, ready for :func:`enrich` or for decoupler directly.
        Features with a missing annotation in any of the columns are left out.

    Raises:
        KeyError: ``var`` has none of the columns named by ``by``.
    """
    requested = COMPOSITES.get(by, by) if isinstance(by, str) else list(by)
    columns = [requested] if isinstance(requested, str) else list(requested)

    var = as_frame(adata.var)
    missing = [column for column in columns if column not in var]
    if missing:
        raise KeyError(f"var has no column(s) {missing}; available: {sorted(var.columns)}")

    known = var[columns].notna().all(axis=1).to_numpy()
    # str.cat, not .agg(join, axis=1): the latter returns a frame rather than a Series over no rows,
    # so an annotation that names no family needed a branch of its own to avoid failing below.
    named = var.loc[known, columns].astype(str)
    labels = named.iloc[:, 0].str.cat(named.iloc[:, 1:], sep="|")
    return pd.DataFrame(
        {
            "source": labels.to_numpy(),
            "target": var.index[known].to_numpy(),
            "weight": 1.0,
        }
    )


def _score(dc: Any, frame: pd.DataFrame, network: pd.DataFrame, method: str) -> np.ndarray:
    """The (wells x sets) score array for one method, taken off the matrix so no obsm score can leak in."""
    return np.asarray(dc.mt.decouple(frame, network, methods=[method], cons=False)[f"score_{method}"], dtype=float)


def _permutation_padj(frame: pd.DataFrame, network: pd.DataFrame, method: str, n_permutations: int) -> np.ndarray:
    """Two-sided permutation p-values for a single method, BH-adjusted per well.

    Scores the observed net once, then the net with its set membership shuffled ``n_permutations`` times,
    counting per (well, set) how often a shuffled score is at least as extreme as the observed one. Only the
    running count is kept, so memory does not grow with ``n_permutations``. The count feeds the two-sided
    empirical p ``(1 + count) / (n_permutations + 1)``, which is then adjusted across the sets of each well,
    the same family decoupler's parametric padj corrects over.
    """
    import decoupler as dc

    n_sources = network["source"].nunique()
    if n_permutations * n_sources > 200_000:
        warnings.warn(
            f"n_permutations={n_permutations} over {n_sources} set(s) scores the net {n_permutations} times, "
            f"which can be slow; lower n_permutations or the set count if it does not finish.",
            stacklevel=2,
        )

    observed = np.abs(_score(dc, frame, network, method))
    count = np.zeros(observed.shape, dtype=np.int64)
    for seed in range(n_permutations):
        shuffled = dc.pp.shuffle_net(network, seed=seed)
        # `nan >= x` is False, so a set the null could not score adds nothing to the count.
        count += np.abs(_score(dc, frame, shuffled, method)) >= observed
    # A set the observed run could not score has no p-value; leave it NaN rather than the smallest one.
    pvalues = np.where(np.isnan(observed), np.nan, (1.0 + count) / (n_permutations + 1.0))
    return np.vstack([benjamini_hochberg(row) for row in pvalues])


def _warn_if_collinear(dc: Any, network: pd.DataFrame, frame: pd.DataFrame) -> None:
    """Warn when two feature sets are near-collinear, so their enrichment scores cannot be told apart.

    A hygiene check only: if net_corr refuses an unusual net the check is skipped rather than allowed
    to break scoring.
    """
    try:
        corr = dc.pp.net_corr(network, data=frame)
    except Exception:
        return
    strong = corr[np.abs(corr["corr"].to_numpy(dtype=float)) > 0.95]
    pairs = [f"{a} and {b}" for a, b in zip(strong["source_a"], strong["source_b"], strict=False)]
    if pairs:
        warnings.warn(
            f"feature sets are near-collinear ({'; '.join(pairs[:2])}); enrichment cannot separate "
            f"near-duplicate sets, so interpret them together or merge them.",
            UserWarning,
            stacklevel=3,
        )


@inplace_or_copy()
def enrich(
    adata: AnnData,
    net: pd.DataFrame | None = None,
    by: str | Sequence[str] = "feature_group",
    method: str = "ulm",
    methods: Sequence[str] | None = None,
    top_fraction: float = 0.05,
    n_permutations: int = 0,
    check_collinearity: bool = True,
    copy: bool = False,
    **decoupler_kwargs: Any,
) -> AnnData | None:
    """Score every profile against every feature set.

    Args:
        adata: Profiles to score. Normalize first, since the methods use the values as given.
        net: A decoupler network with ``source``, ``target`` and ``weight``, for example prior-knowledge sets. Built from ``by`` when omitted.
        by: Passed to :func:`feature_sets` when ``net`` is not given.
        method: One of ``METHODS``. ``"ulm"`` fits a linear model per set and is the usual choice; ``"mlm"`` fits all sets jointly, which handles overlapping sets; ``"ora"`` is an over-representation test on the extremes; ``"aucell"``, ``"gsea"``, ``"gsva"``, ``"zscore"``, ``"waggr"`` and ``"viper"`` are the remaining decoupler scorers. ``"consensus"`` runs a panel of the single methods and combines their calls, decoupler's robustness feature.
        methods: The panel for ``method="consensus"``, each entry one of the single methods (``METHODS`` without ``"consensus"``). Defaults to ``CONSENSUS_PANEL``. Only used with ``method="consensus"``.
        top_fraction: Fraction of features, ranked by value, that ORA counts as extreme, whether ``method="ora"`` or ``"ora"`` sits in a consensus panel. The default 0.05 tests the top twentieth against the rest. Ignored when ``n_up`` is passed, and by the methods that use every feature.
        n_permutations: 0 (the default) keeps the method's own parametric p-values. A positive count replaces ``padj_<method>`` with two-sided permutation p-values, obtained by shuffling the network that many times and comparing each score against the null, at ``n_permutations`` times the scoring cost. Single methods only; a positive count with ``method="consensus"`` raises.
        check_collinearity: Warn when two feature sets are nearly collinear, since enrichment cannot then separate them. Off skips the check.
        copy: Return a modified copy instead of mutating in place.
        decoupler_kwargs: Passed through to decoupler, e.g. ``tmin`` for the smallest usable set. For ``method="consensus"`` an ``args`` mapping of per-method keyword arguments (as ``decoupler.mt.decouple`` takes) is merged with the computed ORA ``n_up``.

    Returns:
        ``None``, or the modified copy.
        decoupler writes ``obsm["score_<method>"]``, and ``obsm["padj_<method>"]`` for the methods that produce one (all but ``"aucell"`` and ``"gsva"``, which write only the score). ``method="consensus"`` writes ``obsm["score_consensus"]`` and ``obsm["padj_consensus"]`` alongside each panel member's own ``score_<method>`` (and its ``padj_<method>``, except for the score-only ``"aucell"`` and ``"gsva"``). All are frames indexed by set name.

    Raises:
        ValueError: ``method`` is not one of ``METHODS``; ``methods`` is given with a non-consensus ``method``, or names an entry that is not a single method; ``n_permutations`` is positive with ``method="consensus"``; no feature set could be built from ``by``; or ``top_fraction`` is outside (0, 1).
    """
    import decoupler as dc

    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    if methods is not None and method != "consensus":
        raise ValueError(f"methods is only used with method='consensus', got method={method!r}")
    if n_permutations and method == "consensus":
        raise ValueError("n_permutations is not supported with method='consensus'")
    network = feature_sets(adata, by) if net is None else net
    if not len(network):
        raise ValueError(f"no feature sets built from by={by!r}: every feature's annotation is missing")

    # One dense frame for the hygiene check, the consensus scoring and the permutation null.
    frame = pd.DataFrame(get_matrix(adata), index=adata.obs_names, columns=adata.var_names)
    if check_collinearity and network["source"].nunique() > 1:
        _warn_if_collinear(dc, network, frame)

    if method == "consensus":
        panel: tuple[str, ...]
        if methods is None:
            panel = CONSENSUS_PANEL
        elif isinstance(methods, str):
            panel = (methods,)
        else:
            panel = tuple(methods)
        invalid = [name for name in panel if name not in SINGLE_METHODS]
        if invalid:
            raise ValueError(f"methods entries must each be one of the single methods {SINGLE_METHODS}, got {invalid}")
        if not panel:
            raise ValueError("methods must name at least one single method for method='consensus'")
        # decouple takes per-method kwargs in `args`; keep the ORA n_up default while letting the caller override it.
        # Copy the inner dicts too, so setdefault does not write n_up into the caller's own mapping.
        args = {name: dict(values) for name, values in decoupler_kwargs.pop("args", {}).items()}
        if "ora" in panel:
            args.setdefault("ora", {}).setdefault("n_up", _ora_n_up(adata, top_fraction))
        # Handing decouple the matrix (not the AnnData) makes it return the panel's scores and build the
        # consensus from only those. A score_* left on obsm by an earlier enrich therefore cannot leak in,
        # which an AnnData input would allow, since cons=True consolidates every score_* it finds on obsm.
        scores = dc.mt.decouple(frame, network, methods=list(panel), args=args, cons=True, **decoupler_kwargs)
        # decouple returns None for a score-only method's padj (aucell, gsva); writing None into obsm
        # corrupts the AnnData, so keep only the frames it actually produced.
        adata.obsm.update({key: value for key, value in scores.items() if value is not None})
        get_logger().info("enrich(consensus) scored %d set(s) with panel %s", network["source"].nunique(), panel)
        return None

    if n_permutations > 0:
        # Score the observed net the stale-safe way, off the matrix, so the padj lines up with a score that
        # cannot fold an earlier enrich's obsm back in. Then replace the parametric padj with the calibrated one.
        observed = dc.mt.decouple(frame, network, methods=[method], cons=False)[f"score_{method}"]
        adata.obsm[f"score_{method}"] = observed
        padj = _permutation_padj(frame, network, method, n_permutations)
        adata.obsm[f"padj_{method}"] = pd.DataFrame(padj, index=observed.index, columns=observed.columns)
        get_logger().info(
            "enrich(%s) scored %d set(s) with %d-permutation p-values", method, network["source"].nunique(), n_permutations
        )
        return None

    if method == "ora":
        decoupler_kwargs.setdefault("n_up", _ora_n_up(adata, top_fraction))
    getattr(dc.mt, method)(adata, network, **decoupler_kwargs)
    get_logger().info("enrich(%s) scored %d set(s)", method, network["source"].nunique())
    return None


@inplace_or_copy()
def rank_features(
    adata: AnnData,
    groupby: str,
    method: str = "wilcoxon",
    key_added: str = "rank_features",
    copy: bool = False,
) -> AnnData | None:
    """Rank features by how well they separate each group, with the annotation attached.

    Wraps :func:`scanpy.tl.rank_genes_groups` and joins the parsed ``var`` annotation onto the result, so each ranked feature carries its object, feature group and channel.

    Args:
        adata: Object to rank.
        groupby: ``obs`` column defining the groups.
        method: Passed to scanpy: ``"wilcoxon"``, ``"t-test"``, ``"logreg"``.
        key_added: Name for the output table.
        copy: Return a modified copy instead of mutating in place.

    Returns:
        ``None``, or the modified copy.
        Writes ``uns["mantispy"][key_added]`` with ``group``, ``feature``, ``score``, ``pvalue``, ``qvalue`` and the ``object``, ``feature_group`` and ``channel`` the feature belongs to.

    Notes:
        scanpy's ``logfoldchanges`` column is dropped.
        Normalized morphology features are signed, so the ratio is often negative and its log is NaN or meaningless.
        Rank by ``score``.
    """
    import scanpy as sc

    var = as_frame(adata.var)
    # A shallow object: X is shared, so ranking a large matrix does not double memory.
    scratch = ad.AnnData(X=get_matrix(adata), obs=as_frame(adata.obs)[[groupby]].astype("category"), var=var[[]])
    with warnings.catch_warnings():
        # scanpy always computes log fold changes, which warn on the negative ratios of signed features.
        # The column is dropped below.
        warnings.filterwarnings("ignore", "invalid value encountered in log2", RuntimeWarning)
        sc.tl.rank_genes_groups(scratch, groupby=groupby, method=method)

    table = sc.get.rank_genes_groups_df(scratch, group=None).rename(
        columns={"names": "feature", "scores": "score", "pvals": "pvalue", "pvals_adj": "qvalue"}
    )
    table = table.drop(columns=["logfoldchanges"], errors="ignore")
    annotation = [column for column in ("object", "feature_group", "channel") if column in var]
    table = table.merge(var[annotation].astype(str), left_on="feature", right_index=True, how="left")
    adata.uns.setdefault("mantispy", {})[key_added] = table
    return None


@inplace_or_copy()
def rank_sets(
    adata: AnnData,
    groupby: str,
    score_key: str = "score_ulm",
    key_added: str = "rank_sets",
    copy: bool = False,
) -> AnnData | None:
    """Rank feature sets by how far each group's score sits from the rest.

    Args:
        adata: Object already scored by :func:`enrich`.
        groupby: ``obs`` column defining the groups.
        score_key: The ``obsm`` key written by :func:`enrich`.
        key_added: Name for the output table.
        copy: Return a modified copy instead of mutating in place.

    Returns:
        ``None``, or the modified copy.
        Writes ``uns["mantispy"][key_added]`` with ``group``, ``set`` and ``score``, the group's mean enrichment score minus the mean over all other rows.

    Raises:
        KeyError: ``obsm`` has no ``score_key``.
        ValueError: ``groupby`` has a single group, leaving nothing to rank it against.
    """
    if score_key not in adata.obsm:
        raise KeyError(f"obsm has no {score_key!r}; run mt.tl.enrich first, which writes it")

    stored = adata.obsm[score_key]
    names = list(stored.columns) if hasattr(stored, "columns") else [str(i) for i in range(np.shape(stored)[1])]
    scores = np.asarray(stored, dtype=float)
    groups = as_frame(adata.obs)[groupby].astype(str).to_numpy()

    records = []
    for group in pd.unique(groups):
        inside = groups == group
        if inside.all():
            raise ValueError(f"{groupby!r} has a single group, so there is nothing to rank it against")
        difference = np.nanmean(scores[inside], axis=0) - np.nanmean(scores[~inside], axis=0)
        records.append(pd.DataFrame({"group": str(group), "set": names, "score": difference}))

    adata.uns.setdefault("mantispy", {})[key_added] = pd.concat(records, ignore_index=True)
    return None
