"""Hierarchical clustering of per-perturbation profiles, with the tree kept for a dendrogram.

One consensus profile per perturbation is the usual input (:func:`~mantispy.tl.consensus`), so a cluster is a
group of perturbations with a shared phenotype. The linkage tree is stored so :func:`~mantispy.pl.dendrogram`
can draw it and :func:`~mantispy.tl.ora` can test each cluster's genes against prior knowledge.

The granularity is chosen automatically by default, by best silhouette or by most stable membership; see
:func:`cluster` for how, and to set it explicitly.
"""

from __future__ import annotations

import numbers
from typing import Literal

import numpy as np
import pandas as pd
from anndata import AnnData

from mantispy._core._reduce import representation
from mantispy._core.logging import get_logger
from mantispy._core.mutation import inplace_or_copy

METHODS = ("hierarchical", "leiden")


def _auto_cut(linkage_matrix: np.ndarray, distances: np.ndarray, max_clusters: int) -> tuple[np.ndarray, float, float]:
    """The fcluster labels whose silhouette is best over the candidate cluster counts.

    Sweeps ``k`` from 2 to ``max_clusters``, scoring each cut's labels against the precomputed distances, and
    returns the winning labels, the height the tree is cut at for that count, and the silhouette that won. A
    single observation, or distances with no spread, has no cut to make.
    """
    from scipy.cluster.hierarchy import fcluster
    from scipy.spatial.distance import squareform
    from sklearn.metrics import silhouette_score

    square = squareform(distances)
    heights = linkage_matrix[:, 2]
    n_obs = square.shape[0]
    best_labels, best_k, best_score = np.ones(n_obs, dtype=np.int64), 1, float("nan")
    for k in range(2, max_clusters + 1):
        labels = fcluster(linkage_matrix, k, criterion="maxclust")
        # A cut can return fewer groups than asked for when merges tie; score only the counts it actually made.
        n_labels = len(set(labels))
        if n_labels < 2:
            continue
        score = float(silhouette_score(square, labels, metric="precomputed"))
        if np.isnan(best_score) or score > best_score:
            best_labels, best_k, best_score = labels, n_labels, score

    # The height that separates best_k clusters sits between the last merge kept and the first merge cut.
    cut = float("nan")
    if best_k >= 2:
        below, above = n_obs - best_k - 1, n_obs - best_k
        cut = float((heights[below] + heights[above]) / 2)
    return best_labels, cut, best_score


def _stability_cut(
    linkage_matrix: np.ndarray,
    max_clusters: int,
    window: tuple[float, float] | None = None,
    n_steps: int = 200,
    ma_frac: float = 0.05,
) -> tuple[np.ndarray, float, float]:
    """The cut height whose cluster membership is most stable across nearby heights.

    Sweeps a grid of cut heights over the tree; each height scores as the fraction of observations that sit in a
    cluster whose exact membership also occurs at the neighboring grid heights, in [0, 1], smoothed with a short
    moving average. Only cuts with 2 to ``max_clusters`` clusters are scored, so the all-in-one-cluster top and the
    near-all-singletons bottom cannot win; this mirrors the bounded correlation window Rohban 2017 swept. Returns
    the labels at the winning height, that height, and its stability score. A tree with no height spread, or a grid
    with no in-range cut, falls back to a single cluster with nan cut and score. Within a window the finest
    perfectly-stable cut wins, since the exact-membership stability saturates to 1.0 across a stable plateau.

    ``window`` is an optional ``(low, high)`` height band the sweep is clipped to, so the plateau of 2 to 3 huge
    clusters near the top of the tree cannot trivially win; ``None`` sweeps the full height range. A window that does
    not overlap the tree's height range raises a ValueError.
    """
    from scipy.cluster.hierarchy import fcluster

    n_obs = int(linkage_matrix.shape[0]) + 1
    heights = linkage_matrix[:, 2]
    lo, hi = float(heights.min()), float(heights.max())
    if window is not None:
        w_lo, w_hi = max(lo, float(window[0])), min(hi, float(window[1]))
        if not w_hi > w_lo:
            raise ValueError(
                f"stability_window {window} does not overlap the tree height range "
                f"[{lo:.3g}, {hi:.3g}]"
            )
        lo, hi = w_lo, w_hi
    if not hi > lo:
        return np.ones(n_obs, dtype=np.int64), float("nan"), float("nan")
    grid = np.linspace(lo, hi, n_steps)
    parts = [fcluster(linkage_matrix, h, criterion="distance") for h in grid]
    sets = [{frozenset(np.flatnonzero(p == c).tolist()) for c in np.unique(p)} for p in parts]
    raw = np.zeros(n_steps)
    for i in range(1, n_steps - 1):
        cur = sets[i]
        if not (2 <= len(cur) <= max_clusters):
            continue
        agree = sum(len(c) for c in cur if c in sets[i - 1]) + sum(len(c) for c in cur if c in sets[i + 1])
        raw[i] = agree / 2 / n_obs  # fraction of observations in a cluster that recurs at neighboring heights
    w = max(1, int(n_steps * ma_frac)) | 1
    # Each raw score is a fraction in [0, 1]; clip keeps the smoothed score there against convolution rounding.
    smooth = np.clip(np.convolve(raw, np.ones(w) / w, mode="same"), 0.0, 1.0)
    if not smooth.any():
        return np.ones(n_obs, dtype=np.int64), float("nan"), float("nan")
    best = int(np.argmax(smooth[1:-1])) + 1
    return parts[best], float(grid[best]), float(smooth[best])


@inplace_or_copy(expects="perturbation")
def cluster(
    adata: AnnData,
    use_rep: str | None = "X_pca",
    method: str = "hierarchical",
    linkage: str = "average",
    metric: str = "correlation",
    distance_cut: float | None = None,
    n_clusters: int | None = None,
    criterion: Literal["silhouette", "stability"] = "silhouette",
    resolution: float = 1.0,
    key_added: str = "cluster",
    copy: bool = False,
    *,
    stability_window: tuple[float, float] | None = None,
) -> AnnData | None:
    """Cluster the profiles and store the labels, with the linkage tree for a dendrogram.

    Args:
        adata: Profiles to cluster, normally one consensus profile per perturbation from :func:`~mantispy.tl.consensus`.
        use_rep: Cluster ``obsm[use_rep]`` (an embedding such as ``sc.pp.pca`` writes) instead of ``X``, or ``None`` for ``X``.
        method: ``"hierarchical"`` (the default) builds a linkage tree with scipy; ``"leiden"`` delegates to :func:`scanpy.tl.leiden` on the neighbors graph and stores no tree.
        linkage: The scipy linkage method for ``method="hierarchical"``, for example ``"average"``, ``"complete"`` or ``"ward"``.
        metric: The scipy pairwise distance for ``method="hierarchical"``. ``"correlation"`` is ``1 - Pearson`` between profiles.
        distance_cut: Cut the tree at this height. Mutually exclusive with ``n_clusters``; ``method="hierarchical"`` only.
        n_clusters: Cut the tree into this many clusters. Mutually exclusive with ``distance_cut``; ``method="hierarchical"`` only.
        criterion: Which score picks the automatic cut for ``method="hierarchical"``: ``"silhouette"`` (the default) keeps the cut with the best silhouette, ``"stability"`` keeps the cut whose cluster membership is most stable across nearby heights. Ignored when ``distance_cut`` or ``n_clusters`` is given, or when ``method != "hierarchical"``.
        resolution: Passed to :func:`scanpy.tl.leiden` for ``method="leiden"``.
        key_added: ``obs`` column the labels are written to.
        copy: Return a modified copy instead of mutating in place.
        stability_window: Optional ``(low, high)`` height band the ``criterion="stability"`` sweep is restricted to, in the same height units as ``distance_cut`` (for ``metric="correlation"``, height is ``1 - correlation``, so the correlation window 0.4 to 0.7 is ``(0.3, 0.6)``). ``None`` (the default) sweeps the full height range. Whenever provided it is validated for shape; it is only applied to the ``criterion="stability"`` automatic cut.

    Returns:
        ``None``, or the modified copy.
        Writes categorical cluster labels to ``obs[key_added]``. For ``method="hierarchical"`` it also writes the
        linkage matrix to ``uns["mantispy"][key_added + "_linkage"]`` and, to ``uns["mantispy"][key_added]``, a summary
        with ``n_clusters``, ``distance_cut``, ``metric``, ``linkage``, ``silhouette``, ``stability`` and the
        ``labels`` the tree's leaves carry, in the object's row order, so :func:`~mantispy.pl.dendrogram` can label
        them. ``silhouette`` is set only when the automatic ``criterion="silhouette"`` cut ran. ``stability`` is set
        when the ``criterion="stability"`` auto-cut selects an in-range cut, and stays ``nan`` on the degenerate
        fallback (a flat tree or no in-range cut). The score that did not run stays ``nan``.

    Raises:
        ValueError: ``method`` is not one of ``METHODS``, both ``distance_cut`` and ``n_clusters`` are given, ``criterion`` is not ``"silhouette"`` or ``"stability"``, ``stability_window`` is not a length-2 ``(low, high)`` pair of numbers with ``low < high`` or does not overlap the tree's height range, or the object has fewer than two rows to cluster.

    Notes:
        With neither ``distance_cut`` nor ``n_clusters`` the granularity is chosen automatically. With
        ``criterion="silhouette"`` the tree is cut into 2 to ``min(n_obs - 1, 25)`` clusters and the cut with the
        best silhouette is kept. With ``criterion="stability"`` a grid of cut heights is swept and the height whose
        cluster membership recurs most at neighboring heights is kept, the way Rohban 2017 cut their dendrogram.
        Pass ``stability_window`` to restrict that sweep to a height band, so the wide plateau of a few huge clusters
        near the top of the tree cannot trivially win and collapse the cut; it only affects ``criterion="stability"``.
        Within a window the criterion favors the finest perfectly-stable cut, since the exact-membership stability
        saturates to 1.0 across a stable plateau.
        Rank clusters by the biology they recover rather than trusting the count, since neither score sees biology.
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    if distance_cut is not None and n_clusters is not None:
        raise ValueError("pass at most one of distance_cut and n_clusters, not both")
    if criterion not in ("silhouette", "stability"):
        raise ValueError(f"criterion must be 'silhouette' or 'stability', got {criterion!r}")
    if stability_window is not None:
        if not isinstance(stability_window, tuple) or len(stability_window) != 2:
            raise ValueError(f"stability_window must be a (low, high) pair of numbers, got {stability_window!r}")
        if not all(isinstance(v, numbers.Real) for v in stability_window):
            raise ValueError(f"stability_window must be a (low, high) pair of numbers, got {stability_window!r}")
        if not stability_window[0] < stability_window[1]:
            raise ValueError(f"stability_window must have low < high, got {stability_window!r}")
    if adata.n_obs < 2:
        raise ValueError(f"clustering needs at least two rows, got {adata.n_obs}")

    if method == "leiden":
        import scanpy as sc

        if use_rep is not None and use_rep not in adata.obsm:
            raise KeyError(f"obsm has no representation {use_rep!r}; run sc.pp.pca first, or pass use_rep=None")
        sc.pp.neighbors(adata, use_rep=use_rep)
        # The igraph backend is scanpy's future default and the one the repo's tests use.
        sc.tl.leiden(adata, resolution=resolution, key_added=key_added, flavor="igraph", n_iterations=2, directed=False)
        get_logger().info("cluster(leiden): %d cluster(s)", adata.obs[key_added].nunique())
        return None

    from scipy.cluster.hierarchy import fcluster
    from scipy.cluster.hierarchy import linkage as scipy_linkage
    from scipy.spatial.distance import pdist

    values = representation(adata, use_rep)
    # metric and linkage are validated by scipy at call time; the stubs type them as Literals, so a runtime str
    # cannot be narrowed to them here.
    distances = pdist(values, metric=metric)  # type: ignore[call-overload]
    # A profile that is constant, or shares a gap pattern, makes a NaN correlation distance; the tree cannot
    # be built over it, so fail with a message rather than let scipy raise deep in the linkage.
    if not np.all(np.isfinite(distances)):
        raise ValueError(
            f"the {metric!r} distance is not finite for every pair (a constant or degenerate profile); "
            "drop such profiles, or choose another metric."
        )
    linkage_matrix = scipy_linkage(distances, method=linkage)  # type: ignore[arg-type]

    silhouette = float("nan")
    stability = float("nan")
    if n_clusters is not None:
        labels = fcluster(linkage_matrix, n_clusters, criterion="maxclust")
        cut = float("nan")
    elif distance_cut is not None:
        labels = fcluster(linkage_matrix, distance_cut, criterion="distance")
        cut = float(distance_cut)
    elif criterion == "silhouette":
        labels, cut, silhouette = _auto_cut(linkage_matrix, distances, min(adata.n_obs - 1, 25))
    else:
        labels, cut, stability = _stability_cut(linkage_matrix, min(adata.n_obs - 1, 25), window=stability_window)
    chosen = int(len(set(labels)))

    adata.obs[key_added] = pd.Categorical([str(label) for label in labels])
    store = adata.uns.setdefault("mantispy", {})
    store[f"{key_added}_linkage"] = np.asarray(linkage_matrix, dtype=float)
    summary = {
        "n_clusters": chosen,
        "distance_cut": cut,
        "metric": metric,
        "linkage": linkage,
        "silhouette": silhouette,
        "stability": stability,
        "labels": [str(name) for name in adata.obs_names],
    }
    store[key_added] = summary
    get_logger().info("cluster(hierarchical): %d cluster(s) over %d profile(s)", chosen, adata.n_obs)
    return None
