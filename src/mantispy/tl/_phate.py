"""PHATE embedding, its affinity graph and its diffusion potential."""

from __future__ import annotations

from typing import Literal

import numpy as np
import scipy.sparse as sp
from anndata import AnnData
from scipy.sparse.csgraph import connected_components

from mantispy._core import _phate
from mantispy._core._distance import pairwise_sqeuclidean
from mantispy._core._reduce import representation
from mantispy._core.logging import get_logger
from mantispy._core.mutation import inplace_or_copy

METRICS = ("euclidean", "cosine")
MDS = ("metric", "classic")

#: Above this many rows the potential is kept only on request, since it has a column per landmark.
POTENTIAL_MAX_OBS = 50_000
#: Diffusion powers the automatic ``t`` searches.
_T_MAX = 100


@inplace_or_copy()
def phate(
    adata: AnnData,
    n_components: int = 2,
    knn: int = 5,
    decay: float | None = 40,
    t: int | None = None,
    gamma: float = 1.0,
    n_landmark: int | None = 2000,
    metric: Literal["euclidean", "cosine"] = "euclidean",
    use_rep: str | None = "X_pca",
    mds: Literal["metric", "classic"] = "metric",
    store_potential: bool | None = None,
    random_state: int = 0,
    key_added: str = "phate",
    copy: bool = False,
) -> AnnData | None:
    """Embed the rows with PHATE, and keep its affinity graph and diffusion potential.

    PHATE (Moon et al., Nat Biotechnol, 2019) builds an alpha-decay affinity between each row and its nearest
    neighbours, turns it into a random walk, and runs the walk for ``t`` steps.
    Each row's ``-log`` transition probabilities form its diffusion potential, and metric MDS on distances between
    potentials gives the embedding.
    The walk smooths noise along the data's manifold, so branches and gradients that a kNN graph splits into
    islands stay connected.

    The affinity graph is written in the layout :func:`scanpy.pp.neighbors` uses, so
    :func:`mantispy.tl.cluster` with ``neighbors_key=key_added``, or :func:`scanpy.tl.leiden` with the same key,
    clusters on it, and :func:`scanpy.pl.embedding` with ``basis=key_added`` draws the embedding.
    The potential is a representation of its own: :func:`mantispy.tl.hit_calling` with
    ``use_rep="X_phate_potential"`` and ``covariance="identity"`` tests each perturbation's distance from the
    controls in it.

    This is a native implementation of the method, not a wrapper of the ``phate`` package.
    Its parameter names and defaults follow ``phate.PHATE``, and the kernel and potential agree with it for the same
    ``t``; the automatic ``t`` uses a published knee rule and can land on a different power than ``phate``.

    Args:
        adata: Profiles to embed, at any resolution.
        n_components: Dimensions of the embedding.
        knn: Neighbours that set each row's kernel bandwidth: the distance to its ``knn``-th nearest other row.
        decay: The alpha of the alpha-decay kernel ``exp(-(d / bandwidth) ** decay)``; a larger value cuts the
            kernel off more sharply past the bandwidth.
            ``None`` uses a plain kNN kernel, one on the ``knn`` neighbours and zero elsewhere.
        t: Steps of the random walk.
            ``None`` (the default) chooses it automatically, at the knee of the von Neumann entropy of the walk's
            spectrum over its powers, where further steps stop removing noise and start removing structure.
        gamma: The informational distance: ``1`` is the log potential of the paper, ``0`` a square-root potential,
            values in between interpolate.
        n_landmark: With more rows than this, the walk runs between this many landmarks, groups of rows found by
            k-means on diffusion coordinates, and each row is placed from its transitions into them.
            ``None`` always runs the exact walk between all rows.
        metric: Distance the neighbours are found by, ``"euclidean"`` or ``"cosine"``.
        use_rep: Embed ``obsm[use_rep]``, normally principal components, or ``X`` with ``None``.
        mds: ``"metric"`` minimises the stress of the embedding by SMACOF, started from ``"classic"``, the
            classical (Torgerson) MDS of the potential distances.
        store_potential: Write the potential to ``obsm``.
            ``None`` (the default) writes it for objects of up to ``POTENTIAL_MAX_OBS`` rows only, since it has a
            column per landmark, or per row on the exact path.
        random_state: Seed for the landmark clustering.
            The exact path has no randomness.
        key_added: Name for the outputs.
        copy: Return a modified copy instead of mutating in place.

    Returns:
        ``None``, or the modified copy.
        Writes the embedding to ``obsm["X_" + key_added]``.
        Writes the symmetric affinity to ``obsp[key_added + "_connectivities"]`` and the neighbour distances it was
        built from to ``obsp[key_added + "_distances"]``, with ``uns[key_added]`` naming both and holding the
        parameters and the ``t`` used, the layout :func:`scanpy.pp.neighbors` writes for its ``key_added``.
        With ``store_potential``, writes the diffusion potential to ``obsm["X_" + key_added + "_potential"]``.

    Raises:
        ValueError: An argument is out of range, or the object has no more rows than ``knn``.
    """
    if metric not in METRICS:
        raise ValueError(f"metric must be one of {METRICS}, got {metric!r}")
    if mds not in MDS:
        raise ValueError(f"mds must be one of {MDS}, got {mds!r}")
    if knn < 1:
        raise ValueError(f"knn must be at least 1, got {knn}")
    if decay is not None and decay <= 0:
        raise ValueError(f"decay must be positive or None, got {decay}")
    if t is not None and t < 1:
        raise ValueError(f"t must be a positive number of steps or None, got {t}")
    if not -1 <= gamma <= 1:
        raise ValueError(f"gamma must be in [-1, 1], got {gamma}")
    if n_components < 1:
        raise ValueError(f"n_components must be at least 1, got {n_components}")
    if n_landmark is not None and n_landmark < 2:
        raise ValueError(f"n_landmark must be at least 2 or None, got {n_landmark}")
    n_obs = adata.n_obs
    if n_obs <= knn:
        raise ValueError(f"PHATE needs more rows than knn={knn}, got {n_obs}")

    x = representation(adata, use_rep)
    if not np.isfinite(x).all():
        raise ValueError("the representation has missing or infinite values; impute or drop them first")
    if metric == "cosine":
        norms = np.linalg.norm(x, axis=1)
        if (norms == 0).any():
            raise ValueError("a row of zeros has no cosine distance; drop it or use metric='euclidean'")
        x = x / norms[:, None]

    idx, dist = _phate.knn_with_tail(np.ascontiguousarray(x), knn, decay)
    if metric == "cosine":
        # Unit rows: the cosine distance is half the squared Euclidean one, in the same order.
        dist = np.where(idx >= 0, dist * dist / 2.0, dist)
    kernel, distances = _phate.affinity(idx, dist, knn, decay)
    del idx, dist
    degree = np.asarray(kernel.sum(axis=1)).ravel()
    walk = (sp.diags(1.0 / degree) @ kernel).tocsr()

    if n_landmark is None or n_obs <= n_landmark:
        transitions = None
        operator = walk
        spectrum_of = (kernel, degree)
    else:
        groups = _phate.landmarks(kernel, degree, n_landmark, random_state)
        membership = sp.csr_matrix((np.ones(n_obs), (np.arange(n_obs), groups)), shape=(n_obs, int(groups.max()) + 1))
        transitions = (walk @ membership).tocsr()
        to_rows = (membership.T @ kernel).tocsr()
        landmark_degree = np.asarray(to_rows.sum(axis=1)).ravel()
        operator = (sp.diags(1.0 / landmark_degree) @ to_rows @ transitions).tocsr()
        # The landmark walk is landmark_degree^-1 times this symmetric matrix, which gives its spectrum.
        spectrum_of = ((to_rows @ sp.diags(1.0 / degree) @ to_rows.T).tocsr(), landmark_degree)

    if t is None:
        entropy = _phate.von_neumann_entropy(_phate.symmetric_spectrum(*spectrum_of), _T_MAX)
        # entropy[i] is that of power i + 1, so the powers on the steep line are 1 .. knee.
        t = _phate.knee(entropy)
    t = int(t)

    landmark_potential = _phate.potential(_phate.diffusion_power(operator, t), gamma)
    potential_distances = np.sqrt(pairwise_sqeuclidean(landmark_potential, landmark_potential))
    np.fill_diagonal(potential_distances, 0.0)
    coordinates = _phate.classic_mds(potential_distances, n_components)
    if mds == "metric":
        coordinates = _phate.smacof(potential_distances, coordinates)
    del potential_distances

    if transitions is None:
        embedding, row_potential = coordinates, landmark_potential
    else:
        embedding = _phate.interpolate(transitions, coordinates)
        row_potential = None

    n_components_found = int(connected_components(kernel, directed=False)[0])
    if n_components_found > 1:
        get_logger().warning(
            "phate: the affinity graph has %d disconnected components",
            n_components_found,
            deep="their relative placement in the embedding is arbitrary; raise knn to connect them",
        )

    if store_potential is None:
        store_potential = n_obs <= POTENTIAL_MAX_OBS
    adata.obsm[f"X_{key_added}"] = np.asarray(embedding, dtype=np.float64)
    if store_potential:
        if row_potential is None:
            row_potential = _phate.interpolate(transitions, landmark_potential, np.float32)
        adata.obsm[f"X_{key_added}_potential"] = np.asarray(row_potential, dtype=np.float32)
    adata.obsp[f"{key_added}_connectivities"] = kernel
    adata.obsp[f"{key_added}_distances"] = distances
    params: dict[str, object] = {
        "method": "phate",
        "knn": int(knn),
        "t": t,
        "gamma": float(gamma),
        "metric": metric,
        "mds": mds,
        "random_state": int(random_state),
    }
    if decay is not None:
        params["decay"] = float(decay)
    if transitions is not None:
        params["n_landmark"] = int(transitions.shape[1])
    if use_rep is not None:
        params["use_rep"] = use_rep
    adata.uns[key_added] = {
        "connectivities_key": f"{key_added}_connectivities",
        "distances_key": f"{key_added}_distances",
        "params": params,
    }
    get_logger().info("phate: t=%d, %s", t, "exact" if transitions is None else f"{transitions.shape[1]} landmarks")
    return None
