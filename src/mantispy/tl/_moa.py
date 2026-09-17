"""Mechanism-of-action retrieval and neighborhood enrichment."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy.stats import hypergeom

from mantispy._core._reduce import representation
from mantispy._core._stats import benjamini_hochberg
from mantispy._core.frames import as_frame
from mantispy._core.logging import get_logger
from mantispy._core.mutation import inplace_or_copy
from mantispy.tl._similarity import similarity_matrix

SCHEMES = ("nn", "nsc", "nscb")


def _blocked_similarity(
    adata: AnnData, metric: str, use_rep: str | None, scheme: str, compound_key: str, batch_key: str
) -> np.ndarray:
    """Similarity with the diagonal and the neighbors the scheme forbids set to -inf."""
    similarity = similarity_matrix(representation(adata, use_rep), metric).astype(np.float64)
    np.fill_diagonal(similarity, -np.inf)
    obs = as_frame(adata.obs)

    blocked = []
    if scheme in ("nsc", "nscb"):
        blocked.append((compound_key, scheme))
    if scheme == "nscb":
        blocked.append((batch_key, "nscb"))
    for column, needed_by in blocked:
        if column not in obs:
            raise KeyError(f"obs has no column {column!r}, which scheme={needed_by!r} needs")
        labels = obs[column].astype(str).to_numpy()
        similarity[labels[:, None] == labels[None, :]] = -np.inf
    return similarity


@inplace_or_copy(expects=("well", "perturbation"))
def nn_moa_classify(
    adata: AnnData,
    moa_key: str = "Metadata_MOA",
    metric: str = "cosine",
    scheme: str = "nsc",
    compound_key: str = "Metadata_Compound",
    batch_key: str = "Metadata_Batch",
    use_rep: str | None = None,
    key_added: str = "moa",
    copy: bool = False,
) -> AnnData | None:
    """Leave-one-out nearest-neighbor mechanism assignment.

    Args:
        adata: Profiles to classify, one row per treatment or per well.
        moa_key: ``obs`` column holding the known mechanism.
        metric: Similarity between profiles, ``"cosine"`` or ``"pearson"``.
        scheme: ``"nn"`` allows any neighbor, which is usually optimistic because a compound can match itself at another dose. ``"nsc"`` (not-same-compound) excludes neighbors of the same compound, as in the published BBBC021 benchmark. ``"nscb"`` also excludes neighbors from the same batch, so a batch effect cannot produce the match.
        compound_key: ``obs`` column read for the ``nsc`` and ``nscb`` exclusions.
        batch_key: ``obs`` column read for the ``nscb`` exclusion.
        use_rep: Classify ``obsm[use_rep]`` instead of ``X``.
        key_added: Name for the outputs.
        copy: Return a modified copy instead of mutating in place.

    Returns:
        ``None``, or the modified copy.
        Writes ``obs[key_added + "_predicted"]``, a summary at ``uns["mantispy"][key_added]`` (``accuracy``, ``n_classified``, ``n_excluded``, ``scheme``) and a tidy ``true``/``predicted``/``count`` table at ``uns["mantispy"][key_added + "_confusion"]``.

    Raises:
        ValueError: ``scheme`` is not one of ``SCHEMES``.
        KeyError: ``obs`` has no ``moa_key``, or no column that the chosen scheme excludes neighbors on.

    Notes:
        A row with no admissible neighbor, for example under ``"nscb"`` on a single batch, is left unclassified and counted in ``n_excluded``.
        Chance level is ``1/n_classes`` only when the classes are balanced; otherwise compare against the largest class's share.

        A profile with no mechanism on file is neither scored nor used as a neighbor, so the accuracy does not depend on the annotated fraction.
    """
    if scheme not in SCHEMES:
        raise ValueError(f"scheme must be one of {SCHEMES}, got {scheme!r}")
    if moa_key not in adata.obs:
        raise KeyError(f"obs has no column {moa_key!r} holding the known mechanism")

    similarity = _blocked_similarity(adata, metric, use_rep, scheme, compound_key, batch_key)
    truth = as_frame(adata.obs)[moa_key].to_numpy(dtype=object)
    annotated = pd.notna(truth)
    # Unannotated profiles cannot be neighbors, so the nearest annotated profile is used even when an unannotated one is closer.
    similarity[:, ~annotated] = -np.inf

    usable = np.isfinite(similarity).any(axis=1) & annotated
    predicted = np.full(adata.n_obs, "", dtype=object)
    predicted[usable] = truth[similarity[usable].argmax(axis=1)]

    if not usable.any():
        warnings.warn(
            f"scheme={scheme!r} excluded every neighbour of every profile, so nothing was classified. "
            "This happens, for example, with scheme='nscb' on a single batch.",
            UserWarning,
            stacklevel=3,
        )

    # pd.Series.eq rather than ==, which raises on an object array holding pd.NA; eq treats a missing value as unequal.
    correct = pd.Series(predicted).eq(pd.Series(truth)).to_numpy() & usable
    labelled = np.where(usable, predicted, np.array(None, dtype=object))
    adata.obs[f"{key_added}_predicted"] = pd.Categorical(labelled)
    store = adata.uns.setdefault("mantispy", {})
    store[key_added] = {
        "accuracy": float(correct.sum() / max(int(usable.sum()), 1)),
        "n_classified": int(usable.sum()),
        "n_excluded": int((~usable).sum()),
        "scheme": scheme,
    }
    store[f"{key_added}_confusion"] = (
        pd.DataFrame({"true": truth[usable], "predicted": predicted[usable]}).value_counts().reset_index(name="count")
    )
    get_logger().info(
        "nn_moa_classify(%s) got %.1f%% of %d profiles right",
        scheme,
        100 * store[key_added]["accuracy"],
        store[key_added]["n_classified"],
    )
    return None


@inplace_or_copy(expects=("well", "perturbation"))
def moa_enrichment(
    adata: AnnData,
    moa_key: str = "Metadata_MOA",
    groupby: str = "Metadata_Perturbation",
    k: int = 10,
    metric: str = "cosine",
    use_rep: str | None = None,
    key_added: str = "moa_enrichment",
    copy: bool = False,
) -> AnnData | None:
    """Test which mechanisms are over-represented among each profile's nearest neighbors.

    For each profile and mechanism, a hypergeometric test asks whether the mechanism is more common among the ``k`` nearest neighbors than among all other profiles.

    Args:
        adata: Profiles to test, one row per treatment or per well.
        moa_key: ``obs`` column holding the mechanism labels.
        groupby: ``obs`` column naming each profile in the output table.
        k: Number of neighbors considered, capped at ``n_obs - 1``. Smaller values are more local and less powerful.
        metric: As in :func:`nn_moa_classify`.
        use_rep: As in :func:`nn_moa_classify`.
        key_added: Name for the output table.
        copy: Return a modified copy instead of mutating in place.

    Returns:
        ``None``, or the modified copy.
        Writes ``uns["mantispy"][key_added]`` with ``group``, ``moa``, ``n_neighbours``, ``pvalue`` and ``qvalue``, one row per annotated profile and per mechanism found among its neighbors.

    Raises:
        ValueError: ``k`` is less than 1, or ``obs[moa_key]`` has no annotated rows.

    Notes:
        Unannotated profiles are not tested, but they can be neighbors.
        They take up places among the ``k`` neighbors without adding to any mechanism's count, and they are part of the population the test draws from.
        The profile itself is excluded from both its neighborhood and the population.
    """
    if k < 1:
        raise ValueError(f"k must be at least 1, got {k}")
    similarity = similarity_matrix(representation(adata, use_rep), metric).astype(np.float64)
    np.fill_diagonal(similarity, -np.inf)

    obs = as_frame(adata.obs)
    # No .astype(str), which turns missing labels into a "nan" mechanism on pandas < 3.
    # Missing labels get code -1 and are masked out with `annotated`.
    labels = pd.Categorical(obs[moa_key])
    annotated = labels.codes >= 0
    if not annotated.any():
        raise ValueError(f"obs[{moa_key!r}] has no annotated rows, so there is no mechanism to test for")
    groups = obs[groupby].astype(str).to_numpy()

    codes = labels.codes
    n_labels = len(labels.categories)
    scored = np.flatnonzero(annotated)

    k = min(k, adata.n_obs - 1)
    top = np.argpartition(-similarity, kth=k - 1, axis=1)[:, :k]

    # Unannotated neighbors take up one of the k places but add to no mechanism's count.
    votes = codes[top[scored]]
    valid = (votes >= 0).ravel()
    found = np.zeros((scored.size, n_labels), dtype=np.int64)
    np.add.at(found, (np.repeat(np.arange(scored.size), k)[valid], votes.ravel()[valid]), 1)

    # The population matches the pool `top` draws from: every other profile, annotated or not.
    population = adata.n_obs - 1
    totals = (
        np.bincount(codes[annotated], minlength=n_labels)[None, :] - np.eye(n_labels, dtype=np.int64)[codes[scored]]
    )
    rows, columns = np.nonzero(found)
    counts = found[rows, columns]
    pvalues = hypergeom.sf(counts - 1, population, totals[rows, columns], k)

    table = pd.DataFrame(
        {
            "group": groups[scored][rows],
            "moa": np.asarray(labels.categories, dtype=object)[columns],
            "n_neighbours": counts,
            "pvalue": pvalues,
        }
    )
    table["qvalue"] = benjamini_hochberg(table["pvalue"].to_numpy())
    adata.uns.setdefault("mantispy", {})[key_added] = table
    return None
