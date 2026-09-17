"""Profile-by-profile similarity, and the replicate-reproducibility metrics built on it."""

from __future__ import annotations

import numpy as np
import pandas as pd
from anndata import AnnData

from mantispy._core._reduce import group_codes, representation
from mantispy._core.masks import reference_mask
from mantispy._core.mutation import inplace_or_copy

METRICS = ("cosine", "pearson")

#: Largest float64 similarity matrix, in bytes, that :func:`similarity_matrix` builds.
#: The cast to float32 adds half as much again at peak.
SIMILARITY_BYTES = 4_000_000_000


def similarity_matrix(values: np.ndarray, metric: str = "cosine") -> np.ndarray:
    """Dense pairwise similarity between rows.

    Pearson is cosine on row-centered data, so one code path serves both.

    Args:
        values: Profiles as rows.
        metric: One of ``METRICS``.

    Returns:
        An ``(n_obs, n_obs)`` float32 matrix with 1.0 on the diagonal.
        Missing values are filled with zero before the similarity is taken.

    Raises:
        ValueError: ``metric`` is not one of ``METRICS``, or the float64 matrix would exceed :data:`SIMILARITY_BYTES`. Memory is quadratic in the number of profiles (50,640 JUMP wells need 30 GB), so aggregate to consensus profiles first.
    """
    if metric not in METRICS:
        raise ValueError(f"metric must be one of {METRICS}, got {metric!r}")
    n_obs = np.shape(values)[0]
    if n_obs**2 * 8 > SIMILARITY_BYTES:
        raise ValueError(
            f"a dense similarity over {n_obs} profiles needs {n_obs**2 * 12 / 1e9:.1f} GB, above the "
            "SIMILARITY_BYTES limit. Aggregate first with adata = mt.tl.consensus(adata), or subset "
            "the rows to compare."
        )
    values = np.nan_to_num(np.asarray(values, dtype=np.float64), nan=0.0)
    if metric == "pearson":
        values = values - values.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    unit = values / np.where(norms == 0, 1.0, norms)
    matrix = unit @ unit.T
    np.fill_diagonal(matrix, 1.0)
    return matrix.astype(np.float32)


@inplace_or_copy(expects=("well", "perturbation"))
def similarity(
    adata: AnnData,
    metric: str = "cosine",
    use_rep: str | None = None,
    key_added: str = "similarity",
    copy: bool = False,
) -> AnnData | None:
    """Store pairwise profile similarity in ``obsp[key_added]``.

    Args:
        adata: Profiles to compare.
        metric: Similarity between profiles, one of ``METRICS``.
        use_rep: Compare ``obsm[use_rep]`` instead of ``X``.
        key_added: ``obsp`` key written.
        copy: Return a modified copy instead of mutating in place.

    Returns:
        ``None``, or the modified copy.
        Writes the dense float32 similarity matrix of :func:`similarity_matrix` to ``obsp[key_added]``.

    Notes:
        The result is dense and quadratic in the number of profiles, so this expects well- or perturbation-level profiles rather than single cells.
    """
    values = representation(adata, use_rep)
    adata.obsp[key_added] = similarity_matrix(values, metric)
    return None


def _non_replicate_pool(matrix: np.ndarray, codes: np.ndarray, block: int = 2048) -> np.ndarray:
    """Every above-diagonal similarity whose two profiles fall in different groups, read row by row.

    This is the null the replicate medians are scored against.
    It is taken a row block at a time, so the pair indices of the whole upper triangle never exist at once: at 20 000 profiles those two index arrays cost 3.2 GB between them, four times the pool they select.
    Values stay ``float32`` as :func:`similarity_matrix` returns them and are widened once drawn, which is exact because every entry is a float32 either way.

    Args:
        matrix: Pairwise similarity, as :func:`similarity_matrix` returns it.
        codes: Per-row integer group codes, so a pair is a non-replicate pair exactly when its two codes differ.
        block: Rows per step, which bounds the mask this builds rather than the pool it returns.

    Returns:
        A 1-D ``float32`` array of the non-replicate similarities, ordered as the upper triangle is read row by row.
    """
    columns = np.arange(matrix.shape[0])
    parts = [
        matrix[start : start + block][
            (columns[None, :] > columns[start : start + block, None])
            & (codes[None, :] != codes[start : start + block, None])
        ]
        for start in range(0, matrix.shape[0], block)
    ]
    return np.concatenate(parts) if parts else np.empty(0, dtype=matrix.dtype)


@inplace_or_copy(expects=("well", "perturbation"))
def percent_replicating(
    adata: AnnData,
    groupby: str = "Metadata_Perturbation",
    metric: str = "pearson",
    use_rep: str | None = None,
    null_size: int = 10_000,
    quantile: float = 0.95,
    seed: int = 0,
    key_added: str = "percent_replicating",
    copy: bool = False,
) -> AnnData | None:
    """Median replicate correlation against a non-replicate null.

    An older readout, largely replaced by mAP.
    It thresholds rather than ranks, and it depends on the number of replicates per perturbation.
    It is included because many published results report it.

    Args:
        adata: Well-level profiles with several replicates per group.
        groupby: Column whose groups are the replicate sets.
        metric: Similarity between profiles, one of ``METRICS``.
        use_rep: Score ``obsm[use_rep]`` instead of ``X``.
        null_size: Number of draws in the non-replicate null.
        quantile: Quantile of the null medians a group has to beat to count as replicating.
        seed: Seed for the null draws.
        key_added: Name for the outputs.
        copy: Return a modified copy instead of mutating in place.

    Returns:
        ``None``, or the modified copy.
        Writes ``uns["mantispy"][key_added]`` with ``group``, ``n_replicates``, ``median_replicate_correlation``, ``null_threshold`` and ``is_replicating``, leaving out groups with a single replicate.
        Writes ``uns["mantispy"][key_added + "_summary"]`` with ``fraction_replicating`` and ``n_groups``.
    """
    # The matrix stays float32, as :func:`similarity_matrix` returns it, and only the drawn values are widened.
    # Every entry is a float32 widened to float64 either way, so the medians below are unchanged.
    # That, with the blocked pool, is what this costs: 20 000 wells x 500 features at 4 replicates per group
    # (200.0M pairs) take 14.7 s and peak at 5.0 GB of Python allocation, against 19.6 s and 9.8 GB for a
    # float64 copy of the matrix and a pool selected through materialized upper-triangle indices.
    matrix = similarity_matrix(representation(adata, use_rep), metric)
    codes, keys = group_codes(adata, groupby)
    generator = np.random.default_rng(seed)
    non_replicate = _non_replicate_pool(matrix, codes)

    records = []
    for group, key in enumerate(keys):
        members = np.flatnonzero(codes == group)
        if members.size < 2:
            continue
        pair_rows, pair_columns = np.triu_indices(members.size, k=1)
        observed = float(np.median(matrix[members[pair_rows], members[pair_columns]].astype(np.float64)))
        # Each null draw takes as many non-replicate pairs as the group has replicate pairs.
        draws = generator.choice(non_replicate, size=(null_size, pair_rows.size), replace=True)
        null_threshold = float(np.quantile(np.median(draws.astype(np.float64), axis=1), quantile))
        records.append(
            {
                "group": str(key),
                "n_replicates": int(members.size),
                "median_replicate_correlation": observed,
                "null_threshold": null_threshold,
                "is_replicating": observed > null_threshold,
            }
        )

    table = pd.DataFrame(records)
    store = adata.uns.setdefault("mantispy", {})
    store[key_added] = table
    store[f"{key_added}_summary"] = {
        "fraction_replicating": float(table["is_replicating"].mean()) if len(table) else float("nan"),
        "n_groups": int(len(table)),
    }
    return None


@inplace_or_copy(expects=("well", "perturbation"))
def grit(
    adata: AnnData,
    groupby: str = "Metadata_Perturbation",
    reference: str = "negcon",
    metric: str = "pearson",
    use_rep: str | None = None,
    key_added: str = "grit",
    copy: bool = False,
) -> AnnData | None:
    """Similarity of each replicate to its group, z-scored against its similarity to the controls.

    For each profile, the similarities to its replicates are z-scored against its similarities to the control profiles and averaged.
    A perturbation's grit is the mean over its replicates.

    Args:
        adata: Well-level profiles with several replicates per group.
        groupby: Column whose groups are the replicate sets.
        reference: Which rows are the controls each similarity is z-scored against.
        metric: Similarity between profiles, one of ``METRICS``.
        use_rep: Score ``obsm[use_rep]`` instead of ``X``.
        key_added: Name for the outputs.
        copy: Return a modified copy instead of mutating in place.

    Returns:
        ``None``, or the modified copy.
        Writes the per-replicate score to ``obs[key_added]`` and ``uns["mantispy"][key_added]`` with ``group``, ``key_added`` (the mean over the group's replicates) and ``n_replicates``.
        A replicate is left ``NaN`` when it is alone in its group, when fewer than two control profiles are available to it, or when those similarities have no spread, and ``n_replicates`` counts only the replicates that were scored.

    Raises:
        ValueError: ``reference`` selects no rows.
    """
    is_control = reference_mask(adata, reference)
    if not is_control.any():
        raise ValueError(f"no reference rows selected by reference={reference!r}")

    matrix = similarity_matrix(representation(adata, use_rep), metric).astype(np.float64)
    codes, keys = group_codes(adata, groupby)
    positions = np.arange(adata.n_obs)

    per_replicate = np.full(adata.n_obs, np.nan)
    for group in range(len(keys)):
        members = np.flatnonzero(codes == group)
        if members.size < 2:
            continue
        for member in members:
            others = members[members != member]
            control_similarity = matrix[member, is_control & (positions != member)]
            if control_similarity.size < 2:
                continue
            spread = control_similarity.std(ddof=1)
            if spread == 0:
                continue
            per_replicate[member] = float(np.mean((matrix[member, others] - control_similarity.mean()) / spread))

    adata.obs[key_added] = per_replicate
    table = (
        pd.DataFrame({"group": keys.astype(str)[codes], key_added: per_replicate})
        .groupby("group", observed=True)[key_added]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": key_added, "count": "n_replicates"})
    )
    adata.uns.setdefault("mantispy", {})[key_added] = table
    return None
