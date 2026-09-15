"""Profile-by-profile similarity, and the replicate-reproducibility metrics built on it."""

from __future__ import annotations

import numpy as np
import pandas as pd
from anndata import AnnData

from mantispy._core._reduce import group_codes, representation
from mantispy._core._utils import inplace_or_copy, reference_mask

METRICS = ("cosine", "pearson")

#: Largest float64 similarity matrix, in bytes, that :func:`similarity_matrix` builds. The
#: cast to float32 adds half as much again at peak.
SIMILARITY_BYTES = 4_000_000_000


def similarity_matrix(values: np.ndarray, metric: str = "cosine") -> np.ndarray:
    """Dense pairwise similarity between rows.

    Pearson is cosine on row-centered data, so one code path serves both.

    Raises:
        ValueError: ``metric`` is not one of ``METRICS``, or the float64 matrix would exceed
            :data:`SIMILARITY_BYTES`. Memory is quadratic in the number of profiles (50,640 JUMP
            wells need 30 GB), so aggregate to consensus profiles first.
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

    Notes:
        The result is dense and quadratic in the number of profiles, so this expects well- or
        perturbation-level profiles rather than single cells.
    """
    values = representation(adata, use_rep)
    adata.obsp[key_added] = similarity_matrix(values, metric)
    return None


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

    An older readout, largely replaced by mAP. It thresholds rather than ranks, and it
    depends on the number of replicates per perturbation. It is included because many
    published results report it.

    Returns:
        ``None``, or the modified copy. Writes a per-group table to
        ``uns["mantispy"][key_added]`` and a summary dict to
        ``uns["mantispy"][key_added + "_summary"]``.
    """
    matrix = similarity_matrix(representation(adata, use_rep), metric).astype(np.float64)
    codes, keys = group_codes(adata, groupby)
    generator = np.random.default_rng(seed)

    rows, columns = np.triu_indices(adata.n_obs, k=1)
    non_replicate = matrix[rows, columns][codes[rows] != codes[columns]]

    records = []
    for group, key in enumerate(keys):
        members = np.flatnonzero(codes == group)
        if members.size < 2:
            continue
        pair_rows, pair_columns = np.triu_indices(members.size, k=1)
        observed = float(np.median(matrix[members[pair_rows], members[pair_columns]]))
        # Each null draw takes as many non-replicate pairs as the group has replicate pairs.
        draws = generator.choice(non_replicate, size=(null_size, pair_rows.size), replace=True)
        null_threshold = float(np.quantile(np.median(draws, axis=1), quantile))
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

    For each profile, the similarities to its replicates are z-scored against its
    similarities to the control profiles and averaged. A perturbation's grit is the mean
    over its replicates.
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
