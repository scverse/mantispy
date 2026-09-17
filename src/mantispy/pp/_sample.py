"""Stratified subsampling.

A screen of a million cells by four thousand features is 16 GB of float32 before any transform allocates its output.
Exploring on a representative sample and confirming on the full data keeps memory use manageable.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from anndata import AnnData

from mantispy._core._reduce import group_codes
from mantispy._core.frames import as_frame
from mantispy._core.logging import get_logger
from mantispy._core.provenance import record_params


def downsample(
    adata: AnnData,
    n_per_group: int = 500,
    groupby: Sequence[str] | str | None = ("Metadata_Plate", "Metadata_Well"),
    stratify: str | None = None,
    seed: int = 0,
) -> AnnData:
    """Return at most ``n_per_group`` rows from each group.

    Args:
        adata: Object to sample from. Never modified.
        n_per_group: Cap per group. Groups smaller than this are kept whole, so groups are capped but not balanced.
        groupby: Columns defining a group. The default caps each well, so every well is represented instead of the densest wells filling the sample.
        stratify: Keep this column's proportions inside each group, so a rare perturbation is not lost to the sampling.
        seed: Seed for reproducibility.

    Returns:
        A new object holding the sampled rows in their original order, with the call recorded in ``uns["mantispy"]["params"]``.

    Raises:
        ValueError: If ``n_per_group`` is below 1.
    """
    if n_per_group < 1:
        raise ValueError(f"n_per_group must be at least 1, got {n_per_group}")

    codes, keys = group_codes(adata, groupby)
    generator = np.random.default_rng(seed)
    labels = as_frame(adata.obs)[stratify].astype(str).to_numpy() if stratify else None
    chosen: list[np.ndarray] = []

    for index in range(len(keys)):
        rows = np.flatnonzero(codes == index)
        if rows.size <= n_per_group:
            chosen.append(rows)
        elif labels is None:
            chosen.append(generator.choice(rows, size=n_per_group, replace=False))
        else:
            inside = labels[rows]
            picked = [
                generator.choice(
                    members := rows[inside == label],
                    size=min(max(1, round(n_per_group * members.size / rows.size)), members.size),
                    replace=False,
                )
                for label in np.unique(inside)
            ]
            # Rounding each share up to at least one can overshoot the cap. Trimming the
            # tail of the concatenation would always drop the same labels, so trim at
            # random instead.
            taken = np.concatenate(picked)
            chosen.append(taken if taken.size <= n_per_group else generator.choice(taken, n_per_group, replace=False))

    keep = np.sort(np.concatenate(chosen)) if chosen else np.array([], dtype=int)
    get_logger().info("downsample kept %d of %d rows in %d group(s)", keep.size, adata.n_obs, len(keys))

    result = adata[keep].copy()
    record_params(
        result,
        "downsample",
        {"n_per_group": n_per_group, "groupby": groupby, "stratify": stratify, "seed": seed},
    )
    return result
