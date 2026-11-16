"""Group control guides into pseudo-genes, to check how often a hit caller calls a gene with no phenotype."""

from __future__ import annotations

import numpy as np
import pandas as pd
from anndata import AnnData

from mantispy._core.frames import as_frame
from mantispy._core.logging import get_logger
from mantispy._core.masks import reference_mask
from mantispy._core.mutation import inplace_or_copy


def guide_codes(obs: pd.DataFrame, groupby: str, replicate: str) -> tuple[np.ndarray, pd.DataFrame]:
    """One integer per row naming its guide, and the guides in code order.

    A guide is a ``(groupby, replicate)`` pair compared as strings, so two control genes that share a guide label keep two guides, and an integer guide column groups like its text.
    A row missing either value has no guide, code ``-1``.
    The guides are sorted by group, then by replicate.
    """
    parts, labels = [], []
    for column in (groupby, replicate):
        codes, uniques = pd.factorize(obs[column])
        text_codes, text = pd.factorize(pd.Index(uniques).astype(str), sort=True)
        parts.append(np.where(codes >= 0, text_codes[np.maximum(codes, 0)], -1).astype(np.int64))
        labels.append(np.asarray(text, dtype=object))
    gene, guide = parts
    n_labels = max(len(labels[1]), 1)
    has = (gene >= 0) & (guide >= 0)
    keys, inverse = np.unique(gene[has] * n_labels + guide[has], return_inverse=True)
    codes = np.full(len(obs), -1, dtype=np.int64)
    codes[has] = inverse
    table = pd.DataFrame({groupby: labels[0][keys // n_labels], replicate: labels[1][keys % n_labels]})
    return codes, table


@inplace_or_copy()
def pseudogenes(
    adata: AnnData,
    groupby: str = "Metadata_Gene",
    replicate: str = "Metadata_sgRNA",
    size: int | None = None,
    reference: str = "negcon",
    match_cells: bool = True,
    seed: int = 0,
    key_added: str = "pseudogene",
    copy: bool = False,
) -> AnnData | None:
    """Group control guides into pseudo-genes, so a hit caller can test them like real genes.

    A pseudo-gene is made of control guides, so it has no phenotype, and a hit caller that calls it has made a false call.
    The control guides are sorted, shuffled with ``seed`` and cut into groups of ``size``; the guides left over, fewer than ``size``, join no pseudo-gene.
    A guide is a ``groupby`` and ``replicate`` pair, so two control genes that share a guide label give two guides.

    A pseudo-gene matches a real gene in guides per gene, through ``size``, and in cells per guide, through ``match_cells``.
    Each pseudo-gene guide draws a cell count from the real guides' cell counts and keeps a random subset of its cells of that size, drawn without replacement.
    A test reads the precision of a guide's median from its cell count, so a pseudo-gene guide with many more cells than a real guide would be tested at another precision than the genes it stands for.
    A guide with fewer cells than its draw keeps all of them and is never upsampled; a warning gives how many guides fall short, and by how many cells.

    The control label in ``groupby`` is left as it is, so every other function sees the controls it saw before.
    Pass ``pseudogenes=key_added`` to :func:`~mantispy.tl.bootstrap_features`.
    It tests each pseudo-gene, on its kept cells, against every control guide but its own; every control cell, kept or not, stays in the null of every other group, so the real genes come out exactly as without pseudo-genes.

    This is modelled on Brieflow's ``create_pseudogene_groups``, with some differences.
    Brieflow keeps the leftover guides as a last, smaller pseudo-gene, groups each control category apart where this pools every reference guide, and does not match cells per guide.
    It also keeps a pseudo-gene's own guides in its null, which makes the pseudo-gene look like its null and understates the rate.

    Args:
        adata: Object with one row per cell.
        groupby: ``obs`` column naming the gene each guide targets.
        replicate: ``obs`` column naming the guide.
            Rows with a missing gene or guide are left out of every pseudo-gene.
        size: Guides per pseudo-gene.
            ``None`` uses the median number of guides per gene outside the reference, rounded half up.
        reference: The control cells: ``"negcon"``, or the name of a boolean ``obs`` column.
        match_cells: Subsample each pseudo-gene guide to a cell count drawn from the real guides'.
            ``False`` keeps every cell of each pseudo-gene guide.
        seed: Seed for the shuffle, the cell counts and the subsets.
        key_added: Name of the new ``obs`` column.
        copy: Return a modified copy instead of mutating in place.

    Returns:
        ``None``, or the modified copy.
        Writes ``obs[key_added]``, a categorical column that names the pseudo-gene, ``"pseudogene_01"``, ``"pseudogene_02"`` and so on, on each kept cell of a pseudo-gene guide, and is missing on every other row.
        Also writes ``uns["mantispy"]["pseudogenes"][key_added]``, the ``groupby`` and ``replicate`` columns that define a guide, which :func:`~mantispy.tl.bootstrap_features` reads to leave a pseudo-gene's own guides out of its null.

    Raises:
        KeyError: ``groupby`` or ``replicate`` is not an ``obs`` column.
        ValueError: ``size`` is below one, ``size=None`` or ``match_cells`` finds no guide outside the reference to match, there are fewer control guides than ``size``, or one pseudo-gene would hold every control guide and so leave itself no null.

    Examples:
        Count how often the bootstrap calls a pseudo-gene at the real genes' cutoff:

        >>> import mantispy as mt
        >>> mt.pp.pseudogenes(adata)  # doctest: +SKIP
        >>> mt.tl.bootstrap_features(adata, pseudogenes="pseudogene")  # doctest: +SKIP
        >>> adata.uns["mantispy"]["bootstrap_summary"]["rates"]  # doctest: +SKIP

        Calibrate a per-gene score against the pseudo-genes:

        >>> table = adata.uns["mantispy"]["bootstrap"]  # doctest: +SKIP
        >>> score = -np.log10(table["pvalue"].clip(lower=1e-300)).groupby(table["group"]).sum()  # doctest: +SKIP
        >>> genes = ad.AnnData(
        ...     obs=pd.DataFrame({"gene": score.index, "score": score.to_numpy()}, index=score.index)
        ... )  # doctest: +SKIP
        >>> labels = list(adata.obs["pseudogene"].cat.categories)  # doctest: +SKIP
        >>> mt.tl.empirical_fdr(genes, control_genes=labels, score="score", group="gene")  # doctest: +SKIP
    """
    obs = as_frame(adata.obs)
    for name, column in (("groupby", groupby), ("replicate", replicate)):
        if column not in obs:
            raise KeyError(f"{name}={column!r} is not an obs column")
    if size is not None and size < 1:
        raise ValueError(f"size must be at least 1, got {size}")

    is_control = reference_mask(adata, reference)
    guide, keys = guide_codes(obs, groupby, replicate)
    real = ~is_control & (guide >= 0)
    real_cells = np.bincount(guide[real], minlength=len(keys))
    if (size is None or match_cells) and not real_cells.any():
        raise ValueError(
            "no guide lies outside the reference to match; pass size= and match_cells=False to group the controls "
            "without matching a real gene"
        )
    if size is None:
        per_gene = keys[real_cells > 0].groupby(groupby).size()
        size = max(1, int(np.floor(float(per_gene.median()) + 0.5)))
        get_logger().info("pseudogenes: size=%d, the median number of guides per gene", size)
    real_cells = real_cells[real_cells > 0]

    control_guides = np.unique(guide[is_control & (guide >= 0)])
    n_groups = control_guides.size // size
    if n_groups == 0:
        raise ValueError(f"there are {control_guides.size} control guide(s), fewer than size={size}")
    if n_groups == 1 and control_guides.size == size:
        raise ValueError(
            f"size={size} puts all {size} control guides in one pseudo-gene, which leaves its null no control guide"
        )

    rng = np.random.default_rng(seed)
    chosen = rng.permutation(control_guides)[: n_groups * size]
    position = np.full(len(keys), -1, dtype=np.int64)
    position[chosen] = np.arange(chosen.size)

    # The chosen guides' control rows, guide by guide in the shuffled order, each guide's in row order.
    rows = np.flatnonzero(is_control & (guide >= 0))
    rows = rows[position[guide[rows]] >= 0]
    rows = rows[np.argsort(position[guide[rows]], kind="stable")]
    available = np.bincount(position[guide[rows]], minlength=chosen.size)
    if match_cells:
        target = rng.choice(real_cells, size=chosen.size)
        kept = np.minimum(available, target)
        short = available < target
        if short.any():
            get_logger().warning(
                "pseudogenes: %d of %d pseudo-gene guide(s) have fewer control cells than the real guide cell count "
                "drawn for them, %d cell(s) short in all, and keep every cell; the median pseudo-gene guide keeps %g "
                "cells, the median real guide has %g",
                int(short.sum()),
                chosen.size,
                int((target - available)[short].sum()),
                float(np.median(kept)),
                float(np.median(real_cells)),
            )
        starts = np.cumsum(available) - available
        take = np.concatenate(
            [start + rng.choice(n, size=k, replace=False) for n, k, start in zip(available, kept, starts, strict=True)]
        ).astype(np.int64)
        rows = rows[take]

    names = [f"pseudogene_{index + 1:0{max(2, len(str(n_groups)))}d}" for index in range(n_groups)]
    codes = np.full(adata.n_obs, -1, dtype=np.int64)
    codes[rows] = position[guide[rows]] // size
    adata.obs[key_added] = pd.Categorical.from_codes(codes, categories=pd.Index(names))
    adata.uns.setdefault("mantispy", {}).setdefault("pseudogenes", {})[key_added] = {
        "groupby": groupby,
        "replicate": replicate,
    }
    return None
