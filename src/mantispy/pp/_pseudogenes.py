"""Group control guides into pseudo-genes, to check how often a hit caller calls a gene with no phenotype."""

from __future__ import annotations

import numpy as np
from anndata import AnnData

from mantispy._core.frames import as_frame
from mantispy._core.masks import reference_mask
from mantispy._core.mutation import inplace_or_copy


@inplace_or_copy()
def pseudogenes(
    adata: AnnData,
    groupby: str = "Metadata_Gene",
    replicate: str = "Metadata_sgRNA",
    size: int = 4,
    reference: str = "negcon",
    seed: int = 0,
    key_added: str | None = None,
    copy: bool = False,
) -> AnnData | None:
    """Relabel random groups of control guides as pseudo-genes, so a hit caller tests them like real genes.

    A pseudo-gene is made of control guides, so it has no phenotype, and the share of pseudo-genes a hit caller calls is its false positive rate on this screen.
    This is Brieflow's ``create_pseudogene_groups``.
    The control guides are sorted, shuffled with ``seed`` and cut into groups of ``size``; the guides left over, fewer than ``size``, stay plain controls.

    Pass ``groupby=key_added`` to :func:`~mantispy.tl.bootstrap_features`, with the same ``reference``.
    It tests each pseudo-gene against every control guide but its own, and keeps the pseudo-gene guides in the null of every other group, so the real genes come out exactly as without pseudo-genes.
    :func:`~mantispy.tl.guide_activity`, :func:`~mantispy.tl.aggregate_guides` and :func:`~mantispy.tl.empirical_fdr` find their controls by label, so a relabelled guide leaves their null; calibrate them on a subset of the control guides instead.

    Args:
        adata: Object with one row per cell or per guide.
        groupby: ``obs`` column naming the gene each guide targets.
        replicate: ``obs`` column naming the guide; the guides are what is grouped.
        size: Guides per pseudo-gene, best matched to the guides per real gene.
        reference: The control cells: ``"negcon"``, or the name of a boolean ``obs`` column.
        seed: Seed for the shuffle.
        key_added: Name of the new gene column; ``None`` writes ``groupby + "_pseudo"``.
        copy: Return a modified copy instead of mutating in place.

    Returns:
        ``None``, or the modified copy.
        Writes ``obs[key_added]``, a copy of ``obs[groupby]`` as strings in which the guides of each pseudo-gene read ``"pseudogene_01"``, ``"pseudogene_02"`` and so on.
        Also writes ``uns["mantispy"]["pseudogenes"][key_added]``, the list of those labels, which is how :func:`~mantispy.tl.bootstrap_features` tells a pseudo-gene from a control label.

    Raises:
        ValueError: ``size`` is below one, or there are fewer control guides than ``size``.

    Examples:
        Check that the bootstrap calls about 5% of pseudo-genes at ``p < 0.05``:

        >>> import mantispy as mt
        >>> mt.pp.pseudogenes(adata, size=4)  # doctest: +SKIP
        >>> mt.tl.bootstrap_features(adata, groupby="Metadata_Gene_pseudo")  # doctest: +SKIP
        >>> table = adata.uns["mantispy"]["bootstrap"]  # doctest: +SKIP
        >>> pseudo = table["group"].str.startswith("pseudogene_")  # doctest: +SKIP
        >>> (table.loc[pseudo, "pvalue"] < 0.05).mean()  # doctest: +SKIP
    """
    if size < 1:
        raise ValueError(f"size must be at least 1, got {size}")
    key_added = f"{groupby}_pseudo" if key_added is None else key_added

    obs = as_frame(adata.obs)
    is_control = reference_mask(adata, reference)
    # Guides are compared as strings, so an integer or categorical guide column gives the same groups as its text.
    guides = obs[replicate].astype(str).to_numpy()
    control_guides = np.unique(guides[is_control])
    n_groups = control_guides.size // size
    if n_groups == 0:
        raise ValueError(f"there are {control_guides.size} control guide(s), fewer than size={size}")

    shuffled = np.random.default_rng(seed).permutation(control_guides)[: n_groups * size]
    width = max(2, len(str(n_groups)))
    names = np.array([f"pseudogene_{index + 1:0{width}d}" for index in range(n_groups)])
    label_of_guide = dict(zip(shuffled, np.repeat(names, size), strict=True))

    labels = obs[groupby].astype(str).to_numpy(dtype=object)
    in_pseudo = is_control & np.isin(guides, shuffled)
    labels[in_pseudo] = [label_of_guide[guide] for guide in guides[in_pseudo]]
    adata.obs[key_added] = labels
    adata.uns.setdefault("mantispy", {}).setdefault("pseudogenes", {})[key_added] = [str(name) for name in names]
    return None
