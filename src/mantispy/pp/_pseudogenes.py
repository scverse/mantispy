"""Group control guides into pseudo-genes, to check how often a hit caller calls a gene with no phenotype."""

from __future__ import annotations

import numpy as np
from anndata import AnnData

from mantispy._core.frames import as_frame
from mantispy._core.logging import get_logger
from mantispy._core.masks import reference_mask
from mantispy._core.mutation import inplace_or_copy


@inplace_or_copy()
def pseudogenes(
    adata: AnnData,
    groupby: str = "Metadata_Gene",
    replicate: str = "Metadata_sgRNA",
    size: int = 4,
    n_groups: int | None = None,
    reference: str = "negcon",
    seed: int = 0,
    keep_in_reference: bool = False,
    key_added: str | None = None,
    copy: bool = False,
) -> AnnData | None:
    """Relabel random groups of control guides as pseudo-genes, so a hit caller tests them like real genes.

    A pseudo-gene is made of control guides, so it has no phenotype, and the share of pseudo-genes a hit caller calls is its false positive rate on this screen.
    This is Brieflow's ``create_pseudogene_groups``.
    The control guides are sorted, shuffled with ``seed``, and the first ``n_groups * size`` of them are cut into groups of ``size``; the rest stay plain controls and form the null the pseudo-genes are tested against.

    The output is a new gene column and a reference column to pass on to a hit caller.
    For :func:`~mantispy.tl.bootstrap_features`, pass ``groupby=key_added`` and ``reference=key_added + "_reference"``.
    For :func:`~mantispy.tl.guide_activity`, :func:`~mantispy.tl.aggregate_guides` and :func:`~mantispy.tl.empirical_fdr`, which find their controls by label, pass the new column as the gene column: the pseudo-genes carry their own labels and the leftover controls keep the control label.

    Args:
        adata: Object with one row per cell or per guide.
        groupby: ``obs`` column naming the gene each guide targets.
        replicate: ``obs`` column naming the guide; the guides are what is grouped.
        size: Guides per pseudo-gene, best matched to the guides per real gene.
        n_groups: How many pseudo-genes to make.
            ``None`` makes as many as half the control guides fill, so the other half is left for the null.
        reference: The control cells: ``"negcon"``, or the name of a boolean ``obs`` column.
        seed: Seed for the shuffle.
        keep_in_reference: Leave the pseudo-gene guides in the reference column too, as Brieflow does.
            By default they are taken out, so a pseudo-gene is not tested against a null that holds its own guides.
        key_added: Name of the new gene column; ``None`` writes ``groupby + "_pseudo"``.
        copy: Return a modified copy instead of mutating in place.

    Returns:
        ``None``, or the modified copy.
        Writes ``obs[key_added]``, a copy of ``obs[groupby]`` as strings in which the guides of each pseudo-gene read ``"pseudogene_01"``, ``"pseudogene_02"`` and so on.
        Also writes ``obs[key_added + "_reference"]``, ``True`` for the control rows left to form the null.

    Raises:
        ValueError: ``size`` or ``n_groups`` is below one, or there are too few control guides to fill the groups.

    Examples:
        Check that the bootstrap calls about 5% of pseudo-genes at ``p < 0.05``:

        >>> import mantispy as mt
        >>> mt.pp.pseudogenes(adata, size=4)  # doctest: +SKIP
        >>> mt.tl.bootstrap_features(  # doctest: +SKIP
        ...     adata, groupby="Metadata_Gene_pseudo", reference="Metadata_Gene_pseudo_reference"
        ... )
        >>> table = adata.uns["mantispy"]["bootstrap"]  # doctest: +SKIP
        >>> pseudo = table["group"].str.startswith("pseudogene_")  # doctest: +SKIP
        >>> (table.loc[pseudo, "pvalue"] < 0.05).mean()  # doctest: +SKIP
    """
    if size < 1:
        raise ValueError(f"size must be at least 1, got {size}")
    if n_groups is not None and n_groups < 1:
        raise ValueError(f"n_groups must be at least 1, got {n_groups}")
    key_added = f"{groupby}_pseudo" if key_added is None else key_added

    obs = as_frame(adata.obs)
    is_control = reference_mask(adata, reference)
    # Guides are compared as strings, so an integer or categorical guide column gives the same groups as its text.
    guides = obs[replicate].astype(str).to_numpy()
    control_guides = np.unique(guides[is_control])
    wanted = control_guides.size // (2 * size) if n_groups is None else n_groups
    if wanted == 0 or wanted * size > control_guides.size:
        raise ValueError(
            f"{control_guides.size} control guide(s) cannot fill {max(wanted, 1)} pseudo-gene(s) of size={size}"
            + (", with half of them left for the null" if n_groups is None else "")
        )
    n_groups = wanted

    shuffled = np.random.default_rng(seed).permutation(control_guides)[: n_groups * size]
    width = max(2, len(str(n_groups)))
    names = np.array([f"pseudogene_{index + 1:0{width}d}" for index in range(n_groups)])
    label_of_guide = dict(zip(shuffled, np.repeat(names, size), strict=True))

    labels = obs[groupby].astype(str).to_numpy(dtype=object)
    in_pseudo = is_control & np.isin(guides, shuffled)
    labels[in_pseudo] = [label_of_guide[guide] for guide in guides[in_pseudo]]
    in_reference = is_control if keep_in_reference else is_control & ~in_pseudo

    adata.obs[key_added] = labels
    adata.obs[f"{key_added}_reference"] = in_reference
    if not in_reference.any():
        get_logger().warning(
            "pseudogenes: every control guide is in a pseudo-gene, so %r selects no control row",
            f"{key_added}_reference",
            deep="lower size so some control guides are left over, or pass keep_in_reference=True",
        )
    return None
