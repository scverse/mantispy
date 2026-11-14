"""Group control guides into pseudo-genes, to check how often a hit caller calls a gene with no phenotype."""

from __future__ import annotations

import numpy as np
import pandas as pd
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
    size: int | None = None,
    reference: str = "negcon",
    seed: int = 0,
    key_added: str | None = None,
    copy: bool = False,
) -> AnnData | None:
    """Relabel random groups of control guides as pseudo-genes, so a hit caller tests them like real genes.

    A pseudo-gene is made of control guides, so it has no phenotype, and a hit caller that calls it has made a false call.
    The control guides are sorted, shuffled with ``seed`` and cut into groups of ``size``; the guides left over, fewer than ``size``, stay plain controls.

    This is modelled on Brieflow's ``create_pseudogene_groups``, with some differences.
    Brieflow keeps the leftover guides as a last, smaller pseudo-gene, and groups each control category apart, where this pools every reference guide.
    The shuffle and the labels differ, so the groups do too.
    Brieflow also keeps a pseudo-gene's own guides in its null, which makes the pseudo-gene look like its null and understates the rate; :func:`~mantispy.tl.bootstrap_features` leaves them out, so its rate comes out higher on the same screen.

    Pass ``groupby=key_added`` to :func:`~mantispy.tl.bootstrap_features`, with the same ``reference``.
    It tests each pseudo-gene against every control guide but its own, and keeps the pseudo-gene guides in the null of every other group, so the real genes come out exactly as without pseudo-genes.
    It reports two rates at the p-value cutoff Benjamini-Hochberg sets for the real genes: the per-test rate, the share of pseudo-gene features called, and the per-gene rate, the share of pseudo-genes with any feature called.
    :func:`~mantispy.tl.guide_activity` and :func:`~mantispy.tl.aggregate_guides` find their controls by label, so a relabelled guide leaves their null, and they are not calibrated by this function; they raise when given its column with the label it thinned.
    :func:`~mantispy.tl.empirical_fdr` can take the pseudo-genes as its null instead, given one row per group of a score from that run and ``control_genes=adata.uns["mantispy"]["pseudogenes"][key_added]`` (see the examples).

    Args:
        adata: Object with one row per cell or per guide.
        groupby: ``obs`` column naming the gene each guide targets.
        replicate: ``obs`` column naming the guide; the guides are what is grouped.
            Rows with a missing guide are left out of every pseudo-gene.
        size: Guides per pseudo-gene.
            ``None`` uses the median number of guides per gene outside the reference, so a pseudo-gene matches a real gene.
        reference: The control cells: ``"negcon"``, or the name of a boolean ``obs`` column.
        seed: Seed for the shuffle.
        key_added: Name of the new gene column; ``None`` writes ``groupby + "_pseudo"``.
        copy: Return a modified copy instead of mutating in place.

    Returns:
        ``None``, or the modified copy.
        Writes ``obs[key_added]``, a copy of ``obs[groupby]`` as strings, missing values kept missing, in which the guides of each pseudo-gene read ``"pseudogene_01"``, ``"pseudogene_02"`` and so on.
        Also writes ``uns["mantispy"]["pseudogenes"][key_added]``, the list of those labels, which is how :func:`~mantispy.tl.bootstrap_features` tells a pseudo-gene from a control label.

    Raises:
        KeyError: ``groupby`` or ``replicate`` is not an ``obs`` column.
        ValueError: ``size`` is below one, ``size=None`` finds no gene outside the reference to match, or there are fewer control guides than ``size``.

    Examples:
        Count how often the bootstrap calls a pseudo-gene at the real genes' cutoff:

        >>> import mantispy as mt
        >>> mt.pp.pseudogenes(adata)  # doctest: +SKIP
        >>> mt.tl.bootstrap_features(adata, groupby="Metadata_Gene_pseudo")  # doctest: +SKIP
        >>> table = adata.uns["mantispy"]["bootstrap"]  # doctest: +SKIP
        >>> real, pseudo = table[~table["is_pseudogene"]], table[table["is_pseudogene"]]  # doctest: +SKIP
        >>> cutoff = real.loc[real["qvalue"] < 0.05, "pvalue"].max()  # doctest: +SKIP
        >>> called = pseudo["pvalue"] <= cutoff  # doctest: +SKIP
        >>> called.mean()  # the per-test rate  # doctest: +SKIP
        >>> called.groupby(pseudo["group"]).any().mean()  # the per-gene rate  # doctest: +SKIP

        Calibrate a per-gene score against the pseudo-genes:

        >>> score = -np.log10(table["pvalue"].clip(lower=1e-300)).groupby(table["group"]).sum()  # doctest: +SKIP
        >>> genes = ad.AnnData(
        ...     obs=pd.DataFrame({"gene": score.index, "score": score.to_numpy()}, index=score.index)
        ... )  # doctest: +SKIP
        >>> labels = adata.uns["mantispy"]["pseudogenes"]["Metadata_Gene_pseudo"]  # doctest: +SKIP
        >>> mt.tl.empirical_fdr(genes, control_genes=labels, score="score", group="gene")  # doctest: +SKIP
    """
    obs = as_frame(adata.obs)
    for name, column in (("groupby", groupby), ("replicate", replicate)):
        if column not in obs:
            raise KeyError(f"{name}={column!r} is not an obs column")
    key_added = f"{groupby}_pseudo" if key_added is None else key_added

    is_control = reference_mask(adata, reference)
    # Guides are compared as strings, so an integer or categorical guide column gives the same groups as its text.
    has_guide = obs[replicate].notna().to_numpy()
    guides = np.where(has_guide, obs[replicate].astype(str).to_numpy(), "")
    labels = np.array(obs[groupby], dtype=object)
    has_gene = pd.notna(labels)
    labels[has_gene] = labels[has_gene].astype(str)

    if size is None:
        genes = ~is_control & has_gene & has_guide
        if not genes.any():
            raise ValueError(
                "size=None matches the guides per gene, but no gene lies outside the reference; pass size="
            )
        per_gene = pd.Series(guides[genes]).groupby(labels[genes]).nunique()
        size = max(1, round(float(per_gene.median())))
        get_logger().info("pseudogenes: size=%d, the median number of guides per gene", size)
    if size < 1:
        raise ValueError(f"size must be at least 1, got {size}")

    control_guides = np.unique(guides[is_control & has_guide])
    n_groups = control_guides.size // size
    if n_groups == 0:
        raise ValueError(f"there are {control_guides.size} control guide(s), fewer than size={size}")

    shuffled = np.random.default_rng(seed).permutation(control_guides)[: n_groups * size]
    width = max(2, len(str(n_groups)))
    names = np.array([f"pseudogene_{index + 1:0{width}d}" for index in range(n_groups)])
    label_of_guide = dict(zip(shuffled, np.repeat(names, size), strict=True))

    in_pseudo = is_control & has_guide & np.isin(guides, shuffled)
    labels[in_pseudo] = [label_of_guide[guide] for guide in guides[in_pseudo]]
    adata.obs[key_added] = labels
    adata.uns.setdefault("mantispy", {}).setdefault("pseudogenes", {})[key_added] = [str(name) for name in names]
    return None


def check_control_label(adata: AnnData, column: str, label: str) -> None:
    """Raise if ``column`` is a pseudo-gene column and ``label`` the control label it left only the leftover guides.

    For callers that find their controls by label and read one row per guide.
    """
    recorded = adata.uns.get("mantispy", {}).get("pseudogenes", {}).get(column)
    if not recorded:
        return
    counts = as_frame(adata.obs)[column].astype(str).value_counts()
    smallest = min(int(counts.get(str(name), 0)) for name in recorded)
    kept = int(counts.get(str(label), 0))
    if kept < smallest:
        raise ValueError(
            f"{column!r} is a pseudo-gene column from mt.pp.pseudogenes, which leaves {label!r} {kept} guide(s), "
            f"fewer than the {smallest} of a pseudo-gene; pass the gene column it was made from"
        )
