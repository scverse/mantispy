from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from anndata import AnnData

from mantispy._core._stats import benjamini_hochberg
from mantispy._core.frames import as_frame
from mantispy._core.logging import get_logger
from mantispy._core.mutation import inplace_or_copy


@inplace_or_copy()
def ora(
    adata: AnnData,
    groupby: str = "cluster",
    net: pd.DataFrame | None = None,
    gene_key: str = "Metadata_Gene",
    source: str = "source",
    target: str = "target",
    tmin: int = 5,
    key_added: str = "ora",
    copy: bool = False,
    *,
    padj_by: Literal["all", "group"] = "all",
) -> AnnData | None:
    """Test each group's genes for over-representation of gene sets.

    Args:
        adata: Object whose ``obs`` carries the grouping and the gene symbols, for example after :func:`~mantispy.tl.cluster`.
        groupby: ``obs`` column defining the groups whose genes are tested.
        net: A gene-set network with ``source`` and ``target`` columns, such as :func:`mantispy.ds.gene_sets` returns.
        gene_key: ``obs`` column holding the gene symbol.
        source: Column of ``net`` naming the set.
            Renamed to ``source`` internally.
        target: Column of ``net`` naming the gene.
            Renamed to ``target`` internally.
        tmin: Smallest number of a set's genes that must be in the universe for the set to be tested.
        key_added: Name for the output table.
        copy: Return a modified copy instead of mutating in place.
        padj_by: Scope of the Benjamini-Hochberg correction.
            ``"all"`` (default) corrects once across every group and set.
            ``"group"`` corrects within each group's tests, so a group's modest enrichment is not penalized by unrelated groups (use this when many groups are tested at once).
            Because the scope is chosen per call, q-values from an ``"all"`` run and a ``"group"`` run are not directly comparable, so keep one scope within a single comparison.

    Returns:
        ``None``, or the modified copy.
        Writes ``uns["mantispy"][key_added]`` with ``group``, ``source`` (the set), ``n`` (the group's genes in that set), ``odds_ratio`` (the Haldane-Anscombe log odds ratio), ``pvalue`` (a two-tailed Fisher exact test) and ``qvalue`` (Benjamini-Hochberg corrected), sorted by q.

    Raises:
        KeyError: ``obs`` has no ``groupby`` or no ``gene_key``.
        ValueError: ``net`` is not given, or its ``source``/``target`` columns are missing.

    Notes:
        The universe is the set of distinct genes in ``obs[gene_key]``, so a set is tested only on its genes that the screen measured, and sets with fewer than ``tmin`` measured genes are skipped.
        Each group and set is tested with a two-tailed Fisher exact test over that universe.
    """
    from scipy.stats import fisher_exact

    if padj_by not in ("all", "group"):
        raise ValueError(f"padj_by must be 'all' or 'group', got {padj_by!r}")
    obs = as_frame(adata.obs)
    for column in (groupby, gene_key):
        if column not in obs:
            raise KeyError(f"obs has no column {column!r}")
    if net is None:
        raise ValueError("net is required; pass a gene-set network, e.g. mt.ds.gene_sets('hallmark')")
    network = net.rename(columns={source: "source", target: "target"})
    if not {"source", "target"} <= set(network.columns):
        raise ValueError(f"net must have columns {source!r} and {target!r}")

    gene_names = obs[gene_key].astype(str).to_numpy()
    present = obs[gene_key].notna().to_numpy() & (gene_names != "")
    in_universe = set(gene_names[present])
    n_bg = len(in_universe)

    network = network.astype({"source": str, "target": str})
    network = network[network["target"].isin(in_universe)]
    set_genes = {name: set(block["target"]) for name, block in network.groupby("source", observed=True)}
    set_genes = {name: targets for name, targets in set_genes.items() if len(targets) >= tmin}

    group_labels = obs[groupby].astype(str).to_numpy()
    groups = pd.unique(group_labels)
    records = []
    for group in groups:
        members = set(gene_names[(group_labels == group) & present]) & in_universe
        k = len(members)
        if k == 0 or k == n_bg:
            continue
        for name, targets in set_genes.items():
            a = len(members & targets)
            n_s = len(targets)
            # 2x2: rows member/non-member genes, columns in-set/out-of-set, over the measured universe.
            b, c = k - a, n_s - a
            d = n_bg - k - c
            records.append(
                {
                    "group": group,
                    "source": str(name),
                    "n": a,
                    # Haldane-Anscombe log odds ratio: +0.5 per cell keeps it finite when a cell is zero.
                    "odds_ratio": float(np.log(((a + 0.5) * (d + 0.5)) / ((b + 0.5) * (c + 0.5)))),
                    "pvalue": float(fisher_exact([[a, b], [c, d]], alternative="two-sided")[1]),
                }
            )

    table = pd.DataFrame(records, columns=["group", "source", "n", "odds_ratio", "pvalue"])
    if padj_by == "group":
        table["qvalue"] = table.groupby("group", observed=True, sort=False)["pvalue"].transform(
            lambda p: benjamini_hochberg(p.to_numpy())
        )
    else:
        table["qvalue"] = benjamini_hochberg(table["pvalue"].to_numpy())
    table = table.sort_values("qvalue").reset_index(drop=True)
    adata.uns.setdefault("mantispy", {})[key_added] = table
    get_logger().info("ora: %d test(s) over %d group(s)", len(table), len(groups))
    return None
