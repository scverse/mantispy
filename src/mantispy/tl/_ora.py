"""Over-representation of prior-knowledge gene sets among the genes of each group.

Given a grouping (typically the clusters :func:`~mantispy.tl.cluster` writes), each group's member genes are
tested against a gene-set network with an over-representation test, so a cluster can be labelled by the
pathways or complexes its genes fall into. This is the discrete counterpart to :func:`~mantispy.tl.enrich`,
which scores continuous profiles.

The test is run with :func:`decoupler.mt.ora`, and the universe is the set of genes measured in the object
(the perturbed set), not every gene in the network, so enrichment is judged against what the screen could have
found.
"""

from __future__ import annotations

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
) -> AnnData | None:
    """Test each group's genes for over-representation of gene sets.

    Args:
        adata: Object whose ``obs`` carries the grouping and the gene symbols, for example after :func:`~mantispy.tl.cluster`.
        groupby: ``obs`` column defining the groups whose genes are tested.
        net: A gene-set network with ``source`` and ``target`` columns, such as :func:`mantispy.ds.gene_sets` returns.
        gene_key: ``obs`` column holding the gene symbol.
        source: Column of ``net`` naming the set. Renamed to ``source`` internally.
        target: Column of ``net`` naming the gene. Renamed to ``target`` internally.
        tmin: Smallest number of a set's genes that must be in the universe for the set to be tested, passed to decoupler.
        key_added: Name for the output table.
        copy: Return a modified copy instead of mutating in place.

    Returns:
        ``None``, or the modified copy.
        Writes ``uns["mantispy"][key_added]`` with ``group``, ``source`` (the set), ``n`` (the group's genes in
        that set), ``odds_ratio`` (decoupler's Haldane-Anscombe log odds ratio), ``pvalue`` (a two-tailed
        Fisher exact test) and ``qvalue`` (Benjamini-Hochberg across every tested group and set), sorted by q.

    Raises:
        KeyError: ``obs`` has no ``groupby`` or no ``gene_key``.
        ValueError: ``net`` is not given, or its ``source``/``target`` columns are missing.

    Notes:
        The universe is the set of distinct genes in ``obs[gene_key]``, so a set is tested only on its genes
        that the screen measured. decoupler's ORA returns p-values already Benjamini-Hochberg adjusted across a
        group's sets, so the raw two-tailed Fisher p is recomputed here from the same contingency table and one
        Benjamini-Hochberg correction is then applied across the whole table, keeping ``pvalue`` and ``qvalue``
        on one consistent footing.
    """
    import decoupler as dc
    from scipy.stats import fisher_exact

    obs = as_frame(adata.obs)
    for column in (groupby, gene_key):
        if column not in obs:
            raise KeyError(f"obs has no column {column!r}")
    if net is None:
        raise ValueError("net is required; pass a gene-set network, e.g. mt.ds.gene_sets('hallmark')")
    network = net.rename(columns={source: "source", target: "target"})
    if not {"source", "target"} <= set(network.columns):
        raise ValueError(f"net must have columns {source!r} and {target!r}")

    genes = obs[gene_key].dropna().astype(str)
    universe = sorted(set(genes))
    n_bg = len(universe)
    in_universe = set(universe)

    # Restrict the net to measured genes so the universe is the perturbed set, then index each set's genes.
    network = network.astype({"source": str, "target": str})
    network = network[network["target"].isin(in_universe)]
    set_genes = {name: set(block["target"]) for name, block in network.groupby("source", observed=True)}

    group_labels = obs[groupby].astype(str)
    records = []
    for group in pd.unique(group_labels):
        members = sorted(set(genes[group_labels.to_numpy() == group]) & in_universe)
        k = len(members)
        if k == 0 or k == n_bg:
            continue
        # A single row over the universe, members ranked on top; decoupler keeps features ranked above n_up,
        # so n_up = n_bg - k selects exactly the k member genes (see tl/_enrich.py::_ora_n_up).
        row = pd.DataFrame([np.isin(universe, members).astype(float)], index=[group], columns=universe)
        es, _ = dc.mt.ora(row, network, tmin=tmin, n_up=n_bg - k, n_bg=n_bg, empty=False, verbose=False)
        member_set = set(members)
        for name in es.columns:
            targets = set_genes.get(name, set())
            a = len(member_set & targets)
            n_s = len(targets)
            # 2x2: rows are member/non-member genes, columns in-set/out-of-set, over the measured universe.
            table = [[a, k - a], [n_s - a, n_bg - k - (n_s - a)]]
            records.append(
                {
                    "group": group,
                    "source": str(name),
                    "n": a,
                    "odds_ratio": float(es[name].iloc[0]),
                    "pvalue": float(fisher_exact(table, alternative="two-sided")[1]),
                }
            )

    table = pd.DataFrame(records, columns=["group", "source", "n", "odds_ratio", "pvalue"])
    table["qvalue"] = benjamini_hochberg(table["pvalue"].to_numpy()) if len(table) else []
    table = table.sort_values("qvalue").reset_index(drop=True)
    adata.uns.setdefault("mantispy", {})[key_added] = table
    get_logger().info("ora: %d test(s) over %d group(s)", len(table), group_labels.nunique())
    return None
