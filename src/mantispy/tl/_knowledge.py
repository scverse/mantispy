"""Gene-set tools for genetic screens.

In a CRISPR or ORF screen each perturbation is a gene, so gene-set resources apply to morphological profiles.
The enrichment used with feature annotations tests pathways when given a pathway network.

The sets come from OmniPath through decoupler, in the same ``source``/``target``/``weight`` format that :func:`~mantispy.tl.feature_sets` produces.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from anndata import AnnData

from mantispy._core._reduce import representation
from mantispy._core._stats import benjamini_hochberg, permutation_pvalue
from mantispy._core.frames import as_frame
from mantispy._core.logging import get_logger
from mantispy._core.mutation import inplace_or_copy
from mantispy.tl._similarity import similarity_matrix

SOURCES = ("hallmark", "progeny", "collectri", "dorothea")


def gene_sets(source: str = "hallmark", organism: str = "human") -> pd.DataFrame:
    """Fetch a gene-set network, or read one from a GMT file.

    Args:
        source: One of ``SOURCES``, or a path ending in ``.gmt``. ``"hallmark"`` (50 broad programs) is a good default for a morphological screen.
        organism: Passed to the OmniPath resource.

    Returns:
        A frame with ``source``, ``target`` and ``weight``.
        Unweighted resources get ``weight = 1.0``.

    Raises:
        ValueError: ``source`` is neither one of ``SOURCES`` nor a path to a ``.gmt`` file.

    Notes:
        Named resources are downloaded on first use and cached by decoupler.
    """
    import decoupler as dc

    if str(source).endswith(".gmt"):
        net = dc.pp.read_gmt(str(source))
    elif source in SOURCES:
        net = getattr(dc.op, source)(organism=organism)
    else:
        raise ValueError(f"source must be one of {SOURCES} or a path to a .gmt file, got {source!r}")

    net = net.rename(columns={column: column.lower() for column in net.columns})
    if "weight" not in net:
        net["weight"] = 1.0
    get_logger().info("gene_sets(%s): %d sets over %d edges", source, net["source"].nunique(), len(net))
    return net[["source", "target", "weight"]]


@inplace_or_copy(expects="perturbation")
def pathway_coherence(
    adata: AnnData,
    net: pd.DataFrame,
    gene_key: str = "Metadata_Gene",
    metric: str = "cosine",
    use_rep: str | None = None,
    min_genes: int = 5,
    n_permutations: int = 1000,
    seed: int = 0,
    key_added: str = "pathway_coherence",
    copy: bool = False,
) -> AnnData | None:
    """Score how similar the profiles of each gene set's genes are.

    For each set, the mean pairwise similarity among the profiles of its genes is compared with random sets of the same number of profiles.
    This checks whether a genetic screen recovers known biology.

    Args:
        adata: One profile per gene, normally the output of :func:`~mantispy.tl.consensus`.
        net: A gene-set network from :func:`gene_sets`, or any frame with ``source`` and ``target``.
        gene_key: ``obs`` column holding the gene symbol.
        metric: Similarity between profiles, ``"cosine"`` or ``"pearson"``.
        use_rep: Measure in ``obsm[use_rep]`` instead of ``X``.
        min_genes: Sets with fewer of their genes present in the screen are skipped.
        n_permutations: Number of random sets in the null.
        seed: Seed for drawing the random sets.
        key_added: Name for the output table.
        copy: Return a modified copy instead of mutating in place.

    Returns:
        ``None``, or the modified copy.
        Writes ``uns["mantispy"][key_added]`` with ``set``, ``n_genes``, ``coherence``, ``pvalue`` and ``qvalue``, sorted by coherence.

    Raises:
        KeyError: ``obs`` has no column ``gene_key``.

    Notes:
        Rank sets by coherence rather than by p-value.
        With 1000 permutations every strongly coherent set reaches the p-value floor of 1/1001 and ties there.
        The p-value shows whether a set is coherent, and the coherence orders the sets that are.
    """
    obs = as_frame(adata.obs)
    if gene_key not in obs:
        raise KeyError(f"obs has no column {gene_key!r} holding the gene symbol")

    values = representation(adata, use_rep)
    matrix = similarity_matrix(values, metric).astype(np.float64)
    np.fill_diagonal(matrix, np.nan)

    genes = obs[gene_key].astype(str).to_numpy()
    positions: dict[str, list[int]] = {}
    for index, gene in enumerate(genes):
        positions.setdefault(gene, []).append(index)

    generator = np.random.default_rng(seed)
    records = []
    for name, block in net.groupby("source", observed=True):
        rows = [index for gene in set(block["target"].astype(str)) for index in positions.get(gene, [])]
        if len(set(genes[rows])) < min_genes:
            continue
        observed = float(np.nanmean(matrix[np.ix_(rows, rows)]))
        draws = np.array(
            [
                np.nanmean(matrix[np.ix_(draw, draw)])
                for draw in (
                    generator.choice(adata.n_obs, size=len(rows), replace=False) for _ in range(n_permutations)
                )
            ]
        )
        records.append(
            {
                "set": str(name),
                "n_genes": int(len(set(genes[rows]))),
                "coherence": observed,
                "pvalue": float(permutation_pvalue(np.array([observed]), draws[None, :])[0]),
            }
        )

    table = pd.DataFrame(records, columns=["set", "n_genes", "coherence", "pvalue"])
    table["qvalue"] = benjamini_hochberg(table["pvalue"].to_numpy()) if len(table) else []
    table = table.sort_values("coherence", ascending=False).reset_index(drop=True)
    adata.uns.setdefault("mantispy", {})[key_added] = table
    get_logger().info("pathway_coherence scored %d set(s) of %d", len(table), net["source"].nunique())
    return None


@inplace_or_copy(expects="perturbation")
def enrich_hits(
    adata: AnnData,
    net: pd.DataFrame,
    gene_key: str = "Metadata_Gene",
    hit_key: str = "hits_qvalue",
    threshold: float = 0.05,
    key_added: str = "enrich_hits",
    copy: bool = False,
) -> AnnData | None:
    """Test which gene sets are over-represented among the hits.

    Args:
        adata: One profile per gene, already scored by :func:`~mantispy.tl.hit_calling`.
        net: A gene-set network from :func:`gene_sets`.
        gene_key: ``obs`` column holding the gene symbol.
        hit_key: ``obs`` column holding the q-value that defines a hit.
        threshold: q-value below which a gene counts as a hit.
        key_added: Name for the output table.
        copy: Return a modified copy instead of mutating in place.

    Returns:
        ``None``, or the modified copy.
        Writes ``uns["mantispy"][key_added]`` with ``set``, ``odds_ratio`` (log, Haldane-Anscombe corrected) and ``qvalue``, the Benjamini-Hochberg adjusted p-value of decoupler's Fisher test.

    Raises:
        KeyError: ``obs`` has no ``gene_key`` or no ``hit_key``.
        ValueError: Every gene or no gene is a hit at ``threshold``, so one side of the comparison is empty.

    Notes:
        Over-representation is coarser than :func:`pathway_coherence`.
        It counts hits per set and ignores whether their phenotypes resemble each other.
        Sets that share many genes, such as the cell-cycle programs, are reported together.

        The background is the set of genes measured in this screen.
        When almost every gene is a hit (190 of 193 on rohban2017), the result mostly reflects the library's composition.
        In that case tighten ``threshold``, or rank by phenotype strength instead.
    """
    import decoupler as dc

    obs = as_frame(adata.obs)
    for column in (gene_key, hit_key):
        if column not in obs:
            raise KeyError(
                f"obs has no column {column!r}"
                + ("; run mt.tl.hit_calling first, which writes obs['hits_qvalue']" if column == hit_key else "")
            )

    genes = obs[gene_key].astype(str).to_numpy()
    hits = obs[hit_key].to_numpy(dtype=float) < threshold
    membership = (
        pd.DataFrame([np.where(hits, 1.0, 0.0)], index=pd.Index(["hits"]), columns=pd.Index(genes))
        .T.groupby(level=0)
        .max()
        .T
    )

    # Count hits per distinct gene, as n_bg does; `hits` still has one entry per row.
    n_bg = int(len(set(genes)))
    n_hits = int(membership.to_numpy().sum())
    if n_hits in (0, n_bg):
        raise ValueError(
            f"{n_hits} of {n_bg} genes are hits at threshold={threshold}, so either the hits or the non-hits "
            "are empty and there is nothing to compare. Loosen the threshold if nothing was called, tighten "
            "it if everything was, or rank by phenotype strength instead."
        )
    # ORA returns log odds ratios and BH-adjusted q-values.
    # The background is the screen's own genes rather than decoupler's genome-wide default.
    # ORA keeps genes ranked above n_up, so n_up = n_bg - n_hits selects the hits and reproduces Fisher's exact test.
    # empty=False keeps sets without hits, so depletion is reported.
    scores, qvalues = dc.mt.ora(membership, net, tmin=1, n_bg=n_bg, n_up=n_bg - n_hits, empty=False, verbose=False)
    table = pd.DataFrame(
        {
            "set": list(scores.columns),
            "odds_ratio": np.asarray(scores, dtype=float).ravel(),
            "qvalue": np.asarray(qvalues, dtype=float).ravel(),
        }
    ).sort_values("qvalue")

    adata.uns.setdefault("mantispy", {})[key_added] = table.reset_index(drop=True)
    get_logger().info("enrich_hits: %d hit gene(s) of %d", n_hits, n_bg)
    return None
